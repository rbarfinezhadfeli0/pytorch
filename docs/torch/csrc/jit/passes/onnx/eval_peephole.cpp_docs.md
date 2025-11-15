# Documentation: `torch/csrc/jit/passes/onnx/eval_peephole.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/eval_peephole.cpp`
- **Size**: 5,156 bytes (5.04 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/eval_peephole.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/torch.h>

#include <c10/util/irange.h>
#include <algorithm>

namespace torch::jit {

namespace onnx {
using namespace ::c10::onnx;
}

static std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        continue;
      }
      inputTensorValues.push_back(itr->second.second.toTensor());
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      continue;
    }
  }
  return inputTensorValues;
}

// This pass fuses Conv and BatchNorm into Conv node
// Conv and BatchNorm can be fused only if inputs for BatchNorm node:
// scale, bias, mean and var are all tensors of same shape (C) and
// if the size of the first dimension (dim 0) is the same between Conv
// input weight and BatchNorm input scale.
static void fuseConvBatchNorm(Block* b, ValueToParamPairMap& valsToParamsMap) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseConvBatchNorm(child_block, valsToParamsMap);
    }
    if (it->kind() == onnx::Conv) {
      auto oldConv = *it;
      if (oldConv->outputs().at(0)->uses().size() != 1) {
        continue;
      }
      auto bnNode = oldConv->outputs().at(0)->uses()[0].user;
      if (bnNode->kind() != onnx::BatchNormalization) {
        continue;
      }

      if (oldConv->outputs().size() !=
          bnNode->outputs().size()) { // BN layer is not in eval mode
        continue;
      }

      auto epsilon = bnNode->f(attr::epsilon);
      auto convInputVals = getValues(oldConv, valsToParamsMap);
      if (convInputVals.empty() ||
          (oldConv->inputs().size() == 3 && convInputVals.size() != 2)) {
        continue;
      }

      auto bnInputVals = getValues(bnNode, valsToParamsMap);
      if (bnInputVals.size() != 4) {
        continue;
      }

      // See
      // https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
      auto bnScale = bnInputVals[0].clone();
      auto bnB = bnInputVals[1].clone();
      auto bnMean = bnInputVals[2].clone();
      auto bnVar = bnInputVals[3].clone();
      // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
      auto convW = convInputVals[0].clone();
      at::Tensor convB;

      if (!bnScale.is_floating_point() || !bnB.is_floating_point() ||
          !bnMean.is_floating_point() || !bnVar.is_floating_point() ||
          !convW.is_floating_point() || bnScale.dim() != 1 || bnB.dim() != 1 ||
          bnMean.dim() != 1 || bnVar.dim() != 1 ||
          !(bnScale.size(0) == bnB.size(0)) ||
          !(bnB.size(0) == bnMean.size(0)) ||
          !(bnMean.size(0) == bnVar.size(0)) || !(convW.dim() > 2) ||
          !(convW.size(0) == bnScale.size(0))) {
        continue;
      }

      bnVar = bnVar.add(epsilon);
      bnVar = bnVar.sqrt();
      bnScale = bnScale.div(bnVar);

      // Calculate weight
      for (const auto i : c10::irange(convW.size(0))) {
        convW[i] = convW[i].mul(bnScale[i]);
      }

      // Calculate bias
      if (oldConv->inputs().size() == 3) {
        convB = convInputVals[1].clone();
        convB = convB.sub(bnMean);
        convB = convB.mul(bnScale);
        convB = convB.add(bnB);
      } else {
        bnMean = bnMean.mul(bnScale);
        bnB = bnB.sub(bnMean);
        convB = bnB;
      }

      Node* newConv = b->owningGraph()->create(onnx::Conv, 1);
      newConv->outputs().at(0)->copyMetadata(bnNode->outputs().at(0));

      newConv->copyAttributes(*oldConv);
      newConv->insertBefore(bnNode);
      newConv->addInput(oldConv->inputs().at(0));
      newConv->copyMetadata(oldConv);

      auto newConvW = b->owningGraph()->addInput();
      valsToParamsMap.insert(
          {newConvW, std::make_pair(newConvW->debugName(), convW)});
      newConvW->inferTypeFrom(convW);
      newConv->addInput(newConvW);

      auto newConvB = b->owningGraph()->addInput();
      valsToParamsMap.insert(
          {newConvB, std::make_pair(newConvB->debugName(), convB)});
      newConvB->inferTypeFrom(convB);
      newConv->addInput(newConvB);

      bnNode->outputs().at(0)->replaceAllUsesWith(newConv->outputs().at(0));
      bnNode->destroy();
      it.destroyCurrent();
    }
  }
}

static void EvalPeepholeONNX(Block* b, ParamMap& paramsDict) {
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  fuseConvBatchNorm(b, valsToParamsMap);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

void EvalPeepholeONNX(std::shared_ptr<Graph>& g, ParamMap& paramsDict) {
  EvalPeepholeONNX(g->block(), paramsDict);
  GRAPH_DUMP("After EvalPeepholeONNX:", g);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `onnx`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/onnx/eval_peephole.h`
- `torch/csrc/jit/passes/onnx/helper.h`
- `torch/torch.h`
- `c10/util/irange.h`
- `algorithm`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/passes/onnx`):

- [`remove_inplace_ops_for_onnx.cpp_docs.md`](./remove_inplace_ops_for_onnx.cpp_docs.md)
- [`list_model_parameters.cpp_docs.md`](./list_model_parameters.cpp_docs.md)
- [`preprocess_for_onnx.h_docs.md`](./preprocess_for_onnx.h_docs.md)
- [`remove_inplace_ops_for_onnx.h_docs.md`](./remove_inplace_ops_for_onnx.h_docs.md)
- [`constant_fold.cpp_docs.md`](./constant_fold.cpp_docs.md)
- [`eliminate_unused_items.cpp_docs.md`](./eliminate_unused_items.cpp_docs.md)
- [`cast_all_constant_to_floating.h_docs.md`](./cast_all_constant_to_floating.h_docs.md)
- [`list_model_parameters.h_docs.md`](./list_model_parameters.h_docs.md)
- [`shape_type_inference.cpp_docs.md`](./shape_type_inference.cpp_docs.md)
- [`constant_map.cpp_docs.md`](./constant_map.cpp_docs.md)


## Cross-References

- **File Documentation**: `eval_peephole.cpp_docs.md`
- **Keyword Index**: `eval_peephole.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

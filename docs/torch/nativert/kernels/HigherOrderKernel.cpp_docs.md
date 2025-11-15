# Documentation: HigherOrderKernel.cpp

## File Metadata
- **Path**: `torch/nativert/kernels/HigherOrderKernel.cpp`
- **Size**: 4247 bytes
- **Lines**: 111
- **Extension**: .cpp
- **Type**: Regular file

## Original Source

```cpp
#include <torch/nativert/kernels/HigherOrderKernel.h>

#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

namespace torch::nativert {

HigherOrderKernel::HigherOrderKernel(
    const Node* node,
    std::vector<std::unique_ptr<GraphExecutorBase>> graphExecutors)
    : OpKernel(node), graphExecutors_(std::move(graphExecutors)) {
  static constexpr std::string_view prefix = "torch.ops.higher_order.";
  TORCH_CHECK(c10::starts_with(node->target(), prefix));
  auto opName = node->target().substr(prefix.size());
  if (opName == "cond") {
    opType_ = OpType::COND;
    // Checking torch.cond schema is as expected:
    // torch.cond(Tensor predicate, Graph graph1, Graph graph2, Tensor[] args)
    // -> Tensor[]
    TORCH_CHECK(node_->attributes().size() == 2);
    TORCH_CHECK(node_->inputs().size() == 2);
  } else if (opName == "while_loop") {
    opType_ = OpType::WHILE_LOOP;
    // Checking torch.while_loop schema is as expected:
    // torch.while_loop(Graph cond, Graph body, Tensor[] args, Tensor[]
    // additional) -> Tensor[]
    TORCH_CHECK(node_->attributes().size() == 2);
    TORCH_CHECK(node_->inputs().size() == 2);
  } else if (opName == "run_const_graph") {
    opType_ = OpType::RUN_CONST_GRAPH;
    // Checking torch.run_const_graph schema is as expected:
    // torch.run_const_graph(Graph graph, Tensor[] args) -> Tensor[]
    TORCH_CHECK(!node_->attributes().empty());
    TORCH_CHECK(node_->inputs().size() == 1);
  } else {
    TORCH_CHECK(false, "Unknown higher order op: ", opName);
  }
}

void HigherOrderKernel::computeInternal(ExecutionFrame& executionFrame) const {
  switch (opType_) {
    case OpType::COND: {
      auto inputs = executionFrame.getIValue(node_->inputs()[1].value->id())
                        .toList()
                        .vec();
      std::vector<c10::IValue> outputs;
      auto cond = executionFrame.getIValue(node_->inputs()[0].value->id());
      size_t branchIdx = 0;
      if (cond.isTensor()) {
        branchIdx = cond.toTensor().item().toBool() ? 0 : 1;
      } else if (cond.isBool()) {
        branchIdx = cond.toBool() ? 0 : 1;
      } else {
        TORCH_CHECK(false, "Unsupported type for cond predicate");
      }
      ExecutionFrame branchFrame(*std::get<std::unique_ptr<Graph>>(
          node_->attributes()[branchIdx].value));
      auto ret =
          graphExecutors_[branchIdx]->execute(branchFrame, std::move(inputs));
      for (size_t i = 0; i < ret.size(); i++) {
        executionFrame.setIValue(node_->outputs()[i]->id(), std::move(ret[i]));
      }
      break;
    }
    case OpType::WHILE_LOOP: {
      auto carriedVals =
          executionFrame.getIValue(node_->inputs()[0].value->id())
              .toList()
              .vec();
      auto additonalVals =
          executionFrame.getIValue(node_->inputs()[1].value->id())
              .toList()
              .vec();
      size_t numCarriedVals = carriedVals.size();
      ExecutionFrame condFrame(
          *std::get<std::unique_ptr<Graph>>(node_->attributes()[0].value));
      ExecutionFrame bodyFrame(
          *std::get<std::unique_ptr<Graph>>(node_->attributes()[1].value));
      while (true) {
        auto inputs = carriedVals;
        inputs.insert(inputs.end(), additonalVals.begin(), additonalVals.end());
        auto cond = graphExecutors_[0]->execute(condFrame, inputs);

        if (cond.at(0).isTensor() && !cond[0].toTensor().item().toBool()) {
          break;
        }
        if (cond.at(0).isBool() && !cond[0].toBool()) {
          break;
        }
        auto out = graphExecutors_[1]->execute(bodyFrame, std::move(inputs));
        TORCH_CHECK(out.size() == numCarriedVals);
        carriedVals = std::move(out);
      }
      for (size_t i = 0; i < carriedVals.size(); i++) {
        executionFrame.setIValue(
            node_->outputs()[i]->id(), std::move(carriedVals[i]));
      }
      break;
    }
    case OpType::RUN_CONST_GRAPH: {
      // run_const_graph op is a special case of higher order op which has
      // been executed during weights loading, therefore at runtime we can
      // just make this a no-op.
      break;
    }
    default:
      TORCH_CHECK(false, "Unknown higher order op");
  }
}

} // namespace torch::nativert

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough


## Key Components

The file contains 350 words across 111 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4247 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

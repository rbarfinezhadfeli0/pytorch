# Documentation: `docs/torch/nativert/kernels/HigherOrderKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/HigherOrderKernel.cpp_docs.md`
- **Size**: 6,727 bytes (6.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/HigherOrderKernel.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/HigherOrderKernel.cpp`
- **Size**: 4,247 bytes (4.15 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

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


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/kernels/HigherOrderKernel.h`
- `c10/util/Exception.h`
- `c10/util/string_view.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/nativert/kernels`):

- [`PrimKernelRegistry.cpp_docs.md`](./PrimKernelRegistry.cpp_docs.md)
- [`KernelRegistry.h_docs.md`](./KernelRegistry.h_docs.md)
- [`AutoFunctionalizeKernel.cpp_docs.md`](./AutoFunctionalizeKernel.cpp_docs.md)
- [`ETCallDelegateKernel.cpp_docs.md`](./ETCallDelegateKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `HigherOrderKernel.cpp_docs.md`
- **Keyword Index**: `HigherOrderKernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/kernels`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/nativert/kernels`):

- [`ETCallDelegateKernel.cpp_docs.md_docs.md`](./ETCallDelegateKernel.cpp_docs.md_docs.md)
- [`TritonKernel.h_kw.md_docs.md`](./TritonKernel.h_kw.md_docs.md)
- [`PrimKernelRegistry.cpp_kw.md_docs.md`](./PrimKernelRegistry.cpp_kw.md_docs.md)
- [`C10Kernel.h_kw.md_docs.md`](./C10Kernel.h_kw.md_docs.md)
- [`CallTorchBindKernel.cpp_docs.md_docs.md`](./CallTorchBindKernel.cpp_docs.md_docs.md)
- [`ETCallDelegateKernel.h_kw.md_docs.md`](./ETCallDelegateKernel.h_kw.md_docs.md)
- [`AutoFunctionalizeKernel.h_kw.md_docs.md`](./AutoFunctionalizeKernel.h_kw.md_docs.md)
- [`HigherOrderKernel.h_docs.md_docs.md`](./HigherOrderKernel.h_docs.md_docs.md)
- [`C10Kernel.h_docs.md_docs.md`](./C10Kernel.h_docs.md_docs.md)
- [`CallTorchBindKernel.h_kw.md_docs.md`](./CallTorchBindKernel.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `HigherOrderKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `HigherOrderKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

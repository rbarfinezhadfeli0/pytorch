# Documentation: `docs/torch/nativert/kernels/AutoFunctionalizeKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/AutoFunctionalizeKernel.cpp_docs.md`
- **Size**: 4,722 bytes (4.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/AutoFunctionalizeKernel.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/AutoFunctionalizeKernel.cpp`
- **Size**: 2,226 bytes (2.17 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/kernels/AutoFunctionalizeKernel.h>

#include <c10/util/Enumerate.h>
#include <c10/util/Exception.h>

namespace torch::nativert {

UnsafeAutoFunctionalizeKernel::UnsafeAutoFunctionalizeKernel(const Node* node)
    : OpKernel(node),
      op_(getOperatorForTarget(
          std::get<std::string>(node->attributes()[0].value))),
      schema_(op_.schema()),
      arguments_(prefillStackWithStaticArgs(node, schema_)),
      numOutputs_(static_cast<int>(schema_.returns().size())) {
  for (const auto& [idx, schemaArg] : c10::enumerate(schema_.arguments())) {
    if (schemaArg.alias_info() != nullptr &&
        schemaArg.alias_info()->isWrite()) {
      mutatingInputArgs_.push_back(node->getInput(schemaArg.name()).value);
    }
  }
}

void UnsafeAutoFunctionalizeKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  // Make a copy of the stack
  std::vector<c10::IValue> stack = arguments_.getStackWithStaticArgs();

  fillDynamicInputs(executionFrame, arguments_, stack);

  // Call the op with the prepared stack.
  try {
    op_.callBoxed(stack);
  } catch (const std::exception& ex) {
    // TODO: this eats the original exception type. ATen returns different
    // exception types that correspond to different Python errors (e.g.
    // IndexError, ValueError). If retaining this information is important
    // to us, we'll have to change this up a little.
    auto stackTrace = node_->getMetadata("stack_trace");
    TORCH_CHECK(
        false,
        "Oringinal Python stacktrace:\n",
        stackTrace ? *stackTrace : "<no stack trace>",
        "\n",
        ex.what())
  }

  const auto& outputValues = node_->outputs();

  for (int i = 0; i < numOutputs_; ++i) {
    executionFrame.setIValue(outputValues[i]->id(), std::move(stack.at(i)));
  }

  // Copy over mutating inputs to outputs
  int mutatingArgStartIndex = (numOutputs_ == 0) ? 1 : numOutputs_;
  for (size_t i = mutatingArgStartIndex; i < outputValues.size(); ++i) {
    executionFrame.setIValue(
        outputValues[i]->id(),
        executionFrame.getIValue(
            mutatingInputArgs_.at(i - mutatingArgStartIndex)->id(),
            true /*  allowNone */));
  }
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

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

- `torch/nativert/kernels/AutoFunctionalizeKernel.h`
- `c10/util/Enumerate.h`
- `c10/util/Exception.h`


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
- [`ETCallDelegateKernel.cpp_docs.md`](./ETCallDelegateKernel.cpp_docs.md)
- [`HigherOrderKernel.cpp_docs.md`](./HigherOrderKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `AutoFunctionalizeKernel.cpp_docs.md`
- **Keyword Index**: `AutoFunctionalizeKernel.cpp_kw.md`
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

- **File Documentation**: `AutoFunctionalizeKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `AutoFunctionalizeKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

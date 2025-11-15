# Documentation: `docs/torch/nativert/kernels/C10Kernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/C10Kernel.cpp_docs.md`
- **Size**: 11,041 bytes (10.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/C10Kernel.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/C10Kernel.cpp`
- **Size**: 8,380 bytes (8.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/kernels/C10Kernel.h>

#include <fmt/ostream.h>

#include <c10/util/Enumerate.h>
#include <c10/util/Exception.h>

#ifdef __SIGRID_USE_GPU__
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#endif

namespace torch::nativert {

C10Kernel::C10Kernel(
    const Node* node,
    OpKernelKind kind,
    AliasingSpec&& aliasingSpec)
    : OpKernel(node, kind),
      op_(getOperatorForTarget(node->target(), node)),
      schema_(op_.schema(), std::move(aliasingSpec), kind_),
      arguments_(prefillStackWithStaticArgs(node, op_.schema())) {}

void C10Kernel::computeInternal(ExecutionFrame& executionFrame) const {
  // Make a copy of the stack
  std::vector<c10::IValue> stack = arguments_.getStackWithStaticArgs();

  fillDynamicInputs(executionFrame, arguments_, stack);

  // Call the op with the prepared stack.
  try {
    op_.callBoxed(stack);
  } catch (const std::exception& ex) {
    auto stackTrace = node_->getMetadata("stack_trace");
    TORCH_CHECK(
        false,
        "Exception while executing node: ",
        *node_,
        "\n"
        "with args:\n",
        readableArgs(op_.schema(), stack),
        "\n",
        ex.what(),
        "\n",
        "Original Python stacktrace:\n",
        stackTrace ? *stackTrace : "<no stack trace>")
  }

  // Write out results
  // TODO: we store intermediates in a single table (symint and tensor alike).
  // This can theoretically lead to name collisions, although based on how
  // these are named I don't think it will ever happen in practice. We need to
  // enforce it though.
  const auto& outputValues = node_->outputs();
  TORCH_CHECK(
      outputValues.size() == stack.size(),
      "Output size mismatch for ",
      node_->toString());
  for (auto&& [i, actualOutput] : c10::enumerate(stack)) {
    executionFrame.setIValue(outputValues[i]->id(), std::move(actualOutput));
  }
}

namespace {
std::unordered_map<std::string, c10::IValue> getSymInputs(
    const ExecutionFrame& executionFrame,
    const Node& node) {
  std::unordered_map<std::string, c10::IValue> inputs;
  for (const auto& input : node.inputs()) {
    const auto& val = executionFrame.getIValue(input.value->id());
    if (val.isInt() || val.isDouble() || val.isBool()) {
      inputs[input.name] = val;
    } else {
      TORCH_CHECK(false, "unsupported type for symbolic input");
    }
  }
  for (const auto& attribute : node.attributes()) {
    if (std::holds_alternative<int64_t>(attribute.value)) {
      inputs[attribute.name] = std::get<int64_t>(attribute.value);
    } else if (std::holds_alternative<double>(attribute.value)) {
      inputs[attribute.name] = std::get<double>(attribute.value);
    } else if (std::holds_alternative<bool>(attribute.value)) {
      inputs[attribute.name] = std::get<bool>(attribute.value);
    } else {
      TORCH_CHECK(false, "unsupported type for symbolic input");
    }
  }
  return inputs;
}

template <typename T>
void computeScalarBinaryOp(
    ExecutionFrame& executionFrame,
    const Node& node,
    std::enable_if_t<true, T> a,
    std::enable_if_t<true, T> b) {
  std::string_view target = node.target();
  T out;

  if (target == "_operator.add") {
    out = a + b;
  } else if (target == "_operator.sub") {
    out = a - b;
  } else if (target == "_operator.mul") {
    out = a * b;
  } else if (target == "_operator.pow") {
    out = std::pow(a, b);
  } else {
    TORCH_CHECK(false, "unsupported operator for scalar binary op: ", target);
  }

  executionFrame.setIValue(node.outputs()[0]->id(), out);
  VLOG(2) << fmt::format(
      "Completed executing node: {} with a={}, b={}, out={}",
      fmt::streamed(node),
      a,
      b,
      out);
}

} // namespace

void ScalarBinaryOpKernel::computeInternal(
    ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  const auto& a = inputs.at("a");
  const auto& b = inputs.at("b");

  auto coerceToDouble = [](const c10::IValue& x) -> double {
    if (x.isInt()) {
      return static_cast<double>(x.toInt());
    } else if (x.isDouble()) {
      return x.toDouble();
    } else {
      TORCH_CHECK(false, "unsupported type for symbolic input");
    }
  };

  if (a.isInt() && b.isInt()) {
    computeScalarBinaryOp<int64_t>(
        executionFrame, *node_, a.toInt(), b.toInt());
  } else {
    computeScalarBinaryOp<double>(
        executionFrame, *node_, coerceToDouble(a), coerceToDouble(b));
  }
}

void SymIntOpKernel::computeInternal(ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  int64_t a = inputs.at("a").toInt();
  std::string_view target = node_->target();
  if (target == "torch.sym_float") {
    double out = static_cast<double>(a);
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
    VLOG(2) << fmt::format(
        "Completed executing node: {} with a={}, out={}",
        fmt::streamed(*node_),
        a,
        out);
    return;
  }
  int64_t b = inputs.at("b").toInt();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t out;

  if (target == "_operator.floordiv") {
    out = a / b;
  } else if (target == "_operator.mod") {
    out = a % b;
  } else if (target == "torch.sym_max") {
    out = std::max(a, b);
  } else if (target == "torch.sym_min") {
    out = std::min(a, b);
  } else {
    TORCH_CHECK(false, "unsupported operator for SymInt: ", node_->target())
  }

  executionFrame.setIValue(node_->outputs()[0]->id(), out);
  VLOG(2) << fmt::format(
      "Completed executing node: {} with a={}, b={}, out={}",
      fmt::streamed(*node_),
      a,
      b,
      out);
}

void SymBoolOpKernel::computeInternal(ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool out;

  const std::string_view target = node_->target();
  if (target == "torch.sym_not") {
    bool a = inputs.at("a").toBool();
    out = !a;
  } else if (target == "_operator.ge") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a >= b;
  } else if (target == "_operator.le") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a <= b;
  } else if (target == "_operator.eq") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a == b;
  } else if (target == "_operator.gt") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a > b;
  } else if (target == "_operator.lt") {
    int64_t a = inputs.at("a").toInt();
    int64_t b = inputs.at("b").toInt();
    out = a < b;
  } else if (target == "_operator.and_") {
    bool a = inputs.at("a").toBool();
    bool b = inputs.at("b").toBool();
    out = a && b;
  } else {
    TORCH_CHECK(false, "unsupported operator for SymBool: ", node_->target())
  }

  executionFrame.setIValue(node_->outputs()[0]->id(), out);
}

void SymFloatOpKernel::computeInternal(ExecutionFrame& executionFrame) const {
  auto inputs = getSymInputs(executionFrame, *node_);

  const std::string_view target = node_->target();
  if (target == "math.trunc") {
    double x = inputs.at("x").toDouble();
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    int64_t out = trunc(x);
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else if (target == "torch._sym_sqrt") {
    double a = inputs.at("a").toDouble();
    double out = std::sqrt(a);
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else if (target == "_operator.neg") {
    auto a = inputs.at("a");
    c10::IValue out;
    if (a.isInt()) {
      out = -a.toInt();
    } else if (a.isDouble()) {
      out = -a.toDouble();
    } else {
      TORCH_CHECK(false, "unsupported type for symbolic input");
    }
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else if (target == "_operator.truediv") {
    auto ia = inputs.at("a");
    double a = ia.isInt() ? static_cast<double>(ia.toInt()) : ia.toDouble();
    auto ib = inputs.at("b");
    double b = ib.isInt() ? static_cast<double>(ib.toInt()) : ib.toDouble();
    double out = a / b;
    executionFrame.setIValue(node_->outputs()[0]->id(), out);
  } else {
    TORCH_CHECK(false, "unsupported operator for SymFloat: ", node_->target());
  }
}
} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/kernels/C10Kernel.h`
- `fmt/ostream.h`
- `c10/util/Enumerate.h`
- `c10/util/Exception.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/Exceptions.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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
- [`HigherOrderKernel.cpp_docs.md`](./HigherOrderKernel.cpp_docs.md)
- [`KernelHandlerRegistry.cpp_docs.md`](./KernelHandlerRegistry.cpp_docs.md)
- [`NativeKernels.cpp_docs.md`](./NativeKernels.cpp_docs.md)
- [`KernelFactory.cpp_docs.md`](./KernelFactory.cpp_docs.md)
- [`AutoFunctionalizeKernel.h_docs.md`](./AutoFunctionalizeKernel.h_docs.md)
- [`HigherOrderKernel.h_docs.md`](./HigherOrderKernel.h_docs.md)


## Cross-References

- **File Documentation**: `C10Kernel.cpp_docs.md`
- **Keyword Index**: `C10Kernel.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `C10Kernel.cpp_docs.md_docs.md`
- **Keyword Index**: `C10Kernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

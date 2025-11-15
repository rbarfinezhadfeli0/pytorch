# Documentation: `docs/torch/nativert/kernels/PrimKernelRegistry.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/kernels/PrimKernelRegistry.cpp_docs.md`
- **Size**: 7,202 bytes (7.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/kernels/PrimKernelRegistry.cpp`

## File Metadata

- **Path**: `torch/nativert/kernels/PrimKernelRegistry.cpp`
- **Size**: 4,412 bytes (4.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/record_function.h>

#include <ATen/CPUFunctions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/static/ops.h>

#include <c10/util/Enumerate.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>

namespace torch::nativert {

C10_DEFINE_REGISTRY(PrimKernelRegistry, OpKernel, const Node*)

namespace {

class OpKernel_prim_listpack : public OpKernel {
 public:
  explicit OpKernel_prim_listpack(const Node* node)
      : OpKernel(node, OpKernelKind::kPrimKernel) {
    auto listType = node->outputs()[0]->type();
    switch (listType.kind()) {
      case Type::Kind::TensorList:
        type_ = c10::TensorType::get();
        break;
      case Type::Kind::SymIntList:
        type_ = c10::IntType::get();
        break;
      case Type::Kind::OptionalTensorList:
        type_ = c10::OptionalType::create(c10::TensorType::get());
        break;
      default:
        TORCH_CHECK(false, "Unsupported list type: ", listType);
    }
  }

  void computeInternal(ExecutionFrame& executionFrame) const override final {
    RECORD_USER_SCOPE("nativert::OpKernel_prim_listpack");
    c10::List<c10::IValue> list(type_);
    list.reserve(numInputs());
    for (size_t i = 0; i < numInputs(); ++i) {
      if (KernelInput(i).isNone()) {
        list.emplace_back();
      } else {
        list.push_back(KernelInput(i));
      }
    }
    KernelOutput(0) = std::move(list);
  }

 private:
  c10::TypePtr type_;
};

} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.ListPack",
    OpKernel_prim_listpack)

REGISTER_PRIM_KERNEL("prim.ListUnpack", prim_listunpack, {
  RECORD_USER_SCOPE("nativert::OpKernel_prim_listunpack");
  auto inputListRef = KernelInput(0).toListRef();
  for (const auto& [i, ivalue] : c10::enumerate(inputListRef)) {
    KernelOutput(i) = ivalue;
  }
})

// Noop for input and output
REGISTER_PRIM_KERNEL("prim.Input", prim_input, {})
REGISTER_PRIM_KERNEL("prim.Output", prim_output, {})

namespace {

class OpKernel_variadic_concat : public OpKernel {
 public:
  explicit OpKernel_variadic_concat(const Node* node)
      : OpKernel(node, OpKernelKind::kPrimKernel) {
    dim_ = !node_->attributes().empty()
        ? constantToIValue(node_->getAttribute("dim").value).toInt()
        : 0;
  }
  void computeInternal(ExecutionFrame& executionFrame) const override final {
    {
      const size_t numNodeInps = numInputs();
      auto numCatInps = numNodeInps;
      auto dim = dim_;
      if (KernelInput(numCatInps - 1).isInt()) {
        dim = KernelInput(numCatInps - 1).toInt();
        numCatInps--;
      }
      std::vector<at::Tensor> inputs(numCatInps);
      for (const auto i : c10::irange(numCatInps)) {
        inputs[i] = KernelInput(i).toTensor();
      }

      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::cat(inputs, dim);
        return;
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::cat_outf(inputs, dim, out_t);
    }
  }

 private:
  int dim_;
};

} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.VarConcat",
    OpKernel_variadic_concat)

namespace {

class OpKernel_variadic_stack : public OpKernel {
 public:
  explicit OpKernel_variadic_stack(const Node* node)
      : OpKernel(node, OpKernelKind::kPrimKernel) {
    dim_ = !node_->attributes().empty()
        ? constantToIValue(node_->getAttribute("dim").value).toInt()
        : 0;
  }
  void computeInternal(ExecutionFrame& executionFrame) const override final {
    {
      const size_t numNodeInps = numInputs();
      auto numStackInps = numNodeInps;
      auto dim = dim_;
      if (KernelInput(numStackInps - 1).isInt()) {
        dim = KernelInput(numStackInps - 1).toInt();
        numStackInps--;
      }
      std::vector<at::Tensor> inputs(numStackInps);
      for (const auto i : c10::irange(numStackInps)) {
        inputs[i] = KernelInput(i).toTensor();
      }
      auto& out = KernelOutput(0);
      if (out.isNone()) {
        out = at::native::_stack_cpu(inputs, dim);
        return;
      }
      auto& out_t = out.toTensor();
      fastResizeToZero(out_t);
      at::native::_stack_out_cpu(inputs, dim, out_t);
    }
  }

 private:
  int64_t dim_;
};
} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.VarStack",
    OpKernel_variadic_stack)

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `C10_REGISTER_TYPED_CLASS`

**Classes/Structs**: `OpKernel_prim_listpack`, `OpKernel_variadic_concat`, `OpKernel_variadic_stack`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/kernels`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/record_function.h`
- `ATen/CPUFunctions.h`
- `c10/core/ScalarType.h`
- `c10/util/irange.h`
- `torch/csrc/jit/runtime/static/ops.h`
- `c10/util/Enumerate.h`
- `torch/nativert/kernels/PrimKernelRegistry.h`


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

Files in the same folder (`torch/nativert/kernels`):

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

- **File Documentation**: `PrimKernelRegistry.cpp_docs.md`
- **Keyword Index**: `PrimKernelRegistry.cpp_kw.md`
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

- May involve **JIT compilation** or compilation optimizations.
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

- **File Documentation**: `PrimKernelRegistry.cpp_docs.md_docs.md`
- **Keyword Index**: `PrimKernelRegistry.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

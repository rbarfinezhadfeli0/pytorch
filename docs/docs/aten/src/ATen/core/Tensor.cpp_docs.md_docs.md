# Documentation: `docs/aten/src/ATen/core/Tensor.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/Tensor.cpp_docs.md`
- **Size**: 8,084 bytes (7.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/Tensor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/core/Tensor.cpp`
- **Size**: 5,497 bytes (5.37 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/FunctionalTensorWrapper.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/MethodOperators.h>
#else
#include <ATen/ops/contiguous_ops.h>
#include <ATen/ops/fill_ops.h>
#include <ATen/ops/to_ops.h>
#include <ATen/ops/zero_ops.h>
#endif

#include <iostream>

namespace at {

const TensorBase& get_tensor_base(const Tensor &t) {
  return t;
}

TensorBase TensorBase::__dispatch_contiguous(c10::MemoryFormat memory_format) const {
  OptionalTensorRef self(*this);
  return at::_ops::contiguous::call(*self, memory_format);
}

const TensorBase& TensorBase::fill_(const c10::Scalar &fill_value) const {
  Tensor self(*this);
  at::_ops::fill__Scalar::call(self, fill_value);
  return *this;
}

const TensorBase& TensorBase::zero_() const {
  Tensor self(*this);
  at::_ops::zero_::call(self);
  return *this;
}

TensorBase TensorBase::to(
    at::TensorOptions options,
    bool non_blocking,
    bool copy,
    std::optional<at::MemoryFormat> memory_format) const {
  Tensor self(*this);
  return at::_ops::to_dtype_layout::call(
      self, optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(), options.device_opt(),
      options.pinned_memory_opt(), non_blocking, copy, memory_format);
}

void TensorBase::enforce_invariants() {
  TORCH_CHECK(
      impl_.get() != nullptr, "TensorImpl with nullptr is not supported");
  // Following line throws if the method is not a POD data type or is not
  // supported by ATen
  scalar_type();
  if (defined()) {
    TORCH_INTERNAL_ASSERT(
        impl_->dtype_initialized(),
        "Partially-initialized tensor not supported by Tensor");
    TORCH_INTERNAL_ASSERT(
        !impl_->is_sparse(),
        "Sparse Tensors are supported by Tensor, but invariant checking isn't implemented.  Please file a bug.");
    TORCH_INTERNAL_ASSERT(
        !impl_->has_storage() || impl_->is_meta() || impl_->storage_initialized(),
        "Partially-initialized tensor not supported by Tensor");
  }
}

void TensorBase::print() const {
  if (defined()) {
    std::cerr << "[" << toString() << " " << sizes() << "]" << '\n';
  } else {
    std::cerr << "[UndefinedTensor]" << '\n';
  }
}

std::string TensorBase::toString() const {
  std::string base_str;
  if (scalar_type() == ScalarType::Undefined) {
    base_str = "UndefinedType";
  } else {
    auto dispatchkey = options().computeDispatchKey();
    std::string dispatchkey_str;
    if (dispatchkey == c10::DispatchKey::PrivateUse1) {
      dispatchkey_str = c10::get_privateuse1_backend();
    } else if (dispatchkey == c10::DispatchKey::AutocastPrivateUse1) {
      dispatchkey_str = "Autocast" + c10::get_privateuse1_backend();
    } else if (dispatchkey == c10::DispatchKey::QuantizedPrivateUse1) {
      dispatchkey_str = "Quantized" + c10::get_privateuse1_backend();
    } else {
      dispatchkey_str = at::toString(dispatchkey);
    }
    base_str = dispatchkey_str + at::toString(scalar_type()) + "Type";
  }
  return base_str;
}

TensorBase TensorBase::variable_data() const {
  return impl::GetVariableHooks()->variable_data(*this);
}

TensorBase TensorBase::tensor_data() const {
  return impl::GetVariableHooks()->tensor_data(*this);
}

bool TensorBase::is_leaf() const {
  return impl::GetVariableHooks()->is_leaf(*this);
}

int64_t TensorBase::output_nr() const {
  return impl::GetVariableHooks()->output_nr(*this);
}

void TensorBase::set_data(const TensorBase & new_data) const {
  impl::GetVariableHooks()->set_data(*this, new_data);
}

TensorBase TensorBase::data() const {
  return impl::GetVariableHooks()->data(*this);
}

int64_t TensorBase::_version() const {
  return impl::GetVariableHooks()->_version(*this);
}

void TensorBase::retain_grad() const {
  impl::GetVariableHooks()->retain_grad(*this);
}

bool TensorBase::retains_grad() const {
  return impl::GetVariableHooks()->retains_grad(*this);
}

void Tensor::_backward(TensorList inputs,
        const std::optional<Tensor>& gradient,
        std::optional<bool> keep_graph,
        bool create_graph) const {
  impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
}

const TensorBase& TensorBase::requires_grad_(bool _requires_grad) const {
  impl::GetVariableHooks()->requires_grad_(*this, _requires_grad);
  return *this;
}

// View Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bool TensorBase::is_view() const {
  return impl::GetVariableHooks()->is_view(*this);
}

const TensorBase& TensorBase::_base() const {
  return impl::GetVariableHooks()->base(*this);
}

const std::string& TensorBase::name() const {
  return impl::GetVariableHooks()->name(*this);
}

const std::shared_ptr<torch::autograd::Node>& TensorBase::grad_fn() const {
  return impl::GetVariableHooks()->grad_fn(*this);
}

void TensorBase::remove_hook(unsigned pos) const {
  impl::GetVariableHooks()->remove_hook(*this, pos);
}

unsigned TensorBase::_register_hook(std::function<TensorBase(const TensorBase&)> hook) const {
  return impl::GetVariableHooks()->_register_hook(*this, std::move(hook));
}

std::optional<ScalarType> TensorBase::grad_dtype() const {
  return impl::GetVariableHooks()->grad_dtype(*this);
}

void TensorBase::set_grad_dtype(const std::optional<ScalarType>& grad_dtype) const {
  return impl::GetVariableHooks()->set_grad_dtype(*this, grad_dtype);
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/core/Formatting.h`
- `ATen/core/VariableHooksInterface.h`
- `ATen/core/LegacyTypeDispatch.h`
- `ATen/FunctionalTensorWrapper.h`
- `ATen/MethodOperators.h`
- `ATen/ops/contiguous_ops.h`
- `ATen/ops/fill_ops.h`
- `ATen/ops/to_ops.h`
- `ATen/ops/zero_ops.h`
- `iostream`


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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `Tensor.cpp_docs.md`
- **Keyword Index**: `Tensor.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Tensor.cpp_docs.md_docs.md`
- **Keyword Index**: `Tensor.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

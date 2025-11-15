# Documentation: `docs/aten/src/ATen/TensorSubclassLikeUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/TensorSubclassLikeUtils.h_docs.md`
- **Size**: 5,778 bytes (5.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/TensorSubclassLikeUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/TensorSubclassLikeUtils.h`
- **Size**: 3,230 bytes (3.15 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/equal.h>
#endif

namespace at {

// Note [Tensor-subclass-like Tensors]
// Tensor-subclass-like is defined as:
// - a Tensor subclass (via __torch_dispatch__ in Python or extending
//   TensorImpl in C++)
// - anything else that shares the same perils as Tensor subclasses.
//   For example, many Tensor subclasses do not have storage and meta Tensors
//   do not have storage either, so meta Tensors belong here.
//
// We should ensure that PyTorch internals supports Tensor-subclass-like
// objects. In particular, Tensor-subclass-like objects struggle with two
// classes of operations that are problematic for Tensor subclasses:
// 1. Because some Tensor subclasses do not have storage, .item() or
//    .data_ptr() calls are not good.
// 2. Certain in-place operations can eliminate the typing of the Tensor
//    subclass. For example:
//    >>> torch.zeros(input.sizes(), grad.options()).diag().copy_(input)
//    If input is a Tensor subclass, then the above ends up either erroring out
//    or returning a regular non-Tensor-subclass Tensor!

constexpr auto kFunctorchWrappedTensors = DispatchKeySet(
    {DispatchKey::FuncTorchGradWrapper,
     DispatchKey::FuncTorchBatched,
     DispatchKey::Functionalize});

constexpr auto kTensorSubclassLike =
    kFunctorchWrappedTensors |
    DispatchKeySet(
        {// WARNING: DO NOT put combined backend component + functionality keys
         // here, you will incorrectly always match on the functionality key
         // no matter the backend component
         DispatchKey::Batched,
         DispatchKey::Sparse,
         DispatchKey::SparseCsr,
         DispatchKey::Python}) |
    DispatchKeySet(BackendComponent::MetaBit);

inline bool isTensorSubclassLike(const Tensor& tensor) {
  if (c10::impl::dispatch_mode_enabled())
    return true;
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  return !(key_set & kTensorSubclassLike).empty();
}

inline bool areAnyTensorSubclassLike(TensorList tensors) {
  if (c10::impl::dispatch_mode_enabled())
    return true;
  return std::any_of(tensors.begin(), tensors.end(), isTensorSubclassLike);
}

inline bool areAnyOptionalTensorSubclassLike(
    const c10::List<std::optional<Tensor>>& tensors) {
  if (c10::impl::dispatch_mode_enabled())
    return true;
  return std::any_of(
      tensors.begin(),
      tensors.end(),
      [](const std::optional<Tensor>& opt_tensor) {
        return (
            opt_tensor.has_value() && isTensorSubclassLike(opt_tensor.value()));
      });
}

// Helper function to deal testing truthfulness of a scalar tensor
// in a Composite Compliant manner.
// NOTE: This function expects a scalar tensor of boolean dtype.
// Eg.
// Non-Composite Compliant Pattern : (t == 0).all().item<bool>()
// Composite Compliant Pattern : is_salar_tensor_true((t == 0).all())
inline bool is_scalar_tensor_true(const Tensor& t) {
  TORCH_INTERNAL_ASSERT(t.dim() == 0)
  TORCH_INTERNAL_ASSERT(t.scalar_type() == kBool)
  return at::equal(t, t.new_ones({}, t.options()));
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Tensor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/List.h`
- `ATen/core/Tensor.h`
- `c10/core/impl/TorchDispatchModeTLS.h`
- `ATen/Functions.h`
- `ATen/ops/equal.h`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `TensorSubclassLikeUtils.h_docs.md`
- **Keyword Index**: `TensorSubclassLikeUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `TensorSubclassLikeUtils.h_docs.md_docs.md`
- **Keyword Index**: `TensorSubclassLikeUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

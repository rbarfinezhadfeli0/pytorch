# Documentation: `docs/aten/src/ATen/EmptyTensor.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/EmptyTensor.h_docs.md`
- **Size**: 7,062 bytes (6.90 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/EmptyTensor.h`

## File Metadata

- **Path**: `aten/src/ATen/EmptyTensor.h`
- **Size**: 4,692 bytes (4.58 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

inline void check_size_nonnegative(ArrayRef<int64_t> size) {
  for (const auto& x : size) {
    TORCH_CHECK(
        x >= 0,
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

inline void check_size_nonnegative(ArrayRef<c10::SymInt> size) {
  for (const auto& x : size) {
    TORCH_SYM_CHECK(
        x.sym_ge(0),
        "Trying to create tensor with negative dimension ",
        x,
        ": ",
        size);
  }
}

TORCH_API size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize,
    size_t storage_offset = 0);
TORCH_API SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize,
    const SymInt& storage_offset = 0);
TORCH_API size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize,
    size_t storage_offset = 0);
TORCH_API SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize,
    const SymInt& storage_offset = 0);

TORCH_API TensorBase empty_generic(
    IntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API TensorBase empty_generic_symint(
    SymIntArrayRef size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API TensorBase empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type);

TORCH_API TensorBase empty_strided_symint_generic(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type);

TORCH_API TensorBase empty_cpu(
    IntArrayRef size,
    ScalarType dtype,
    bool pin_memory = false,
    std::optional<c10::MemoryFormat> memory_format_opt = std::nullopt);

TORCH_API TensorBase empty_cpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API TensorBase empty_cpu(IntArrayRef size, const TensorOptions& options);

TORCH_API TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    bool pin_memory = false);

TORCH_API TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

TORCH_API TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options);

TORCH_API TensorBase empty_meta(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<c10::MemoryFormat> memory_format_opt = std::nullopt);

TORCH_API TensorBase empty_meta(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API TensorBase empty_symint_meta(
    SymIntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_API TensorBase empty_meta(IntArrayRef size, const TensorOptions& options);

TORCH_API TensorBase
empty_strided_meta(IntArrayRef size, IntArrayRef stride, ScalarType dtype);

TORCH_API TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

TORCH_API TensorBase empty_strided_meta(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options);

TORCH_API TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    ScalarType dtype);

TORCH_API TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt);

TORCH_API TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,
    SymIntArrayRef stride,
    const TensorOptions& options);

} // namespace at::detail

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/TensorBase.h`


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

- **File Documentation**: `EmptyTensor.h_docs.md`
- **Keyword Index**: `EmptyTensor.h_kw.md`
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

- **File Documentation**: `EmptyTensor.h_docs.md_docs.md`
- **Keyword Index**: `EmptyTensor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

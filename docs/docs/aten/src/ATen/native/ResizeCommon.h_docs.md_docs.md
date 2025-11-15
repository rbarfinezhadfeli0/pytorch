# Documentation: `docs/aten/src/ATen/native/ResizeCommon.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/ResizeCommon.h_docs.md`
- **Size**: 4,935 bytes (4.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/ResizeCommon.h`

## File Metadata

- **Path**: `aten/src/ATen/native/ResizeCommon.h`
- **Size**: 2,491 bytes (2.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

template <typename T>
inline T storage_size_for(ArrayRef<T> size, ArrayRef<T> stride) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(size.size() == stride.size(),
      "storage_size_for(size, stride) requires that size and stride ",
      "have the same size as a precondition.");
  T storage_size = 1;
  for (const auto dim : c10::irange(size.size())) {
    if (size[dim] == 0) {
      storage_size = 0;
      break;
    }
    storage_size += (size[dim] - 1) * stride[dim];
  }
  return storage_size;
}

inline const Tensor& resize_named_tensor_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(self.has_names());
  TORCH_CHECK(
      self.sizes() == size,
      "Cannot resize named tensor with resize_ or resize_as_ (tried to resize "
      "Tensor",
      self.names(),
      " with size ",
      self.sizes(),
      " to ",
      size,
      "). This may be caused by passing a named tensor ",
      "as an `out=` argument; please ensure that the sizes are the same. ");
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "Unsupported memory format for named tensor resize ",
      optional_memory_format.value());
  return self;
}

// For deterministic output, fill new elements that were added after a storage
// resize with NaN or MAX_INT. `old_storage_nbytes` is the size of the storage
// before the resize happened.
inline const Tensor& fill_resize_deterministic_(const Tensor& tensor, int64_t old_storage_nbytes) {
  const at::Storage& storage = tensor.unsafeGetTensorImpl()->unsafe_storage();
  int64_t new_storage_nbytes = storage.nbytes();
  int64_t old_storage_numel = old_storage_nbytes / tensor.itemsize();
  int64_t new_storage_numel = new_storage_nbytes / tensor.itemsize();
  if (new_storage_numel > old_storage_numel) {
    at::Tensor tensor_view = at::empty({}, at::TensorOptions().dtype(tensor.scalar_type()).device(tensor.device()));
    tensor_view.set_(
      storage,
      /*storage_offset=*/old_storage_numel,
      /*size=*/{new_storage_numel - old_storage_numel},
      /*stride=*/{1});
    at::native::fill_empty_deterministic_(tensor_view);
  }
  return tensor;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/native/TensorFactories.h`
- `ATen/NamedTensorUtils.h`
- `c10/util/irange.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `ResizeCommon.h_docs.md`
- **Keyword Index**: `ResizeCommon.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ResizeCommon.h_docs.md_docs.md`
- **Keyword Index**: `ResizeCommon.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/TensorProperties.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/TensorProperties.cpp_docs.md`
- **Size**: 7,505 bytes (7.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/TensorProperties.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorProperties.cpp`
- **Size**: 4,321 bytes (4.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/TensorProperties.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nested_tensor_size_native.h>
#include <ATen/ops/contiguous_native.h>
#include <ATen/ops/cudnn_is_acceptable_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/is_same_size_native.h>
#include <ATen/ops/is_set_to_native.h>
#include <ATen/ops/size_native.h>
#include <ATen/ops/stride_native.h>
#include <ATen/ops/sym_is_contiguous_native.h>
#include <ATen/ops/sym_numel_native.h>
#include <ATen/ops/sym_size_native.h>
#include <ATen/ops/sym_storage_offset_native.h>
#include <ATen/ops/sym_stride_native.h>
#endif

#include <c10/util/irange.h>

namespace at::native {

bool is_same_size(const Tensor& self, const Tensor& other) {
  return self.sym_sizes().equals(other.sym_sizes());
}

bool nested_is_same_size(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.is_nested() && other.is_nested(),
      "Expected both self and other to be nested tensors. ",
      "Self ",
      self.is_nested() ? "is " : "is not ",
      "nested. While Other ",
      other.is_nested() ? "is " : "is not ",
      "nested.")
  const auto self_nt_size = _nested_tensor_size(self);
  const auto other_nt_size = _nested_tensor_size(other);
  return at::equal(self_nt_size, other_nt_size);
}
int64_t size(const Tensor& self, int64_t dim) {
  return self.size(dim);
}

int64_t stride(const Tensor& self, int64_t dim) {
  return self.stride(dim);
}

c10::SymInt sym_size(const Tensor& self, int64_t dim) {
  return self.sym_size(dim);
}

c10::SymBool sym_is_contiguous(
    const Tensor& self,
    c10::MemoryFormat memory_format) {
  return self.sym_is_contiguous(memory_format);
}

c10::SymInt sym_stride(const Tensor& self, int64_t dim) {
  return self.sym_stride(dim);
}

c10::SymInt sym_numel(const Tensor& self) {
  return self.sym_numel();
}

c10::SymInt sym_storage_offset(const Tensor& self) {
  return self.sym_storage_offset();
}

int64_t size(const Tensor& self, Dimname dim) {
  size_t pos_dim = dimname_to_position(self, dim);
  return self.sizes()[pos_dim];
}

int64_t stride(const Tensor& self, Dimname dim) {
  size_t pos_dim = dimname_to_position(self, dim);
  return self.strides()[pos_dim];
}

bool cudnn_is_acceptable(const TensorBase& self) {
  if (!globalContext().userEnabledCuDNN())
    return false;
  if (!self.is_cuda())
    return false;
  if (!detail::getCUDAHooks().compiledWithCuDNN())
    return false;
  // cuDNN functions like grid_sampler returns CUDNN_STATUS_BAD_PARAM on empty
  // tensors. Maybe some cuDNN functions actually support empty tensors, but
  // native/THNN kernels shouldn't be much slower because the output is also
  // likely empty.
  if (self.sym_numel() == 0)
    return false;
  // NB: In the old Python code, there was also a test to see if the
  // cuDNN library was actually dynamically linked or not.  I'm not
  // sure if we can actually test this.
  return true;
}

bool cudnn_is_acceptable(const Tensor& self) {
  return cudnn_is_acceptable(static_cast<const TensorBase&>(self));
}

Tensor& detach_(Tensor& self) {
  // this just exists to give us a hook in VariableType and an entry in
  // Declarations.yaml
  // TORCH_CHECK(false, "detach_ is not implemented for Tensor");
  return self;
}

Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous_or_false(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  return self.clone(memory_format);
}

bool is_set_to(const Tensor& self, const Tensor& src) {
  if (self.storage().unsafeGetStorageImpl() ==
          src.storage().unsafeGetStorageImpl() &&
      self.storage_offset() == src.storage_offset() &&
      self.dim() == src.dim()) {
    for (const auto d : c10::irange(self.dim())) {
      if (self.size(d) != src.size(d) || self.stride(d) != src.stride(d)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

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

- `ATen/Context.h`
- `ATen/NamedTensorUtils.h`
- `ATen/core/Tensor.h`
- `ATen/detail/CUDAHooksInterface.h`
- `ATen/native/TensorProperties.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_nested_tensor_size_native.h`
- `ATen/ops/contiguous_native.h`
- `ATen/ops/cudnn_is_acceptable_native.h`
- `ATen/ops/detach_native.h`
- `ATen/ops/equal.h`
- `ATen/ops/is_same_size_native.h`
- `ATen/ops/is_set_to_native.h`
- `ATen/ops/size_native.h`
- `ATen/ops/stride_native.h`
- `ATen/ops/sym_is_contiguous_native.h`
- `ATen/ops/sym_numel_native.h`
- `ATen/ops/sym_size_native.h`
- `ATen/ops/sym_storage_offset_native.h`
- `ATen/ops/sym_stride_native.h`
- `c10/util/irange.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `TensorProperties.cpp_docs.md`
- **Keyword Index**: `TensorProperties.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `TensorProperties.cpp_docs.md_docs.md`
- **Keyword Index**: `TensorProperties.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

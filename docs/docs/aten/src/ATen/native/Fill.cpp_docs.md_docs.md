# Documentation: `docs/aten/src/ATen/native/Fill.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Fill.cpp_docs.md`
- **Size**: 7,425 bytes (7.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Fill.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Fill.cpp`
- **Size**: 4,779 bytes (4.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// Functions that fill Tensors with constants.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/Fill.h>
#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/fill_diagonal_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zero_native.h>
#endif

namespace at::native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor& fill_out(Tensor& self, const Scalar& value) {
  if (self.device() == at::kCPU && self.numel() == 1) {
    return at::detail::scalar_fill(self, value);
  }
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
    .check_all_same_dtype(false)
    .add_output(self)
    .resize_outputs(false)
    .build();
  fill_stub(iter.device_type(), iter, value);
  return self;
}

static Tensor& fill_out_quantized(Tensor& self, const Scalar& value) {
  at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
  out = out.to(self.device()).to(self.suggest_memory_format());
  // Trust the `copy_` to handle the quantization and the boundary checks.
  self.copy_(out);
  return self;
}

Tensor& fill_(Tensor& self, const Scalar& value) {
  return fill_out(self, value);
}

Tensor& fill_quantized_(Tensor& self, const Scalar& value) {
  return fill_out_quantized(self, value);
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  if (self.device() != value.device()){
    return fill_out(self, value.item());
  }
  // Check if value is a view of self and if it is we clone
  // it to avoid overwriting self prematurely
  if(self.is_alias_of(value)) {
    self.copy_(value.clone());
  } else{
    self.copy_(value);
  }
  return self;
}

Tensor& fill_quantized_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return fill_out_quantized(self, value.item());
}

Tensor& fill_meta_(Tensor& self, const Scalar& value) {
  return self;
}

Tensor& fill_meta_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return self;
}

Tensor fill(const Tensor& self, const Scalar& value) {
  return at::empty_like(self).fill_(value);
}

Tensor fill(const Tensor& self, const Tensor& value) {
  return at::empty_like(self).fill_(value);
}

DEFINE_DISPATCH(fill_stub);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill_diagonal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& fill_diagonal_(Tensor& self, const Scalar& fill_value, bool wrap) {
  int64_t nDims = self.dim();
  TORCH_CHECK(nDims >= 2, "dimensions must larger than 1");

  auto height = self.sym_size(0);
  auto width = self.sym_size(1);

  if (nDims > 2) {
    for (const auto i : c10::irange(1, nDims)) {
      if (self.sym_size(i) != height) {
        TORCH_CHECK(false, "all dimensions of input must be of equal length");
      }
    }
  }

  auto storage_offset = self.sym_storage_offset();
  auto size = std::min(height, width);

  int64_t stride = 0;
  for (const auto i : c10::irange(nDims)) {
    stride += self.stride(i);
  }
  std::vector<SymInt> strides{stride};
  std::vector<SymInt> sizes{size};

  auto main_diag = self.as_strided_symint(sizes, strides, storage_offset);
  main_diag.fill_(fill_value);

  if (wrap && nDims == 2 && height > width + 1) {
    auto step = width + 1;
    auto wrap_size = ((self.numel() + step - 1) / step) - size;
    std::vector<SymInt> wrap_sizes{wrap_size};

    auto offset = self.stride(0) * (width + 1);

    auto wrap_diag = self.as_strided_symint(wrap_sizes, strides, storage_offset + offset);
    wrap_diag.fill_(fill_value);
  }

  return self;
}

static Tensor& zero_cpu_(Tensor &self, int64_t nelements) {
  void* ptr = self.data_ptr();
  if (nullptr == ptr) {
    return self.fill_(0);
  }
  auto size_bytes = nelements * self.dtype().itemsize();
  if (size_bytes > 0) {
    std::memset(ptr, 0, size_bytes);
  }
  return self;
}

Tensor& zero_(Tensor &self) {
  int64_t nelements = c10::multiply_integers(self.sizes());
  if (self.device() == at::kCPU &&
      self.is_non_overlapping_and_dense() &&
      nelements < internal::GRAIN_SIZE) {
    return zero_cpu_(self, nelements);
  }
  return self.fill_(0);
}

Tensor& zero_meta_(Tensor& self) {
  return self;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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

- `ATen/native/Fill.h`
- `ATen/core/Tensor.h`
- `ATen/ScalarOps.h`
- `ATen/TensorIterator.h`
- `ATen/TensorOperators.h`
- `c10/util/accumulate.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/fill_diagonal_native.h`
- `ATen/ops/fill_native.h`
- `ATen/ops/ones.h`
- `ATen/ops/zero_native.h`


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
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `Fill.cpp_docs.md`
- **Keyword Index**: `Fill.cpp_kw.md`
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

- **File Documentation**: `Fill.cpp_docs.md_docs.md`
- **Keyword Index**: `Fill.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

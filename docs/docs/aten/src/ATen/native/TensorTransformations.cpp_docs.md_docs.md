# Documentation: `docs/aten/src/ATen/native/TensorTransformations.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/TensorTransformations.cpp_docs.md`
- **Size**: 10,801 bytes (10.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/TensorTransformations.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorTransformations.cpp`
- **Size**: 7,695 bytes (7.51 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/IndexKernel.h> // for flip_stub
#include <ATen/native/TensorTransformations.h>

#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/atleast_1d_native.h>
#include <ATen/ops/atleast_2d_native.h>
#include <ATen/ops/atleast_3d_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chalf_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/flip_native.h>
#include <ATen/ops/fliplr_native.h>
#include <ATen/ops/flipud_native.h>
#include <ATen/ops/roll_native.h>
#include <ATen/ops/rot90_native.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

#include <algorithm>
#include <utility>
#include <vector>

namespace at::native {

Tensor flip(const Tensor& self, IntArrayRef dims) {
  TORCH_CHECK(
      self.scalar_type() != at::kQUInt4x2 &&
          self.scalar_type() != at::kQUInt2x4,
      "flip is not supported for tensor with data type ",
      self.scalar_type());
  const int64_t total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  Tensor out_tensor = at::empty_like(self, MemoryFormat::Preserve);

  // Count dimensions in which we need to do work
  int n = 0;
  auto strides = DimVector(self.strides());
  for (const auto i : c10::irange(total_dims)) {
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      n++;
      strides[i] = 0;
    }
  }

  // Nothing to do, we return fast
  if (n == 0 || self.numel() <= 1) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  // create dummy output with 0 strides at flipped dimension, to prevent
  // tensorIterator from coalescing flipped dims
  const auto restrided_self = self.as_strided(self.sizes(), strides);
  auto iter =
      TensorIteratorConfig()
          .set_check_mem_overlap(false)
          .check_all_same_dtype(false)
          .declare_static_dtype_and_device(self.scalar_type(), self.device())
          .add_output(out_tensor)
          .add_const_input(self)
          .add_const_input(restrided_self)
          .build();

  auto* data = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto sizes = iter.shape();
  // This is a SmallVector of _signed_ ints
  auto strides_bytes = DimVector(iter.strides(0));
  const auto strides_self = iter.strides(1);
  const auto strides_dummy = iter.strides(2);

  // To understand this transformation, think of a 3D cube.
  //   - The data ptr points to the lower-left most vertex of the cube
  //   - The strides tell us how to move in each dimension,
  //     that is, data + stride[i] advances one element in the dimension i
  // To flip a dimension:
  //   - We move the pointer to the opposite vertex of the cube
  //   - We iterate in the opposite direction (invert the strides)

  for (const auto i : c10::irange(iter.ndim())) {
    // We know that an dimension has a zero stride and self[i] does not, as we
    // defined above Note that it may be the case that strides_dummy[i] = 0 not
    // because we set it, but because strides_self[i] == 0. We do not want to do
    // anything there
    if (strides_dummy[i] == 0 && strides_self[i] != 0) {
      data += strides_bytes[i] * (sizes[i] - 1);
      strides_bytes[i] *= -1;
    }
  }
  iter._unsafe_set_arg_strides(0, strides_bytes);
  iter._unsafe_set_arg_data(0, reinterpret_cast<void*>(data));

  flip_stub(iter.device_type(), iter, self.is_quantized());

  return out_tensor;
}

Tensor roll(
    const Tensor& self,
    IntArrayRef shifts,
    IntArrayRef dims) { // Used by CPU and MPS dispatch.
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }
  // avoid a div zero error below.
  if (self.numel() == 0) {
    return self.clone(at::MemoryFormat::Preserve);
  }
  int64_t dim = dims[0];
  int64_t size = self.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if (start < 0) {
    start = start + size;
  }
  auto t0 = self.narrow(dim, start, size - start);
  auto t1 = self.narrow(dim, 0, start);
  return at::cat({std::move(t0), std::move(t1)}, dim);
}

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) {
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size();

  TORCH_CHECK(
      total_rot_dims == 2,
      "expected total rotation dims == 2, but got dims = ",
      total_rot_dims);

  TORCH_CHECK(
      total_dims >= 2,
      "expected total dims >= 2, but got total dims = ",
      total_dims);

  TORCH_CHECK(
      dims[0] != dims[1] && std::abs(dims[0] - dims[1]) != total_dims,
      "expected rotation dims to be different, but got dim0 = ",
      dims[0],
      " and dim1 = ",
      dims[1]);

  // check range of dims
  TORCH_CHECK(
      dims[0] < total_dims && dims[0] >= -total_dims,
      "Rotation dim0 out of range, dim0 = ",
      dims[0]);

  TORCH_CHECK(
      dims[1] < total_dims && dims[1] >= -total_dims,
      "Rotation dim1 out of range, dim1 = ",
      dims[1]);

  // handle modulo with negative k
  k = (4 + (k % 4)) % 4;

  switch (k) {
    case 1:
      return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
    case 2:
      return self.flip(dims);
    case 3:
      return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
    default:
      return self.clone(at::MemoryFormat::Contiguous);
  }
}

Tensor fliplr(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 2, "Input must be >= 2-d.");

  return self.flip({1});
}

Tensor flipud(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 1, "Input must be >= 1-d.");

  return self.flip({0});
}

Tensor atleast_1d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1});
    default:
      return self;
  }
}

std::vector<Tensor> atleast_1d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_1d(input);
  };
  std::transform(
      tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_2d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1});
    case 1: {
      return self.unsqueeze(0);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_2d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_2d(input);
  };
  std::transform(
      tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_3d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1, 1});
    case 1: {
      return self.unsqueeze(0).unsqueeze(-1);
    }
    case 2: {
      return self.unsqueeze(-1);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_3d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_3d(input);
  };
  std::transform(
      tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor chalf(const Tensor& self, std::optional<MemoryFormat> memory_format) {
  return self.to(kComplexHalf, false, false, memory_format);
}

DEFINE_DISPATCH(flip_stub);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

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

- `ATen/native/IndexKernel.h`
- `ATen/native/TensorTransformations.h`
- `ATen/Parallel.h`
- `ATen/TensorIterator.h`
- `ATen/WrapDimUtilsMulti.h`
- `ATen/core/DimVector.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/atleast_1d_native.h`
- `ATen/ops/atleast_2d_native.h`
- `ATen/ops/atleast_3d_native.h`
- `ATen/ops/cat.h`
- `ATen/ops/chalf_native.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/flip_native.h`
- `ATen/ops/fliplr_native.h`
- `ATen/ops/flipud_native.h`
- `ATen/ops/roll_native.h`
- `ATen/ops/rot90_native.h`
- `ATen/ops/zeros_like_ops.h`
- `algorithm`
- `utility`
- `vector`


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

- **File Documentation**: `TensorTransformations.cpp_docs.md`
- **Keyword Index**: `TensorTransformations.cpp_kw.md`
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

- **File Documentation**: `TensorTransformations.cpp_docs.md_docs.md`
- **Keyword Index**: `TensorTransformations.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/TensorFactories.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/TensorFactories.h_docs.md`
- **Size**: 7,995 bytes (7.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/TensorFactories.h`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorFactories.h`
- **Size**: 5,398 bytes (5.27 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/scalar_tensor.h>
#endif

namespace at::native {
// Different combinations of row, col, and offset can lead to two cases:
//
// Case 1 - Trapezoid (Triangle as a special case): row + offset <= col
//    Example A: offset > 0
//      1 1 0 0 0
//      1 1 1 0 0
//      1 1 1 1 0
//    Example B: offset <= 0
//      0 0 0
//      1 0 0
//      1 1 0
//    In this case, we calculate the number of elements in the first row and
//    last row of the tril respectively, and then compute the tril size.
//
// Case 2 - Trapezoid + Rectangle: row + offset > col
//    Example:
//      1 1 0
//      1 1 1
//      1 1 1
//    In this case, we first calculate the size of top trapezoid, and then
//    calculate the size of the bottom rectangle.
inline int64_t get_tril_size(int64_t row, int64_t col, int64_t offset) {
  // If either dimension is 0 then the there is no tril
  if (row == 0 || col == 0) {
    return 0;
  }
  // number of elements in the first row of the tril
  auto m_first_row = offset > 0 ? std::min<int64_t>(col, 1 + offset)
                                : // upper bounded by col
      row + offset > 0; // either 0 or 1
  // number of elements in the last row of the tril, bounded by [0, col]
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  // number of rows, bounded by [0, row]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);

  // calculate # of elements in the top trapezoid
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }

  return tril_size;
}

inline void check_args(
    int64_t row,
    int64_t col,
    std::optional<Layout> layout_opt) {
  TORCH_CHECK(row >= 0, "row must be non-negative, got", row);
  TORCH_CHECK(col >= 0, "col must be non-negative, got", col);
  if (layout_opt.has_value()) {
    TORCH_CHECK(
        *layout_opt == at::kStrided,
        "only support layout=torch.strided, got",
        *layout_opt)
  }
}

using at::check_size_nonnegative;

// assumes maximum value in created tensor is n-1 (e.g., torch.randperm(n))
inline void check_supported_max_int_with_precision(
    int64_t n,
    const Tensor& tensor) {
  // match defined() to behavior of checks below
  TORCH_CHECK(
      at::scalar_tensor(n > 0 ? n - 1 : n, tensor.options()).defined(),
      "n is too large for result tensor type: '",
      tensor.toString(),
      "'");

  // Ensure sufficient precision for floating point representation.
  switch (tensor.scalar_type()) {
    case at::ScalarType::Half:
      TORCH_CHECK(
          n <= (int64_t(1) << 11) + 1,
          "n cannot be greater than 2049 for Half type.");
      break;
    case at::ScalarType::Float:
      TORCH_CHECK(
          n <= (int64_t(1) << 24) + 1,
          "n cannot be greater than 2^24+1 for Float type.");
      break;
    case at::ScalarType::Double: // Unlikely to happen, but doesn't hurt to
                                 // check
      TORCH_CHECK(
          n <= (int64_t(1) << 53) + 1,
          "n cannot be greater than 2^53+1 for Double type.");
      break;
    default:
      break;
  }
}

// Called by `empty*` functions when deterministic algorithms are enabled to
// fill the tensor with NaN if it is floating point or complex type, or fill
// with max value if it is integer type
inline Tensor& fill_empty_deterministic_(Tensor& tensor) {
  if (tensor.is_floating_point() || tensor.is_complex()) {
    AT_DISPATCH_V2(
        tensor.scalar_type(),
        "fill_empty_deterministic_",
        AT_WRAP([&]() {
          tensor.fill_(std::numeric_limits<scalar_t>::quiet_NaN());
        }),
        AT_EXPAND(AT_FLOATING_TYPES),
        AT_EXPAND(AT_COMPLEX_TYPES),
        AT_EXPAND(AT_FLOAT8_TYPES),
        kBFloat16,
        kHalf,
        kComplexHalf);
  } else {
    AT_DISPATCH_V2(
        tensor.scalar_type(),
        "fill_empty_deterministic_",
        AT_WRAP([&]() { tensor.fill_(std::numeric_limits<scalar_t>::max()); }),
        kBool,
        AT_EXPAND(AT_INTEGRAL_TYPES_V2));
  }
  return tensor;
}

// The ZeroTensor allocator ignores whatever allocation is requested and always
// gives you nullptr
struct ZeroTensorAllocator final : public at::Allocator {
  ZeroTensorAllocator(at::Device device) : device_(device) {}
  ~ZeroTensorAllocator() override = default;
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  DataPtr allocate(const size_t /*nbytes*/) override {
    return {nullptr, nullptr, &deleter, device_};
  }
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
  void copy_data(
      void* dest [[maybe_unused]],
      const void* src [[maybe_unused]],
      std::size_t count [[maybe_unused]]) const final {}
  at::Device device_;
};

using binary_fn = void (*)(TensorIterator&);

DECLARE_DISPATCH(binary_fn, complex_stub)
DECLARE_DISPATCH(binary_fn, polar_stub)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `ZeroTensorAllocator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Dispatch.h`
- `ATen/Dispatch_v2.h`
- `ATen/EmptyTensor.h`
- `ATen/TensorIterator.h`
- `ATen/core/Tensor.h`
- `ATen/native/DispatchStub.h`
- `ATen/Functions.h`
- `ATen/ops/scalar_tensor.h`


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

- **File Documentation**: `TensorFactories.h_docs.md`
- **Keyword Index**: `TensorFactories.h_kw.md`
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

- **File Documentation**: `TensorFactories.h_docs.md_docs.md`
- **Keyword Index**: `TensorFactories.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/SegmentReduce.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/SegmentReduce.cpp_docs.md`
- **Size**: 26,195 bytes (25.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/SegmentReduce.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/SegmentReduce.cpp`
- **Size**: 23,440 bytes (22.89 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SegmentReduce.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_segment_reduce_backward_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/segment_reduce_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

DEFINE_DISPATCH(_segment_reduce_lengths_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_stub);
DEFINE_DISPATCH(_segment_reduce_lengths_backward_stub);
DEFINE_DISPATCH(_segment_reduce_offsets_backward_stub);

namespace {

template <typename T, bool is_offsets_like=false>
void _segment_reduce_lengths_cpu_kernel1(
    ReductionType reduction,
    const Tensor& data,
    const T* lengths_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& output,
    int64_t segment_count,
    int64_t lengths_stride_axis) {
  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++)
      outer_offset *= output.size(d);
  for (int64_t d = axis + 1; d < output.dim(); d++)
      inner_offset *= output.size(d);
  int64_t lengths_size_axis = is_offsets_like ? segment_count + 1 : segment_count;
  auto data_stride_axis = data.stride(axis);
  auto data_size_axis = data.size(axis);
  auto output_stride_axis = output.stride(axis);
  auto output_size_axis = output.size(axis);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_cpu", [&]() {
        auto* output_data = output.data_ptr<scalar_t>();
        const auto* values_data = data.const_data_ptr<scalar_t>();
        for (const auto outer_idx : c10::irange(outer_offset)) {
          int64_t segment_start, segment_length;
          int64_t segment_end = is_offsets_like ?
                                lengths_data[outer_idx * lengths_stride_axis * lengths_size_axis] :
                                0;
          for (const auto dim_idx : c10::irange(segment_count)) {
            segment_start = segment_end;
            auto lengths_idx = outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;
            if (is_offsets_like) {
              segment_end = lengths_data[lengths_idx + 1];
              segment_length = segment_end - segment_start;
            } else {
              segment_length = lengths_data[lengths_idx];
              segment_end += segment_length;
            }
            for (const auto inner_idx : c10::irange(inner_offset)) {
              // ===== step1: initialize starting value
              scalar_t initial_value;
              if (initial.has_value()) {
                initial_value = initial.value().to<scalar_t>();
              } else if (reduction == ReductionType::MAX) {
                initial_value = -std::numeric_limits<scalar_t>::infinity();
              } else if (
                  reduction == ReductionType::MEAN ||
                  reduction == ReductionType::SUM) {
                initial_value = 0;
              } else if (reduction == ReductionType::MIN) {
                initial_value = std::numeric_limits<scalar_t>::infinity();
              } else if (reduction == ReductionType::PROD) {
                initial_value = 1;
              }

              // ===== step2: apply reduction
              for (const auto j : c10::irange(segment_start, segment_end)) {
                int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                     + j * data_stride_axis + inner_idx;
                const auto val = values_data[data_index];
                if (reduction == ReductionType::MAX) {
                  initial_value = at::_isnan(val)
                      ? val
                      : std::max<scalar_t>(initial_value, val);
                } else if (
                    reduction == ReductionType::MEAN ||
                    reduction == ReductionType::SUM) {
                  initial_value = initial_value + val;
                } else if (reduction == ReductionType::MIN) {
                  initial_value = at::_isnan(val)
                      ? val
                      : std::min<scalar_t>(initial_value, val);
                } else if (reduction == ReductionType::PROD) {
                  initial_value = initial_value * val;
                }
              }

              // ===== step3: finalize reduction
              TORCH_CHECK(segment_length >= 0);

              if (segment_length == 0 && !initial.has_value() &&
                  reduction == ReductionType::MEAN) {
                initial_value = static_cast<scalar_t>(NAN);
              } else if (
                  reduction == ReductionType::MEAN &&
                  segment_length > 0 && !at::_isnan(initial_value)) {
                initial_value = initial_value / segment_length;
              }
              int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                                     + dim_idx * output_stride_axis + inner_idx;
              output_data[output_index] = initial_value;
            }
          }
        }
      });
}

Tensor _segment_reduce_lengths_cpu_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  // data and lengths should be contiguous from the call to .contiguous in segment_reduce_kernel
  TORCH_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
  TORCH_CHECK(lengths.is_contiguous(), "Expected lengths to be contiguous.");
  // reduction axis should always be the last dimension of lengths
  axis = lengths.dim() - 1;
  int64_t segment_count = lengths.size(axis);
  int64_t lengths_stride_axis = lengths.stride(axis);
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "_segment_reduce_lengths_cpu_kernel1", [&]() {
    const auto* lengths_data = lengths.const_data_ptr<index_t>();
    _segment_reduce_lengths_cpu_kernel1(
        reduction, data, lengths_data, axis, initial, output, segment_count, lengths_stride_axis);
  });

  return output;
}

Tensor _segment_reduce_offsets_cpu_kernel(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  // data and lengths should be contiguous from the call to .contiguous in segment_reduce_kernel
  TORCH_CHECK(data.is_contiguous(), "Expected data to be contiguous.");
  TORCH_CHECK(offsets.is_contiguous(), "Expected offsets to be contiguous.");
  // reduction axis should always be the last dimension of lengths
  axis = offsets.dim() - 1;
  int64_t segment_count = offsets.size(axis) - 1;
  int64_t offsets_stride_axis = offsets.stride(axis);
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "_segment_reduce_offsets_cpu_kernel1", [&]() {
    const auto* offsets_data = offsets.const_data_ptr<index_t>();
    _segment_reduce_lengths_cpu_kernel1<index_t, /*is_offsets_like=*/true>(
        reduction, data, offsets_data, axis, initial, output, segment_count, offsets_stride_axis);
  });

  return output;
}

template <typename T, bool is_offsets_like = false>
void _segment_reduce_cpu_lengths_backward_kernel1(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const T* lengths_data,
    int64_t axis,
    const std::optional<Scalar>& initial,
    Tensor& grad_input,
    int64_t segment_count,
    int64_t lengths_stride_axis) {
  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++)
      outer_offset *= output_contig.size(d);
  for (int64_t d = axis + 1; d < output_contig.dim(); d++)
      inner_offset *= output_contig.size(d);
  int64_t lengths_size_axis = is_offsets_like ? segment_count + 1 : segment_count;
  auto data_stride_axis = data_contig.stride(axis);
  auto data_size_axis = data_contig.size(axis);
  auto output_stride_axis = output_contig.stride(axis);
  auto output_size_axis = output_contig.size(axis);
  // TODO: Switch to TensorIterator for better maintainablility and
  // readability
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      data_contig.scalar_type(),
      "_segment_reduce_cpu",
      [&]() {
        auto* output_data = output_contig.const_data_ptr<scalar_t>();
        auto* grad_data = grad_contig.const_data_ptr<scalar_t>();
        auto* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
        const auto* values_data = data_contig.const_data_ptr<scalar_t>();
        // Used to calculate exclusive prod
        scalar_t initial_prod_value;
        if (reduction == ReductionType::PROD) {
          if (initial.has_value()) {
            initial_prod_value = initial.value().to<scalar_t>();
          } else {
            initial_prod_value = 1;
          }
        }

        for (const auto outer_idx : c10::irange(outer_offset)) {
          // int64_t lengths_cum_sum = 0;
          int64_t segment_start, segment_length;
          int64_t segment_end = is_offsets_like ?
                                lengths_data[outer_idx * lengths_stride_axis * lengths_size_axis] :
                                0;
          for (const auto dim_idx : c10::irange(segment_count)) {
            // int64_t segment_length = lengths_data[outer_idx * lengths_stride_axis * segment_count + dim_idx];
            segment_start = segment_end;
            auto lengths_idx = outer_idx * lengths_stride_axis * lengths_size_axis + dim_idx;
            if (is_offsets_like) {
              segment_end = lengths_data[lengths_idx + 1];
              segment_length = segment_end - segment_start;
            } else {
              segment_length = lengths_data[lengths_idx];
              segment_end += segment_length;
            }
            if (segment_length == 0) {
              continue;
            }
            for (const auto inner_idx : c10::irange(inner_offset)) {
              int64_t output_index = outer_idx * output_stride_axis * output_size_axis
                                     + dim_idx * output_stride_axis + inner_idx;
              if (reduction == ReductionType::MAX ||
                  reduction == ReductionType::MIN) {
                int64_t counter = 0;
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  if (at::_isnan(values_data[data_index]) ||
                      values_data[data_index] == output_data[output_index]) {
                    grad_input_data[data_index] = grad_data[output_index];
                    counter++;
                  }
                }
                // Average gradient based on number of maximum elements in
                // the segment
                if (counter < 2) {
                  continue;
                }
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  if (grad_input_data[data_index] > 0) {
                    grad_input_data[data_index] =
                        grad_input_data[data_index] / counter;
                  }
                }
              } else if (reduction == ReductionType::MEAN) {
                auto grad_val = grad_data[output_index] / segment_length;
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  grad_input_data[data_index] = grad_val;
                }
              } else if (reduction == ReductionType::SUM) {
                const auto& grad_val = grad_data[output_index];
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  grad_input_data[data_index] = grad_val;
                }
              } else if (reduction == ReductionType::PROD) {
                const auto& grad_val = grad_data[output_index] * output_data[output_index];
                for (const auto j : c10::irange(segment_start, segment_end)) {
                  int64_t data_index = outer_idx * data_stride_axis * data_size_axis
                                       + j * data_stride_axis + inner_idx;
                  if (at::_isnan(values_data[data_index]) ||
                      values_data[data_index] == 0) {
                    // explicitly compute exclusive prod
                    scalar_t exclusive_prod = initial_prod_value;
                    int64_t idx;
                    for (const auto k : c10::irange(segment_start, segment_end)) {
                      if (k != j) {
                        idx = outer_idx * data_stride_axis * data_size_axis
                              + k * data_stride_axis + inner_idx;
                        exclusive_prod *= values_data[idx];
                      }
                    }
                    grad_input_data[data_index] = grad_data[output_index] * exclusive_prod;
                  } else {
                    grad_input_data[data_index] = grad_val / values_data[data_index];
                  }
                }
              }
            }
          }
        }
      });
}

Tensor _segment_reduce_cpu_lengths_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  axis = lengths_contig.dim() - 1;
  int64_t segment_count = lengths_contig.size(axis);
  int64_t lengths_stride_axis = lengths_contig.stride(axis);
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.scalar_type(), "_segment_reduce_cpu_lengths_backward_kernel1", [&] {
        const auto* lengths_data = lengths_contig.const_data_ptr<index_t>();
        _segment_reduce_cpu_lengths_backward_kernel1(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            lengths_data,
            axis,
            initial,
            grad_input,
            segment_count,
            lengths_stride_axis);
      });

  return grad_input;
}


Tensor _segment_reduce_cpu_offsets_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  axis = offsets_contig.dim() - 1;
  int64_t segment_count = offsets_contig.size(axis) - 1;
  int64_t offsets_stride_axis = offsets_contig.stride(axis);
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(), "_segment_reduce_cpu_offsets_backward_kernel1", [&] {
        const auto* offsets_data = offsets_contig.const_data_ptr<index_t>();
        _segment_reduce_cpu_lengths_backward_kernel1<index_t, /*is_offsets_like=*/true>(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            offsets_data,
            axis,
            initial,
            grad_input,
            segment_count,
            offsets_stride_axis);
      });

  return grad_input;
}

} // namespace

Tensor segment_reduce_kernel(
    const Tensor& data,
    std::string_view reduce,
    const std::optional<Tensor>& lengths,
    const std::optional<Tensor>& indices,
    const std::optional<Tensor>& offsets,
    int64_t axis,
    bool unsafe,
    const std::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  TORCH_CHECK(data.numel() >= 0);

  // check that one of lengths or offsets is defined
  auto lengths_has_value = lengths.has_value();
  auto offsets_has_value = offsets.has_value();
  TORCH_CHECK(
    !indices.has_value(),
    "segment_reduce(): indices based reduction is not supported yet.");
  TORCH_CHECK(
      lengths_has_value || offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.")

  auto reduction = get_reduction_enum(reduce);
  const auto data_contig = data.contiguous();

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();

    // offsets related checks
    TORCH_CHECK(data.get_device() == offsets_value.get_device());
    TORCH_CHECK(data.dim() >= offsets_value.dim());
    TORCH_CHECK(axis == offsets_value.dim() - 1,
                "segment_reduce(): Expected axis to be the last dimension of offsets but got ", axis, ".");

    // TODO: add checks when !unsafe

    const auto offsets_contig = offsets_value.contiguous();

    return _segment_reduce_offsets_stub(
      data_contig.device().type(),
      reduction,
      data_contig,
      offsets_contig,
      axis,
      initial);

  } else {
    const auto& lengths_value = lengths.value();

    // length related checks
    TORCH_CHECK(data.get_device() == lengths_value.get_device());
    TORCH_CHECK(data.dim() >= lengths_value.dim());
    TORCH_CHECK(axis == lengths_value.dim() - 1,
                "segment_reduce(): Expected axis to be the last dimension of lengths but got ", axis, ".");

    if (!unsafe) {
      auto min_length = lengths_value.min().item<int64_t>();
      TORCH_CHECK((min_length >= 0), "lengths contains negative value!");
      TORCH_CHECK(all(lengths_value.sum({-1}) == data.size(axis)).item<bool>(),
                  "segment_reduce(): Expected all rows of lengths along axis ",
                  "to sum to data.size(lengths.dim()-1) when !unsafe.");
    }

    const auto lengths_contig = lengths_value.contiguous();

    return _segment_reduce_lengths_stub(
      data_contig.device().type(),
      reduction,
      data_contig,
      lengths_contig,
      axis,
      initial);
  }
}

REGISTER_ARCH_DISPATCH(
    _segment_reduce_lengths_stub,
    DEFAULT,
    &_segment_reduce_lengths_cpu_kernel)
REGISTER_AVX2_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel)
REGISTER_AVX512_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel)
REGISTER_VSX_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel)
REGISTER_SVE256_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cpu_kernel)

// offsets dispatches
REGISTER_ARCH_DISPATCH(
    _segment_reduce_offsets_stub,
    DEFAULT,
    &_segment_reduce_offsets_cpu_kernel)
REGISTER_AVX2_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel)
REGISTER_AVX512_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel)
REGISTER_VSX_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel)
REGISTER_SVE256_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cpu_kernel)

// Currently some computation is being duplicated across forward and backward.
// TODO: Cache indices in forward pass to reuse in backward
Tensor _segment_reduce_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    std::string_view reduce,
    const std::optional<Tensor>& lengths,
    const std::optional<Tensor>& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  // check that one of lengths or offsets is defined
  // codegen for derivatives.yaml passes an undefined Tensor for None rather than a std::optional
  // so checking .has_value() doesn't work unlike in the forward pass
  auto lengths_has_value = lengths.has_value() && lengths.value().defined();
  auto offsets_has_value = offsets.has_value() && offsets.value().defined();
  TORCH_CHECK(
      lengths_has_value ||  offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.");

  const auto grad_contig = grad.contiguous();
  const auto output_contig = output.contiguous();
  const auto data_contig = data.contiguous();
  auto reduction = get_reduction_enum(reduce);

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();
    const auto offsets_contig = offsets_value.contiguous();
    return _segment_reduce_offsets_backward_stub(
      grad_contig.device().type(),
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      offsets_contig,
      axis,
      initial);
  } else {
    const auto& lengths_value = lengths.value();
    const auto lengths_contig = lengths_value.contiguous();
    return _segment_reduce_lengths_backward_stub(
      grad_contig.device().type(),
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      lengths_contig,
      axis,
      initial);
  }
}

REGISTER_ARCH_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    DEFAULT,
    &_segment_reduce_cpu_lengths_backward_kernel)
REGISTER_AVX512_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel)
REGISTER_AVX2_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel)
REGISTER_VSX_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel)
REGISTER_ZVECTOR_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel)
REGISTER_SVE256_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_cpu_lengths_backward_kernel)

REGISTER_ARCH_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    DEFAULT,
    &_segment_reduce_cpu_offsets_backward_kernel)
REGISTER_AVX512_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel)
REGISTER_AVX2_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel)
REGISTER_VSX_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel)
REGISTER_ZVECTOR_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel)
REGISTER_SVE256_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &_segment_reduce_cpu_offsets_backward_kernel)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 36 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/SegmentReduce.h`
- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/NumericUtils.h`
- `ATen/TensorOperators.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_segment_reduce_backward_native.h`
- `ATen/ops/all.h`
- `ATen/ops/empty.h`
- `ATen/ops/segment_reduce_native.h`
- `ATen/ops/zeros.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `SegmentReduce.cpp_docs.md`
- **Keyword Index**: `SegmentReduce.cpp_kw.md`
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

- **File Documentation**: `SegmentReduce.cpp_docs.md_docs.md`
- **Keyword Index**: `SegmentReduce.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

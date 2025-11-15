# Documentation: `aten/src/ATen/native/LossMultiMargin.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/LossMultiMargin.cpp`
- **Size**: 9,677 bytes (9.45 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LossMulti.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/multi_margin_loss_backward_native.h>
#include <ATen/ops/multi_margin_loss_native.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t>
inline scalar_t multi_margin_inner_sum_cpu(
    const scalar_t* input_data,
    const scalar_t* weight_data,
    const int p,
    const scalar_t margin,
    const int64_t dim,
    const int64_t target_idx) {
  const scalar_t input_target = input_data[target_idx];
  scalar_t sum = 0;
  for (const auto d : c10::irange(dim)) {
    if (d == target_idx) {
      continue;
    }

    const scalar_t z = margin - input_target + input_data[d];
    if (z > 0) {
      scalar_t h = (p == 1) ? z : z * z;
      if (weight_data != nullptr) {
        h *= weight_data[target_idx];
      }
      sum += h;
    }
  }

  sum /= dim;
  return sum;
}

inline int64_t target_index_checked(
    const int64_t* target_data,
    const int64_t index,
    const int64_t dim) {
  const int64_t idx = target_data[index];
  TORCH_CHECK(idx >= 0 && idx < dim, "target out of range");
  return idx;
}

template <typename scalar_t>
inline void multi_margin_loss_cpu_kernel(
    Tensor& output,
    const scalar_t* input_data,
    const int64_t* target_data,
    const int p,
    scalar_t margin,
    const scalar_t* weight_data,
    const int64_t nframe,
    const int64_t dim,
    const int64_t reduction) {
  using accscalar_t = at::acc_type<scalar_t, false>;

  // dim() != 0 check is for 1d input which produces a scalar output (that
  // cannot be handled by TensorAccessor)
  if (reduction == Reduction::None && output.dim() > 0) {
    auto output_acc = output.accessor<scalar_t, 1>();
    for (const auto t : c10::irange(nframe)) {
      const auto idx = target_index_checked(target_data, t, dim);
      auto sum = multi_margin_inner_sum_cpu(
          input_data, weight_data, p, margin, dim, idx);
      output_acc[t] = sum;
      input_data += dim;
    }
  } else {
    accscalar_t sum = 0;
    auto output_acc = output.data_ptr<scalar_t>();
    for (const auto t : c10::irange(nframe)) {
      const auto idx = target_index_checked(target_data, t, dim);
      sum += multi_margin_inner_sum_cpu(
          input_data, weight_data, p, margin, dim, idx);
      input_data += dim;
    }
    if (reduction == Reduction::Mean) {
      sum /= nframe;
    }
    output_acc[0] = sum;
  }
}

void multi_margin_loss_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction) {
  int64_t nframe = 0, dim = 0;
  const auto ndims = input.dim();

  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");

  multi_margin_loss_shape_check(nframe, dim, ndims, input, target, weight);

  // produce a scalar output for 1d input
  if (reduction == Reduction::None && target.dim() > 0) {
    output.resize_({nframe});
  } else {
    output.resize_({});
  }
  if (input.numel() == 0) {
    return;
  }

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  Tensor weight_contiguous;
  if (weight && weight->defined()) {
    weight_contiguous = weight->contiguous();
  }

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multi_margin_loss_cpu_kernel", [&] {
        auto input_data = input_contiguous.const_data_ptr<scalar_t>();
        auto target_data = target_contiguous.const_data_ptr<int64_t>();
        auto weight_data =
            weight_contiguous.defined() ? weight_contiguous.const_data_ptr<scalar_t>() : nullptr;
        multi_margin_loss_cpu_kernel<scalar_t>(
            output,
            input_data,
            target_data,
            p,
            margin.to<scalar_t>(),
            weight_data,
            nframe,
            dim,
            reduction);
      });
}

template <typename scalar_t>
void multi_margin_loss_backward_cpu_kernel(
    scalar_t* grad_input_data,
    const Tensor& grad_output,
    const scalar_t* input_data,
    const int64_t* target_data,
    int p,
    scalar_t margin,
    scalar_t g,
    const scalar_t* weight_data,
    int64_t nframe,
    int64_t dim,
    int64_t reduction) {
  scalar_t* grad_input_row_data = grad_input_data;
  for (const auto t : c10::irange(nframe)) {
    int64_t target_idx = target_index_checked(target_data, t, dim);
    scalar_t input_target = input_data[target_idx];
    scalar_t grad_input_target = 0;
    for (const auto d : c10::irange(dim)) {
      scalar_t z = margin - input_target + input_data[d];
      if (d == target_idx) {
        continue;
      }

      if (z > 0) {
        scalar_t h = (p == 1) ? g : 2 * g * z;
        if (weight_data != nullptr) {
          h *= weight_data[target_idx];
        }
        grad_input_target -= h;
        grad_input_row_data[d] = h;
      } else {
        grad_input_row_data[d] = 0;
      }
    }
    grad_input_row_data[target_idx] = grad_input_target;

    input_data += dim;
    grad_input_row_data += dim;
  }

  if (reduction != Reduction::None || grad_output.dim() == 0) {
    assert(
        reduction != Reduction::None || grad_output.dim() > 0 ||
        nframe == 1); // check 1d scalar fallback-case
    const auto d = *grad_output.const_data_ptr<scalar_t>();
    for (int64_t t = 0; t < nframe * dim; t++) {
      grad_input_data[t] *= d;
    }
  } else {
    auto grad_output_acc = grad_output.accessor<const scalar_t, 1>();
    for (const auto t : c10::irange(nframe)) {
      for (const auto d : c10::irange(dim)) {
        grad_input_data[t * dim + d] *= grad_output_acc[t];
      }
    }
  }
}

void multi_margin_loss_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int p,
    const Scalar& margin,
    const Tensor& weight,
    int64_t reduction) {
  int64_t nframe = 0, dim = 0;
  const auto ndims = input.dim();

  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");

  multi_margin_loss_shape_check(nframe, dim, ndims, input, target, weight);
  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  if (input.numel() == 0) {
    return;
  }

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weight_contiguous = weight.contiguous();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multi_margin_loss_backward_cpu_kernel", [&] {
        auto grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
        auto input_data = input_contiguous.const_data_ptr<scalar_t>();
        auto target_data = target_contiguous.const_data_ptr<int64_t>();
        auto weight_data = weight_contiguous.defined()
            ? weight_contiguous.const_data_ptr<scalar_t>()
            : nullptr;
        scalar_t g = reduction == Reduction::Mean
            ? static_cast<scalar_t>(1. / (nframe * dim))
            : static_cast<scalar_t>(1. / dim);
        multi_margin_loss_backward_cpu_kernel<scalar_t>(
            grad_input_data,
            grad_output,
            input_data,
            target_data,
            p,
            margin.to<scalar_t>(),
            g,
            weight_data,
            nframe,
            dim,
            reduction);
      });
}

} // namespace

Tensor multi_margin_loss_cpu(
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction) {
  auto output = at::empty({0}, input.options());
  multi_margin_loss_out_cpu_template(
      output, input, target, p.toInt(), margin, weight, reduction);
  return output;
}

Tensor& multi_margin_loss_cpu_out(const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& output) {
  multi_margin_loss_out_cpu_template(
      output, input, target, p.toInt(), margin, weight, reduction);
  return output;
}

Tensor multi_margin_loss_cpu_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin, const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  auto grad_input = at::empty({0}, input.options());
  multi_margin_loss_backward_out_cpu_template(
      grad_input,
      grad_output,
      input,
      target,
      p.toInt(),
      margin,
      weight,
      reduction);
  return grad_input;
}

Tensor& multi_margin_loss_cpu_backward_out(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin, const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  multi_margin_loss_backward_out_cpu_template(
      grad_input,
      grad_output,
      input,
      target,
      p.toInt(),
      margin,
      weight,
      reduction);
  return grad_input;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

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

- `ATen/core/Tensor.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/native/LossMulti.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/multi_margin_loss_backward_native.h`
- `ATen/ops/multi_margin_loss_native.h`


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

- **File Documentation**: `LossMultiMargin.cpp_docs.md`
- **Keyword Index**: `LossMultiMargin.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

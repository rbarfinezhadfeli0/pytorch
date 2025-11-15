# Documentation: `aten/src/ATen/native/Correlation.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Correlation.cpp`
- **Size**: 4,869 bytes (4.75 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/complex.h>
#include <ATen/ops/corrcoef_native.h>
#include <ATen/ops/cov.h>
#include <ATen/ops/cov_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/real.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/true_divide.h>
#endif

namespace at::native {

Tensor cov(
    const Tensor& self,
    int64_t correction,
    const std::optional<Tensor>& fweights,
    const std::optional<Tensor>& aweights) {
  constexpr int64_t OBSERVATIONS_DIM = 1;

  TORCH_CHECK(
      self.ndimension() <= 2,
      "cov(): expected input to have two or fewer dimensions but got an input with ",
      self.ndimension(),
      " dimensions");

  TORCH_CHECK(
      self.scalar_type() != kBool,
      "cov(): bool dtype is not supported for input");

  // View input tensor as 2D (variables, observations)
  auto in = self.ndimension() < 2 ? self.view({1, -1}) : self;
  const auto num_observations = in.size(OBSERVATIONS_DIM);

  // The product of frequencies (fweights) and weights (aweights).
  Tensor w;

  if (fweights.has_value()) {
    w = fweights.value();
    TORCH_CHECK(
        w.ndimension() <= 1,
        "cov(): expected fweights to have one or fewer dimensions but got fweights with ",
        w.ndimension(),
        " dimensions");
    TORCH_CHECK(
        at::isIntegralType(w.scalar_type(), false),
        "cov(): expected fweights to have integral dtype but got fweights with ",
        w.scalar_type(),
        " dtype");
    TORCH_CHECK(
        w.numel() == num_observations,
        "cov(): expected fweights to have the same numel as there are observations in the input but got ",
        w.numel(),
        " != ",
        num_observations);
    TORCH_CHECK(
        num_observations == 0 || at::is_scalar_tensor_true(w.min().ge(0)),
        "cov(): fweights cannot be negative");
  }

  if (aweights.has_value()) {
    const auto& aw = aweights.value();
    TORCH_CHECK(
        aw.ndimension() <= 1,
        "cov(): expected aweights to have one or fewer dimensions but got aweights with ",
        aw.ndimension(),
        " dimensions");
    TORCH_CHECK(
        at::isFloatingType(aw.scalar_type()),
        "cov(): expected aweights to have floating point dtype but got aweights with ",
        aw.scalar_type(),
        " dtype");
    TORCH_CHECK(
        aw.numel() == num_observations,
        "cov(): expected aweights to have the same numel as there are observations in the input but got ",
        aw.numel(),
        " != ",
        num_observations);
    TORCH_CHECK(
        num_observations == 0 || at::is_scalar_tensor_true(aw.min().ge(0)),
        "cov(): aweights cannot be negative");
    w = w.defined() ? w * aw : aw;
  }

  // Compute a weighted average of the observations
  const auto w_sum = w.defined()
      ? w.sum()
      : at::scalar_tensor(num_observations, in.options().dtype(kLong));

  TORCH_CHECK(
      !w.defined() || at::is_scalar_tensor_true(w_sum.ne(0)),
      "cov(): weights sum to zero, can't be normalized");

  const auto avg = (w.defined() ? in * w : in).sum(OBSERVATIONS_DIM) / w_sum;

  // Compute the normalization factor
  Tensor norm_factor;

  if (w.defined() && aweights.has_value() && correction != 0) {
    norm_factor = w_sum - correction * (w * aweights.value()).sum() / w_sum;
  } else {
    norm_factor = w_sum - correction;
  }

  if (at::is_scalar_tensor_true(norm_factor.le(0))) {
    TORCH_WARN("cov(): degrees of freedom is <= 0. Correction should be strictly less than the number of observations.");
    norm_factor.zero_();
  }

  // Compute covariance matrix
  in = in - avg.unsqueeze(1);
  const auto c = at::mm(in, (w.defined() ? in * w : in).t().conj());
  return at::true_divide(c, norm_factor).squeeze();
}

Tensor corrcoef(const Tensor& self) {
  TORCH_CHECK(
      self.ndimension() <= 2,
      "corrcoef(): expected input to have two or fewer dimensions but got an input with ",
      self.ndimension(),
      " dimensions");

  auto c = at::cov(self);

  if (c.ndimension() == 0) {
    // scalar covariance, return nan if c in {nan, inf, 0}, 1 otherwise
    return c / c;
  }

  // normalize covariance
  const auto d = c.diagonal();
  const auto stddev = at::sqrt(d.is_complex() ? at::real(d) : d);
  c = c / stddev.view({-1, 1});
  c = c / stddev.view({1, -1});

  // due to floating point rounding the values may be not within [-1, 1], so
  // to improve the result we clip the values just as NumPy does.
  return c.is_complex()
      ? at::complex(at::real(c).clip(-1, 1), at::imag(c).clip(-1, 1))
      : c.clip(-1, 1);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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
- `ATen/TensorOperators.h`
- `ATen/TensorSubclassLikeUtils.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/complex.h`
- `ATen/ops/corrcoef_native.h`
- `ATen/ops/cov.h`
- `ATen/ops/cov_native.h`
- `ATen/ops/imag.h`
- `ATen/ops/mm.h`
- `ATen/ops/real.h`
- `ATen/ops/scalar_tensor.h`
- `ATen/ops/sqrt.h`
- `ATen/ops/true_divide.h`


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

- **File Documentation**: `Correlation.cpp_docs.md`
- **Keyword Index**: `Correlation.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

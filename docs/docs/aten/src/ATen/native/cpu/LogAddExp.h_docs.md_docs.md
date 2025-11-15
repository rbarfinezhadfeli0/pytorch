# Documentation: `docs/aten/src/ATen/native/cpu/LogAddExp.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/LogAddExp.h_docs.md`
- **Size**: 4,945 bytes (4.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/LogAddExp.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/LogAddExp.h`
- **Size**: 2,446 bytes (2.39 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/complex.h>
#include <ATen/NumericUtils.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

// custom min and max to be used in logcumsumexp for complex arguments
template <typename scalar_t>
std::pair<c10::complex<scalar_t>, c10::complex<scalar_t>> _logcumsumexp_minmax(c10::complex<scalar_t> x, c10::complex<scalar_t> y) {
  if (at::_isnan(y)) {  // either real is nan or imag is nan
    return std::make_pair(y, y);
  } else if (at::_isnan(x)) {  // either real is nan or imag is nan
    return std::make_pair(x, x);
  } else {
    return (x.real() < y.real()) ? std::make_pair(x, y) : std::make_pair(y, x);
  }
}

template <typename scalar_t>
scalar_t _log_add_exp_helper(scalar_t x, scalar_t y) {
  // Reference : https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
  scalar_t min = at::_isnan(y) ? y : std::min(x, y); // std::min returns first arg if one of the args is nan
  scalar_t max = at::_isnan(y) ? y : std::max(x, y); // std::max returns first arg if one of the args is nan
  if (min != max || std::isfinite(min)) {
    // nan will be propagated here
    return std::log1p(std::exp(min - max)) + max;
  } else {
    // special case to correctly handle infinite cases
    return x;
  }
}

template <typename scalar_t>
c10::complex<scalar_t> _log_add_exp_helper(const c10::complex<scalar_t>& x, const c10::complex<scalar_t>& y) {
  auto [min, max] = _logcumsumexp_minmax<scalar_t>(x, y);
  auto min_real = std::real(min);
  auto max_real = std::real(max);

  if (at::_isnan(min)) {  // either real is nan or imag is nan
    // handling the "infectious" NaNs
    return {std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::quiet_NaN()};
  } else if (!std::isfinite(min_real) && (min_real == max_real)) {
    if (min_real < 0) {
      // handle the -inf case, the imaginary part here does not really matter as the exp(value)
      // will be around 0.0 and the angle (i.e. the imaginary part) cannot be determined.
      // It does not matter if we're taking the exp of this value
      return min;
    } else {
      // handle the +inf case, we don't need the special precision for log1p for small values
      // and to avoid producing nan in case of real(max) == real(min) == +inf
      return std::log(std::exp(min) + std::exp(max));
    }
  } else {
    return std::log1p(std::exp(min - max)) + max;
  }
}

} // end namespace
} //end at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `CPU_CAPABILITY`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/complex.h`
- `ATen/NumericUtils.h`


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

Files in the same folder (`aten/src/ATen/native/cpu`):

- [`UpSampleKernelAVXAntialias.h_docs.md`](./UpSampleKernelAVXAntialias.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`UnfoldBackwardKernel.cpp_docs.md`](./UnfoldBackwardKernel.cpp_docs.md)
- [`int8mm_kernel.cpp_docs.md`](./int8mm_kernel.cpp_docs.md)
- [`LerpKernel.cpp_docs.md`](./LerpKernel.cpp_docs.md)
- [`UpSampleKernel.cpp_docs.md`](./UpSampleKernel.cpp_docs.md)
- [`scaled_modified_bessel_k0.cpp_docs.md`](./scaled_modified_bessel_k0.cpp_docs.md)
- [`DistributionKernels.cpp_docs.md`](./DistributionKernels.cpp_docs.md)
- [`CopyKernel.cpp_docs.md`](./CopyKernel.cpp_docs.md)
- [`SampledAddmmKernel.cpp_docs.md`](./SampledAddmmKernel.cpp_docs.md)


## Cross-References

- **File Documentation**: `LogAddExp.h_docs.md`
- **Keyword Index**: `LogAddExp.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/cpu`):

- [`BinaryOpsKernel.cpp_docs.md_docs.md`](./BinaryOpsKernel.cpp_docs.md_docs.md)
- [`MultinomialKernel.cpp_kw.md_docs.md`](./MultinomialKernel.cpp_kw.md_docs.md)
- [`AmpGradScalerKernels.cpp_docs.md_docs.md`](./AmpGradScalerKernels.cpp_docs.md_docs.md)
- [`FusedSGDKernel.cpp_docs.md_docs.md`](./FusedSGDKernel.cpp_docs.md_docs.md)
- [`scaled_modified_bessel_k1.cpp_docs.md_docs.md`](./scaled_modified_bessel_k1.cpp_docs.md_docs.md)
- [`int_mm_kernel.h_docs.md_docs.md`](./int_mm_kernel.h_docs.md_docs.md)
- [`IsContiguous.h_docs.md_docs.md`](./IsContiguous.h_docs.md_docs.md)
- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`WeightNormKernel.cpp_kw.md_docs.md`](./WeightNormKernel.cpp_kw.md_docs.md)
- [`FusedAdamKernel.cpp_docs.md_docs.md`](./FusedAdamKernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `LogAddExp.h_docs.md_docs.md`
- **Keyword Index**: `LogAddExp.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

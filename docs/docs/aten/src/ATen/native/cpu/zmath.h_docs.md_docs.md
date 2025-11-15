# Documentation: `docs/aten/src/ATen/native/cpu/zmath.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/zmath.h_docs.md`
- **Size**: 9,111 bytes (8.90 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/zmath.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/zmath.h`
- **Size**: 6,622 bytes (6.47 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// Complex number math operations that act as no-ops for other dtypes.
#include <c10/util/complex.h>
#include <c10/util/MathConstants.h>
#include<ATen/NumericUtils.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
inline VALUE_TYPE zabs (SCALAR_TYPE z) {
  return z;
}

template<>
inline c10::complex<float> zabs <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(std::abs(z));
}

template<>
inline float zabs <c10::complex<float>, float> (c10::complex<float> z) {
  return std::abs(z);
}

template<>
inline c10::complex<double> zabs <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(std::abs(z));
}

template<>
inline double zabs <c10::complex<double>, double> (c10::complex<double> z) {
  return std::abs(z);
}

// This overload corresponds to non-complex dtypes.
// The function is consistent with its NumPy equivalent
// for non-complex dtypes where `pi` is returned for
// negative real numbers and `0` is returned for 0 or positive
// real numbers.
// Note: `nan` is propagated.
template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
inline VALUE_TYPE angle_impl (SCALAR_TYPE z) {
  if (at::_isnan(z)) {
    return z;
  }
  return z < 0 ? c10::pi<double> : 0;
}

template<>
inline c10::complex<float> angle_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(std::arg(z), 0.0);
}

template<>
inline float angle_impl <c10::complex<float>, float> (c10::complex<float> z) {
  return std::arg(z);
}

template<>
inline c10::complex<double> angle_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(std::arg(z), 0.0);
}

template<>
inline double angle_impl <c10::complex<double>, double> (c10::complex<double> z) {
  return std::arg(z);
}

template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
constexpr VALUE_TYPE real_impl (SCALAR_TYPE z) {
  return z; //No-Op
}

template<>
constexpr c10::complex<float> real_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(z.real(), 0.0);
}

template<>
constexpr float real_impl <c10::complex<float>, float> (c10::complex<float> z) {
  return z.real();
}

template<>
constexpr c10::complex<double> real_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(z.real(), 0.0);
}

template<>
constexpr double real_impl <c10::complex<double>, double> (c10::complex<double> z) {
  return z.real();
}

template <typename SCALAR_TYPE, typename VALUE_TYPE=SCALAR_TYPE>
constexpr VALUE_TYPE imag_impl (SCALAR_TYPE /*z*/) {
  return 0;
}

template<>
constexpr c10::complex<float> imag_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(z.imag(), 0.0);
}

template<>
constexpr float imag_impl <c10::complex<float>, float> (c10::complex<float> z) {
  return z.imag();
}

template<>
constexpr c10::complex<double> imag_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(z.imag(), 0.0);
}

template<>
constexpr double imag_impl <c10::complex<double>, double> (c10::complex<double> z) {
  return z.imag();
}

template <typename TYPE>
inline TYPE conj_impl (TYPE z) {
  return z; //No-Op
}

template<>
inline c10::complex<at::Half> conj_impl <c10::complex<at::Half>> (c10::complex<at::Half> z) {
  return c10::complex<at::Half>{z.real(), -z.imag()};
}

template<>
inline c10::complex<float> conj_impl <c10::complex<float>> (c10::complex<float> z) {
  return c10::complex<float>(z.real(), -z.imag());
}

template<>
inline c10::complex<double> conj_impl <c10::complex<double>> (c10::complex<double> z) {
  return c10::complex<double>(z.real(), -z.imag());
}

template <typename TYPE>
inline TYPE ceil_impl (TYPE z) {
  return std::ceil(z);
}

template <>
inline c10::complex<float> ceil_impl (c10::complex<float> z) {
  return c10::complex<float>(std::ceil(z.real()), std::ceil(z.imag()));
}

template <>
inline c10::complex<double> ceil_impl (c10::complex<double> z) {
  return c10::complex<double>(std::ceil(z.real()), std::ceil(z.imag()));
}

template<typename T>
inline c10::complex<T> sgn_impl (c10::complex<T> z) {
  if (z == c10::complex<T>(0, 0)) {
    return c10::complex<T>(0, 0);
  } else {
    return z / zabs(z);
  }
}

template <typename TYPE>
inline TYPE floor_impl (TYPE z) {
  return std::floor(z);
}

template <>
inline c10::complex<float> floor_impl (c10::complex<float> z) {
  return c10::complex<float>(std::floor(z.real()), std::floor(z.imag()));
}

template <>
inline c10::complex<double> floor_impl (c10::complex<double> z) {
  return c10::complex<double>(std::floor(z.real()), std::floor(z.imag()));
}

template <typename TYPE>
inline TYPE round_impl (TYPE z) {
  return std::nearbyint(z);
}

template <>
inline c10::complex<float> round_impl (c10::complex<float> z) {
  return c10::complex<float>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <>
inline c10::complex<double> round_impl (c10::complex<double> z) {
  return c10::complex<double>(std::nearbyint(z.real()), std::nearbyint(z.imag()));
}

template <typename TYPE>
inline TYPE trunc_impl (TYPE z) {
  return std::trunc(z);
}

template <>
inline c10::complex<float> trunc_impl (c10::complex<float> z) {
  return c10::complex<float>(std::trunc(z.real()), std::trunc(z.imag()));
}

template <>
inline c10::complex<double> trunc_impl (c10::complex<double> z) {
  return c10::complex<double>(std::trunc(z.real()), std::trunc(z.imag()));
}

template <typename TYPE, std::enable_if_t<!c10::is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::max(a, b);
  }
}

template <typename TYPE, std::enable_if_t<c10::is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    return b;
  } else {
    return std::abs(a) > std::abs(b) ? a : b;
  }
}

template <typename TYPE, std::enable_if_t<!c10::is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::min(a, b);
  }
}

template <typename TYPE, std::enable_if_t<c10::is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    return b;
  } else {
    return std::abs(a) < std::abs(b) ? a : b;
  }
}

} // end namespace
} //end at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

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
- `c10/util/MathConstants.h`


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

- **File Documentation**: `zmath.h_docs.md`
- **Keyword Index**: `zmath.h_kw.md`
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

- **File Documentation**: `zmath.h_docs.md_docs.md`
- **Keyword Index**: `zmath.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

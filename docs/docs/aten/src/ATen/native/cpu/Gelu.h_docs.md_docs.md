# Documentation: `docs/aten/src/ATen/native/cpu/Gelu.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/Gelu.h_docs.md`
- **Size**: 5,604 bytes (5.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/Gelu.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/Gelu.h`
- **Size**: 3,104 bytes (3.03 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// On Windows, math.h needs to be included with _USE_MATH_DEFINES defined to
// access constants such as M_SQRT2 and M_2_SQRTPI.
#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#endif // _WIN32

#include <ATen/cpu/vec/vec.h>
#include <c10/util/BFloat16.h> // For c10::is_reduced_floating_point_v.

namespace at::native {
inline namespace CPU_CAPABILITY {
constexpr double kGeluBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
constexpr double kGeluKappa = 0.044715;

template <typename T>
using reduced_fp_to_float_t = std::conditional_t<c10::is_reduced_floating_point_v<T>, float, T>;

template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
float reduced_fp_to_float(T x) {
  return float(x);
}

template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
T reduced_fp_to_float(T x) {
  return x;
}

template <typename T>
T scalar_gelu_approximated_with_tanh(T x) {
  using opmath_t = reduced_fp_to_float_t<T>;
  auto x_float = reduced_fp_to_float(x);
  auto x_cube = x_float * x_float * x_float;
  auto inner = opmath_t(kGeluBeta) * (x_float + opmath_t(kGeluKappa) * x_cube);
  return opmath_t(0.5) * x_float * (opmath_t(1) + std::tanh(inner));
}

template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kGeluBetaVec((T(kGeluBeta)));
  const vec::Vectorized<T> kGeluKappaVec((T(kGeluKappa)));
  auto x_cube = x * x * x;
  vec::Vectorized<T> inner_vec = kGeluBetaVec * (x + kGeluKappaVec * x_cube);
  return kPointFiveVec * x * (kOneVec + inner_vec.tanh());
}

template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu_approximated_with_tanh(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(
      vectorized_gelu_approximated_with_tanh(x0),
      vectorized_gelu_approximated_with_tanh(x1));
}


template <typename T>
T scalar_gelu(T x) {
  using opmath_t = reduced_fp_to_float_t<T>;
  const auto kAlpha = opmath_t(M_SQRT1_2);
  return reduced_fp_to_float(x) * opmath_t(0.5) * (opmath_t(1) + std::erf(reduced_fp_to_float(x) * kAlpha));
}

template<typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  const vec::Vectorized<T> kAlphaVec(T(M_SQRT1_2));
  const vec::Vectorized<T> kOneVec(T(1));
  const vec::Vectorized<T> kPointFiveVec(T(0.5));
  return x * kPointFiveVec * (kOneVec + (x * kAlphaVec).erf());
}

template<typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
vec::Vectorized<T> vectorized_gelu(vec::Vectorized<T> x) {
  auto [x0, x1] = at::vec::convert_to_float<T>(x);
  return at::vec::convert_from_float<T>(vectorized_gelu(x0), vectorized_gelu(x1));
}

} // namespace CPU_CAPABILITY
} // namespace at::native

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

- `cmath`
- `math.h`
- `ATen/cpu/vec/vec.h`
- `c10/util/BFloat16.h`


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

- **File Documentation**: `Gelu.h_docs.md`
- **Keyword Index**: `Gelu.h_kw.md`
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

- **File Documentation**: `Gelu.h_docs.md_docs.md`
- **Keyword Index**: `Gelu.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/cpu/Elu.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/Elu.h_docs.md`
- **Size**: 5,351 bytes (5.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/Elu.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/Elu.h`
- **Size**: 2,866 bytes (2.80 KB)
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
#endif // _WIN32

#include <ATen/cpu/vec/vec.h>
#include <c10/util/BFloat16.h> // For c10::is_reduced_floating_point_v.

namespace at::native {
inline namespace CPU_CAPABILITY {
/**
 * Return a function object that calculates ELU with the given
 * parameters on its input element.  ParamT is the type of the input
 * and output to the ELU, and MathT is the type (possibly
 * higher-precision, e.g. float if ParamT is reduced-precision float)
 * in which to do intermediate calculations.
 */
template <typename ParamT, typename MathT=ParamT>
auto get_scalar_elu_elementwise_func(MathT alpha, MathT scale, MathT input_scale) {
  const auto negcoef = alpha * scale;
  const auto poscoef = scale;
  const auto negiptcoef = input_scale;
  return [negcoef, negiptcoef, poscoef](ParamT a) -> ParamT {
    return MathT(a) < MathT(0)
      ? std::expm1(MathT(a) * negiptcoef) * negcoef
      : MathT(a) * poscoef;
  };
}

/**
 * Return a function object that calculates ELU with the given
 * parameters on its input element. The function object takes and
 * returns Vectorized<T>.
 */
template <typename T, std::enable_if_t<!c10::is_reduced_floating_point_v<T>, bool> = true>
auto get_vectorized_elu_elementwise_func(T alpha, T scale, T input_scale) {
  const vec::Vectorized<T> negcoef_vec(alpha * scale);
  const vec::Vectorized<T> poscoef_vec(scale);
  const vec::Vectorized<T> negiptcoef_vec(input_scale);
  const vec::Vectorized<T> zero_vec(static_cast<T>(0));
  return [negcoef_vec, poscoef_vec, negiptcoef_vec, zero_vec](vec::Vectorized<T> a) -> vec::Vectorized<T> {
    const auto cmp = a >= zero_vec;
    if (!cmp.zero_mask()) {
      return a * poscoef_vec;
    } else {
      return vec::Vectorized<T>::blendv((a * negiptcoef_vec).expm1() * negcoef_vec, a * poscoef_vec, cmp);
    }
  };
}

/**
 * Return a function object that calculates ELU with the given
 * parameters on its input element. The function object takes and
 * returns Vectorized<ParamT>, and Vectorized<MathT> is the type
 * (possibly higher-precision) in which to do intermediate
 * calculations.
 */
template <typename T, std::enable_if_t<c10::is_reduced_floating_point_v<T>, bool> = true>
auto get_vectorized_elu_elementwise_func(float alpha, float scale, float input_scale) {
  // Takes float->float.
  const auto float_func = get_vectorized_elu_elementwise_func<float>(alpha, scale, input_scale);
  return [float_func](vec::Vectorized<T> a) -> vec::Vectorized<T> {
    auto [a0, a1] = vec::convert_to_float<T>(a);
    auto res0 = float_func(a0);
    auto res1 = float_func(a1);
    return vec::convert_from_float<T>(res0, res1);
  };
}
} // namespace CPU_CAPABILITY
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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

- **File Documentation**: `Elu.h_docs.md`
- **Keyword Index**: `Elu.h_kw.md`
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

- **File Documentation**: `Elu.h_docs.md_docs.md`
- **Keyword Index**: `Elu.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

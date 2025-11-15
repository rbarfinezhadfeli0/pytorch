# Documentation: `docs/aten/src/ATen/cpu/vml.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cpu/vml.h_docs.md`
- **Size**: 8,347 bytes (8.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cpu/vml.h`

## File Metadata

- **Path**: `aten/src/ATen/cpu/vml.h`
- **Size**: 6,073 bytes (5.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/complex.h>

// This header implements various unary operations using a MKL VML style
// interface.

// It implements various functions with a simple interface
// For example it enables the user to call vsin(float* out, const float* in,
// size) This functions takes a pointer to a continuous output array of floats and
// a constant input array. It will then apply sin to each value in the input
// array and write the result into the output array. out and in may point to the
// same memory, i.e. this fully supports in-place operations. These functions
// also implement their own parallelization, so take precautions when calling
// these from threaded functions.

// When MKL is available it will call into MKL's VML library similar to NumPy
// If MKL is not available it will use SLEEF.

// This file might be compiled under AVX or AVX2 when called from e.g.
// UnaryOpsKernel.cpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if AT_MKL_ENABLED() && !defined(__APPLE__)
#include <mkl.h>
#endif


namespace at::vml {
inline namespace CPU_CAPABILITY {

using namespace vec;

template <typename scalar_t>
inline void vrsqrt(scalar_t* out, scalar_t* in, int64_t size) {
  parallel_for(0, size, 2048, [out, in](int64_t begin, int64_t end) {
    map(
        [](const Vectorized<scalar_t>& x) {
          return Vectorized<scalar_t>((scalar_t)1) / x.sqrt();
        },
        out + begin,
        in + begin,
        end - begin);
  });
}

// NB: We ignore numerical errors by convention and leave them to the user

#define IMPLEMENT_VML(op)                                               \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    using vec_t = Vectorized<vec_scalar_t<scalar_t>>;                   \
    vec::map([](vec_t x) { return x.op(); }, out, in, size);            \
  }                                                                     \

IMPLEMENT_VML(abs)
IMPLEMENT_VML(acos)
IMPLEMENT_VML(asin)
IMPLEMENT_VML(atan)
IMPLEMENT_VML(atanh)
IMPLEMENT_VML(ceil)
IMPLEMENT_VML(cos)
// IMPLEMENT_VML(cosh)
IMPLEMENT_VML(erf)
IMPLEMENT_VML(erfc)
IMPLEMENT_VML(erfinv)
IMPLEMENT_VML(exp)
IMPLEMENT_VML(expm1)
IMPLEMENT_VML(floor)
IMPLEMENT_VML(i0)
IMPLEMENT_VML(i0e)
IMPLEMENT_VML(digamma)
IMPLEMENT_VML(reciprocal)
IMPLEMENT_VML(log)
IMPLEMENT_VML(log10)
IMPLEMENT_VML(log1p)
IMPLEMENT_VML(log2)
IMPLEMENT_VML(neg)
IMPLEMENT_VML(sin)
// IMPLEMENT_VML(sinh)
IMPLEMENT_VML(sqrt)
IMPLEMENT_VML(round)
IMPLEMENT_VML(rsqrt)
IMPLEMENT_VML(tan)
IMPLEMENT_VML(tanh)
IMPLEMENT_VML(trunc)
IMPLEMENT_VML(lgamma)


#if AT_MKL_ENABLED() && !defined(__APPLE__)

// NB: LP64 MKL is the most commonly used and thus we assume it here. That means
// we need to expect MKL_INT to be of type int, which implies int32_t or int64_t in most
// cases.
static_assert(
    std::is_same_v<MKL_INT, int32_t> || std::is_same_v<MKL_INT, int64_t>,
    "MKL_INT is assumed to be int32_t or int64_t");
#define IMPLEMENT_VML_MKL_STUB(op, mklop, type, mkltype)                \
  template <>                                                           \
  inline void v##op(type * out, const type * in, int64_t size) {        \
    auto constexpr max_mkl_ind = std::numeric_limits<MKL_INT>::max();   \
    if (size <= static_cast<int64_t>(max_mkl_ind)) {                    \
      vm##mkltype##mklop(                                               \
          size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE); \
    } else {                                                            \
      int64_t ind = 0;                                                  \
      int64_t chunks = size / max_mkl_ind;                              \
      int64_t rest = size % max_mkl_ind;                                \
      for (; ind < chunks; ind++) {                                     \
        vm##mkltype##mklop(                                             \
            max_mkl_ind,                                                \
            in + ind * max_mkl_ind,                                     \
            out + ind * max_mkl_ind,                                    \
            VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);              \
      }                                                                 \
      vm##mkltype##mklop(                                               \
          rest,                                                         \
          in + ind * max_mkl_ind,                                       \
          out + ind * max_mkl_ind,                                      \
          VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);                \
    }                                                                   \
  }

#define IMPLEMENT_VML_MKL(op, mklop)          \
  IMPLEMENT_VML_MKL_STUB(op, mklop, float, s) \
  IMPLEMENT_VML_MKL_STUB(op, mklop, double, d)

// NB: abs, cosh and sinh were temporarily disabled due to issues with Apple
// NB: expm1 is disabled because on some configs it produces expm1(nan)=-1
IMPLEMENT_VML_MKL(acos, Acos)
IMPLEMENT_VML_MKL(asin, Asin)
IMPLEMENT_VML_MKL(atan, Atan)
IMPLEMENT_VML_MKL(cos, Cos)
// IMPLEMENT_VML_MKL(cosh, Cosh)
IMPLEMENT_VML_MKL(erf, Erf)
IMPLEMENT_VML_MKL(erfc, Erfc)
IMPLEMENT_VML_MKL(erfinv, ErfInv)
IMPLEMENT_VML_MKL(exp, Exp)
// IMPLEMENT_VML_MKL(expm1, Expm1)
IMPLEMENT_VML_MKL(log, Ln)
IMPLEMENT_VML_MKL(log10, Log10)
IMPLEMENT_VML_MKL(sin, Sin)
// IMPLEMENT_VML_MKL(sinh, Sinh)
IMPLEMENT_VML_MKL(sqrt, Sqrt)
IMPLEMENT_VML_MKL(tan, Tan)
IMPLEMENT_VML_MKL(tanh, Tanh)
IMPLEMENT_VML_MKL(trunc, Trunc)

// Not vectorized in MKL version tested
// IMPLEMENT_VML_MKL(abs, Abs)
// IMPLEMENT_VML_MKL(log1p, Log1p)

#if INTEL_MKL_VERSION >= 20180406
IMPLEMENT_VML_MKL(log2, Log2)
#endif

#endif

} // namespace
} // namespace at::vml

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vec`, `CPU_CAPABILITY`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/Parallel.h`
- `ATen/OpMathType.h`
- `ATen/cpu/vec/functional.h`
- `ATen/cpu/vec/vec.h`
- `c10/util/complex.h`
- `algorithm`
- `cstddef`
- `cstdint`
- `cstring`
- `type_traits`
- `mkl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/cpu`):

- [`FlushDenormal.cpp_docs.md`](./FlushDenormal.cpp_docs.md)
- [`Utils.h_docs.md`](./Utils.h_docs.md)
- [`FlushDenormal.h_docs.md`](./FlushDenormal.h_docs.md)
- [`Utils.cpp_docs.md`](./Utils.cpp_docs.md)


## Cross-References

- **File Documentation**: `vml.h_docs.md`
- **Keyword Index**: `vml.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen/cpu`):

- [`FlushDenormal.cpp_docs.md_docs.md`](./FlushDenormal.cpp_docs.md_docs.md)
- [`FlushDenormal.cpp_kw.md_docs.md`](./FlushDenormal.cpp_kw.md_docs.md)
- [`FlushDenormal.h_docs.md_docs.md`](./FlushDenormal.h_docs.md_docs.md)
- [`Utils.cpp_docs.md_docs.md`](./Utils.cpp_docs.md_docs.md)
- [`vml.h_kw.md_docs.md`](./vml.h_kw.md_docs.md)
- [`Utils.cpp_kw.md_docs.md`](./Utils.cpp_kw.md_docs.md)
- [`FlushDenormal.h_kw.md_docs.md`](./FlushDenormal.h_kw.md_docs.md)
- [`Utils.h_kw.md_docs.md`](./Utils.h_kw.md_docs.md)
- [`Utils.h_docs.md_docs.md`](./Utils.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `vml.h_docs.md_docs.md`
- **Keyword Index**: `vml.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

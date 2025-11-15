# Documentation: `caffe2/perfkernels/batch_box_cox_avx512.cc`

## File Metadata

- **Path**: `caffe2/perfkernels/batch_box_cox_avx512.cc`
- **Size**: 3,278 bytes (3.20 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef CAFFE2_PERF_USE_MKL
#include <immintrin.h>

// Enable compiler vectorized version only if numerical consistency is not
// required between dev and opt versions - disabled for now
#ifndef FAST_VECTORIZED_KERNEL
#define CPU_CAPABILITY_AVX512
#include <ATen/cpu/vec/vec.h>

namespace at::vec {
namespace {
// Implements the vectorized version of std::max() operation,
// which DOESNOT propagates NaN for second argument
template <typename scalar_t>
Vectorized<scalar_t> max(const Vectorized<scalar_t>& a, const Vectorized<scalar_t>& b);

template <>
Vectorized<double> max(const Vectorized<double>& a, const Vectorized<double>& b) {
  // std::max(NaN, nonNan) -> NaN
  return _mm512_max_pd(b, a);
}

template <>
Vectorized<float> max(const Vectorized<float>& a, const Vectorized<float>& b) {
  // std::max(NaN, nonNan) -> NaN
  return _mm512_max_ps(b, a);
}

// Implements recieprocal method based on newton-rapson method
// 1. user RCP approximiation
// 2. update with RCP = RCP * (2 - X * RCP)
template <typename scalar_t>
Vectorized<scalar_t> fast_recieprocal(const Vectorized<scalar_t>& b);
template <typename scalar_t>
scalar_t fast_recieprocal(scalar_t b);

template<>
Vectorized<float> fast_recieprocal(const Vectorized<float>& b) {
  auto minus2 = _mm512_set1_ps(-2.f);
  auto rcp = _mm512_rcp14_ps(b);
  rcp = _mm512_mul_ps(rcp,  _mm512_fnmsub_ps(rcp, b, minus2));
  rcp = _mm512_mul_ps(rcp,  _mm512_fnmsub_ps(rcp, b, minus2));
  return rcp;
}

template <>
float fast_recieprocal(float b) {
  auto minus2 = _mm_set_ss(-2.f);
  auto b_reg = _mm_set_ss(b);
  auto rcp = _mm_rcp_ss(b_reg);
  rcp = _mm_mul_ss(rcp,  _mm_fnmsub_ss(rcp, b_reg, minus2));
  rcp = _mm_mul_ss(rcp,  _mm_fnmsub_ss(rcp, b_reg, minus2));
  return _mm_cvtss_f32(rcp);
}

template<>
Vectorized<double> fast_recieprocal(const Vectorized<double>& b) {
  auto minus2 = _mm512_set1_pd(-2.);
  auto rcp = _mm512_rcp14_pd(b);
  rcp = _mm512_mul_pd(rcp,  _mm512_fnmsub_pd(rcp, b, minus2));
  rcp = _mm512_mul_pd(rcp,  _mm512_fnmsub_pd(rcp, b, minus2));
  return rcp;
}

template <>
double fast_recieprocal(double b) {
  return 1./b;
}
} // namespace
} // namespace at::vec
#endif

#include "caffe2/perfkernels/batch_box_cox_vec.h"

namespace caffe2::details {

template <typename T>
void compute_batch_box_cox__avx512(
    std::size_t N,
    std::size_t D,
    std::size_t block_size,
    const T* self_data,
    const T* __restrict lambda1_data,
    const T* __restrict lambda2_data,
    T* output_data) {
      compute_batch_box_cox_vec_fma<T>(
          N,
          D,
          block_size,
          self_data,
          lambda1_data,
          lambda2_data,
          output_data);
    }

// Vectorized version specializations for float and double
template
void compute_batch_box_cox__avx512<float>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const float* self_data,
  const float* __restrict lambda1_data,
  const float* __restrict lambda2_data,
  float* output_data);

template
void compute_batch_box_cox__avx512<double>(
  std::size_t N,
  std::size_t D,
  std::size_t block_size,
  const double* self_data,
  const double* __restrict lambda1_data,
  const double* __restrict lambda2_data,
  double* output_data);

} // namespace caffe2::detail
#endif // CAFFE2_PERF_USE_MKL

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `caffe2/perfkernels`, which is part of the **Caffe2** deep learning framework.



## Dependencies

### Import Dependencies

This file includes:

- `immintrin.h`
- `ATen/cpu/vec/vec.h`
- `caffe2/perfkernels/batch_box_cox_vec.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`caffe2/perfkernels`):

- [`embedding_lookup_idx_avx2.cc_docs.md`](./embedding_lookup_idx_avx2.cc_docs.md)
- [`common.h_docs.md`](./common.h_docs.md)
- [`hp_emblookup_codegen.py_docs.md`](./hp_emblookup_codegen.py_docs.md)
- [`common_avx2.cc_docs.md`](./common_avx2.cc_docs.md)
- [`embedding_lookup_idx.cc_docs.md`](./embedding_lookup_idx.cc_docs.md)
- [`batch_box_cox_vec.h_docs.md`](./batch_box_cox_vec.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`sve_emblookup_codegen.py_docs.md`](./sve_emblookup_codegen.py_docs.md)
- [`embedding_lookup_idx_sve.cc_docs.md`](./embedding_lookup_idx_sve.cc_docs.md)


## Cross-References

- **File Documentation**: `batch_box_cox_avx512.cc_docs.md`
- **Keyword Index**: `batch_box_cox_avx512.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

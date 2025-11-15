# Documentation: `docs/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp_docs.md`
- **Size**: 52,868 bytes (51.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp`
- **Size**: 181,255 bytes (177.01 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TopKImpl.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UpSample.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/quantized/FakeQuantAffine.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>
#include <c10/util/Unroll.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#endif

#include <cmath>
#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_NEON__) || defined(__aarch64__)
#include <ATen/quantized/Quantizer.h>
#include <arm_neon.h>
#endif


// NOLINTBEGIN(*-c-arrays)
namespace at::native {
namespace {

void check_tensor_memory_format(const Tensor& ref, const Tensor& other) {
  TORCH_CHECK(
      ref.is_contiguous(ref.suggest_memory_format()),
      "Quantized tensor should be contiguous");
  TORCH_CHECK(
      other.is_contiguous(ref.suggest_memory_format()),
      "Float tensor should be contiguous "
      "in same memory format as quantized tensor");
}

// ****************** HEY YOU! YES YOU! Read this! ********************
//
// Please read the README.md in this directory before editing this file

template <bool ReLUFused = false>
Tensor qcat_nhwc_kernel(
    const MaterializedITensorListRef& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  const at::Tensor& qx0 = qxs[0];
  int64_t C_out = 0;
  std::vector<int64_t> Cs_in;
  // Prefix sum of input channels for fast indexing
  std::vector<int64_t> Cs_sum;
  std::vector<double> scales;
  std::vector<int64_t> zero_pts;
  std::vector<void*> data_ptrs;
  std::vector<bool> is_fast_path;

  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(
        qx.dim() == qx0.dim(),
        "Tensors must have the same number of dimensions: got ",
        qx.dim(),
        " and ",
        qx0.dim());
#define CHECK_DIM(d)                                            \
  TORCH_CHECK(                                                  \
      qx.size(d) == qx0.size(d),                                \
      "Sizes of tensors must match expect in dimension 1. Got", \
      qx.size(d),                                               \
      " and ",                                                  \
      qx0.size(d));
    CHECK_DIM(0);
    CHECK_DIM(2);
    CHECK_DIM(3);
    TORCH_CHECK(
        qx.scalar_type() == qx0.scalar_type(),
        "Expected object of scalar type ",
        toString(qx0.scalar_type()),
        " but got scalar type ",
        toString(qx.scalar_type()));
    Cs_in.push_back(qx.size(1));
    Cs_sum.push_back(C_out);
    C_out += qx.size(1);
    scales.push_back(qx.q_scale());
    zero_pts.push_back(qx.q_zero_point());
    data_ptrs.push_back(qx.data_ptr());
    is_fast_path.push_back(
        qx.q_scale() == scale &&
        qx.q_zero_point() == zero_point);
  }

  const int64_t N = qx0.size(0);
  const int64_t H = qx0.size(2);
  const int64_t W = qx0.size(3);
  float inv_scale = static_cast<float>(1.0 / scale);

  auto output = at::_empty_affine_quantized(
      {N, C_out, H, W},
      qx0.options().memory_format(MemoryFormat::ChannelsLast),
      scale,
      zero_point,
      std::nullopt);

  // N, H, and W are explicitly captured here because there's a bug in GCC5
  // and clang5 which causes an internal compiler error if they're not
  AT_DISPATCH_QINT_TYPES(output.scalar_type(), "qcat_nhwc", [&, N, H, W]() {
    using Vec = Vectorized<scalar_t>;
    at::parallel_for(0, N * H * W, 0, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        // loop over input tensors
        for (const auto tidx : c10::irange(Cs_in.size())) {
          scalar_t::underlying* optr =
              reinterpret_cast<scalar_t::underlying*>(output.data_ptr()) +
              i * C_out + Cs_sum[tidx];

          auto curr_C = Cs_in[tidx];
          float curr_scale = scales[tidx];
          int64_t curr_zero_pt = zero_pts[tidx];

          scalar_t::underlying* iptr =
              reinterpret_cast<scalar_t::underlying*>(data_ptrs[tidx]) +
              i * curr_C;

          if (is_fast_path[tidx] && !ReLUFused) {
            std::memcpy(optr, iptr, curr_C * sizeof(typename scalar_t::underlying));
            continue;
          }

          constexpr auto VLEN = Vec::size();
          int64_t c = 0;

          // Vectorized loop
          if (c + VLEN <= curr_C) {
            auto curr_scale_vec = Vectorized<float>(curr_scale);
            auto curr_zero_pt_vec = Vectorized<float>(curr_zero_pt);
            auto scale_neg_zp_premul = curr_scale_vec * curr_zero_pt_vec.neg();
            for (; c + VLEN <= curr_C; c += VLEN) {
              auto inp_vec = Vec::loadu(iptr + c);
              auto float_values = inp_vec.dequantize(
                  curr_scale_vec, curr_zero_pt_vec, scale_neg_zp_premul);
              Vec::float_vec_return_type retvals;
              for (int i = 0; i < Vec::float_num_vecs(); ++i) {
                if constexpr (ReLUFused) {
                  retvals[i] =
                      vec::maximum(float_values[i], Vectorized<float>(0.0f));
                } else {
                  retvals[i] = float_values[i];
                }
              }
              auto quantized =
                  Vec::quantize(retvals, scale, zero_point, inv_scale);
              quantized.store(optr + c);
            }
          }

          // Vectorized loop for channel between 8 and 32 (avx2)
          constexpr auto kVLEN = Vectorized<float>::size();
          int64_t elem_size = curr_C - c;
          if ((VLEN == 4 * kVLEN) && elem_size >= kVLEN) {
            auto curr_scale_vec = Vectorized<float>(curr_scale);
            auto curr_zero_pt_vec = Vectorized<float>(curr_zero_pt);
            auto scale_neg_zp_premul = curr_scale_vec * curr_zero_pt_vec.neg();
            int64_t vec_num = elem_size / kVLEN;
            std::array<typename scalar_t::underlying, VLEN> buf_in{};
            memcpy(buf_in.data(), iptr + c, vec_num * kVLEN);
            auto inp_vec = Vec::loadu(buf_in.data());
            auto float_values = inp_vec.dequantize(
                curr_scale_vec, curr_zero_pt_vec, scale_neg_zp_premul);
            Vec::float_vec_return_type retvals;
            for (int i = 0; i < vec_num; ++i) {
              if constexpr (ReLUFused) {
                retvals[i] =
                    vec::maximum(float_values[i], Vectorized<float>(0.0f));
              } else {
                retvals[i] = float_values[i];
              }
            }
            auto quantized =
                Vec::quantize(retvals, scale, zero_point, inv_scale);
            quantized.store(optr + c, vec_num * kVLEN);
            c += vec_num * kVLEN;
          }

          // Scalar loop
          for (; c < curr_C; ++c) {
            auto float_val = at::native::dequantize_val(
                curr_scale,
                curr_zero_pt,
                reinterpret_cast<scalar_t*>(iptr)[c]);
            if constexpr (ReLUFused) {
              float_val = std::max(0.0f, float_val);
            }
            optr[c] = at::native::quantize_val<scalar_t>(
                          scale, zero_point, float_val)
                          .val_;
          } // for c
        } // for tidx
      } // for i
    });
  });

  return output;
}

// horizontal sum over a range of uint8_t
int64_t hsum(const uint8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_v = _mm256_setzero_si256();
  __m256i one_epi16_v = _mm256_set1_epi16(1);
  __m256i one_epi8_v = _mm256_set1_epi8(1);
  // vectorized
  for (; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    sum_v = _mm256_add_epi32(
      sum_v,
      _mm256_madd_epi16(
        // first argument is unsigned, second is signed
        _mm256_maddubs_epi16(src_v, one_epi8_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v = _mm512_setzero_si512();
  __m512i one_epi16_v = _mm512_set1_epi16(1);
  __m512i one_epi8_v = _mm512_set1_epi8(1);
  // vectorized
  for (; i < len / 64 * 64; i += 64) {
    __m512i src_v = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    sum_v = _mm512_add_epi32(
      sum_v,
      _mm512_madd_epi16(
        // first argument is unsigned, second is signed
        _mm512_maddubs_epi16(src_v, one_epi8_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[16];
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);
  for (const auto k : c10::irange(16)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

  return row_sum;
}

// horizontal sum over a range of int8_t
int64_t hsum(const int8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_v = _mm256_setzero_si256();
  __m256i one_epi16_v = _mm256_set1_epi16(1);
  __m256i one_epi8_v = _mm256_set1_epi8(1);
  // vectorized
  for (; i < len / 32 * 32; i += 32) {
    __m256i src_v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    sum_v = _mm256_add_epi32(
      sum_v,
      _mm256_madd_epi16(
        // first argument is unsigned, second is signed
        _mm256_maddubs_epi16(one_epi8_v, src_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v = _mm512_setzero_si512();
  __m512i one_epi16_v = _mm512_set1_epi16(1);
  __m512i one_epi8_v = _mm512_set1_epi8(1);
  // vectorized
  for (; i < len / 64 * 64; i += 64) {
    __m512i src_v = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    sum_v = _mm512_add_epi32(
      sum_v,
      _mm512_madd_epi16(
        // first argument is unsigned, second is signed
        _mm512_maddubs_epi16(one_epi8_v, src_v),
      one_epi16_v)
    );
  }

  alignas(64) int32_t temp[16];
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v);
  for (const auto k : c10::irange(16)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

  return row_sum;
}

// horizontal sum over a range of int32_t
int64_t hsum(const int32_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256i sum_epi64 = _mm256_setzero_si256();
  // vectorized
  for (; i < len / 8 * 8; i += 8) {
    __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    // widen
    __m128i src_lo_epi32 = _mm256_castsi256_si128(src_epi32);
    __m128i src_hi_epi32 = _mm256_extracti128_si256(src_epi32, 1);
    __m256i src_lo_epi64 = _mm256_cvtepi32_epi64(src_lo_epi32);
    __m256i src_hi_epi64 = _mm256_cvtepi32_epi64(src_hi_epi32);
    // add
    sum_epi64 = _mm256_add_epi64(sum_epi64, src_lo_epi64);
    sum_epi64 = _mm256_add_epi64(sum_epi64, src_hi_epi64);
  }

  alignas(64) int64_t temp[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_epi64);
  for (const auto k : c10::irange(4)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_epi64 = _mm512_setzero_si512();
  // vectorized
  for (; i < len / 16 * 16; i += 16) {
    __m512i src_epi32 = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    // widen
    __m256i src_lo_epi32 = _mm512_castsi512_si256(src_epi32);
    __m256i src_hi_epi32 = _mm512_extracti32x8_epi32(src_epi32, 1);
    __m512i src_lo_epi64 = _mm512_cvtepi32_epi64(src_lo_epi32);
    __m512i src_hi_epi64 = _mm512_cvtepi32_epi64(src_hi_epi32);
    // add
    sum_epi64 = _mm512_add_epi64(sum_epi64, src_lo_epi64);
    sum_epi64 = _mm512_add_epi64(sum_epi64, src_hi_epi64);
  }

  alignas(64) int64_t temp[8];
  _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_epi64);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i];
  }

  return row_sum;
}

// horizontal sum of squares over a range of uint8_t
int64_t hsum_sq(const uint8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  // vectorized
  __m256i sum_v_epu32 = _mm256_setzero_si256();
  alignas(64) int32_t temp[8];
  int overflow_threshold = 262144; // 2147483647(max of int32)/(256*256)*8 = 262144
  int loop = len / overflow_threshold + 1;
  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
      // (i15, ..., i0)
      __m128i src_epu8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
      __m256i src_epu16 = _mm256_cvtepu8_epi16(src_epu8);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_epu16 = _mm256_mullo_epi16(src_epu16, src_epu16);
      // (i7 ^ 2, ..., i0 ^ 2)
      __m128i sq_lo_epu16 = _mm256_castsi256_si128(sq_epu16);
      // (i15 ^ 2, ..., i8 ^ 2)
      __m128i sq_hi_epu16 = _mm256_extractf128_si256(sq_epu16, 1);
      // widen to epu32
      __m256i sq_lo_epu32 = _mm256_cvtepu16_epi32(sq_lo_epu16);
      __m256i sq_hi_epu32 = _mm256_cvtepu16_epi32(sq_hi_epu16);
      // add to running sum
      sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_lo_epu32);
      sum_v_epu32 = _mm256_add_epi32(sum_v_epu32, sq_hi_epu32);
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epu32);
    for (const auto k : c10::irange(8)) {
      row_sum += temp[k];
    }
    sum_v_epu32 = _mm256_setzero_si256();
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512i sum_v_epu32 = _mm512_setzero_si512();
  alignas(64) int32_t temp[16];
  int overflow_threshold = 262144; // 2147483647(max of int32)/(512*512)*8 = 262144
  int loop = len / overflow_threshold + 1;
  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 32 * 32)); i += 32) {
      // (i31, ..., i0)
      __m256i src_epu8 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
      __m512i src_epu16 = _mm512_cvtepu8_epi16(src_epu8);
      // (i31 ^ 2, ..., i0 ^ 2)
      __m512i sq_epu16 = _mm512_mullo_epi16(src_epu16, src_epu16);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_lo_epu16 = _mm512_castsi512_si256(sq_epu16);
      // (i31 ^ 2, ..., i16 ^ 2)
      __m256i sq_hi_epu16 = _mm512_extracti32x8_epi32(sq_epu16, 1);
      // widen to epu32
      __m512i sq_lo_epu32 = _mm512_cvtepu16_epi32(sq_lo_epu16);
      __m512i sq_hi_epu32 = _mm512_cvtepu16_epi32(sq_hi_epu16);
      // add to running sum
      sum_v_epu32 = _mm512_add_epi32(sum_v_epu32, sq_lo_epu32);
      sum_v_epu32 = _mm512_add_epi32(sum_v_epu32, sq_hi_epu32);
    }
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v_epu32);
    for (const auto k : c10::irange(16)) {
      row_sum += temp[k];
    }
    sum_v_epu32 = _mm512_setzero_si512();
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i] * A[i];
  }

  return row_sum;
}

// horizontal sum of squares over a range of int8_t
int64_t hsum_sq(const int8_t* A, int len) {
  int64_t row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  // vectorized
  __m256i sum_v_epi32 = _mm256_setzero_si256();
  alignas(64) int32_t temp[8];

  int overflow_threshold = 1048576; //2147483647/(128*128)*8 = 1048576
  int loop = len / overflow_threshold + 1;

  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 16 * 16)); i += 16) {
      // (i15, ..., i0)
      __m128i src_epi8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(A + i));
      __m256i src_epi16 = _mm256_cvtepi8_epi16(src_epi8);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_epi16 = _mm256_mullo_epi16(src_epi16, src_epi16);
      // (i7 ^ 2, ..., i0 ^ 2)
      __m128i sq_lo_epi16 = _mm256_castsi256_si128(sq_epi16);
      // (i15 ^ 2, ..., i8 ^ 2)
      __m128i sq_hi_epi16 = _mm256_extractf128_si256(sq_epi16, 1);
      // widen to epi32
      __m256i sq_lo_epi32 = _mm256_cvtepi16_epi32(sq_lo_epi16);
      __m256i sq_hi_epi32 = _mm256_cvtepi16_epi32(sq_hi_epi16);
      // add to running sum
      sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_lo_epi32);
      sum_v_epi32 = _mm256_add_epi32(sum_v_epi32, sq_hi_epi32);
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(temp), sum_v_epi32);

    for (const auto k : c10::irange(8)) {
      row_sum += temp[k];
    }
    sum_v_epi32 = _mm256_setzero_si256();
  }
#elif defined(CPU_CAPABILITY_AVX512)
  // vectorized
  __m512i sum_v_epi32 = _mm512_setzero_si512();
  alignas(64) int32_t temp[16];

  int overflow_threshold = 1048576; //2147483647/(256*256)*8 = 1048576
  int loop = len / overflow_threshold + 1;

  for(int j=0; j<=loop; j++){
    for (; ((i < overflow_threshold * j) && (i < len / 32 * 32)); i += 32) {
      // (i31, ..., i0)
      __m256i src_epi8 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
      __m512i src_epi16 = _mm512_cvtepi8_epi16(src_epi8);
      // (i31 ^ 2, ..., i0 ^ 2)
      __m512i sq_epi16 = _mm512_mullo_epi16(src_epi16, src_epi16);
      // (i15 ^ 2, ..., i0 ^ 2)
      __m256i sq_lo_epi16 = _mm512_castsi512_si256(sq_epi16);
      // (i31 ^ 2, ..., i16 ^ 2)
      __m256i sq_hi_epi16 = _mm512_extracti32x8_epi32(sq_epi16, 1);
      // widen to epi32
      __m512i sq_lo_epi32 = _mm512_cvtepi16_epi32(sq_lo_epi16);
      __m512i sq_hi_epi32 = _mm512_cvtepi16_epi32(sq_hi_epi16);
      // add to running sum
      sum_v_epi32 = _mm512_add_epi32(sum_v_epi32, sq_lo_epi32);
      sum_v_epi32 = _mm512_add_epi32(sum_v_epi32, sq_hi_epi32);
    }
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_v_epi32);

    for (const auto k : c10::irange(16)) {
      row_sum += temp[k];
    }
    sum_v_epi32 = _mm512_setzero_si512();
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    row_sum += A[i] * A[i];
  }

  return row_sum;
}

// horizontal sum os squares over a range of int32_t
// floats throughout are necessary to prevent overflow
float hsum_sq(const int32_t* A, int len) {
  float row_sum = 0;
  int i = 0;

#ifdef CPU_CAPABILITY_AVX2
  __m256 sum_ps = _mm256_setzero_ps();
  // vectorized
  for (; i < len / 8 * 8; i += 8) {
    __m256i src_epi32 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(A + i));
    __m256 src_ps = _mm256_cvtepi32_ps(src_epi32);
    sum_ps = _mm256_add_ps(sum_ps, _mm256_mul_ps(src_ps, src_ps));
  }

  alignas(64) float temp[8];
  _mm256_store_ps(temp, sum_ps);
  for (const auto k : c10::irange(8)) {
    row_sum += temp[k];
  }
#elif defined(CPU_CAPABILITY_AVX512)
  __m512 sum_ps = _mm512_setzero_ps();
  // vectorized
  for (; i < len / 16 * 16; i += 16) {
    __m512i src_epi32 = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(A + i));
    __m512 src_ps = _mm512_cvtepi32_ps(src_epi32);
    sum_ps = _mm512_add_ps(sum_ps, _mm512_mul_ps(src_ps, src_ps));
  }

  alignas(64) float temp[16];
  _mm512_store_ps(temp, sum_ps);
  for (const auto k : c10::irange(16)) {
    row_sum += temp[k];
  }
#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

  // scalar
  for (; i < len; ++i) {
    int64_t cur = static_cast<int64_t>(A[i]);
    row_sum += (float)cur * (float)cur;
  }

  return row_sum;
}

void qrelu_kernel(const Tensor& qx, Tensor& qy) {
  const auto zero_point = qx.q_zero_point();
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qrelu", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto zero_point_vec = Vec(scalar_t(zero_point));
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        },
        [&](Vec value) -> Vec { return value.relu(zero_point_vec); });
  });
}

void leaky_qrelu_out_kernel(Tensor& out, const Tensor& qx,
                                   const Scalar& negval_) {
  int64_t i_zp = qx.q_zero_point();
  float i_scale = static_cast<float>(qx.q_scale());

  int64_t o_zp = out.q_zero_point();
  float o_scale = static_cast<float>(out.q_scale());
  float o_inv_scale = 1.0f / o_scale;

  float negval = negval_.to<float>();

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "leaky_qrelu", [&] {
    using Vec = Vectorized<float>;  // Naive implementation uses dequant/quant loop.
    using qVec = Vectorized<scalar_t>;
    Vec zero_vec = Vec(0.0f);
    Vec one_vec = Vec(1.0f);

    Vec i_scale_vec = Vec(i_scale);
    Vec i_zp_vec = Vec(i_zp);
    Vec i_scale_zp_neg_premul_vec = i_scale_vec * i_zp_vec.neg();

    Vec negval_vec = Vec(negval);

    auto iter = TensorIterator::unary_op(out, qx);

    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          auto value_dx = at::native::dequantize_val(i_scale, i_zp, value_qx);
          auto value_dy = value_dx > 0 ? value_dx : value_dx * negval;
          return at::native::quantize_val<scalar_t>(o_scale, o_zp, value_dy);
        },
        [&](qVec qx_vec) -> qVec {
          /* Vectorized implementation creates a multiplicand vector, which has
           * "alpha" for all negative dx values and ones-vector for all
           * positive values of dx. The multiplicand then is multiplied by the
           * input.
           */
          auto dx_vec_vec = qx_vec.dequantize(i_scale_vec, i_zp_vec,
                                              i_scale_zp_neg_premul_vec);
          for (auto & dx_vec : dx_vec_vec) {
            const auto multiplicand = Vec::blendv(negval_vec, one_vec,
                                                  dx_vec > zero_vec);
            dx_vec *= multiplicand;
          }
          return qVec::quantize(dx_vec_vec, o_scale, o_zp, o_inv_scale);
        });
  });
}

void qprelu_out_kernel(Tensor& out,
                              const Tensor& qx,
                              const Tensor& qw) {
  int32_t i_zp = static_cast<int32_t>(qx.q_zero_point());
  float i_scale = static_cast<float>(qx.q_scale());

  int32_t w_zp = static_cast<int32_t>(qw.q_zero_point());
  float w_scale = static_cast<float>(qw.q_scale());

  int32_t o_zp = static_cast<int32_t>(out.q_zero_point());
  float o_scale = static_cast<float>(out.q_scale());
  float o_inv_scale = 1.0f / o_scale;

  float multiplier = i_scale * w_scale * o_inv_scale;

  int64_t input_ndim = qx.dim();
  TORCH_CHECK(input_ndim > 0, "qprelu: zero-dim input tensor is not allowed.");

  // This logic is present in at::prelu and repeated here, as this path can be
  // hit via quantized::prelu, which is registered under quantized/cpu/qprelu.cpu
  auto qw_nd = qw;
  if (input_ndim != qw_nd.dim()) {
    DimVector dim_w(input_ndim, 1);
    if (input_ndim > 1) {
      dim_w[1] = qw.numel();
    }
    // This will always be a view in CPU/CUDA, but some backends
    // like MKLDNN do not support views
    qw_nd = qw_nd.reshape(dim_w);
  }

  auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_input(qx)
    .add_input(qw_nd)
    .build();

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qprelu", [&] {
    using qVec = Vectorized<scalar_t>;
    qVec i_zp_vec = qVec(static_cast<scalar_t>(i_zp));
    qVec w_zp_vec = qVec(static_cast<scalar_t>(w_zp));

    // Quantized one as weight
    auto qw_one = at::native::quantize_val<scalar_t>(w_scale, w_zp, 1.0f);
    qVec vec_qw_one = qVec(qw_one);
    auto vec_qw_one_sub_zp = vec_qw_one.widening_subtract(w_zp_vec)[0];
    int32_t qw_one_sub_zp = qw_one.val_ - w_zp;

    cpu_kernel_vec(
      iter,
      [=](scalar_t val_qx, scalar_t val_qw) -> scalar_t {
        int32_t qx_pos = std::max(static_cast<int32_t>(val_qx.val_), i_zp);
        int32_t qx_neg = std::min(static_cast<int32_t>(val_qx.val_), i_zp);
        int32_t qx_pos_sub_zp = qx_pos - i_zp;
        int32_t qx_neg_sub_zp = qx_neg - i_zp;
        int32_t qw_sub_zp = val_qw.val_ - w_zp;
        auto qy_sub_zp = qx_pos_sub_zp * qw_one_sub_zp + qx_neg_sub_zp * qw_sub_zp;
        return at::native::requantize_from_int<scalar_t>(
            multiplier, o_zp, qy_sub_zp);
      },
      [=](qVec vec_qx, qVec vec_qw) -> qVec {
        auto vec_qx_pos = vec_qx.maximum(i_zp_vec);
        auto vec_qx_neg = vec_qx.minimum(i_zp_vec);
        qVec::int_vec_return_type qx_pos_sub_zp = vec_qx_pos.widening_subtract(i_zp_vec);
        qVec::int_vec_return_type qx_neg_sub_zp = vec_qx_neg.widening_subtract(i_zp_vec);
        qVec::int_vec_return_type qw_sub_zp = vec_qw.widening_subtract(w_zp_vec);
        qVec::int_vec_return_type qy_sub_zp;
        for (const auto i : c10::irange(qVec::int_num_vecs())) {
          qy_sub_zp[i] = qx_pos_sub_zp[i] * vec_qw_one_sub_zp + qx_neg_sub_zp[i] * qw_sub_zp[i];
        }
        return qVec::requantize_from_int(qy_sub_zp, multiplier, o_zp);
      });
  });

}

void qgelu_kernel(const Tensor& qx, Tensor& qy, GeluType approximate) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();
  int64_t output_zero_point = zero_point;
  float output_scale = scale;
  float inv_output_scale = 1.0 / output_scale;
  const auto kAlphaVec = Vectorized<float>(M_SQRT1_2);
  const auto kBetaVec = Vectorized<float>(M_SQRT2 * M_2_SQRTPI * 0.5);
  const auto kKappaVec = Vectorized<float>(0.044715);
  const auto kOneVec = Vectorized<float>(1);
  const auto kPointFiveVec = Vectorized<float>(0.5);

  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qgelu", [&]() {
      qy = at::_empty_affine_quantized(
          qx.sizes(),
          at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          output_scale,
          output_zero_point,
          std::nullopt);
      auto iter = TensorIterator::unary_op(qy, qx);

      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(
          iter,
          [&](scalar_t value_qx) -> scalar_t {
            const auto value_dx =
                at::native::dequantize_val(scale, zero_point, value_qx);

            const auto kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
            const auto kKappa = 0.044715;
            const auto x_cube = value_dx * value_dx * value_dx;
            const auto inner = kBeta * (value_dx + kKappa * x_cube);
            const auto value_dy = 0.5 * value_dx * (1.0 + std::tanh(inner));

            return at::native::quantize_val<scalar_t>(
                output_scale, output_zero_point, value_dy);
          },
          [&](Vec value_qx) -> Vec {
            auto value_dx = value_qx.dequantize(
                scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
            for (auto & value : value_dx) {
              auto value_cube = value * value * value;
              auto inner = kBetaVec * (value + kKappaVec * value_cube);
              value = kPointFiveVec * value * (kOneVec + inner.tanh());
            }
            return Vec::quantize(
                value_dx, output_scale, output_zero_point, inv_output_scale);
          });
    });
  } else {
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qgelu", [&]() {
      qy = at::_empty_affine_quantized(
          qx.sizes(),
          at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
          output_scale,
          output_zero_point,
          std::nullopt);
      auto iter = TensorIterator::unary_op(qy, qx);

      using Vec = Vectorized<scalar_t>;
      cpu_kernel_vec(
          iter,
          [&](scalar_t value_qx) -> scalar_t {
            const auto value_dx =
                at::native::dequantize_val(scale, zero_point, value_qx);
            const auto value_dy =
                value_dx * 0.5 * (1 + std::erf(value_dx * M_SQRT1_2));
            return at::native::quantize_val<scalar_t>(
                output_scale, output_zero_point, value_dy);
          },
          [&](Vec value_qx) -> Vec {
            auto value_dx = value_qx.dequantize(
                scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
            for (auto & value : value_dx) {
              value = value * kPointFiveVec * (kOneVec + (value * kAlphaVec).erf());
            }
            return Vec::quantize(
                value_dx, output_scale, output_zero_point, inv_output_scale);
          });
    });
  }
}


void qsigmoid_kernel(
    const Tensor& qx, Tensor& qy, double output_scale, int64_t output_zero_point ) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        std::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          const auto value_dy = 1.0f / (1.0 + std::exp((-value_dx)));
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, value_dy);
        },
        [&](Vec value_qx) -> Vec {
          auto value_dx = value_qx.dequantize(scale_vec, zero_point_vec);
          for (auto & value : value_dx) {
            value = value.neg();
            value = value.exp();
            value = Vectorized<float>(1.0f) + value;
            value = value.reciprocal();
          }
          return Vec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

void qhardsigmoid_kernel(const Tensor& qx, Tensor& qy) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardsigmoid", [&]() {

    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    float output_scale = 0.00390625;  // 1.0 / 2^8
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    }
    float inv_output_scale = 1.0 / output_scale;

    // The default zero-point is zero.  As a one-off optimization for
    // kQInt8, we set the zero-point to -128 to maximize precision in the
    // [0, 1] output range. kQInt32 can be handled in a future PR if needed.
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE),
        output_scale,
        output_zero_point,
        qx.suggest_memory_format());
    auto iter = TensorIterator::unary_op(qy, qx);

    using qVec = Vectorized<scalar_t>;
    using fVec = Vectorized<float>;
    fVec kZeroVec(0.0f);
    fVec kThreeVec(3.0f);
    fVec kSixVec(6.0f);

    // Naive implementation: uses dequantize/execute/quantize routine
    cpu_kernel_vec(
        iter,
        [&](scalar_t qx) -> scalar_t {
          auto x = at::native::dequantize_val(scale, zero_point, qx);
          const auto y = std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, y);
        },
        [&](qVec value_qx) -> qVec {
          auto value_dx = value_qx.dequantize(
              scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
          for (auto & value : value_dx) {
            value =
                vec::minimum(
                    vec::maximum(value + kThreeVec, kZeroVec),
                    kSixVec) /
                kSixVec;
          }
          return qVec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

void qclamp_kernel(
    const Tensor& qx,
    const Scalar& min_scalar,
    const Scalar& max_scalar,
    Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    auto max = max_scalar.to<float>();
    scalar_t min_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), min);
    scalar_t max_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), max);
    auto min_vec = Vec(min_q);
    auto max_vec = Vec(max_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          underlying_t min_clamped =
              std::max<underlying_t>(value.val_, min_q.val_);
          return scalar_t(std::min<underlying_t>(min_clamped, max_q.val_));
        },
        [&](Vec val) -> Vec {
          auto min_clamped = val.maximum(min_vec);
          return min_clamped.minimum(max_vec);
        });
  });
}

void qclamp_min_kernel(const Tensor& qx, const Scalar& min_scalar, Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto min = min_scalar.to<float>();
    scalar_t min_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), min);
    auto min_vec = Vec(min_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, min_q.val_));
        },
        [&](Vec val) -> Vec { return val.maximum(min_vec); });
  });
}

void qclamp_max_kernel(const Tensor& qx, const Scalar& max_scalar, Tensor& qy) {
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qclamp", [&]() {
    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU)
            .dtype(SCALAR_TYPE)
            .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        std::nullopt);
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    auto max = max_scalar.to<float>();
    scalar_t max_q = at::native::quantize_val<scalar_t>(
        qx.q_scale(), qx.q_zero_point(), max);
    auto max_vec = Vec(max_q);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          return scalar_t(std::min<underlying_t>(value.val_, max_q.val_));
        },
        [&](Vec val) -> Vec { return val.minimum(max_vec); });
  });
}

void qthreshold_kernel(
  // TODO: For future tasks, since output quantization parameters are set equal to
  // the input ones, it might make sense to implement this completely in the
  // quantized domain.
   const Tensor& qx,
   const Scalar& threshold_scalar,
   const Scalar& value_scalar,
   Tensor& qy) {

  // defines input and output scales and zero_points
  int64_t input_zero_point = qx.q_zero_point();
  float input_scale = static_cast<float>(qx.q_scale());
  int64_t output_zero_point = qy.q_zero_point();
  float output_scale = static_cast<float>(qy.q_scale());
  float inv_output_scale = static_cast<float>(1.0 / output_scale);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qthreshold", [&]() {
    qy = at::_empty_affine_quantized(
      qx.sizes(),
      at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
      qx.q_scale(),
      qx.q_zero_point(),
      std::nullopt);

    // vectorized
    using Vec = Vectorized<float>;
    using qVec = Vectorized<scalar_t>;
    // defines the iterator
    auto iter = TensorIterator::unary_op(qy, qx);
    // defines the vectorized versions
    Vec input_scale_vec = Vec(input_scale);
    Vec input_zero_point_vec = Vec(input_zero_point);
    Vec input_scale_neg_zp_premul_vec = input_scale_vec * input_zero_point_vec.neg();
    // defines the floating-point versions of threshold and value
    float threshold_float = threshold_scalar.to<float>();
    float value_float = value_scalar.to<float>();
    Vec threshold_vec = Vec(threshold_float);
    Vec value_vec = Vec(value_float);

    // Naive implementation: uses dequantize/execute/quantize routine
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          // dequantize
          const auto x = at::native::dequantize_val(input_scale, input_zero_point, value_qx);
          // Applies the Threshold operation
          const auto y = x > threshold_float ? x : value_float;
          // quantize
          return at::native::quantize_val<scalar_t>(output_scale, output_zero_point, y);
        },
        [&](qVec value_qx) -> qVec {
          // dequantize
          auto dx_vec = value_qx.dequantize(
            input_scale_vec, input_zero_point_vec, input_scale_neg_zp_premul_vec);
          for (auto & value : dx_vec) {
            // check if any elements are below threshold
            const auto cmp_to_threshold = value > threshold_vec;
            if (cmp_to_threshold.zero_mask()) {
              // blend
              value = Vec::blendv(value_vec, value, cmp_to_threshold);
            }
          }
          // quantize
          return qVec::quantize(dx_vec, output_scale, output_zero_point, inv_output_scale);
        });
  });
}


void qhardswish_kernel(const Tensor& qx, Tensor& qy) {
  const auto i_scale = qx.q_scale();
  const auto i_zero_point = qx.q_zero_point();

  const auto o_scale = qy.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const float o_inv_scale = static_cast<float>(1.0 / o_scale);

  using fVec = Vectorized<float>;
  fVec i_scale_vec(i_scale);
  fVec i_zero_point_vec(i_zero_point);
  fVec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();
  fVec zero_vec(0.0f);
  fVec three_vec(3.0f);
  fVec six_vec(6.0f);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qhardswish", [&]() {
    using qVec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(qy, qx);
    cpu_kernel_vec(
        iter,
        [&](scalar_t value) -> scalar_t {
          const auto x =
              at::native::dequantize_val(i_scale, i_zero_point, value);
          const auto y = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
          return at::native::quantize_val<scalar_t>(o_scale, o_zero_point, y);
        },
        [&](qVec value) -> qVec {
          auto value_dx = value.dequantize(i_scale_vec, i_zero_point_vec,
                                           i_scale_neg_zp_premul_vec);
          for (auto & value : value_dx) {
            value = value * vec::minimum(
              vec::maximum(value + three_vec, zero_vec),
              six_vec
            ) / six_vec;
          }
          return qVec::quantize(value_dx, o_scale, o_zero_point, o_inv_scale);
        });
  });
}


void qtanh_kernel(const Tensor& qx, Tensor& qy) {
  int64_t zero_point = qx.q_zero_point();
  float scale = static_cast<float>(qx.q_scale());
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>(zero_point);
  auto scale_neg_zp_premul_vec = scale_vec * zero_point_vec.neg();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qtanh", [&]() {
    // Naive implementation: uses dequantize/execute/quantize routine
    // - Output scale is set to 2.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    float output_scale = 0.0078125;  // 2.0 / 512
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 4.656612873077393e-10;  // 2.0 / 2^32
    } else if (SCALAR_TYPE == at::kQUInt8) {
      output_zero_point = 128;
    }
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        std::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, std::tanh(value_dx));
        },
        [&](Vec value_qx) -> Vec {
          const auto value_dx = value_qx.dequantize(
              scale_vec, zero_point_vec, scale_neg_zp_premul_vec);
          Vec::float_vec_return_type retvals;
          for (const auto idx : c10::irange(Vec::float_num_vecs())) {
            retvals[idx] = value_dx[idx].tanh();
          }
          return Vec::quantize(
              retvals, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

void qelu_kernel(
    const Tensor& qx,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& qy) {
  // scale and input_scale arguments refer to a generalized ELU formula
  // if x >= 0, ELU(x) = x * scale
  // if x <= 0, ELU(x) = (exp(x * input_scale) - 1) * scale
  // in the normal ELU formula, both are equal to 1
  // they are NOT related to the quantization scale term

  int64_t i_zp = qx.q_zero_point();
  float i_scale = static_cast<float>(qx.q_scale());

  // In a future PR, we can improve on output scale and zero_point
  // selection.
  int64_t o_zp = qy.q_zero_point();
  float o_scale = static_cast<float>(qy.q_scale());
  float inv_o_scale = static_cast<float>(1.0 / o_scale);

  float alpha_float = alpha.to<float>();
  float scale_coef = scale.to<float>();
  float input_scale_coef = input_scale.to<float>();

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qelu_kernel", [&] {

    auto iter = TensorIterator::unary_op(qy, qx);

    // vectorized
    using Vec = Vectorized<float>;
    using qVec = Vectorized<scalar_t>;

    Vec zero_vec = Vec(0.0f);
    Vec one_vec = Vec(1.0f);
    Vec alpha_vec = Vec(alpha_float);
    Vec scale_coef_vec = Vec(scale_coef);
    Vec input_scale_coef_vec = Vec(input_scale_coef);
    Vec i_scale_vec = Vec(i_scale);
    Vec i_zero_point_vec = Vec(i_zp);
    Vec i_scale_neg_zp_premul_vec = i_scale_vec * i_zero_point_vec.neg();

    cpu_kernel_vec(
      iter,
      [&](scalar_t value_qx) -> scalar_t {
        // dequantize
        const auto x = at::native::dequantize_val(i_scale, i_zp, value_qx);
        // ELU
        const auto y = x >= 0
          ? x * scale_coef
          : ((std::exp(x * input_scale_coef) - 1) * alpha_float * scale_coef);

        // quantize
        return at::native::quantize_val<scalar_t>(o_scale, o_zp, y);
      },
      [&](qVec value_qx) -> qVec {
        // dequantize
        auto dx_vec_vec = value_qx.dequantize(i_scale_vec, i_zero_point_vec,
                                            i_scale_neg_zp_premul_vec);
        for (auto & value : dx_vec_vec) {
          // quickly check if any elements are below zero
          const auto cmp_to_zero = value > zero_vec;

          if (cmp_to_zero.zero_mask()) {

            Vec dx_vec_copy_neg_elu = value * one_vec;
            // calculate the negative part of ELU on the copy
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * input_scale_coef_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu.exp();
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu - one_vec;
            dx_vec_copy_neg_elu = dx_vec_copy_neg_elu * alpha_vec;
            // blend
            value = Vec::blendv(dx_vec_copy_neg_elu, value,
                                        value > zero_vec);
          }

          value = value * scale_coef_vec;
        }
        // quantize
        return qVec::quantize(dx_vec_vec, o_scale, o_zp, inv_o_scale);
      }
    );

  });
}

// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self and out are of the same dtype.
// Note: other is already assumed to be in int32, i.e., it's
// round(float/self_scale)
template <bool ReLUFused = false>
void qadd_scalar_kernel(Tensor& out, const Tensor& self, const Scalar& other) {
  int64_t zero_point = out.q_zero_point();
  float scale = static_cast<float>(out.q_scale());
  float inv_scale = 1.0f / scale;
  int64_t self_zero_point = self.q_zero_point();
  float self_scale = static_cast<float>(self.q_scale());

  float multiplier = self_scale * inv_scale;

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    using Vec = Vectorized<scalar_t>;
    auto iter = TensorIterator::unary_op(out, self);
    auto other_val = other.to<int32_t>();
    auto other_vec = Vectorized<c10::qint32>(static_cast<c10::qint32>(other_val));
    cpu_kernel_vec(
        iter,
        [&](scalar_t a) -> scalar_t {
          int32_t a_sub_z = static_cast<int32_t>(a.val_) -
              static_cast<int32_t>(self_zero_point);
          int32_t c = a_sub_z + other_val;
          scalar_t res = at::native::requantize_from_int<scalar_t>(
              multiplier, zero_point, c);
          if constexpr (ReLUFused) {
            res.val_ = std::max<scalar_t::underlying>(res.val_, zero_point);
          }
          return res;
        },
        [&](Vec a) -> Vec {
          Vec::int_vec_return_type a_sub_z =
              a.widening_subtract(Vec(static_cast<scalar_t>(self_zero_point)));
          Vec::int_vec_return_type c;
          for (const auto i : c10::irange(Vec::int_num_vecs())) {
            c[i] = a_sub_z[i] + other_vec;
          }
          Vec rv = Vec::requantize_from_int(c, multiplier, zero_point);
          if constexpr (ReLUFused) {
            rv = rv.maximum(Vec(static_cast<scalar_t>(zero_point)));
          }
          return rv;
        });
  });
}
// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self, other, out are of the same dtype.
template <bool ReLUFused = false>
void qadd_kernel(Tensor& out, const Tensor& self, const Tensor& other) {
  int64_t zero_point = out.q_zero_point();
  float scale = static_cast<float>(out.q_scale());
  float inv_scale = 1.0f / scale;
  int64_t self_zero_point = self.q_zero_point();
  float self_scale = static_cast<float>(self.q_scale());
  int64_t other_zero_point = other.q_zero_point();
  float other_scale = static_cast<float>(other.q_scale());

  // Broadcast out the parameters here to amortize out that cost across
  // loop iterations.
  // TODO: we can optimize dequantization by doing a premultiplication
  // of the zero point by scale and doing FMA on scale*x_q - (scale*zero_point)
  auto self_zero_point_vec = Vectorized<float>(self_zero_point);
  auto self_scale_vec = Vectorized<float>(self_scale);
  auto other_zero_point_vec = Vectorized<float>(other_zero_point);
  auto other_scale_vec = Vectorized<float>(other_scale);

  auto self_scale_neg_zp_premul_vec = self_scale_vec * self_zero_point_vec.neg();
  auto other_scale_zp_premul_vec = other_scale_vec * other_zero_point_vec.neg();

  auto iter = TensorIterator::borrowing_binary_op(out, self, other);

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t a, scalar_t b) -> scalar_t {
          const auto da =
              at::native::dequantize_val(self_scale, self_zero_point, a);
          const auto db =
              at::native::dequantize_val(other_scale, other_zero_point, b);
          float c = da + db;
          if (ReLUFused) {
            c = std::max<float>(c, 0.0);
          }
          return at::native::quantize_val<scalar_t>(scale, zero_point, c);
        },
        [&](Vec a, Vec b) -> Vec {
          const auto da = a.dequantize(
              self_scale_vec, self_zero_point_vec, self_scale_neg_zp_premul_vec);
          const auto db = b.dequantize(
              other_scale_vec, other_zero_point_vec, other_scale_zp_premul_vec);
          Vec::float_vec_return_type retvals;
          for (const auto i : c10::irange(Vec::float_num_vecs())) {
            auto c = da[i] + db[i];
            if constexpr (ReLUFused) {
              c = vec::maximum(c, Vectorized<float>(0.0f));
          
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/kernels`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/kernels`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/kernels`):

- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)
- [`QuantizedOpKernels.cpp_kw.md_docs.md`](./QuantizedOpKernels.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `QuantizedOpKernels.cpp_docs.md_docs.md`
- **Keyword Index**: `QuantizedOpKernels.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

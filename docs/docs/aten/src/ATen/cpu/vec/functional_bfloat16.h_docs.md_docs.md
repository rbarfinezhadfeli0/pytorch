# Documentation: `docs/aten/src/ATen/cpu/vec/functional_bfloat16.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cpu/vec/functional_bfloat16.h_docs.md`
- **Size**: 27,780 bytes (27.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cpu/vec/functional_bfloat16.h`

## File Metadata

- **Path**: `aten/src/ATen/cpu/vec/functional_bfloat16.h`
- **Size**: 25,411 bytes (24.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec.h>

namespace at::vec {
// BFloat16 specification
template <typename scalar_t>
struct VecScalarType {
  using type = scalar_t;
};
template <>
struct VecScalarType<BFloat16> {
  using type = float;
};
template <>
struct VecScalarType<Half> {
  using type = float;
};

// This is different from at::acc_type since we only need to specialize BFloat16
template <typename scalar_t>
using vec_scalar_t = typename VecScalarType<scalar_t>::type;

// Vector conversion between float and bfloat16/half
template <>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float<
    BFloat16>(const Vectorized<BFloat16>& a) {
  return convert_bfloat16_float(a);
}

template <>
inline std::tuple<Vectorized<float>, Vectorized<float>> convert_to_float<Half>(
    const Vectorized<Half>& a) {
  return convert_half_float(a);
}

template <>
inline Vectorized<BFloat16> convert_from_float<BFloat16>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return convert_float_bfloat16(a, b);
}

template <>
inline Vectorized<Half> convert_from_float<Half>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return convert_float_half(a, b);
}

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void load_to_float(
    const scalar_t* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2);

template <>
inline void load_to_float<BFloat16>(
    const BFloat16* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  load_fp32_from_bf16(data, out1, out2);
}

template <>
inline void load_to_float<Half>(
    const Half* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  load_fp32_from_fp16(data, out1, out2);
}

template <
    typename scalar_t,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void load_to_float(const scalar_t* data, Vectorized<float>& out);

template <>
inline void load_to_float<BFloat16>(
    const BFloat16* data,
    Vectorized<float>& out) {
  load_fp32_from_bf16(data, out);
}

template <>
inline void load_to_float<Half>(const Half* data, Vectorized<float>& out) {
  load_fp32_from_fp16(data, out);
}

// Note that we already have specialized member of Vectorized<scalar_t> for
// BFloat16 so the following functions would run smoothly:
//   using Vec = Vectorized<BFloat16>;
//   Vec one = Vec(BFloat16(1));
//   vec::map([](Vec x) { return one / (one + x.exp()); }, y_ptr, x_ptr, N);
//
// Then why we still need to specialize "functional"?
//   If we do specialization at Vectorized<> level, the above example would need
//   3 pairs of conversion of bf16->fp32/fp32->bf16, each for ".exp()", "+" and
//   "/". If we do specialization at vec::map<>() level, we have only 1 pair of
//   conversion of bf16->fp32/fp32->bf16, for the input and output BFloat16
//   vector only.
//
// The following BFloat16 functionality will only do data type conversion for
// input and output vector (reduce functionality will only convert the final
// scalar back to bf16). Compared to Vectorized<> specialization,
//   1. better performance since we have less data type conversion;
//   2. less rounding error since immediate results are kept in fp32;
//   3. accumulation done on data type of fp32.
//
//  If you plan to extend this file, please ensure adding unit tests at
//    aten/src/ATen/test/vec_test_all_types.cpp
//
template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline float reduce_all(const Op& vec_fun, const scalar_t* data, int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    if (size > fVec::size()) {
      data_fvec0 = fVec::set(
          data_fvec0, vec_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(vec_fun, data_fvec0, fVec::size());
    } else {
      return vec_reduce_all<float>(vec_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = vec_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    if (size - d > fVec::size()) {
      acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(
          acc_fvec1, vec_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      acc_fvec0 =
          fVec::set(acc_fvec0, vec_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = vec_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(vec_fun, acc_fvec0);
}

template <
    typename scalar_t,
    typename Op1,
    typename Op2,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline std::pair<float, float> reduce2_all(
    const Op1& vec_fun1,
    const Op2& vec_fun2,
    const scalar_t* data,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    if (size > fVec::size()) {
      fVec acc1_fvec = fVec::set(
          data_fvec0, vec_fun1(data_fvec0, data_fvec1), size - fVec::size());
      fVec acc2_fvec = fVec::set(
          data_fvec0, vec_fun2(data_fvec0, data_fvec1), size - fVec::size());
      return std::pair<scalar_t, scalar_t>(
          vec_reduce_all<float>(vec_fun1, acc1_fvec, fVec::size()),
          vec_reduce_all<float>(vec_fun2, acc2_fvec, fVec::size()));
    } else {
      return std::pair<scalar_t, scalar_t>(
          vec_reduce_all<float>(vec_fun1, data_fvec0, size),
          vec_reduce_all<float>(vec_fun2, data_fvec0, size));
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  auto [acc1_fvec0, acc1_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  auto [acc2_fvec0, acc2_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
    acc1_fvec1 = vec_fun1(acc1_fvec1, data_fvec1);
    acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
    acc2_fvec1 = vec_fun2(acc2_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    if (size - d > fVec::size()) {
      acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
      acc1_fvec1 = fVec::set(
          acc1_fvec1,
          vec_fun1(acc1_fvec1, data_fvec1),
          size - d - fVec::size());
      acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
      acc2_fvec1 = fVec::set(
          acc2_fvec1,
          vec_fun2(acc2_fvec1, data_fvec1),
          size - d - fVec::size());
    } else {
      acc1_fvec0 =
          fVec::set(acc1_fvec0, vec_fun1(acc1_fvec0, data_fvec0), size - d);
      acc2_fvec0 =
          fVec::set(acc2_fvec0, vec_fun2(acc2_fvec0, data_fvec0), size - d);
    }
  }
  acc1_fvec0 = vec_fun1(acc1_fvec0, acc1_fvec1);
  acc2_fvec0 = vec_fun2(acc2_fvec0, acc2_fvec1);
  return std::pair<scalar_t, scalar_t>(
      vec_reduce_all<float>(vec_fun1, acc1_fvec0),
      vec_reduce_all<float>(vec_fun2, acc2_fvec0));
}

template <
    typename scalar_t,
    typename MapOp,
    typename ReduceOp,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline float map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0);
      data_fvec1 = map_fun(data_fvec1);
      data_fvec0 = fVec::set(
          data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  acc_fvec0 = map_fun(acc_fvec0);
  acc_fvec1 = map_fun(acc_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    data_fvec0 = map_fun(data_fvec0);
    data_fvec1 = map_fun(data_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0);
      data_fvec1 = map_fun(data_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(
          acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0);
      acc_fvec0 =
          fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <
    typename scalar_t,
    typename MapOp,
    typename ReduceOp,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline float map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(data2, size);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      data_fvec0 = fVec::set(
          data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  bVec acc2_bvec = bVec::loadu(data2);
  auto [acc2_fvec0, acc2_fvec1] = convert_to_float<scalar_t>(acc2_bvec);
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0);
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    data_fvec0 = map_fun(data_fvec0, data2_fvec0);
    data_fvec1 = map_fun(data_fvec1, data2_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(
          acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      acc_fvec0 =
          fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <
    typename scalar_t,
    typename MapOp,
    typename ReduceOp,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline float map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    const scalar_t* data3,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(data2, size);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3, size);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      data_fvec0 = fVec::set(
          data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  auto [acc_fvec0, acc_fvec1] = convert_to_float<scalar_t>(acc_bvec);
  bVec acc2_bvec = bVec::loadu(data2);
  auto [acc2_fvec0, acc2_fvec1] = convert_to_float<scalar_t>(acc2_bvec);
  bVec acc3_bvec = bVec::loadu(data3);
  auto [acc3_fvec0, acc3_fvec1] = convert_to_float<scalar_t>(acc3_bvec);
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0, acc3_fvec0);
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1, acc3_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3 + d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
    data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3 + d, size - d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(
          acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      acc_fvec0 =
          fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !(!detail::should_prefer_converting_through_float_v<scalar_t> &&
          std::is_invocable_v<Op, vec::Vectorized<scalar_t>>),
        int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const float* input_data,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    fVec data_fvec0 = fVec::loadu(input_data + d);
    fVec data_fvec1 = fVec::loadu(input_data + d + fVec::size());
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    fVec data_fvec0, data_fvec1;
    if (size - d > fVec::size()) {
      data_fvec0 = fVec::loadu(input_data + d);
      data_fvec1 =
          fVec::loadu(input_data + d + fVec::size(), size - d - fVec::size());
    } else {
      // choose to align with behaviour of bVec::loadu(ptr, size),
      // which leaves data_fvec1 uninitialized
      data_fvec0 = fVec::loadu(input_data + d, size - d);
    }
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !(!detail::should_prefer_converting_through_float_v<scalar_t> &&
          std::is_invocable_v<
              Op,
              vec::Vectorized<scalar_t>,
              vec::Vectorized<scalar_t>>),
        int> = 0>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    auto [data_fvec0, data_fvec1] = convert_to_float<scalar_t>(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !(!detail::should_prefer_converting_through_float_v<scalar_t> &&
          std::is_invocable_v<
              Op,
              vec::Vectorized<scalar_t>,
              vec::Vectorized<scalar_t>,
              vec::Vectorized<scalar_t>>),
        int> = 0>
inline void map3(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data1_bvec = bVec::loadu(input_data1 + d);
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data1_bvec = bVec::loadu(input_data1 + d, size - d);
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d, size - d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <
    typename scalar_t,
    typename Op,
    typename std::enable_if_t<
        !(!detail::should_prefer_converting_through_float_v<scalar_t> &&
          std::is_invocable_v<
              Op,
              vec::Vectorized<scalar_t>,
              vec::Vectorized<scalar_t>,
              vec::Vectorized<scalar_t>,
              vec::Vectorized<scalar_t>>),
        int> = 0>
inline void map4(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    const scalar_t* input_data4,
    int64_t size) {
  using bVec = vec::Vectorized<scalar_t>;
  using fVec = vec::Vectorized<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data1_bvec = bVec::loadu(input_data1 + d);
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    bVec data4_bvec = bVec::loadu(input_data4 + d);
    auto [data4_fvec0, data4_fvec1] = convert_to_float<scalar_t>(data4_bvec);
    fVec output_fvec0 =
        vec_fun(data1_fvec0, data2_fvec0, data3_fvec0, data4_fvec0);
    fVec output_fvec1 =
        vec_fun(data1_fvec1, data2_fvec1, data3_fvec1, data4_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data1_bvec = bVec::loadu(input_data1 + d, size - d);
    auto [data1_fvec0, data1_fvec1] = convert_to_float<scalar_t>(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    auto [data2_fvec0, data2_fvec1] = convert_to_float<scalar_t>(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d, size - d);
    auto [data3_fvec0, data3_fvec1] = convert_to_float<scalar_t>(data3_bvec);
    bVec data4_bvec = bVec::loadu(input_data4 + d, size - d);
    auto [data4_fvec0, data4_fvec1] = convert_to_float<scalar_t>(data4_bvec);
    fVec output_fvec0 =
        vec_fun(data1_fvec0, data2_fvec0, data3_fvec0, data4_fvec0);
    fVec output_fvec1 =
        vec_fun(data1_fvec1, data2_fvec1, data3_fvec1, data4_fvec1);
    bVec output_bvec = convert_from_float<scalar_t>(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

} // namespace at::vec

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `VecScalarType`, `VecScalarType`, `VecScalarType`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cpu/vec`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cpu/vec/vec.h`


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

Files in the same folder (`aten/src/ATen/cpu/vec`):

- [`vec_n.h_docs.md`](./vec_n.h_docs.md)
- [`functional_base.h_docs.md`](./functional_base.h_docs.md)
- [`intrinsics.h_docs.md`](./intrinsics.h_docs.md)
- [`vec_quant.h_docs.md`](./vec_quant.h_docs.md)
- [`vec_base.h_docs.md`](./vec_base.h_docs.md)
- [`functional.h_docs.md`](./functional.h_docs.md)
- [`vec_half.h_docs.md`](./vec_half.h_docs.md)
- [`vec.h_docs.md`](./vec.h_docs.md)
- [`vec_mask.h_docs.md`](./vec_mask.h_docs.md)


## Cross-References

- **File Documentation**: `functional_bfloat16.h_docs.md`
- **Keyword Index**: `functional_bfloat16.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cpu/vec`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cpu/vec`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/cpu/vec`):

- [`vec_half.h_kw.md_docs.md`](./vec_half.h_kw.md_docs.md)
- [`functional.h_docs.md_docs.md`](./functional.h_docs.md_docs.md)
- [`vec_mask.h_kw.md_docs.md`](./vec_mask.h_kw.md_docs.md)
- [`vec_mask.h_docs.md_docs.md`](./vec_mask.h_docs.md_docs.md)
- [`vec_n.h_kw.md_docs.md`](./vec_n.h_kw.md_docs.md)
- [`vec_base.h_kw.md_docs.md`](./vec_base.h_kw.md_docs.md)
- [`vec.h_kw.md_docs.md`](./vec.h_kw.md_docs.md)
- [`intrinsics.h_docs.md_docs.md`](./intrinsics.h_docs.md_docs.md)
- [`functional_base.h_docs.md_docs.md`](./functional_base.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `functional_bfloat16.h_docs.md_docs.md`
- **Keyword Index**: `functional_bfloat16.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

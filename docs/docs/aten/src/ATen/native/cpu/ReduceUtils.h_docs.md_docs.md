# Documentation: `docs/aten/src/ATen/native/cpu/ReduceUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/ReduceUtils.h_docs.md`
- **Size**: 11,450 bytes (11.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/ReduceUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/ReduceUtils.h`
- **Size**: 8,709 bytes (8.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Parallel.h>
#include <ATen/NumericUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReductionType.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>
#include <ATen/native/cpu/utils.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

using namespace vec;

#define AT_DISPATCH_REDUCTION_TYPES(op, ...)                                   \
  [&] {                                                                        \
    switch (op) {                                                              \
      case ReductionType::SUM: {                                               \
        static constexpr auto reduce = ReductionType::SUM;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MEAN: {                                              \
        static constexpr auto reduce = ReductionType::MEAN;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MIN: {                                               \
        static constexpr auto reduce = ReductionType::MIN;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::MAX: {                                               \
        static constexpr auto reduce = ReductionType::MAX;                     \
        return __VA_ARGS__();                                                  \
      }                                                                        \
      case ReductionType::PROD: {                                              \
        static constexpr auto reduce = ReductionType::PROD;                    \
        return __VA_ARGS__();                                                  \
      }                                                                        \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType reduce>
inline vec_scalar_t<scalar_t> init_value() {
  using acc_t = vec_scalar_t<scalar_t>;
  acc_t val;
  if (reduce == ReductionType::SUM ||
      reduce == ReductionType::MEAN) {
    val = static_cast<acc_t>(0);
  } else if (reduce == ReductionType::PROD) {
    val = static_cast<acc_t>(1);
  } else if (reduce == ReductionType::MAX) {
    val = -std::numeric_limits<acc_t>::infinity();
  } else {
    TORCH_INTERNAL_ASSERT(reduce == ReductionType::MIN);
    val = std::numeric_limits<acc_t>::infinity();
  }
  return val;
}

template <typename scalar_t, ReductionType reduce>
inline vec_scalar_t<scalar_t> init_value(const std::optional<Scalar>& initial) {
  using acc_t = vec_scalar_t<scalar_t>;
  if (initial.has_value()) {
    return initial.value().to<acc_t>();
  } else {
    return init_value<scalar_t, reduce>();
  }
}

template <typename scalar_t>
inline void init(scalar_t* out, int64_t size, const vec_scalar_t<scalar_t>& val) {
  using Vec = Vectorized<vec_scalar_t<scalar_t>>;
  map<scalar_t>(
      [val](Vec x) { return Vec(val); },
      out,
      out,
      size);
}

template <typename scalar_t, ReductionType reduce>
inline void init(scalar_t* out, int64_t size, const std::optional<Scalar>& initial) {
  using acc_t = vec_scalar_t<scalar_t>;
  acc_t val = init_value<scalar_t, reduce>(initial);
  init(out, size, val);
}

// overload with `include_self`, used by scatter_reduce
template <typename scalar_t, ReductionType reduce>
inline void init(scalar_t* out, int64_t size, bool include_self = false) {
  using acc_t = vec_scalar_t<scalar_t>;
  if (!include_self) {
    acc_t val = init_value<scalar_t, reduce>();
    init(out, size, val);
  }
}

template <typename scalar_t, ReductionType reduce>
inline void _init(scalar_t* self_ptr, at::opmath_type<scalar_t>* buffer_ptr, int64_t size, bool include_self) {
  if (!include_self) {
    init<at::opmath_type<scalar_t>, reduce>(buffer_ptr, size, include_self);
  } else {
    vec::convert(self_ptr, buffer_ptr, size);
  }
}

template <typename scalar_t>
inline std::enable_if_t<!std::is_same_v<scalar_t, Vec2>, scalar_t>
_max(const scalar_t& x, const scalar_t& y) {
  return at::_isnan(y) ? y : std::max(x, y);
}

template <typename scalar_t>
inline Vectorized<scalar_t> _max(const Vectorized<scalar_t>& x, const Vectorized<scalar_t>& y) {
  // vec::maximum propagates NaN
  return vec::maximum(x, y);
}

template <typename vec_t>
inline std::enable_if_t<std::is_same_v<vec_t, Vec2>, Vec2>
_max(const vec_t& x, const vec_t& y) {
  // vec::maximum propagates NaN
  return maximum(x, y);
}

template <typename scalar_t>
inline std::enable_if_t<!std::is_same_v<scalar_t, Vec2>, scalar_t>
_min(const scalar_t& x, const scalar_t& y) {
  return at::_isnan(y) ? y : std::min(x, y);
}

template <typename scalar_t>
inline Vectorized<scalar_t> _min(const Vectorized<scalar_t>& x, const Vectorized<scalar_t>& y) {
  // vec::minimum propagates NaN
  return vec::minimum(x, y);
}

template <typename vec_t>
inline std::enable_if_t<std::is_same_v<vec_t, Vec2>, Vec2>
_min(const vec_t& x, const vec_t& y) {
  // vec::minimum propagates NaN
  return minimum(x, y);
}

template <typename scalar_t, typename accumut, typename Op,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void map_acc(
    const Op& vec_fun,
    accumut* output_data,
    const accumut* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  using aVec = vec::Vectorized<accumut>;
  int64_t d = 0;
  constexpr int64_t kVecSize = Vec::size();
  constexpr int64_t kaVecSize = aVec::size();
  for (d = 0; d < size - (size % kVecSize); d += kVecSize) {
    Vec data2_vec = Vec::loadu(input_data2 + d);
    auto [data2_avec0, data2_avec1] = convert_to_float<scalar_t>(data2_vec);
    aVec input_vec0 = aVec::loadu(input_data + d);
    aVec input_vec1 = aVec::loadu(input_data + d + kaVecSize);
    vec_fun(input_vec0, data2_avec0).store(output_data + d);
    vec_fun(input_vec1, data2_avec1).store(output_data + d + kaVecSize);
  }
  if (size - d > 0) {
    int64_t tail_size = size - d;
    Vec data2_vec = Vec::loadu(input_data2 + d, tail_size);
    auto [data2_avec0, data2_avec1] = convert_to_float<scalar_t>(data2_vec);
    if (tail_size > kaVecSize) {
      aVec input_vec0 = aVec::loadu(input_data + d);
      aVec input_vec1 = aVec::loadu(input_data + d + kaVecSize, tail_size - kaVecSize);
      vec_fun(input_vec0, data2_avec0).store(output_data + d);
      vec_fun(input_vec1, data2_avec1).store(output_data + d + kaVecSize, tail_size - kaVecSize);
    } else {
      aVec input_vec0 = aVec::loadu(input_data + d, tail_size);
      vec_fun(input_vec0, data2_avec0).store(output_data + d, tail_size);
    }
  }
}

// for Max and Min, propagate NaN:
template <typename T, ReductionType reduce>
inline T update(const T& x, const T& y) {
  if (reduce == ReductionType::SUM ||
      reduce == ReductionType::MEAN) {
    return x + y;
  } else if (reduce == ReductionType::PROD) {
    return x * y;
  } else if (reduce == ReductionType::MAX) {
    return _max(x, y);
  } else {
    TORCH_INTERNAL_ASSERT(reduce == ReductionType::MIN);
    return _min(x, y);
  }
}

template <typename scalar_t, ReductionType reduce>
inline void update(scalar_t* out, const scalar_t* data, int64_t K) {
  using Vec = vec::Vectorized<vec_scalar_t<scalar_t>>;
  map2<scalar_t>(
      [](Vec x, Vec y) { return update<Vec, reduce>(x, y); },
      out,
      out,
      data,
      K);
}

template <typename scalar_t, ReductionType reduce,
          typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline void update(at::opmath_type<scalar_t>* out, const scalar_t* data, int64_t K) {
  using opmath_t = at::opmath_type<scalar_t>;
  using Vec = vec::Vectorized<opmath_t>;
  map_acc<scalar_t, opmath_t>(
      [](Vec x, Vec y) { return update<Vec, reduce>(x, y); },
      out,
      out,
      data,
      K);
}

template <typename scalar_t, ReductionType reduce>
inline void write(scalar_t* out, int64_t count, int64_t K) {
  using Vec = vec::Vectorized<vec_scalar_t<scalar_t>>;
  if (reduce == ReductionType::MEAN) {
    if (count > 0) {
      vec::map<scalar_t>(
          [count](Vec x) { return x / Vec(count); },
          out,
          out,
          K);
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vec`, `CPU_CAPABILITY`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Parallel.h`
- `ATen/NumericUtils.h`
- `ATen/cpu/vec/vec.h`
- `ATen/cpu/vec/functional.h`
- `ATen/native/ReductionType.h`
- `c10/util/irange.h`
- `ATen/OpMathType.h`
- `ATen/native/cpu/utils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `ReduceUtils.h_docs.md`
- **Keyword Index**: `ReduceUtils.h_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `ReduceUtils.h_docs.md_docs.md`
- **Keyword Index**: `ReduceUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

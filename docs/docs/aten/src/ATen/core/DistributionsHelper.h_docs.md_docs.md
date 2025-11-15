# Documentation: `docs/aten/src/ATen/core/DistributionsHelper.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/DistributionsHelper.h_docs.md`
- **Size**: 15,546 bytes (15.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/DistributionsHelper.h`

## File Metadata

- **Path**: `aten/src/ATen/core/DistributionsHelper.h`
- **Size**: 12,603 bytes (12.31 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/TransformationHelper.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/MathConstants.h>
#include <c10/macros/Macros.h>

#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

/**
 * Distributions kernel adapted from THRandom.cpp
 * The kernels try to follow std::random distributions signature
 * For instance: in ATen
 *      auto gen = at::detail::createCPUGenerator();
 *      at::uniform_real_distribution<double> uniform(0, 1);
 *      auto sample = uniform(gen.get());
 *
 *      vs std::random
 *
 *      std::mt19937 gen;
 *      std::uniform_real_distribution uniform(0, 1);
 *      auto sample = uniform(gen);
 */


namespace at {
namespace {

/**
 * Samples a discrete uniform distribution in the range [base, base+range) of type T
 */
template <typename T>
struct uniform_int_from_to_distribution {

  C10_HOST_DEVICE inline uniform_int_from_to_distribution(uint64_t range, int64_t base) : range_(range), base_(base) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
#ifdef FBCODE_CAFFE2
    if ((
      std::is_same_v<T, int64_t> ||
      std::is_same_v<T, double> ||
      std::is_same_v<T, float> ||
      std::is_same_v<T, at::BFloat16>) && range_ >= 1ULL << 32)
#else
    if (range_ >= 1ULL << 28) // allow approx 5% skew in uniform int generation using %
#endif
    {
      return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
    } else {
      return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
    }
  }

  private:
    uint64_t range_;
    int64_t base_;
};

/**
 * Samples a discrete uniform distribution in the range [min_value(int64_t), max_value(int64_t)]
 */
template <typename T>
struct uniform_int_full_range_distribution {

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    return transformation::uniform_int_full_range<T>(generator->random64());
  }

};

/**
 * Samples a discrete uniform distribution in the range [0, max_value(T)] for integral types
 * and [0, 2^mantissa] for floating-point types.
 */
template <typename T>
struct uniform_int_distribution {

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, int64_t>) {
      return transformation::uniform_int<T>(generator->random64());
    } else {
      return transformation::uniform_int<T>(generator->random());
    }
  }

};

/**
 * Samples a uniform distribution in the range [from, to) of type T
 */
template <typename T>
struct uniform_real_distribution {

  C10_HOST_DEVICE inline uniform_real_distribution(T from, T to) : from_(from), to_(to) {
    TORCH_CHECK_IF_NOT_ON_CUDA(from <= to);
    TORCH_CHECK_IF_NOT_ON_CUDA(to - from <= std::numeric_limits<T>::max());
  }

  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator){
    if constexpr (std::is_same_v<T, double>) {
      return transformation::uniform_real<T>(generator->random64(), from_, to_);
    } else {
      return transformation::uniform_real<T>(generator->random(), from_, to_);
    }
  }

  private:
    T from_;
    T to_;
};

// The SFINAE checks introduced in #39816 looks overcomplicated and must revisited
// https://github.com/pytorch/pytorch/issues/40052
#define DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(member)              \
template <typename T>                                                \
struct has_member_##member                                           \
{                                                                    \
    typedef char yes;                                                \
    typedef long no;                                                 \
    template <typename U> static yes test(decltype(&U::member));     \
    template <typename U> static no test(...);                       \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); \
}

DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_double_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_double_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(next_float_normal_sample);
DISTRIBUTION_HELPER_GENERATE_HAS_MEMBER(set_next_float_normal_sample);

#define DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(TYPE)                                      \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            has_member_next_##TYPE##_normal_sample<RNG>::value &&                                   \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* generator, ret_type* ret) {  \
  if (generator->next_##TYPE##_normal_sample()) {                                                   \
    *ret = *(generator->next_##TYPE##_normal_sample());                                             \
    generator->set_next_##TYPE##_normal_sample(std::optional<TYPE>());                              \
    return true;                                                                                    \
  }                                                                                                 \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            !has_member_next_##TYPE##_normal_sample<RNG>::value ||                                  \
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type* /*ret*/) {  \
  return false;                                                                                     \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline void maybe_set_next_##TYPE##_normal_sample(RNG* generator, ret_type cache) { \
  generator->set_next_##TYPE##_normal_sample(cache);                                                \
}                                                                                                   \
                                                                                                    \
template <typename RNG, typename ret_type,                                                          \
          typename std::enable_if_t<(                                                               \
            !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 \
          ), int> = 0>                                                                              \
C10_HOST_DEVICE inline void maybe_set_next_##TYPE##_normal_sample(RNG* /*generator*/, ret_type /*cache*/) { \
}

DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(double)
DISTRIBUTION_HELPER_GENERATE_NEXT_NORMAL_METHODS(float)

/**
 * Samples a normal distribution using the Box-Muller method
 * Takes mean and standard deviation as inputs
 * Note that Box-muller method returns two samples at a time.
 * Hence, we cache the "next" sample in the CPUGeneratorImpl class.
 */
template <typename T>
struct normal_distribution {

  C10_HOST_DEVICE inline normal_distribution(T mean_in, T stdv_in) : mean(mean_in), stdv(stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in >= 0, "stdv_in must be positive: ", stdv_in);
  }

  template <typename RNG>
  C10_HOST_DEVICE inline dist_acctype<T> operator()(RNG generator){
    dist_acctype<T> ret;
    // return cached values if available
    if constexpr (std::is_same_v<T, double>) {
      if (maybe_get_next_double_normal_sample(generator, &ret)) {
        return transformation::normal(ret, mean, stdv);
      }
    } else {
      if (maybe_get_next_float_normal_sample(generator, &ret)) {
        return transformation::normal(ret, mean, stdv);
      }
    }
    // otherwise generate new normal values
    uniform_real_distribution<T> uniform(0.0, 1.0);
    const dist_acctype<T> u1 = uniform(generator);
    const dist_acctype<T> u2 = uniform(generator);
    const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log1p(-u2));
    const dist_acctype<T> theta = static_cast<T>(2.0) * c10::pi<T> * u1;
    if constexpr (std::is_same_v<T, double>) {
      maybe_set_next_double_normal_sample(generator, r * ::sin(theta));
    } else {
      maybe_set_next_float_normal_sample(generator, r * ::sin(theta));
    }
    ret = r * ::cos(theta);
    return transformation::normal(ret, mean, stdv);
  }

  private:
    T mean;
    T stdv;
};

template <typename T>
struct DiscreteDistributionType { using type = float; };

template <> struct DiscreteDistributionType<double> { using type = double; };

/**
 * Samples a bernoulli distribution given a probability input
 */
template <typename T>
struct bernoulli_distribution {

  C10_HOST_DEVICE inline bernoulli_distribution(T p_in) : p(p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in >= 0 && p_in <= 1);
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::bernoulli<T>(uniform(generator), p);
  }

  private:
    T p;
};

/**
 * Samples a geometric distribution given a probability input
 */
template <typename T>
struct geometric_distribution {

  C10_HOST_DEVICE inline geometric_distribution(T p_in) : p(p_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(p_in > 0 && p_in < 1);
  }

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::geometric<T>(uniform(generator), p);
  }

  private:
    T p;
};

/**
 * Samples an exponential distribution given a lambda input
 */
template <typename T>
struct exponential_distribution {

  C10_HOST_DEVICE inline exponential_distribution(T lambda_in) : lambda(lambda_in) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::exponential<T>(uniform(generator), lambda);
  }

  private:
    T lambda;
};

/**
 * Samples a cauchy distribution given median and sigma as inputs
 */
template <typename T>
struct cauchy_distribution {

  C10_HOST_DEVICE inline cauchy_distribution(T median_in, T sigma_in) : median(median_in), sigma(sigma_in) {}

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    uniform_real_distribution<T> uniform(0.0, 1.0);
    return transformation::cauchy<T>(uniform(generator), median, sigma);
  }

  private:
    T median;
    T sigma;
};

/**
 * Samples a lognormal distribution
 * Takes mean and standard deviation as inputs
 * Outputs two samples at a time
 */
template <typename T>
struct lognormal_distribution {

  C10_HOST_DEVICE inline lognormal_distribution(T mean_in, T stdv_in) : mean(mean_in), stdv(stdv_in) {
    TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in > 0);
  }

  template<typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator){
    normal_distribution<T> normal(mean, stdv);
    return transformation::log_normal<T>(normal(generator));
  }

  private:
    T mean;
    T stdv;
};
}
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 27 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `uniform_int_from_to_distribution`, `uniform_int_full_range_distribution`, `uniform_int_distribution`, `uniform_real_distribution`, `has_member_`, `normal_distribution`, `DiscreteDistributionType`, `DiscreteDistributionType`, `bernoulli_distribution`, `geometric_distribution`, `exponential_distribution`, `cauchy_distribution`, `lognormal_distribution`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/TransformationHelper.h`
- `c10/util/Half.h`
- `c10/util/BFloat16.h`
- `c10/util/MathConstants.h`
- `c10/macros/Macros.h`
- `cmath`
- `limits`
- `optional`
- `type_traits`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`aten/src/ATen/core`):

- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `DistributionsHelper.h_docs.md`
- **Keyword Index**: `DistributionsHelper.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `DistributionsHelper.h_docs.md_docs.md`
- **Keyword Index**: `DistributionsHelper.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

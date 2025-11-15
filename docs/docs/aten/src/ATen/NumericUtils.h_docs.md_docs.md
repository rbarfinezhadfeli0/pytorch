# Documentation: `docs/aten/src/ATen/NumericUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/NumericUtils.h_docs.md`
- **Size**: 7,813 bytes (7.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/NumericUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/NumericUtils.h`
- **Size**: 5,137 bytes (5.02 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <cmath>
#include <type_traits>

namespace at {

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <typename T, std::enable_if_t<c10::is_complex<T>::value, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return std::isnan(val.real()) || std::isnan(val.imag());
}

template <typename T, std::enable_if_t<std::is_same_v<T, at::Half>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return at::_isnan(static_cast<float>(val));
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::BFloat16>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(static_cast<float>(val));
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e5m2>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e4m3fn>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e5m2fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Float8_e4m3fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// std::isinf isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val);
#else
  return std::isinf(val);
#endif
}

inline C10_HOST_DEVICE bool _isinf(at::Half val) {
  return at::_isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(at::BFloat16 val) {
  return at::_isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e5m2 val) {
  return val.isinf();
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e4m3fn val [[maybe_unused]]) {
  return false;
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e5m2fnuz val [[maybe_unused]]) {
  return false;
}

inline C10_HOST_DEVICE bool _isinf(at::Float8_e4m3fnuz val [[maybe_unused]]) {
  return false;
}

template <typename T>
C10_HOST_DEVICE inline T exp(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __expf fast approximation for peak bandwidth
  return __expf(x);
#else
  return ::exp(x);
#endif
}

template <>
C10_HOST_DEVICE inline double exp<double>(double x) {
  return ::exp(x);
}

template <typename T>
C10_HOST_DEVICE inline T log(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  return __logf(x);
#else
  return ::log(x);
#endif
}

template <>
C10_HOST_DEVICE inline double log<double>(double x) {
  return ::log(x);
}

template <typename T>
C10_HOST_DEVICE inline T log1p(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  // NOTE: There is no __log1pf so unfortunately we lose precision.
  return __logf(1.0f + x);
#else
  return ::log1p(x);
#endif
}

template <>
C10_HOST_DEVICE inline double log1p<double>(double x) {
  return ::log1p(x);
}

template <typename T>
C10_HOST_DEVICE inline T tan(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __tanf fast approximation for peak bandwidth
  return __tanf(x);
#else
  return ::tan(x);
#endif
}

template <>
C10_HOST_DEVICE inline double tan<double>(double x) {
  return ::tan(x);
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 32 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `hip/hip_runtime.h`
- `c10/macros/Macros.h`
- `c10/util/BFloat16.h`
- `c10/util/Float8_e4m3fn.h`
- `c10/util/Float8_e4m3fnuz.h`
- `c10/util/Float8_e5m2.h`
- `c10/util/Float8_e5m2fnuz.h`
- `c10/util/Half.h`
- `c10/util/complex.h`
- `cmath`
- `type_traits`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `NumericUtils.h_docs.md`
- **Keyword Index**: `NumericUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `NumericUtils.h_docs.md_docs.md`
- **Keyword Index**: `NumericUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

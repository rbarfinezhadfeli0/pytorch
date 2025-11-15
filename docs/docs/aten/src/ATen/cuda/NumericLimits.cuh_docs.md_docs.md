# Documentation: `docs/aten/src/ATen/cuda/NumericLimits.cuh_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/NumericLimits.cuh_docs.md`
- **Size**: 7,639 bytes (7.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/NumericLimits.cuh`

## File Metadata

- **Path**: `aten/src/ATen/cuda/NumericLimits.cuh`
- **Size**: 5,214 bytes (5.09 KB)
- **Type**: CUDA Header File
- **Extension**: `.cuh`

## File Purpose

This is a cuda header file that is part of the PyTorch project.

## Original Source

```
#pragma once

#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <float.h>

// NumericLimits.cuh is a holder for numeric limits definitions of commonly used
// types. This header is very specific to ROCm HIP and may be removed in the future.
// This header is derived from the legacy THCNumerics.cuh.

// The lower_bound and upper_bound constants are same as lowest and max for
// integral types, but are -inf and +inf for floating point types. They are
// useful in implementing min, max, etc.

namespace at {

template <typename T>
struct numeric_limits {
};

// WARNING: the following at::numeric_limits definitions are there only to support
//          HIP compilation for the moment. Use std::numeric_limits if you are not
//          compiling for ROCm.
//          from @colesbury: "The functions on numeric_limits aren't marked with
//          __device__ which is why they don't work with ROCm. CUDA allows them
//          because they're constexpr."

namespace {
  // ROCm doesn't like INFINITY too.
  constexpr double inf = INFINITY;
}

template <>
struct numeric_limits<bool> {
  static inline __host__ __device__ bool lowest() { return false; }
  static inline __host__ __device__ bool max() { return true; }
  static inline __host__ __device__ bool lower_bound() { return false; }
  static inline __host__ __device__ bool upper_bound() { return true; }
};

template <>
struct numeric_limits<uint8_t> {
  static inline __host__ __device__ uint8_t lowest() { return 0; }
  static inline __host__ __device__ uint8_t max() { return UINT8_MAX; }
  static inline __host__ __device__ uint8_t lower_bound() { return 0; }
  static inline __host__ __device__ uint8_t upper_bound() { return UINT8_MAX; }
};

template <>
struct numeric_limits<int8_t> {
  static inline __host__ __device__ int8_t lowest() { return INT8_MIN; }
  static inline __host__ __device__ int8_t max() { return INT8_MAX; }
  static inline __host__ __device__ int8_t lower_bound() { return INT8_MIN; }
  static inline __host__ __device__ int8_t upper_bound() { return INT8_MAX; }
};

template <>
struct numeric_limits<int16_t> {
  static inline __host__ __device__ int16_t lowest() { return INT16_MIN; }
  static inline __host__ __device__ int16_t max() { return INT16_MAX; }
  static inline __host__ __device__ int16_t lower_bound() { return INT16_MIN; }
  static inline __host__ __device__ int16_t upper_bound() { return INT16_MAX; }
};

template <>
struct numeric_limits<int32_t> {
  static inline __host__ __device__ int32_t lowest() { return INT32_MIN; }
  static inline __host__ __device__ int32_t max() { return INT32_MAX; }
  static inline __host__ __device__ int32_t lower_bound() { return INT32_MIN; }
  static inline __host__ __device__ int32_t upper_bound() { return INT32_MAX; }
};

template <>
struct numeric_limits<int64_t> {
#ifdef _MSC_VER
  static inline __host__ __device__ int64_t lowest() { return _I64_MIN; }
  static inline __host__ __device__ int64_t max() { return _I64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return _I64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return _I64_MAX; }
#else
  static inline __host__ __device__ int64_t lowest() { return INT64_MIN; }
  static inline __host__ __device__ int64_t max() { return INT64_MAX; }
  static inline __host__ __device__ int64_t lower_bound() { return INT64_MIN; }
  static inline __host__ __device__ int64_t upper_bound() { return INT64_MAX; }
#endif
};

template <>
struct numeric_limits<at::Half> {
  static inline __host__ __device__ at::Half lowest() { return at::Half(0xFBFF, at::Half::from_bits()); }
  static inline __host__ __device__ at::Half max() { return at::Half(0x7BFF, at::Half::from_bits()); }
  static inline __host__ __device__ at::Half lower_bound() { return at::Half(0xFC00, at::Half::from_bits()); }
  static inline __host__ __device__ at::Half upper_bound() { return at::Half(0x7C00, at::Half::from_bits()); }
};

template <>
struct numeric_limits<at::BFloat16> {
  static inline __host__ __device__ at::BFloat16 lowest() { return at::BFloat16(0xFF7F, at::BFloat16::from_bits()); }
  static inline __host__ __device__ at::BFloat16 max() { return at::BFloat16(0x7F7F, at::BFloat16::from_bits()); }
  static inline __host__ __device__ at::BFloat16 lower_bound() { return at::BFloat16(0xFF80, at::BFloat16::from_bits()); }
  static inline __host__ __device__ at::BFloat16 upper_bound() { return at::BFloat16(0x7F80, at::BFloat16::from_bits()); }
};

template <>
struct numeric_limits<float> {
  static inline __host__ __device__ float lowest() { return -FLT_MAX; }
  static inline __host__ __device__ float max() { return FLT_MAX; }
  static inline __host__ __device__ float lower_bound() { return -static_cast<float>(inf); }
  static inline __host__ __device__ float upper_bound() { return static_cast<float>(inf); }
};

template <>
struct numeric_limits<double> {
  static inline __host__ __device__ double lowest() { return -DBL_MAX; }
  static inline __host__ __device__ double max() { return DBL_MAX; }
  static inline __host__ __device__ double lower_bound() { return -inf; }
  static inline __host__ __device__ double upper_bound() { return inf; }
};

} // namespace at

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cuda.h`
- `limits.h`
- `math.h`
- `float.h`


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

Files in the same folder (`aten/src/ATen/cuda`):

- [`CublasHandlePool.cpp_docs.md`](./CublasHandlePool.cpp_docs.md)
- [`llvm_basic.cpp_docs.md`](./llvm_basic.cpp_docs.md)
- [`CUDABlas.h_docs.md`](./CUDABlas.h_docs.md)
- [`jiterator.cu_docs.md`](./jiterator.cu_docs.md)
- [`CUDAGraph.h_docs.md`](./CUDAGraph.h_docs.md)
- [`llvm_jit_strings.h_docs.md`](./llvm_jit_strings.h_docs.md)
- [`llvm_complex.cpp_docs.md`](./llvm_complex.cpp_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md`](./CUDAGeneratorImpl.cpp_docs.md)
- [`cub_definitions.cuh_docs.md`](./cub_definitions.cuh_docs.md)
- [`jiterator_impl.h_docs.md`](./jiterator_impl.h_docs.md)


## Cross-References

- **File Documentation**: `NumericLimits.cuh_docs.md`
- **Keyword Index**: `NumericLimits.cuh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/cuda`):

- [`PhiloxCudaState.h_docs.md_docs.md`](./PhiloxCudaState.h_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md_docs.md`](./CUDAGeneratorImpl.cpp_docs.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_kw.md_docs.md`](./CUDAGeneratorImpl.cpp_kw.md_docs.md)
- [`Sleep.h_docs.md_docs.md`](./Sleep.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-2.cu_kw.md_docs.md`](./cub-RadixSortPairs-int64-2.cu_kw.md_docs.md)
- [`CUDASparseDescriptors.h_kw.md_docs.md`](./CUDASparseDescriptors.h_kw.md_docs.md)
- [`jiterator_impl.h_docs.md_docs.md`](./jiterator_impl.h_docs.md_docs.md)
- [`CUDAContext.h_docs.md_docs.md`](./CUDAContext.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-4.cu_docs.md_docs.md`](./cub-RadixSortPairs-int64-4.cu_docs.md_docs.md)


## Cross-References

- **File Documentation**: `NumericLimits.cuh_docs.md_docs.md`
- **Keyword Index**: `NumericLimits.cuh_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

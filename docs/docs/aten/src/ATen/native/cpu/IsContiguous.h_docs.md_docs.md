# Documentation: `docs/aten/src/ATen/native/cpu/IsContiguous.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cpu/IsContiguous.h_docs.md`
- **Size**: 4,918 bytes (4.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cpu/IsContiguous.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cpu/IsContiguous.h`
- **Size**: 2,369 bytes (2.31 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

namespace at::native { inline namespace CPU_CAPABILITY {

// n: number of function arguments (arity)
// traits: function_traits (see FunctionTraits.h)
// s: index of scalar argument or -1
template <int n, int stride_index, typename traits, int s=-1>
struct IsContiguous {
  static bool eval(const int64_t* strides) {
    using type = typename traits::template arg<n - 1>::type;
    return strides[stride_index] == (s == n ? 0 : sizeof(type)) &&
           IsContiguous<n - 1, stride_index - 1, traits, s>::eval(strides);
  }
};

// will be called when there is an output exists
template <typename traits, int s>
struct IsContiguous<0, 0, traits, s> {
  static bool eval(const int64_t* strides) {
    return strides[0] == sizeof(typename traits::result_type);
  }
};

// will be called when there is no output
template <typename traits, int s>
struct IsContiguous<0, -1, traits, s> {
  static bool eval(const int64_t* /*strides*/) {
    return true;
  }
};

// output and all inputs are contiguous
template <
    typename traits,
    std::enable_if_t<std::is_void_v<typename traits::result_type>>* =
        nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  return IsContiguous<traits::arity, traits::arity - 1, traits>::eval(strides);
}

template <typename traits,
    std::enable_if_t<!std::is_void_v<typename traits::result_type>>* = nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  return IsContiguous<traits::arity, traits::arity, traits>::eval(strides);
}

// input at `s` is scalar (stride 0); output and other inputs are contiguous
// NB: output is typically at strides[0] so first input corresponds to s=1
template <typename traits, int s,
    std::enable_if_t<std::is_void_v<typename traits::result_type>>* = nullptr>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  return IsContiguous<traits::arity, traits::arity - 1, traits, s>::eval(strides);
}

template <typename traits, int s,
    std::enable_if_t<!std::is_void_v<typename traits::result_type>>* = nullptr>
static inline bool is_contiguous_scalar(const int64_t* strides) {
  static_assert(s > 0 && s <= traits::arity, "scalar argument index out of bounds");
  return IsContiguous<traits::arity, traits::arity, traits, s>::eval(strides);
}

}}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `CPU_CAPABILITY`, `at`

**Classes/Structs**: `IsContiguous`, `IsContiguous`, `IsContiguous`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*No includes detected.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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

- **File Documentation**: `IsContiguous.h_docs.md`
- **Keyword Index**: `IsContiguous.h_kw.md`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

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
- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`WeightNormKernel.cpp_kw.md_docs.md`](./WeightNormKernel.cpp_kw.md_docs.md)
- [`FusedAdamKernel.cpp_docs.md_docs.md`](./FusedAdamKernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `IsContiguous.h_docs.md_docs.md`
- **Keyword Index**: `IsContiguous.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

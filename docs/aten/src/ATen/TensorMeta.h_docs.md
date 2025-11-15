# Documentation: `aten/src/ATen/TensorMeta.h`

## File Metadata

- **Path**: `aten/src/ATen/TensorMeta.h`
- **Size**: 5,034 bytes (4.92 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/DimVector.h>
#include <ATen/core/Dimname.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/strides.h>

namespace at {

class Tensor;

namespace impl {

// Use this to define the prototype for a meta function.  There are two
// versions; one that takes one argument (just the operator name), or FUNC2
// variant that takes two arguments (operator name and overload name).
//
// Example usage:
//
//    TORCH_META_FUNC2(add, Tensor) (
//      const Tensor& self, const Tensor& other
//    ) {
//      ... compute sizes and options ...
//      set_output(sizes, options);
//    }
//
#define TORCH_META_FUNC(name) void structured_##name::meta
#define TORCH_META_FUNC2(name, overload) \
  void structured_##name##_##overload::meta

// These are versions of TORCH_META_FUNC(2) that include a precompute_out struct
// as a return value. They should be used when the kernel in question has
// precomputed values declared in native_functions.yaml and the corresponding
// implementation should return an instance of the aforementioned struct.
#define TORCH_PRECOMPUTE_META_FUNC(name) \
  structured_##name::meta_return_ty structured_##name::meta
#define TORCH_PRECOMPUTE_META_FUNC2(name, overload) \
  structured_##name##_##overload::meta_return_ty    \
      structured_##name##_##overload::meta

// Use this to create a precompute struct in a meta function.
#define TORCH_PRECOMPUTE_STRUCT(name) structured_##name::precompute_out<>
#define TORCH_PRECOMPUTE_STRUCT2(name, overload) \
  structured_##name##_##overload::precompute_out<>

// Use this to define the prototype for an implementation.  This takes only
// one argument, which is the name of the dispatch key entry you're
// implementing.
//
// Example usage:
//
//    TORCH_IMPL_FUNC(add_cpu) (
//      Tensor& result, const Tensor& self, const Tensor& other
//    ) {
//      ... do the actual implementation ...
//    }
//
#define TORCH_IMPL_FUNC(name) void structured_##name::impl

// Base class for all structured kernel classes.  The set_output virtual
// method is varied depending whether or not the operator is
// functional/out/inplace, and could also be specialized for CPU/CUDA/etc
// (although presently it isn't).
//
// A notable subclass of this interface is TensorIteratorBase.
struct TORCH_API MetaBase {
  MetaBase() = default;
  MetaBase(const MetaBase&) = default;
  MetaBase& operator=(const MetaBase&) = default;
  MetaBase(MetaBase&&) noexcept = default;
  MetaBase& operator=(MetaBase&&) noexcept = default;
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;

  // Note: [set_output_*]
  // See: https://github.com/pytorch/pytorch/issues/69813
  // Whenever defining the output properties in the META function of a
  // structured kernel (what was usually done with `set_output`), use one of
  // these 3 variants, instead. In order to decide which variant to use, check
  // the following decision tree:
  //
  // - Can the kernel you are going to implement support output tensors
  //   with arbitrary strides?
  //     |
  //     -- YES: `set_output_raw_strided`
  //     |
  //     -- NO: Should the output tensor strides be contiguous?
  //         |
  //         -- YES: `set_output_contiguous`
  //         |
  //         -- NO: `set_output_strided`
  //
  // Use this function whenever the kernel requires specific strides for the
  // output. If `strides` does not match the given output strides, proxy outputs
  // will be created and passed to the IMPL function.
  virtual void set_output_strided(
      int64_t output_idx [[maybe_unused]],
      IntArrayRef sizes [[maybe_unused]],
      IntArrayRef strides [[maybe_unused]],
      TensorOptions options [[maybe_unused]],
      DimnameList names [[maybe_unused]] = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // Use this function whenever the kernel knows how to handle arbitrary strided
  // outputs. This function has the same behavior as the old `set_output`: it
  // will only re-stride if the given output was resized.
  virtual void set_output_raw_strided(
      int64_t output_idx [[maybe_unused]],
      IntArrayRef sizes [[maybe_unused]],
      IntArrayRef strides_hint [[maybe_unused]],
      TensorOptions options [[maybe_unused]],
      DimnameList names [[maybe_unused]] = {}) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // Use this function if the kernel requires contiguous strides.
  // Alias for `set_output_strided`, but with contiguous strides.
  void set_output_contiguous(
      int64_t output_idx,
      IntArrayRef sizes,
      TensorOptions options,
      DimnameList names = {}) {
    auto strides = c10::contiguous_strides(sizes);
    set_output_strided(output_idx, sizes, strides, options, names);
  }

  // Returns a reference to an undefined tensor if there is no presupplied
  // output
  const Tensor& maybe_get_output() {
    return maybe_get_output(0);
  }
  virtual ~MetaBase() = default;
};

} // namespace impl

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `impl`, `at`

**Classes/Structs**: `Tensor`, `in`, `for`, `of`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/DimVector.h`
- `ATen/core/Dimname.h`
- `c10/core/TensorOptions.h`
- `c10/util/strides.h`


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

- **File Documentation**: `TensorMeta.h_docs.md`
- **Keyword Index**: `TensorMeta.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

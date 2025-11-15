# Documentation: `docs/aten/src/ATen/InferSize.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/InferSize.h_docs.md`
- **Size**: 6,489 bytes (6.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/InferSize.h`

## File Metadata

- **Path**: `aten/src/ATen/InferSize.h`
- **Size**: 3,928 bytes (3.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/DimVector.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <optional>
#include <sstream>
#include <vector>

namespace at {

// Infers the size of a dim with size -1, if it exists. Also checks that new
// shape is compatible with the number of elements.
//
// templated to handle std::vector<int64_t> and DimVector use cases, see
// below
//
template <typename InputArrayRef, typename NumelType, typename ResultVec>
inline void infer_size_impl(
    InputArrayRef shape,
    NumelType numel,
    ResultVec& res) {
  NumelType newsize = 1;
  // N.B. this is an index, not a sym dim!
  std::optional<int64_t> infer_dim;
  for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
    if (TORCH_GUARD_OR_FALSE(sym_eq(shape[dim], -1))) {
      TORCH_CHECK(!infer_dim, "only one dimension can be inferred");
      infer_dim = dim;
    } else {
      // in case of unbacked shape[dim] we assume it's not -1 and add a runtime
      // assertion.
      TORCH_MAYBE_SYM_CHECK(
          sym_gt(shape[dim], -1),
          "invalid shape dimension ",
          shape[dim],
          " at index ",
          dim,
          " of shape ",
          shape);
      newsize *= shape[dim];
    }
  }

  if (infer_dim) {
    // numel is the product of known sizes, it has to be divisible by newsize.
    // and newsize should be positive unless newsize == numel (we throw
    // different) error message in that case.
    if constexpr (std::is_same_v<NumelType, c10::SymInt>) {
      auto v = newsize.maybe_as_int();
      if (v and *v == 0) {
        // Avoid div by 0 when sym_eq(numel % newsize, 0) is constructed!
        // which may happen when newsize is not a symbol! if its a symbol
        // division won't happen anyway during compile.
        TORCH_MAYBE_SYM_CHECK(
            numel == newsize,
            "shape '",
            shape,
            "' is invalid for input of size ",
            numel);
      } else {
        auto cond = sym_gt(newsize, 0)
                        .sym_and(sym_eq(numel % newsize, 0))
                        .sym_or(sym_eq(numel, newsize));
        TORCH_MAYBE_SYM_CHECK(
            cond, "shape '", shape, "' is invalid for input of size ", numel);
      }

    } else {
      TORCH_CHECK(
          (newsize > 0 && (numel % newsize == 0)) || numel == newsize,
          "shape '",
          shape,
          "' is invalid for input of size ",
          numel);
    }

    // We have a degree of freedom here to select the dimension size; follow
    // NumPy semantics and just bail.  However, a nice error message is needed
    // because users often use `view` as a way to flatten & unflatten
    // dimensions and will otherwise be confused why
    //   empty_tensor.view( 0, 0)
    // works yet
    //   empty_tensor.view(-1, 0)
    // doesn't.
    TORCH_MAYBE_SYM_CHECK(
        newsize != 0,
        "cannot reshape tensor of 0 elements into shape ",
        shape,
        " because the unspecified dimension size -1 can be any "
        "value and is ambiguous");

    res[*infer_dim] = numel / newsize;
    return;
  }

  TORCH_MAYBE_SYM_CHECK(
      sym_eq(numel, newsize),
      "shape '",
      shape,
      "' is invalid for input of size ",
      numel);
}

inline std::vector<int64_t> infer_size(IntArrayRef shape, int64_t numel) {
  auto res = shape.vec();
  infer_size_impl(shape, numel, res);
  return res;
}

inline at::DimVector infer_size_dv(IntArrayRef shape, int64_t numel) {
  auto res = at::DimVector(shape);
  infer_size_impl(shape, numel, res);
  return res;
}

inline at::SymDimVector infer_size_dv(
    c10::SymIntArrayRef shape,
    c10::SymInt numel) {
  auto res = at::SymDimVector(shape);
  infer_size_impl<c10::SymIntArrayRef, c10::SymInt, at::SymDimVector>(
      shape, std::move(numel), res);
  return res;
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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

- `ATen/DimVector.h`
- `c10/core/ScalarType.h`
- `c10/core/SymIntArrayRef.h`
- `c10/util/DimVector.h`
- `c10/util/Exception.h`
- `optional`
- `sstream`
- `vector`


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

- **File Documentation**: `InferSize.h_docs.md`
- **Keyword Index**: `InferSize.h_kw.md`
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

- **File Documentation**: `InferSize.h_docs.md_docs.md`
- **Keyword Index**: `InferSize.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

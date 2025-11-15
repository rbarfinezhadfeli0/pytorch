# Documentation: `docs/aten/src/ATen/CollapseDims.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/CollapseDims.h_docs.md`
- **Size**: 4,943 bytes (4.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/CollapseDims.h`

## File Metadata

- **Path**: `aten/src/ATen/CollapseDims.h`
- **Size**: 2,560 bytes (2.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#include <c10/util/Exception.h>
#include <utility>

namespace at {

/*
[collapse dims] Updates sizes, and strides to reflect a "collapse" of
the info, possibly excluding the optional excludeDim. A "collapsed" version
of the info is the fewest dims that order the tensor's elements in the same
way as the original info. If excludeDim is specified, the collapse is the
fewest dims that order the tensor's elements as the original and preserve the
excluded dimension, unless the tensor collapses to a point.

This function returns a pair of values.

1) The (new) index of the preserved dimension if excludeDim is
specified. 0 if the tensor is collapsed to a point. -1
otherwise.

2) The new number of dimensions.
*/
template <typename T>
inline std::pair<int64_t, int64_t> collapse_dims(
    T* sizes,
    T* strides,
    int64_t dims,
    const int excludeDim = -1) {
  TORCH_CHECK(
      excludeDim >= -1 && excludeDim < dims,
      "expected excluded dim between -1 and dims - 1");

  int64_t stopDim = (excludeDim == -1) ? dims : excludeDim;
  int64_t newIndex = -1;
  int64_t oldIndex = 0;
  int64_t remappedExcludedDim = -1;

  while (oldIndex < dims) {
    // Finds a dimension to collapse into
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;
      }

      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      ++oldIndex;
      break;
    }

    // Collapses dims
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;
      }

      if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
        sizes[newIndex] *= sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      } else {
        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      }
    }

    // Handles excludeDim being set (oldIndex == excludeDim)
    if (oldIndex != dims) {
      // Preserves excluded dimension
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      remappedExcludedDim = newIndex;

      // Restarts iteration after excludeDim
      ++oldIndex;
      stopDim = dims;
    }
  }

  // Handles special case of all dims size 1
  if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    return std::pair<int64_t, int64_t>(0, 1);
  }

  dims = newIndex + 1;
  return std::pair<int64_t, int64_t>(remappedExcludedDim, dims);
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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

- `c10/util/Exception.h`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

- **File Documentation**: `CollapseDims.h_docs.md`
- **Keyword Index**: `CollapseDims.h_kw.md`
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

- **File Documentation**: `CollapseDims.h_docs.md_docs.md`
- **Keyword Index**: `CollapseDims.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

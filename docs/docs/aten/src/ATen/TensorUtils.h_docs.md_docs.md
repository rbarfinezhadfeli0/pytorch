# Documentation: `docs/aten/src/ATen/TensorUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/TensorUtils.h_docs.md`
- **Size**: 8,547 bytes (8.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/TensorUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/TensorUtils.h`
- **Size**: 5,958 bytes (5.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/DimVector.h>
#include <ATen/EmptyTensor.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/Utils.h>

#include <utility>

// These functions are NOT in Utils.h, because this file has a dep on Tensor.h

#define TORCH_CHECK_TENSOR_ALL(cond, ...) \
  TORCH_CHECK((cond)._is_all_true().item<bool>(), __VA_ARGS__);

namespace at {

// The following are utility functions for checking that arguments
// make sense.  These are particularly useful for native functions,
// which do NO argument checking by default.

struct TORCH_API TensorArg {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const Tensor& tensor;
  const char* name;
  int pos; // 1-indexed
  TensorArg(const Tensor& tensor, const char* name, int pos)
      : tensor(tensor), name(name), pos(pos) {}
  // Try to mitigate any possibility of dangling reference to temporaries.
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  TensorArg(Tensor&& tensor, const char* name, int pos) = delete;
  const Tensor* operator->() const {
    return &tensor;
  }
  const Tensor& operator*() const {
    return tensor;
  }
};

struct TORCH_API TensorGeometryArg {
  TensorGeometry tensor;
  const char* name;
  int pos; // 1-indexed
  /* implicit */ TensorGeometryArg(TensorArg arg)
      : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos) {}
  TensorGeometryArg(TensorGeometry tensor, const char* name, int pos)
      : tensor(std::move(tensor)), name(name), pos(pos) {}
  const TensorGeometry* operator->() const {
    return &tensor;
  }
  const TensorGeometry& operator*() const {
    return tensor;
  }
};

// A string describing which function did checks on its input
// arguments.
// TODO: Consider generalizing this into a call stack.
using CheckedFrom = const char*;

// The undefined convention: singular operators assume their arguments
// are defined, but functions which take multiple tensors will
// implicitly filter out undefined tensors (to make it easier to perform
// tests which should apply if the tensor is defined, and should not
// otherwise.)
//
// NB: This means that the n-ary operators take lists of TensorArg,
// not TensorGeometryArg, because the Tensor to TensorGeometry
// conversion will blow up if you have undefined tensors.

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    const TensorGeometryArg& t);
TORCH_API void checkDim(
    CheckedFrom c,
    const Tensor& tensor,
    const char* name,
    int pos, // 1-indexed
    int64_t dim);
TORCH_API void checkDim(CheckedFrom c, const TensorGeometryArg& t, int64_t dim);
// NB: this is an inclusive-exclusive range
TORCH_API void checkDimRange(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim_start,
    int64_t dim_end);
TORCH_API void checkSameDim(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
TORCH_API void checkContiguous(CheckedFrom c, const TensorGeometryArg& t);
TORCH_API void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts);
TORCH_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    IntArrayRef sizes);
TORCH_API void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    c10::SymIntArrayRef sizes);
TORCH_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    int64_t size);
TORCH_API void checkSize_symint(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    const c10::SymInt& size);
TORCH_API void checkNumel(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t numel);
TORCH_API void checkSameNumel(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkScalarType(CheckedFrom c, const TensorArg& t, ScalarType s);
TORCH_API void checkScalarTypes(
    CheckedFrom c,
    const TensorArg& t,
    at::ArrayRef<ScalarType> l);
TORCH_API void checkSameGPU(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkSameType(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkSameSize(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
TORCH_API void checkAllSameSize(CheckedFrom c, ArrayRef<TensorArg> tensors);
TORCH_API void checkDefined(CheckedFrom c, const TensorArg& t);
TORCH_API void checkAllDefined(CheckedFrom c, at::ArrayRef<TensorArg> t);

// FixMe: does TensorArg slow things down?
TORCH_API void checkBackend(
    CheckedFrom c,
    at::ArrayRef<Tensor> t,
    at::Backend backend);

TORCH_API void checkDeviceType(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::DeviceType device_type);

TORCH_API void checkLayout(CheckedFrom c, const Tensor& t, Layout layout);

TORCH_API void checkLayout(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::Layout layout);

// Methods for getting data_ptr if tensor is defined
TORCH_API void* maybe_data_ptr(const Tensor& tensor);
TORCH_API void* maybe_data_ptr(const TensorArg& tensor);

TORCH_API void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size);

namespace detail {
TORCH_API std::vector<int64_t> defaultStrides(IntArrayRef sizes);

TORCH_API std::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape);

TORCH_API std::optional<SymDimVector> computeStride(
    c10::SymIntArrayRef oldshape,
    c10::SymIntArrayRef oldstride,
    c10::SymIntArrayRef newshape);

TORCH_API std::optional<DimVector> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    const DimVector& newshape);

} // namespace detail
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 31 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `at`

**Classes/Structs**: `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/DimVector.h`
- `ATen/EmptyTensor.h`
- `ATen/Tensor.h`
- `ATen/TensorGeometry.h`
- `ATen/Utils.h`
- `utility`


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

- **File Documentation**: `TensorUtils.h_docs.md`
- **Keyword Index**: `TensorUtils.h_kw.md`
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

- **File Documentation**: `TensorUtils.h_docs.md_docs.md`
- **Keyword Index**: `TensorUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

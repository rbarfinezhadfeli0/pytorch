# Documentation: `docs/aten/src/ATen/NamedTensorUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/NamedTensorUtils.h_docs.md`
- **Size**: 9,260 bytes (9.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/NamedTensorUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/NamedTensorUtils.h`
- **Size**: 6,755 bytes (6.60 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/NamedTensor.h>
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>

namespace at {

using NameVector = SmallVector<Dimname, kDimVectorStaticSize>;

inline bool has_names(const ITensorListRef& tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const Tensor& t) {
    return t.has_names();
  });
}

// Converts dim to an positional index. Errors if `dim` cannot be used to
// refer to any dimension of tensor.
TORCH_API int64_t dimname_to_position(const Tensor& tensor, Dimname dim);
TORCH_API std::vector<int64_t> dimnames_to_positions(
    const Tensor& tensor,
    DimnameList dims);

// Unifies two DimnameList to produce a third. This is useful for implementing
// the named inference rule for binary broadcasting operations like add.
//
// There are three main constraints:
// 1) Check matching: Names must match positionally from the right.
// 2) Check misaligned: If a name `n` is in `names`, then it must appear at
//    the same index from the right in other.
// 3) The output names are obtained by unifying the names individually from the
// right.
TORCH_API std::vector<Dimname> unify_from_right(
    DimnameList names,
    DimnameList other,
    const char* action = "broadcast");

[[noreturn]] inline void reportNYIDimnameOverload(const char* op_name) {
  TORCH_CHECK(
      false,
      op_name,
      ": You passed a dimname (string) to this op in place of a dimension "
      "index but it does not yet support this behavior. Please pass a dimension "
      "index to work around this.");
}

// [NOTE] Writing name inference rules
//
// Operators that support named tensors are either composed of operations that
// support named tensors or implement some name inference rule. An op that
// implements its own name inference rule generally looks like the following:
//
// Tensor op(...) {
//   perform_shape_checks(...);
//   # (1)
//   auto maybe_outnames = compute_outnames(...);
//   auto result = [&]() {
//     NoNamesGuard guard;
//     return op_impl(...);
//   }();
//   # (2)
//   propagate_names_if_nonempty(result, maybe_outnames);
//
// Each op has (1) a compute outnames step and (2) a propagate names step.
//
// compute_outnames is responsible for checking that input names match and
// determining what the output names should be. It returns either:
// - {} (if the inputs tensors are all unnamed)
// - non-empty outnames.
//
// propagate_names_if_nonempty propagates the outnames if they exist to the
// result tensors.
//
// The {} case is an optimization; if the user does not use named tensors they
// pay no perf cost for it.

namespace namedinference {

const Tensor& propagate_names_if_present_and_nonempty(
    const Tensor& result,
    std::optional<DimnameList> maybe_names,
    bool validate_names = false);
// Propagates `names` to `result` if `names` is not empty.
// `names` can be empty; see [NOTE] Writing name inference rules
// If `names` is not empty, `names.size()` should equal `result.dim()`.
// When in doubt, use this overload instead of the others.
TORCH_API const Tensor& propagate_names_if_nonempty(
    const Tensor& result,
    DimnameList maybe_names,
    bool validate_names = false);

// Propagates `names` to `result`. Only use this if we are certain that there
// are names to propagate (that names is not empty).
TORCH_API const Tensor& propagate_names(
    const Tensor& result,
    DimnameList names,
    bool validate_names = false);

// Propagates all names from src to result.
TORCH_API void propagate_names(const Tensor& result, const Tensor& src);

// Propagates all names except for those at the excluded_idxs.
TORCH_API void propagate_names_except(
    const Tensor& result,
    const Tensor& src,
    IntArrayRef excluded_idxs);

// Used for reduction ops that have a `keepdim` arg.
TORCH_API void propagate_names_for_reduction(
    const Tensor& result,
    const Tensor& src,
    IntArrayRef excluded_idxs,
    bool keepdim);

TORCH_API void propagate_names_for_expand(
    const Tensor& result,
    const Tensor& self);

TORCH_API std::vector<Dimname> compute_cat_outnames(
    const MaterializedITensorListRef& tensors);

TORCH_API std::vector<Dimname> compute_broadcast_outnames(
    const Tensor& self,
    const Tensor& other);

TORCH_API std::vector<Dimname> broadcast_to_outnames(
    const Tensor& tensor,
    const Tensor& reference_tensor,
    const char* op_name);

TORCH_API std::vector<Dimname> compute_matmul_outnames(
    const Tensor& self,
    const Tensor& other);

TORCH_API std::vector<Dimname> compute_cdist_outnames(
    const Tensor& self,
    const Tensor& other);

TORCH_API std::vector<Dimname> compute_bmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other);

TORCH_API std::vector<Dimname> compute_squeeze_outnames(const Tensor& tensor);
TORCH_API std::vector<Dimname> compute_squeeze_outnames(
    const Tensor& tensor,
    std::bitset<dim_bitset_size> dims);

std::vector<Dimname> compute_diagonal_outnames(
    const Tensor& tensor,
    int64_t dim1,
    int64_t dim2);

// TensorImpl* overloads for Legacy TH/THC code. Use these sparingly.

TORCH_API TensorImpl* propagate_names_if_nonempty(
    TensorImpl* result,
    DimnameList maybe_names,
    bool validate_names = false);

TORCH_API TensorImpl* propagate_names(
    TensorImpl* result,
    DimnameList names,
    bool validate_names = false);

TORCH_API void propagate_names(TensorImpl* result, /*const */ TensorImpl* src);

inline void propagate_names(
    const TensorBase& result,
    DimnameList names,
    bool validate_names = false) {
  propagate_names(result.unsafeGetTensorImpl(), names, validate_names);
}

inline void propagate_names_if_nonempty(
    const TensorBase& result,
    DimnameList names,
    bool validate_names = false) {
  propagate_names_if_nonempty(
      result.unsafeGetTensorImpl(), names, validate_names);
}

inline void propagate_names(const TensorBase& result, const TensorBase& src) {
  propagate_names(result.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
}

// result = m1 @ m2 + bias
TORCH_API std::vector<Dimname> propagate_names_for_addmm(
    const Tensor& m1,
    const Tensor& m2,
    const Tensor& bias);

TORCH_API std::vector<Dimname> propagate_names_for_addmv(
    const Tensor& mat,
    const Tensor& vec,
    const Tensor& bias);

TORCH_API void check_names_for_dot(TensorImpl* vec1, TensorImpl* vec2);

TORCH_API std::vector<Dimname> compute_baddbmm_outnames(
    const Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const Tensor& bias);

TORCH_API bool are_names_equal(TensorImpl* self, TensorImpl* other);

} // namespace namedinference

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namedinference`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/NamedTensor.h`
- `ATen/TensorNames.h`
- `ATen/WrapDimUtilsMulti.h`
- `ATen/core/DimVector.h`
- `ATen/core/Tensor.h`


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

- **File Documentation**: `NamedTensorUtils.h_docs.md`
- **Keyword Index**: `NamedTensorUtils.h_kw.md`
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

- **File Documentation**: `NamedTensorUtils.h_docs.md_docs.md`
- **Keyword Index**: `NamedTensorUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

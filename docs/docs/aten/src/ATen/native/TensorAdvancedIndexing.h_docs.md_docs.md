# Documentation: `docs/aten/src/ATen/native/TensorAdvancedIndexing.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/TensorAdvancedIndexing.h_docs.md`
- **Size**: 5,427 bytes (5.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/TensorAdvancedIndexing.h`

## File Metadata

- **Path**: `aten/src/ATen/native/TensorAdvancedIndexing.h`
- **Size**: 2,897 bytes (2.83 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// Indexing tensors by tensors

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReductionType.h>

namespace at {
struct TensorIterator;
}

namespace at::native {

using index_put_with_sort_fn = void (*)(
    Tensor&,
    const c10::List<std::optional<Tensor>>&,
    const Tensor&,
    bool accumulate,
    bool unsafe);
using index_put_with_sort_quantized_fn = void (*)(
    Tensor& self,
    const c10::List<std::optional<Tensor>>& indices,
    const Tensor& value,
    double scale,
    int zero_point,
    bool unsafe);
using gather_fn = void (*)(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index);
using scatter_fn = void (*)(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);
using scatter_fill_fn = void (*)(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src);
using scatter_add_fn = void (*)(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);
using scatter_reduce_fn = void (*)(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);
using scatter_scalar_reduce_fn = void (*)(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const ReductionType& reduce);
using scatter_reduce_two_fn = void (*)(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

DECLARE_DISPATCH(index_put_with_sort_fn, index_put_with_sort_stub)
DECLARE_DISPATCH(
    index_put_with_sort_quantized_fn,
    index_put_with_sort_quantized_stub)
DECLARE_DISPATCH(gather_fn, gather_stub)
DECLARE_DISPATCH(scatter_fn, scatter_stub)
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub)
DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub)
DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub)
DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub)
DECLARE_DISPATCH(scatter_reduce_two_fn, scatter_reduce_two_stub)

TORCH_API Tensor& index_out(
    Tensor& result,
    const Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices);

using scatter_add_expanded_index_fn =
    void (*)(const Tensor&, const Tensor&, const Tensor&);
using scatter_reduce_expanded_index_fn = void (*)(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const ReductionType& reduce,
    bool);
using gather_expanded_index_fn =
    void (*)(const Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(scatter_add_expanded_index_fn, scatter_add_expanded_index_stub)
DECLARE_DISPATCH(
    scatter_reduce_expanded_index_fn,
    scatter_reduce_expanded_index_stub)
DECLARE_DISPATCH(gather_expanded_index_fn, gather_expanded_index_stub)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TensorIterator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/List.h`
- `ATen/core/Tensor.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/ReductionType.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `TensorAdvancedIndexing.h_docs.md`
- **Keyword Index**: `TensorAdvancedIndexing.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TensorAdvancedIndexing.h_docs.md_docs.md`
- **Keyword Index**: `TensorAdvancedIndexing.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/sparse/SparseFactories.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/sparse/SparseFactories.cpp_docs.md`
- **Size**: 6,047 bytes (5.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/sparse/SparseFactories.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/SparseFactories.cpp`
- **Size**: 3,372 bytes (3.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/sparse/SparseFactories.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_spdiags_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {

DEFINE_DISPATCH(spdiags_kernel_stub);

Tensor spdiags(
    const Tensor& diagonals,
    const Tensor& offsets,
    IntArrayRef shape,
    std::optional<Layout> layout) {
  auto diagonals_2d = diagonals.dim() == 1 ? diagonals.unsqueeze(0) : diagonals;
  TORCH_CHECK(diagonals_2d.dim() == 2, "Diagonals must be vector or matrix");
  TORCH_CHECK(shape.size() == 2, "Output shape must be 2d");
  auto offsets_1d = offsets.dim() == 0 ? offsets.unsqueeze(0) : offsets;
  TORCH_CHECK(offsets_1d.dim() == 1, "Offsets must be scalar or vector");
  TORCH_CHECK(
      diagonals_2d.size(0) == offsets_1d.size(0),
      "Number of diagonals (",
      diagonals_2d.size(0),
      ") does not match the number of offsets (",
      offsets_1d.size(0),
      ")");
  if (layout) {
    TORCH_CHECK(
        (*layout == Layout::Sparse) || (*layout == Layout::SparseCsc) ||
            (*layout == Layout::SparseCsr),
        "Only output layouts (Sparse, SparseCsc, SparseCsr) are supported, got ",
        *layout);
  }
  TORCH_CHECK(
      offsets_1d.scalar_type() == at::kLong,
      "Offset Tensor must have dtype Long but got ",
      offsets_1d.scalar_type());

  TORCH_CHECK(
      offsets_1d.numel() == std::get<0>(at::_unique(offsets_1d)).numel(),
      "Offset tensor contains duplicate values");

  auto nnz_per_diag = at::where(
      offsets_1d.le(0),
      offsets_1d.add(shape[0]).clamp_max_(diagonals_2d.size(1)),
      offsets_1d.add(-std::min<int64_t>(shape[1], diagonals_2d.size(1))).neg());

  auto nnz_per_diag_cumsum = nnz_per_diag.cumsum(-1);
  const auto nnz = diagonals_2d.size(0) > 0
      ? nnz_per_diag_cumsum.select(-1, -1).item<int64_t>()
      : int64_t{0};
  // Offsets into nnz for each diagonal
  auto result_mem_offsets = nnz_per_diag_cumsum.sub(nnz_per_diag);
  // coo tensor guts
  auto indices = at::empty({2, nnz}, offsets_1d.options());
  auto values = at::empty({nnz}, diagonals_2d.options());
  // We add this indexer to lookup the row of diagonals we are reading from at
  // each iteration
  const auto n_diag = offsets_1d.size(0);
  Tensor diag_index = at::arange(n_diag, offsets_1d.options());
  // cpu_kernel requires an output
  auto dummy = at::empty({1}, offsets_1d.options()).resize_({0});
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .add_output(dummy)
                  .add_input(diag_index)
                  .add_input(offsets_1d)
                  .add_input(result_mem_offsets)
                  .add_input(nnz_per_diag)
                  .build();
  spdiags_kernel_stub(iter.device_type(), iter, diagonals_2d, values, indices);
  auto result_coo = at::sparse_coo_tensor(indices, values, shape);
  if (layout) {
    if (*layout == Layout::SparseCsr) {
      return result_coo.to_sparse_csr();
    }
    if (*layout == Layout::SparseCsc) {
      return result_coo.to_sparse_csc();
    }
  }
  return result_coo;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/sparse`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Dispatch.h`
- `ATen/TensorIterator.h`
- `ATen/native/sparse/SparseFactories.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_spdiags_native.h`
- `ATen/ops/_unique.h`
- `ATen/ops/arange.h`
- `ATen/ops/empty.h`
- `ATen/ops/sparse_coo_tensor.h`
- `ATen/ops/where.h`


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

Files in the same folder (`aten/src/ATen/native/sparse`):

- [`SparseBinaryOpIntersectionCommon.h_docs.md`](./SparseBinaryOpIntersectionCommon.h_docs.md)
- [`ParamUtils.cpp_docs.md`](./ParamUtils.cpp_docs.md)
- [`SparseTensor.cpp_docs.md`](./SparseTensor.cpp_docs.md)
- [`ValidateCompressedIndicesKernel.cpp_docs.md`](./ValidateCompressedIndicesKernel.cpp_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`SparseBlas.h_docs.md`](./SparseBlas.h_docs.md)
- [`SparseStubs.h_docs.md`](./SparseStubs.h_docs.md)
- [`SparseCsrTensorMath.h_docs.md`](./SparseCsrTensorMath.h_docs.md)
- [`SparseTensorMath.h_docs.md`](./SparseTensorMath.h_docs.md)


## Cross-References

- **File Documentation**: `SparseFactories.cpp_docs.md`
- **Keyword Index**: `SparseFactories.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/sparse`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/sparse`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/sparse`):

- [`ValidateCompressedIndicesKernel.cpp_docs.md_docs.md`](./ValidateCompressedIndicesKernel.cpp_docs.md_docs.md)
- [`SparseTensorMath.h_docs.md_docs.md`](./SparseTensorMath.h_docs.md_docs.md)
- [`SparseBlasImpl.h_kw.md_docs.md`](./SparseBlasImpl.h_kw.md_docs.md)
- [`SparseCsrTensorMath.h_docs.md_docs.md`](./SparseCsrTensorMath.h_docs.md_docs.md)
- [`SparseBlas.h_docs.md_docs.md`](./SparseBlas.h_docs.md_docs.md)
- [`FlattenIndicesKernel.cpp_kw.md_docs.md`](./FlattenIndicesKernel.cpp_kw.md_docs.md)
- [`SoftMax.cpp_docs.md_docs.md`](./SoftMax.cpp_docs.md_docs.md)
- [`SparseTensor.cpp_kw.md_docs.md`](./SparseTensor.cpp_kw.md_docs.md)
- [`SparseBinaryOpIntersectionKernel.cpp_docs.md_docs.md`](./SparseBinaryOpIntersectionKernel.cpp_docs.md_docs.md)
- [`SparseCsrTensor.cpp_docs.md_docs.md`](./SparseCsrTensor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `SparseFactories.cpp_docs.md_docs.md`
- **Keyword Index**: `SparseFactories.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

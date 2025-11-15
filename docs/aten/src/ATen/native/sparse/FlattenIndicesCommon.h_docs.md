# Documentation: `aten/src/ATen/native/sparse/FlattenIndicesCommon.h`

## File Metadata

- **Path**: `aten/src/ATen/native/sparse/FlattenIndicesCommon.h`
- **Size**: 3,324 bytes (3.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sparse/Macros.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/SparseTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/tensor.h>
#endif

#ifdef GPUCC
#define NAME "flatten_indices_cuda"
#else
#define NAME "flatten_indices_cpu"
#endif

namespace at::native {

namespace {

template <template <typename func_t> class kernel_t>
struct KernelLauncher {
  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

template <
  template <typename func_t> class kernel_t,
  typename index_t,
  int64_t max_static_len = 0>
Tensor _flatten_indices_impl(const Tensor& indices, IntArrayRef size) {
  TORCH_INTERNAL_ASSERT(indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size());

  // Need owning storage in case of the Tensor class.
  const auto hash_coeffs_storage = [&]() -> auto {
    auto strides = c10::contiguous_strides(size);
    return at::sparse::TensorGeometryHolder<max_static_len>(strides, strides, indices.options());
  }();
  const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);

  const auto hash_indices = [&]() -> Tensor {
    const auto sparse_dim = indices.size(0);
    const auto indices_dim_stride = indices.stride(0);
    const auto indices_nnz_stride = indices.stride(1);

    auto hash = at::arange(indices.size(1), indices.options().dtype(kLong));

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_output(hash)
      .add_input(hash)
      .build();

    {
      const auto* RESTRICT ptr_indices = indices.const_data_ptr<index_t>();

      KernelLauncher<kernel_t>::launch(iter,
          // NOTE: capture by value required by CUDA
          [=] FUNCAPI (int64_t nnz_idx) -> int64_t {
          const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
          auto hash = static_cast<int64_t>(0);
          for (int64_t dim = 0; dim < sparse_dim; ++dim) {
            const auto dim_hash_coeff = hash_coeffs[dim];
            const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
            hash += dim_index * dim_hash_coeff;
          }
          return hash;
      });
    }

    return hash;
  }();

  return hash_indices;
}

template <template <typename func_t> class kernel_t>
Tensor _flatten_indices(const Tensor& indices, IntArrayRef size) {
  TORCH_CHECK(indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size(),
      NAME, "(): the dimensionality of sparse `indices` and the length of `size` must match. ",
            "Got `indices.size(0) == ", indices.size(0), "` != `size.size() == ", size.size(), "`.");
  Tensor flattened_indices;
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), NAME, [&] () {
    constexpr int64_t max_sparse_dims = 8;
    if (indices.size(0) <= max_sparse_dims) {
      flattened_indices = _flatten_indices_impl<kernel_t, index_t, max_sparse_dims>(indices, size);
    } else {
      flattened_indices = _flatten_indices_impl<kernel_t, index_t>(indices, size);
    }
  });
  return flattened_indices;
}

}

} // at::native

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `kernel_t`, `KernelLauncher`, `kernel_t`, `kernel_t`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/sparse`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `ATen/native/TensorIterator.h`
- `ATen/Dispatch.h`
- `ATen/native/sparse/Macros.h`
- `ATen/ExpandUtils.h`
- `ATen/native/SparseTensorUtils.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/arange.h`
- `ATen/ops/tensor.h`


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

Files in the same folder (`aten/src/ATen/native/sparse`):

- [`SparseBinaryOpIntersectionCommon.h_docs.md`](./SparseBinaryOpIntersectionCommon.h_docs.md)
- [`SparseFactories.cpp_docs.md`](./SparseFactories.cpp_docs.md)
- [`ParamUtils.cpp_docs.md`](./ParamUtils.cpp_docs.md)
- [`SparseTensor.cpp_docs.md`](./SparseTensor.cpp_docs.md)
- [`ValidateCompressedIndicesKernel.cpp_docs.md`](./ValidateCompressedIndicesKernel.cpp_docs.md)
- [`SparseBlas.cpp_docs.md`](./SparseBlas.cpp_docs.md)
- [`SparseBlas.h_docs.md`](./SparseBlas.h_docs.md)
- [`SparseStubs.h_docs.md`](./SparseStubs.h_docs.md)
- [`SparseCsrTensorMath.h_docs.md`](./SparseCsrTensorMath.h_docs.md)
- [`SparseTensorMath.h_docs.md`](./SparseTensorMath.h_docs.md)


## Cross-References

- **File Documentation**: `FlattenIndicesCommon.h_docs.md`
- **Keyword Index**: `FlattenIndicesCommon.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

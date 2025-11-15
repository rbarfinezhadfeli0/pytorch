# Documentation: `aten/src/ATen/native/NonSymbolicBC.h`

## File Metadata

- **Path**: `aten/src/ATen/native/NonSymbolicBC.h`
- **Size**: 2,876 bytes (2.81 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <ATen/core/IListRef.h>

namespace at::native {
// This file contains non-symbolic signatures for ops that we have sym-intified the signature of.
// However, in certain cases (such as static runtime), we call the native versions of the ops directly.
// In those cases, we will duplicate the signature here with non-symbolic ints, and also duplicate the C++ implementation.
TORCH_API at::Tensor reshape(const at::Tensor& self, at::IntArrayRef proposed_shape);
TORCH_API at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length);
TORCH_API at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, std::optional<at::ScalarType> dtype=std::nullopt, std::optional<at::Layout> layout=std::nullopt, std::optional<at::Device> device=std::nullopt, std::optional<bool> pin_memory=std::nullopt, std::optional<bool> is_coalesced=std::nullopt);
TORCH_API at::Tensor nll_loss(const at::Tensor & self, const at::Tensor & target, const std::optional<at::Tensor>& weight_opt, int64_t reduction, int64_t ignore_index);
TORCH_API at::Tensor nll_loss2d(const at::Tensor & self, const at::Tensor & target, const std::optional<at::Tensor>& weight_opt, int64_t reduction, int64_t ignore_index);
// The below ops don't get a duplicated C++ implementation.
// They are backward ops, which make them very unlikely to be called directly
// by external code (at::native::trace_backward).
// They get their own declaration for BC purposes however.
TORCH_API at::Tensor _embedding_bag_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx=-1);
TORCH_API at::Tensor _embedding_bag_sparse_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx=-1);
TORCH_API at::Tensor value_selecting_reduction_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, at::IntArrayRef sizes, bool keepdim);
TORCH_API at::Tensor trace_backward(const at::Tensor & grad, at::IntArrayRef sizes);
TORCH_API at::Tensor index_select_backward(const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index);
TORCH_API at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index);
TORCH_API std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim);
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `c10/util/irange.h`
- `ATen/core/IListRef.h`


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

- **File Documentation**: `NonSymbolicBC.h_docs.md`
- **Keyword Index**: `NonSymbolicBC.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

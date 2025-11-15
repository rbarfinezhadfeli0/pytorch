# Documentation: `aten/src/ATen/native/cuda/FusedAdagradKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/FusedAdagradKernel.cu`
- **Size**: 2,566 bytes (2.51 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adagrad_impl.cuh>

namespace at::native {

void _fused_adagrad_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params, grads, and state_sums must have same dtype, device, and layout");
  _fused_adagrad_cuda_impl_(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

void _fused_adagrad_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adagrad_kernel_cuda_(
        params,
        grads,
        state_sums,
        state_steps,
        lr.item<double>(),
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  // Manually check devices since we specify no device check in
  // native_functions.yaml
  Device param_device = params[0].device();
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same GPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == param_device,
      "lr must be on the same GPU device as the params");

  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params and grads must have same dtype, device, and layout");
  _fused_adagrad_cuda_impl_(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

} // namespace at::native
```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/TypeDefault.h`
- `ATen/native/ForeachUtils.h`
- `ATen/native/cuda/fused_adagrad_impl.cuh`


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

Files in the same folder (`aten/src/ATen/native/cuda`):

- [`LogcumsumexpKernel.cu_docs.md`](./LogcumsumexpKernel.cu_docs.md)
- [`WeightNorm.cu_docs.md`](./WeightNorm.cu_docs.md)
- [`SparseBinaryOpIntersectionKernel.cu_docs.md`](./SparseBinaryOpIntersectionKernel.cu_docs.md)
- [`jit_utils.cpp_docs.md`](./jit_utils.cpp_docs.md)
- [`ReduceNormKernel.cu_docs.md`](./ReduceNormKernel.cu_docs.md)
- [`BinaryMiscOpsKernels.cu_docs.md`](./BinaryMiscOpsKernels.cu_docs.md)
- [`RowwiseScaledMM.h_docs.md`](./RowwiseScaledMM.h_docs.md)
- [`fused_adamw_amsgrad_impl.cuh_docs.md`](./fused_adamw_amsgrad_impl.cuh_docs.md)
- [`Col2Im.cu_docs.md`](./Col2Im.cu_docs.md)
- [`DistributionRandomKernel.cu_docs.md`](./DistributionRandomKernel.cu_docs.md)


## Cross-References

- **File Documentation**: `FusedAdagradKernel.cu_docs.md`
- **Keyword Index**: `FusedAdagradKernel.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

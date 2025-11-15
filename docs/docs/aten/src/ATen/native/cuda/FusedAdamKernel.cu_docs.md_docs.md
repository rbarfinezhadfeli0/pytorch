# Documentation: `docs/aten/src/ATen/native/cuda/FusedAdamKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/FusedAdamKernel.cu_docs.md`
- **Size**: 7,502 bytes (7.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/FusedAdamKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/FusedAdamKernel.cu`
- **Size**: 4,731 bytes (4.62 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/fused_adam_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adam_impl.cuh>

namespace at::native {

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile,
// defensively split instantiations into _impl files. this is only for CUDA 11.3
// for which it took about 20 minutes and 28 minutes in my workstation and CI,
// respectively. As a data point, it took about 20 seconds for CUDA 11.7
// installed in my environment. See
// https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adam_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_amsgrad_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

// The following overload simply has a Tensor lr
void _fused_adam_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adam_kernel_cuda_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr.item<double>(),
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
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

  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_amsgrad_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
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
- `c10/util/Exception.h`
- `ATen/native/cuda/fused_adam_amsgrad_impl.cuh`
- `ATen/native/cuda/fused_adam_impl.cuh`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `FusedAdamKernel.cu_docs.md`
- **Keyword Index**: `FusedAdamKernel.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/native/cuda`):

- [`DeviceSqrt.cuh_kw.md_docs.md`](./DeviceSqrt.cuh_kw.md_docs.md)
- [`UnaryGeometricAsinKernel.cu_kw.md_docs.md`](./UnaryGeometricAsinKernel.cu_kw.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`fused_adamw_impl.cu_docs.md_docs.md`](./fused_adamw_impl.cu_docs.md_docs.md)
- [`TensorTopK.h_kw.md_docs.md`](./TensorTopK.h_kw.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`FusedSgdKernel.cu_docs.md_docs.md`](./FusedSgdKernel.cu_docs.md_docs.md)
- [`Distributions.cu_kw.md_docs.md`](./Distributions.cu_kw.md_docs.md)
- [`block_reduce.cuh_docs.md_docs.md`](./block_reduce.cuh_docs.md_docs.md)
- [`fused_adagrad_impl.cuh_kw.md_docs.md`](./fused_adagrad_impl.cuh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `FusedAdamKernel.cu_docs.md_docs.md`
- **Keyword Index**: `FusedAdamKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

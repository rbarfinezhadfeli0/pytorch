# Documentation: FusedAdamKernel.mm

## File Metadata
- **Path**: `aten/src/ATen/native/mps/operations/FusedAdamKernel.mm`
- **Size**: 6085 bytes
- **Lines**: 141
- **Extension**: .mm
- **Type**: Regular file

## Original Source

```mm
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/mps/operations/FusedAdamAmsgradKernelImpl.h>
#include <ATen/native/mps/operations/FusedAdamKernelImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adam_native.h>
#endif

namespace at::native {
using namespace mps;

void _fused_adam_kernel_mps_(TensorList params,
                             TensorList grads,
                             TensorList exp_avgs,
                             TensorList exp_avg_sqs,
                             TensorList max_exp_avg_sqs,
                             TensorList state_steps,
                             const double lr,
                             const double beta1,
                             const double beta2,
                             const double weight_decay,
                             const double eps,
                             const bool amsgrad,
                             const bool maximize,
                             const std::optional<Tensor>& grad_scale,
                             const std::optional<Tensor>& found_inf) {
  if (amsgrad) {
    TORCH_CHECK(native::check_fast_path_restrictions({params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
                "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_amsgrad_mps_impl_(params,
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
    TORCH_CHECK(native::check_fast_path_restrictions({params, grads, exp_avgs, exp_avg_sqs}),
                "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_mps_impl_(params,
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
void _fused_adam_kernel_mps_(TensorList params,
                             TensorList grads,
                             TensorList exp_avgs,
                             TensorList exp_avg_sqs,
                             TensorList max_exp_avg_sqs,
                             TensorList state_steps,
                             const Tensor& lr,
                             const double beta1,
                             const double beta2,
                             const double weight_decay,
                             const double eps,
                             const bool amsgrad,
                             const bool maximize,
                             const std::optional<Tensor>& grad_scale,
                             const std::optional<Tensor>& found_inf) {
  if (lr.is_cpu()) {
    return _fused_adam_kernel_mps_(params,
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
  }

  // Manually check devices since we specify no device check in
  // native_functions.yaml
  Device param_device = params[0].device();
  TORCH_CHECK(lr.device() == param_device, "lr must be on the same GPU device as the params");

  if (amsgrad) {
    TORCH_CHECK(native::check_fast_path_restrictions({params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
                "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_amsgrad_mps_impl_(params,
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
    TORCH_CHECK(native::check_fast_path_restrictions({params, grads, exp_avgs, exp_avg_sqs}),
                "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adam_mps_impl_(params,
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

This file is part of the PyTorch repository. It is a source or configuration file.

## Detailed Walkthrough


## Key Components

The file contains 308 words across 141 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6085 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

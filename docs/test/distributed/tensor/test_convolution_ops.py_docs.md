# Documentation: `test/distributed/tensor/test_convolution_ops.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_convolution_ops.py`
- **Size**: 9,407 bytes (9.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.nn import functional as F
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


ITER_TIME = 10
LR = 0.001


def _conv_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    for name, param in module.named_parameters():
        dist_spec = [Replicate()]
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        name = "_".join(name.split("."))
        module.register_parameter(name, dist_param)


class DistConvolutionOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # hard code world size to 2
        return 2

    @with_comms
    def test_downsampling_convolution(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(3)]

        input_list = torch.rand(ITER_TIME, 7, 3, 512, 1024)
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        model = nn.Conv2d(3, 256, kernel_size=4, stride=4, padding=0).to(
            self.device_type
        )
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        model_gt = copy.deepcopy(model).to(self.device_type)

        # training with dtensor
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            output = model(inp_dtensor)
            grad_output = grad_output_list[i].to(self.device_type)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output.backward(grad_output_dtensor)
            optimizer.step()

        # training with plain tensor
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            output = model_gt(inp)
            grad_output = grad_output_list[i].to(self.device_type)
            output.backward(grad_output)
            optimizer_gt.step()

        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            f"Too large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            f"Too large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            f"Too large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            f"Too large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )

    # TODO: test_depthwise_convolution is broken in CI with gloo backend.
    # Temporarily disable it to unblock CI.
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_depthwise_convolution(self):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(3)]

        input_list = torch.rand(ITER_TIME, 7, 256, 128, 256)
        grad_output_list = torch.rand(ITER_TIME, 7, 256, 128, 256) * 1e-3

        model = nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256).to(
            self.device_type
        )
        nn.init.ones_(model.weight)
        nn.init.zeros_(model.bias)
        model_gt = copy.deepcopy(model).to(self.device_type)

        # training with dtensor
        model = distribute_module(
            model, device_mesh, _conv_fn, input_fn=None, output_fn=None
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            inp_dtensor = distribute_tensor(inp, device_mesh, shard_spec)
            output = model(inp_dtensor)
            grad_output = grad_output_list[i].to(self.device_type)
            grad_output_dtensor = distribute_tensor(
                grad_output, device_mesh, shard_spec
            )
            output.backward(grad_output_dtensor)
            optimizer.step()

        # training with plain tensor
        optimizer_gt = torch.optim.SGD(model_gt.parameters(), lr=LR)
        for i in range(ITER_TIME):
            optimizer_gt.zero_grad()
            inp = input_list[i].to(self.device_type).requires_grad_()
            output = model_gt(inp)
            grad_output = grad_output_list[i].to(self.device_type)
            output.backward(grad_output)
            optimizer_gt.step()

        weight_diff_abs = model.weight.to_local() - model_gt.weight
        bias_diff_abs = model.bias.to_local() - model_gt.bias
        weight_diff_rel = weight_diff_abs / (torch.abs(model_gt.weight) + 1e-8)
        bias_diff_rel = bias_diff_abs / (torch.abs(model_gt.bias) + 1e-8)
        weight_mse_abs = torch.mean(weight_diff_abs * weight_diff_abs).item()
        bias_mse_abs = torch.mean(bias_diff_abs * bias_diff_abs).item()
        weight_mse_rel = torch.mean(weight_diff_rel * weight_diff_rel).item()
        bias_mse_rel = torch.mean(bias_diff_rel * bias_diff_rel).item()
        self.assertTrue(
            weight_mse_abs <= 1e-6,
            f"Too large absolute mse for weight tensor, expected less equal 1e-6, got {weight_mse_abs}",
        )
        self.assertTrue(
            bias_mse_abs <= 1e-6,
            f"Too large absolute mse for bias tensor, expected less equal 1e-6, got {bias_mse_abs}",
        )
        self.assertTrue(
            weight_mse_rel <= 1e-6,
            f"Too large relative mse for weight tensor, expected less equal 1e-6, got {weight_mse_rel}",
        )
        self.assertTrue(
            bias_mse_rel <= 1e-6,
            f"Too large relative mse for bias tensor, expected less equal 1e-6, got {bias_mse_rel}",
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_conv_backward_none_grad_inp(self):
        device_mesh = self.build_device_mesh()
        conv = nn.Conv2d(64, 64, 3, padding=1).train()
        x = torch.randn(1, 64, 32, 32)
        x_dt = DTensor.from_local(x, device_mesh, [Replicate()])
        w = conv.weight
        w_dt = torch.nn.Parameter(DTensor.from_local(w, device_mesh, [Replicate()]))

        b = conv.bias
        b_dt = torch.nn.Parameter(DTensor.from_local(b, device_mesh, [Replicate()]))

        res = F.conv2d(x_dt, w_dt, b_dt, padding=1)
        dres = torch.rand_like(res)
        res.backward(dres)
        self.assertTrue(w_dt.grad is not None)
        self.assertTrue(b_dt.grad is not None)
        self.assertTrue(x_dt.grad is None)

    def _run_single_arg_fwd(
        self, model, arg, placements=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given model and arg, runs fwd model local and distbuted given device_mesh"""
        device_mesh = self.build_device_mesh()
        model_copy = copy.deepcopy(model).to(device=self.device_type)
        dist_model = distribute_module(model, device_mesh, _conv_fn)
        arg_dt = DTensor.from_local(arg, device_mesh, placements)
        out_dt = dist_model(arg_dt.to(device=self.device_type))
        out = model_copy(arg_dt.full_tensor())
        return (out_dt.full_tensor(), out)

    @with_comms
    def test_conv1d(self):
        model = nn.Conv1d(64, 64, 3, padding=1)
        x = torch.randn(1, 64, 8, device=self.device_type)
        out_dt, out = self._run_single_arg_fwd(model, x)
        self.assertEqual(out_dt, out)

    @with_comms
    def test_conv3d(self):
        model = nn.Conv3d(64, 64, 3, padding=1)
        x = torch.randn(1, 64, 8, 8, 8, device=self.device_type)
        out_dt, out = self._run_single_arg_fwd(model, x, [Shard(0)])
        self.assertEqual(out_dt, out)


DistConvolutionOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistConvolutionOpsTest,
    # Send / recv ops are not supported
    skipped_tests=[
        "test_conv_backward_none_grad_inp",
        "test_depthwise_convolution",
        "test_downsampling_convolution",
    ],
)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DistConvolutionOpsTest`

**Functions defined**: `_conv_fn`, `world_size`, `test_downsampling_convolution`, `test_depthwise_convolution`, `test_conv_backward_none_grad_inp`, `_run_single_arg_fwd`, `test_conv1d`, `test_conv3d`

**Key imports**: copy, torch, torch.nn as nn, DeviceMesh, functional as F, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch.nn as nn`
- `torch.distributed`: DeviceMesh
- `torch.nn`: functional as F
- `torch.testing._internal.common_utils`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python test/distributed/tensor/test_convolution_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_dtensor.py_docs.md`](./test_dtensor.py_docs.md)
- [`test_dtensor_testbase.py_docs.md`](./test_dtensor_testbase.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_dtensor_dispatch_overhead.py_docs.md`](./test_dtensor_dispatch_overhead.py_docs.md)
- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_matrix_ops.py_docs.md`](./test_matrix_ops.py_docs.md)
- [`test_op_schema.py_docs.md`](./test_op_schema.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_convolution_ops.py_docs.md`
- **Keyword Index**: `test_convolution_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

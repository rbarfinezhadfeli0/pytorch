# Documentation: `docs/test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py_docs.md`
- **Size**: 7,684 bytes (7.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py`

## File Metadata

- **Path**: `test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py`
- **Size**: 4,169 bytes (4.07 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import copy

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler, OptState
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, MLP
from torch.testing._internal.common_utils import run_tests


device_type = torch.device(get_devtype())


class TestFullyShardGradientScaler(FSDPTest):
    @skip_if_lt_x_gpu(4)
    def test_gradient_scaler(self):
        self.run_subtests(
            {"has_inf": [True, False], "test_2d": [True, False]},
            self._test_gradient_scaler,
        )

    def _test_gradient_scaler(self, has_inf: bool, test_2d: bool):
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device=device_type, bias=False) for _ in range(2)]
        )
        for layer in model:
            fully_shard(layer)
        fully_shard(model)
        input = torch.randn([4, 4], device=device_type)

        if test_2d:
            mesh_2d = init_device_mesh(
                device_type.type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
            )
            dp_mesh, tp_mesh = mesh_2d["dp"], mesh_2d["tp"]
            model = nn.Sequential(MLP(2), MLP(2), MLP(2))
            tp_parallelize_plan = {
                "0.in_proj": ColwiseParallel(),
                "0.out_proj": RowwiseParallel(),
                "1.in_proj": ColwiseParallel(),
                "1.out_proj": RowwiseParallel(),
                "2.in_proj": ColwiseParallel(),
                "2.out_proj": RowwiseParallel(),
            }
            model = parallelize_module(
                model,
                device_mesh=tp_mesh,
                parallelize_plan=tp_parallelize_plan,
            )
            for module in model:
                fully_shard(module, mesh=dp_mesh)
            fully_shard(model, mesh=dp_mesh)
            input = torch.randn((2,), device=device_type)

        loss = model(input).sum()
        scaler = GradScaler(init_scale=2.0, enabled=True, device=device_type.type)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        scaler.scale(loss).backward()
        inv_scale = scaler._scale.double().reciprocal().float()
        if (
            has_inf is True
            and opt.param_groups[0]["params"][0].grad._local_tensor.device.index == 1
        ):
            opt.param_groups[0]["params"][0].grad._local_tensor[0, 0].fill_(
                float("inf")
            )
        initial_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()

        scaler.unscale_(opt)
        for found_inf in scaler._per_optimizer_states[id(opt)][
            "found_inf_per_device"
        ].values():
            self.assertEqual(found_inf, has_inf)
        self.assertEqual(
            scaler._per_optimizer_states[id(opt)]["stage"].value,
            OptState.UNSCALED.value,
        )
        unscaled_grad = opt.param_groups[0]["params"][0].grad.to_local().clone()
        self.assertEqual(unscaled_grad, initial_grad * inv_scale)
        initial_scale = scaler.get_scale()
        initial_state = copy.copy(opt.state)

        scaler.step(opt)
        steped_state = copy.copy(opt.state)
        if has_inf:
            # assert parameters are the same before/after
            self.assertEqual(steped_state, initial_state)
        else:
            # new parameters here if no inf found during .unscale_()
            self.assertNotEqual(steped_state.items(), initial_state.items())

        scaler.update()
        updated_scale = scaler.get_scale()
        if has_inf:
            # assert scale is updated
            backoff_factor = scaler.get_backoff_factor()
            self.assertEqual(updated_scale, initial_scale * backoff_factor)
        else:
            # scale is not updated
            self.assertEqual(updated_scale, initial_scale)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFullyShardGradientScaler`

**Functions defined**: `test_gradient_scaler`, `_test_gradient_scaler`

**Key imports**: copy, torch, torch.nn as nn, GradScaler, OptState, init_device_mesh, fully_shard, skip_if_lt_x_gpu, FSDPTest, get_devtype, MLP, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_composable/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch.nn as nn`
- `torch.amp.grad_scaler`: GradScaler, OptState
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: fully_shard
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTest, get_devtype, MLP
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
python test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_composable/fsdp`):

- [`test_fully_shard_extensions.py_docs.md`](./test_fully_shard_extensions.py_docs.md)
- [`test_fully_shard_logging.py_docs.md`](./test_fully_shard_logging.py_docs.md)
- [`test_fully_shard_mixed_precision.py_docs.md`](./test_fully_shard_mixed_precision.py_docs.md)
- [`test_fully_shard_ignore_params.py_docs.md`](./test_fully_shard_ignore_params.py_docs.md)
- [`test_fully_shard_frozen.py_docs.md`](./test_fully_shard_frozen.py_docs.md)
- [`test_fully_shard_clip_grad_norm_.py_docs.md`](./test_fully_shard_clip_grad_norm_.py_docs.md)
- [`test_fully_shard_state.py_docs.md`](./test_fully_shard_state.py_docs.md)
- [`test_fully_shard_overlap.py_docs.md`](./test_fully_shard_overlap.py_docs.md)
- [`test_fully_shard_state_dict.py_docs.md`](./test_fully_shard_state_dict.py_docs.md)
- [`test_fully_shard_init.py_docs.md`](./test_fully_shard_init.py_docs.md)


## Cross-References

- **File Documentation**: `test_fully_shard_grad_scaler.py_docs.md`
- **Keyword Index**: `test_fully_shard_grad_scaler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_composable/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_composable/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/_composable/fsdp/test_fully_shard_grad_scaler.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_composable/fsdp`):

- [`test_fully_shard_clip_grad_norm_.py_docs.md_docs.md`](./test_fully_shard_clip_grad_norm_.py_docs.md_docs.md)
- [`test_fully_shard_autograd.py_kw.md_docs.md`](./test_fully_shard_autograd.py_kw.md_docs.md)
- [`test_fully_shard_ignore_params.py_kw.md_docs.md`](./test_fully_shard_ignore_params.py_kw.md_docs.md)
- [`test_fully_shard_comm.py_docs.md_docs.md`](./test_fully_shard_comm.py_docs.md_docs.md)
- [`test_fully_shard_state.py_docs.md_docs.md`](./test_fully_shard_state.py_docs.md_docs.md)
- [`test_fully_shard_ignore_params.py_docs.md_docs.md`](./test_fully_shard_ignore_params.py_docs.md_docs.md)
- [`test_fully_shard_clip_grad_norm_.py_kw.md_docs.md`](./test_fully_shard_clip_grad_norm_.py_kw.md_docs.md)
- [`test_fully_shard_state.py_kw.md_docs.md`](./test_fully_shard_state.py_kw.md_docs.md)
- [`test_fully_shard_mixed_precision.py_docs.md_docs.md`](./test_fully_shard_mixed_precision.py_docs.md_docs.md)
- [`test_fully_shard_state_dict.py_kw.md_docs.md`](./test_fully_shard_state_dict.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fully_shard_grad_scaler.py_docs.md_docs.md`
- **Keyword Index**: `test_fully_shard_grad_scaler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

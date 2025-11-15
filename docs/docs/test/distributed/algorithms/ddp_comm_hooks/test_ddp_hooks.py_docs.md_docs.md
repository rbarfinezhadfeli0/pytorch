# Documentation: `docs/test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py_docs.md`
- **Size**: 10,256 bytes (10.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py`

## File Metadata

- **Path**: `test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py`
- **Size**: 7,440 bytes (7.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch import nn


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.distributed.algorithms.ddp_comm_hooks import (
    DDPCommHookType,
    register_ddp_comm_hook,
)
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


if TEST_WITH_DEV_DBG_ASAN:
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)
    sys.exit(0)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def gpus_for_rank(world_size):
    visible_devices = list(range(torch.accelerator.device_count()))
    gpus_per_process = torch.accelerator.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class Task(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.p = nn.Parameter(torch.randn(40, 20))

    def forward(self, x):
        return self.p * x


class TestDdpCommHook(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        return self.t0(x ** (1 + rank))


class DistributedDataParallelCommHookTest(DistributedTestBase):
    @property
    def world_size(self):
        return 2

    def _local_model(self):
        local_model = TestDdpCommHook().cpu()

        return local_model

    def _get_grads(self, process_group, hook_type=None):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            TestDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        # Register DDP Communication Hook if defined
        if hook_type is not None:
            register_ddp_comm_hook(
                comm_hook_type=hook_type, model=gpu_model, state=process_group
            )

        return self._run_and_get_grads(gpu_model)

    def _run_and_get_grads(self, model):
        torch.manual_seed(2020)
        input = torch.randn(40, 20)
        # Run forward
        output = model(input, self.rank)

        # Run backward
        output.mean().backward()

        # The only layer
        param = next(model.parameters())
        return param.grad

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook(self):
        """
        This unit test verifies the ``allreduce`` hook registered case gives same result
        with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.ALLREDUCE)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=0)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_fp16compress_hook(self):
        """
        This unit test verifies the ``fp16 compress`` hook registered case
        gives close result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.FP16_COMPRESS)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_tensor_hook(self):
        """
        This unit test verifies the ``quantize per tensor`` hook registered case
        gives close result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.QUANTIZE_PER_TENSOR)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_channel_hook(self):
        """
        This unit test verifies the ``quantize per channel`` hook registered case
        gives close result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(
            process_group, DDPCommHookType.QUANTIZE_PER_CHANNEL
        )

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_noop_hook(self):
        """
        This unit test verifies the ``noop`` hook registered case and a subsequent allreduce
        gives same result with no hook registered case.
        """
        process_group = self.create_pg(device_type)

        # No hook registered case, get the reference grads.
        reference_grads = self._get_grads(process_group, None)
        # Register hook case, get the hook grads.
        hook_grads = self._get_grads(process_group, DDPCommHookType.NOOP)
        # Apply a subsequent allreduce to average grads.
        hook_grads.div_(self.world_size)
        dist.all_reduce(hook_grads, group=process_group)

        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=0)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_is_last_hook(self):
        process_group = self.create_pg(device_type)

        def hook(flags, bucket):
            flags.append(bucket.is_last())
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        flags = []
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = nn.Sequential(
            nn.Linear(2, 4000, bias=False),
            *[nn.Linear(4000, 4000, bias=False) for _ in range(10)],
        )
        gpu_model = DistributedDataParallel(
            model.to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )
        gpu_model.register_comm_hook(state=flags, hook=hook)
        input = torch.randn(10, 2)
        gpu_model(input).sum().backward()
        self.assertTrue(flags[-1])
        self.assertFalse(any(flags[:-1]))


if __name__ == "__main__":
    assert not torch.cuda._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Task`, `TestDdpCommHook`, `DistributedDataParallelCommHookTest`

**Functions defined**: `gpus_for_rank`, `__init__`, `forward`, `__init__`, `forward`, `world_size`, `_local_model`, `_get_grads`, `_run_and_get_grads`, `test_ddp_comm_hook_allreduce_hook`, `test_ddp_comm_hook_fp16compress_hook`, `test_ddp_comm_hook_quantize_per_tensor_hook`, `test_ddp_comm_hook_quantize_per_channel_hook`, `test_ddp_comm_hook_noop_hook`, `test_is_last_hook`, `hook`

**Key imports**: sys, torch, torch.distributed as dist, nn, DistributedDataParallel, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/algorithms/ddp_comm_hooks`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed as dist`
- `torch.nn.parallel`: DistributedDataParallel
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/algorithms/ddp_comm_hooks`):



## Cross-References

- **File Documentation**: `test_ddp_hooks.py_docs.md`
- **Keyword Index**: `test_ddp_hooks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/algorithms/ddp_comm_hooks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/algorithms/ddp_comm_hooks`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python docs/test/distributed/algorithms/ddp_comm_hooks/test_ddp_hooks.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/algorithms/ddp_comm_hooks`):

- [`test_ddp_hooks.py_kw.md_docs.md`](./test_ddp_hooks.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_ddp_hooks.py_docs.md_docs.md`
- **Keyword Index**: `test_ddp_hooks.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

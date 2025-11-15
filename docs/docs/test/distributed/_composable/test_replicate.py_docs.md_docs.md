# Documentation: `docs/test/distributed/_composable/test_replicate.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_composable/test_replicate.py_docs.md`
- **Size**: 14,181 bytes (13.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_composable/test_replicate.py`

## File Metadata

- **Path**: `test/distributed/_composable/test_replicate.py`
- **Size**: 10,627 bytes (10.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import os
import unittest
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_XPU


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
device_module = torch.get_device_module(device_type)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class ReplicateStateDictTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def _check_state_dict_parity(self, sd_1, sd_2):
        for k1, k2 in zip(sd_1.keys(), sd_2.keys()):
            self.assertEqual(k1, k2)

        for v1, v2 in zip(sd_1.values(), sd_2.values()):
            self.assertEqual(v1, v2)

    def test_replicate_single_module_save_load(self):
        """
        Tests that replicate() on a single module state_dict
        matches local module state_dict.
        """
        self._init_pg()
        model = Net()
        replicate_model = replicate(deepcopy(model))
        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)

    def test_replicate_non_root_multiple_save_load(self):
        """
        Tests the replicate() on multiple submodules matches
        local module state_dict.
        """
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)

        local_sd = model.state_dict()
        ddp_sd = replicate_model.state_dict()
        self._check_state_dict_parity(local_sd, ddp_sd)


class ReplicateTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )

    def _compare_module(self, mod, replicate_mod):
        local_batch_size = 1
        global_batch_size = self.world_size * local_batch_size
        input = torch.randn(global_batch_size, 2)
        target = torch.randn(global_batch_size, 2)

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        for iteration in range(2):
            step_model(mod, input, target)
            step_model(
                replicate_mod,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            self.assertEqual(
                len(list(mod.parameters())),
                len(list(replicate_mod.parameters())),
            )
            for i, j in zip(mod.parameters(), replicate_mod.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(iteration)
            input = input[torch.randperm(global_batch_size)]

    def test_replicate_single_module(self):
        self._init_pg()
        model = Net()
        replicate_model = replicate(deepcopy(model))
        self._compare_module(model, replicate_model)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_move_args_kwargs_to_device(self):
        class MyNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(2, 2)

            def forward(self, inp, *, kwarg=None):
                if kwarg is not None:
                    inp = inp @ kwarg
                return self.a(inp)

        self._init_pg()
        torch.accelerator.set_device_index(self.rank)
        model = MyNet().to(device_type)
        replicate(model, device_id=torch.accelerator.current_device_index())
        # CPU input ensures replicate can move arg and kwargs to device.
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        model(a, kwarg=b).sum().backward()

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_ignore_module(self):
        self._init_pg()
        torch.accelerator.set_device_index(self.rank)
        # Seed ensures diff input and thus different local grads across ranks.
        torch.manual_seed(self.rank)
        device_module.manual_seed(self.rank)
        model = Net().to(device_type)
        replicate(model, ignored_modules=[model.fc1])
        # CPU input ensures that replicate can move input to GPU as DDP does.
        inp = torch.randn(5, 2, device=device_type) * (self.rank + 1)
        out = model(inp) * 10
        out.sum().backward()
        # FC1 grads should not be synchronized, FC2 and 3 should be.
        fc1_grad = model.fc1.weight.grad
        tensor_list = [torch.zeros_like(fc1_grad) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, fc1_grad)
        grad, rest = tensor_list[0], tensor_list[1:]
        for g in rest:
            self.assertNotEqual(grad, g)

        for dp_grad in [model.fc2.weight.grad, model.fc3.weight.grad]:
            tensor_list = [
                torch.zeros_like(dp_grad) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list, dp_grad)
            grad, rest = tensor_list[0], tensor_list[1:]
            for g in rest:
                self.assertEqual(grad, g)

    def test_replicate_multi_module(self):
        self._init_pg()
        model = Net()
        replicate_model = deepcopy(model)
        replicate(replicate_model.fc1)
        replicate(replicate_model.fc2)
        replicate(replicate_model.fc3)
        self._compare_module(model, replicate_model)

    def test_replicate_with_kwargs(self):
        self._init_pg()
        model = Net()
        replicate_model = replicate(
            deepcopy(model), bucket_cap_mb=1, gradient_as_bucket_view=True
        )
        self._compare_module(model, replicate_model)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_device_id(self):
        self._init_pg()
        model = Net()
        model_cuda = deepcopy(model).to(device_type)
        model_cuda2 = deepcopy(model_cuda)
        replicate(model, device_id=torch.device("cpu"))
        # DDP instance is attached in first pre forward
        model(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model)._ddp_weakref()
        # Should be None for CPU training
        self.assertEqual(None, replicate_ddp_weakref.device_ids)

        replicate(
            model_cuda, device_id=torch.device(torch.accelerator.current_device_index())
        )
        # DDP instance is attached in first pre forward
        model_cuda(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model_cuda)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)
        # Pass in int as device_id
        replicate(model_cuda2, device_id=int(torch.accelerator.current_device_index()))
        # DDP instance is attached in first pre forward
        model_cuda2(torch.randn(2, 2))
        replicate_ddp_weakref = replicate.state(model_cuda2)._ddp_weakref()
        self.assertEqual([0], replicate_ddp_weakref.device_ids)

    def test_replicate_wrong_device_id_type(self):
        self._init_pg()
        model = Net()
        with self.assertRaisesRegex(
            RuntimeError, "Expected device_id to be int or torch.device"
        ):
            replicate(model, device_id=[torch.device("cpu")])


class ReplicateFullyShardInit(ReplicateTest):
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    def test_replicate_fully_shard_init(self):
        class ToyModel(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.linears = nn.Sequential(
                    nn.Linear(dim, dim, bias=False),
                    nn.Linear(dim, dim, bias=False),
                    nn.Linear(dim, dim, bias=False),
                )
                self.proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor):
                y = self.linears(x)
                y = self.proj(y)
                return y

        self._init_pg()
        torch.accelerator.set_device_index(self.rank)
        dim = 3
        bz = 2
        model = ToyModel(dim).to(device_type)
        for linear in model.linears:
            fully_shard(linear)
        fully_shard(model.linears)
        replicate(model, device_id=torch.accelerator.current_device_index())
        for linear in model.linears:
            self.assertTrue(isinstance(linear.weight, DTensor))
        inp = torch.rand(bz, dim)
        # trigger lazy init
        model(inp).sum()
        for linear in model.linears:
            self.assertTrue(isinstance(linear.weight, DTensor))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 6 class(es) and 26 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Net`, `ReplicateStateDictTest`, `ReplicateTest`, `MyNet`, `ReplicateFullyShardInit`, `ToyModel`

**Functions defined**: `__init__`, `forward`, `setUp`, `tearDown`, `_init_pg`, `_check_state_dict_parity`, `test_replicate_single_module_save_load`, `test_replicate_non_root_multiple_save_load`, `world_size`, `setUp`, `tearDown`, `_init_pg`, `_compare_module`, `step_model`, `test_replicate_single_module`, `test_replicate_move_args_kwargs_to_device`, `__init__`, `forward`, `test_replicate_ignore_module`, `test_replicate_multi_module`

**Key imports**: os, unittest, deepcopy, torch, torch.distributed as dist, torch.nn.functional as F, nn, replicate, fully_shard, DTensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_composable`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest`
- `copy`: deepcopy
- `torch`
- `torch.distributed as dist`
- `torch.nn.functional as F`
- `torch.distributed._composable.replicate`: replicate
- `torch.distributed.fsdp`: fully_shard
- `torch.distributed.tensor`: DTensor
- `torch.testing._internal.common_utils`: run_tests, TEST_XPU


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/distributed/_composable/test_replicate.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_composable`):

- [`test_contract.py_docs.md`](./test_contract.py_docs.md)
- [`test_replicate_with_fsdp.py_docs.md`](./test_replicate_with_fsdp.py_docs.md)
- [`test_checkpoint.py_docs.md`](./test_checkpoint.py_docs.md)
- [`test_replicate_mixed_precision.py_docs.md`](./test_replicate_mixed_precision.py_docs.md)
- [`test_replicate_with_compiler.py_docs.md`](./test_replicate_with_compiler.py_docs.md)
- [`test_replicate_training.py_docs.md`](./test_replicate_training.py_docs.md)


## Cross-References

- **File Documentation**: `test_replicate.py_docs.md`
- **Keyword Index**: `test_replicate.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_composable`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_composable`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/test/distributed/_composable/test_replicate.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_composable`):

- [`test_checkpoint.py_kw.md_docs.md`](./test_checkpoint.py_kw.md_docs.md)
- [`test_replicate_with_compiler.py_docs.md_docs.md`](./test_replicate_with_compiler.py_docs.md_docs.md)
- [`test_replicate_with_fsdp.py_docs.md_docs.md`](./test_replicate_with_fsdp.py_docs.md_docs.md)
- [`test_replicate_mixed_precision.py_kw.md_docs.md`](./test_replicate_mixed_precision.py_kw.md_docs.md)
- [`test_replicate_training.py_kw.md_docs.md`](./test_replicate_training.py_kw.md_docs.md)
- [`test_replicate_training.py_docs.md_docs.md`](./test_replicate_training.py_docs.md_docs.md)
- [`test_checkpoint.py_docs.md_docs.md`](./test_checkpoint.py_docs.md_docs.md)
- [`test_contract.py_docs.md_docs.md`](./test_contract.py_docs.md_docs.md)
- [`test_contract.py_kw.md_docs.md`](./test_contract.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_replicate.py_docs.md_docs.md`
- **Keyword Index**: `test_replicate.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

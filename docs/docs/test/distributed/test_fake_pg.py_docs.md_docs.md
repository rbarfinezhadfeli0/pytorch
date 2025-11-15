# Documentation: `docs/test/distributed/test_fake_pg.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_fake_pg.py_docs.md`
- **Size**: 15,190 bytes (14.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_fake_pg.py`

## File Metadata

- **Path**: `test/distributed/test_fake_pg.py`
- **Size**: 10,940 bytes (10.68 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys
import unittest

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._C._distributed_c10d import FakeProcessGroup
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_distributed import HAS_ACCELERATOR
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import run_tests, skipIfHpu, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

device_type = get_devtype().type


class TestFakePG(TestCase):
    def tearDown(self):
        super().tearDown()
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass

    def test_all_reduce(self):
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_allgather(self):
        dist.init_process_group(backend="fake", rank=1, world_size=2)

        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)
        for out_tensor in output_tensors:
            self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_reduce_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)

        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        self.assertEqual(tuple(output_tensor.shape), (3, 3))

    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_construct_fsdp(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        FSDP(nn.Linear(2, 3, device=device_type))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_fake_e2e(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        my_module = nn.Sequential(
            nn.Linear(2, 3, device=device_type),
            nn.ReLU(),
            nn.Linear(3, 2, device=device_type),
        )
        sharded_module = FSDP(my_module, use_orig_params=True)
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        input = torch.randn(2, 2)
        x = sharded_module(input)
        loss = x.sum()
        loss.backward()
        optim.step()

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fake_pg_tracing(self):
        store = dist.HashStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            return funcol.all_gather_tensor(tensor, 0, default_pg)

        gm = make_fx(allgather_fn)(torch.randn(2, 2, device=device_type))
        FileCheck().check("all_gather").check("wait_tensor").run(str(gm.graph))

    def test_broadcast(self):
        dist.init_process_group(backend="fake", rank=0, world_size=2)

        # src == rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=0)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.broadcast(output, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # src == rank
        output = torch.ones(3, 3)
        to_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        dist.scatter(output, to_scatter)
        self.assertEqual(tuple(output.shape), (3, 3))

        # src != rank
        output = torch.ones(3, 3)
        dist.scatter(output, None, src=1)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output_list = [torch.ones(3, 3) for _ in range(2)]
        input_list = [torch.ones(3, 3) for _ in range(2)]
        dist.all_to_all(output_list, input_list)
        self.assertEqual(len(output_list), 2)
        for output in output_list:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall_base(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        out_tensor = torch.ones(3, 3)
        in_tensor = torch.ones(3, 3)
        output_split = [1, 1]
        input_split = [1, 1]
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        self.assertEqual(tuple(out_tensor.shape), (3, 3))

    def test_send(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        tensor = torch.ones(3, 3)
        dist.send(tensor, 1)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    def test_recv(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        output = torch.ones(3, 3)
        dist.recv(output, 1)
        self.assertEqual(tuple(output.shape), (3, 3))

    @skipIfHpu
    @unittest.skipIf(not HAS_ACCELERATOR, "No accelerator")
    def test_fsdp_tp_fake_e2e(self):
        world_size = 4
        tp_size = 2

        store = dist.HashStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        device_mesh = DeviceMesh(
            device_type, torch.arange(0, world_size).view(-1, tp_size)
        )
        device_mesh = init_device_mesh(
            device_type, (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
        )

        sequence_parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0)),
            "net2": RowwiseParallel(output_layouts=Shard(0)),
        }
        pairwise_parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        for parallel_plan in [sequence_parallelize_plan, pairwise_parallelize_plan]:
            my_module = parallelize_module(
                MLPModule(device=device_type),
                device_mesh["tp"],
                parallel_plan,
            )

            sharded_module = FSDP(
                my_module, use_orig_params=True, device_mesh=device_mesh["dp"]
            )
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

            for i in range(10):
                dp_rank = dist.get_rank()
                torch.manual_seed(i + dp_rank)
                input = torch.randn(20, 10, device=f"{device_type}:{dp_rank}")
                x = sharded_module(input)
                loss = x.sum()
                loss.backward()
                optim.step()

    def test_error_on_collective(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        # Test with error_on_collective=False (default behavior)
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # These should work normally
        tensor = torch.ones(3, 3)
        dist.all_reduce(tensor)
        self.assertEqual(tuple(tensor.shape), (3, 3))

        dist.destroy_process_group()

        # Test with error_on_collective=True
        from torch._C._distributed_c10d import FakeProcessGroup

        options = FakeProcessGroup.Options()
        options.error_on_collective = True

        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=2, store=store, pg_options=options
        )

        # These should now raise errors
        tensor = torch.ones(3, 3)
        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.all_reduce(tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            output_tensors = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(output_tensors, tensor)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.broadcast(tensor, src=0)

        with self.assertRaisesRegex(
            RuntimeError, "FakeProcessGroup collective operation error"
        ):
            dist.barrier()

    def test_fake_process_group_direct_usage_error(self):
        class SimpleTensorMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with self.assertRaisesRegex(TypeError, r"No constructor defined"):
            fake_pg = FakeProcessGroup(rank=0, world_size=3)

            with SimpleTensorMode():
                tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                dist.all_reduce(tensor, group=fake_pg)

    def test_fake_process_group_proper_usage_dispatch(self):
        class SimpleTensorMode(TorchDispatchMode):
            def __init__(self):
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.ops.append(str(func))
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=3)

        with SimpleTensorMode() as mode:
            tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            dist.all_reduce(tensor)

        op_names = [str(op) for op in mode.ops]
        self.assertIn("aten.lift_fresh.default", op_names)
        self.assertIn("c10d.allreduce_.default", op_names)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 21 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFakePG`, `SimpleTensorMode`, `SimpleTensorMode`

**Functions defined**: `tearDown`, `test_all_reduce`, `test_allgather`, `test_reduce_scatter`, `test_construct_fsdp`, `test_fsdp_fake_e2e`, `test_fake_pg_tracing`, `allgather_fn`, `test_broadcast`, `test_scatter`, `test_alltoall`, `test_alltoall_base`, `test_send`, `test_recv`, `test_fsdp_tp_fake_e2e`, `test_error_on_collective`, `test_fake_process_group_direct_usage_error`, `__torch_dispatch__`, `test_fake_process_group_proper_usage_dispatch`, `__init__`

**Key imports**: sys, unittest, torch, torch.distributed as dist, torch.distributed._functional_collectives as funcol, torch.nn as nn, FakeProcessGroup, init_device_mesh, FullyShardedDataParallel as FSDP, DeviceMesh, Shard


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `torch`
- `torch.distributed as dist`
- `torch.distributed._functional_collectives as funcol`
- `torch.nn as nn`
- `torch._C._distributed_c10d`: FakeProcessGroup
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.tensor`: DeviceMesh, Shard
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.testing`: FileCheck
- `torch.testing._internal.common_distributed`: HAS_ACCELERATOR
- `torch.testing._internal.common_fsdp`: get_devtype
- `torch.testing._internal.common_utils`: run_tests, skipIfHpu, TestCase
- `torch.testing._internal.distributed._tensor.common_dtensor`: MLPModule
- `torch.testing._internal.distributed.fake_pg`: FakeStore
- `torch.utils._python_dispatch`: TorchDispatchMode


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
python test/distributed/test_fake_pg.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_c10d_logger.py_docs.md`](./test_c10d_logger.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_fake_pg.py_docs.md`
- **Keyword Index**: `test_fake_pg.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/test_fake_pg.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`test_inductor_collectives.py_docs.md_docs.md`](./test_inductor_collectives.py_docs.md_docs.md)
- [`test_control_collectives.py_kw.md_docs.md`](./test_control_collectives.py_kw.md_docs.md)
- [`test_c10d_gloo.py_docs.md_docs.md`](./test_c10d_gloo.py_docs.md_docs.md)
- [`test_collective_utils.py_kw.md_docs.md`](./test_collective_utils.py_kw.md_docs.md)
- [`test_data_parallel.py_kw.md_docs.md`](./test_data_parallel.py_kw.md_docs.md)
- [`test_overlap_bucketing_unit.py_kw.md_docs.md`](./test_overlap_bucketing_unit.py_kw.md_docs.md)
- [`test_c10d_nccl.py_kw.md_docs.md`](./test_c10d_nccl.py_kw.md_docs.md)
- [`test_multi_threaded_pg.py_docs.md_docs.md`](./test_multi_threaded_pg.py_docs.md_docs.md)
- [`argparse_util_test.py_kw.md_docs.md`](./argparse_util_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fake_pg.py_docs.md_docs.md`
- **Keyword Index**: `test_fake_pg.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

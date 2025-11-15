# Documentation: `docs/torch/testing/_internal/distributed/_shard/sharded_tensor/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/distributed/_shard/sharded_tensor/__init__.py_docs.md`
- **Size**: 5,704 bytes (5.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/distributed/_shard/sharded_tensor/__init__.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/_shard/sharded_tensor/__init__.py`
- **Size**: 3,212 bytes (3.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    tp_transports,
)


TEST_GPU_NUM = 4


class ShardedTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self):
        return TEST_GPU_NUM

    def init_pg(self, backend="nccl"):
        if backend not in ["nccl", "gloo", "mpi", "hccl"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    def init_rpc(self):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            _transports=tp_transports()
        )
        rpc_backend_options.init_method = f"file://{self.file_name}"
        for rank in range(self.world_size):
            rpc_backend_options.set_device_map(
                f"worker{rank}", {rank: self.rank, self.rank: rank}
            )

        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

    def init_comms(self, init_rpc=True, backend="nccl"):
        if init_rpc:
            self.init_rpc()
        self.init_pg(backend=backend)

    def destroy_comms(self, destroy_rpc=True):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()

        if destroy_rpc:
            rpc.shutdown()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def assert_sharded_tensor_equal(self, st1, st2):
        st1_local_shards = st1.local_shards()
        st2_local_shards = st2.local_shards()
        self.assertEqual(len(st1_local_shards), len(st2_local_shards))
        for i, st1_local_shard in enumerate(st1_local_shards):
            self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
            self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)

        self.assertEqual(st1.metadata(), st2.metadata())
        self.assertEqual(st1.sharding_spec(), st2.sharding_spec())
        self.assertEqual(len(st1.remote_shards()), len(st2.remote_shards()))


# wrapper to initialize comms (processgroup + rpc)
def with_comms(func=None, init_rpc=True, backend="nccl"):
    if func is None:
        return partial(
            with_comms,
            init_rpc=init_rpc,
            backend=backend,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.init_comms(init_rpc=init_rpc, backend=backend)
        func(self, *args, **kwargs)
        self.destroy_comms(destroy_rpc=init_rpc)

    return wrapper

```



## High-Level Overview


This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ShardedTensorTestBase`

**Functions defined**: `world_size`, `init_pg`, `init_rpc`, `init_comms`, `destroy_comms`, `setUp`, `assert_sharded_tensor_equal`, `with_comms`, `wrapper`

**Key imports**: sys, partial, wraps, torch, torch.distributed as dist, rpc


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `functools`: partial, wraps
- `torch`
- `torch.distributed as dist`
- `torch.distributed`: rpc


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

This is a test file. Run it with:

```bash
python torch/testing/_internal/distributed/_shard/sharded_tensor/__init__.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/distributed/_shard/sharded_tensor`):

- [`_test_st_common.py_docs.md`](./_test_st_common.py_docs.md)
- [`_test_ops_common.py_docs.md`](./_test_ops_common.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal/distributed/_shard/sharded_tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python docs/torch/testing/_internal/distributed/_shard/sharded_tensor/__init__.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal/distributed/_shard/sharded_tensor`):

- [`_test_st_common.py_kw.md_docs.md`](./_test_st_common.py_kw.md_docs.md)
- [`_test_ops_common.py_kw.md_docs.md`](./_test_ops_common.py_kw.md_docs.md)
- [`_test_ops_common.py_docs.md_docs.md`](./_test_ops_common.py_docs.md_docs.md)
- [`_test_st_common.py_docs.md_docs.md`](./_test_st_common.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `test/distributed/rpc/test_share_memory.py`

## File Metadata

- **Path**: `test/distributed/rpc/test_share_memory.py`
- **Size**: 2,351 bytes (2.30 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import contextlib
import copyreg
import os
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.rpc as rpc
import torch.multiprocessing.reductions as TorchMpReductions
from torch import multiprocessing
from torch.distributed.rpc.api import _use_rpc_pickler
from torch.distributed.rpc.internal import _InternalRPCPickler
from torch.testing._internal.common_utils import run_tests, TestCase


@contextlib.contextmanager
def fs_sharing():
    prev_strategy = multiprocessing.get_sharing_strategy()
    multiprocessing.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        multiprocessing.set_sharing_strategy(prev_strategy)


class ShareMemoryRPCPickler(_InternalRPCPickler):
    def __init__(self) -> None:
        super().__init__()
        self._dispatch_table
        # pyre-fixme[4]: Attribute must be annotated.
        self._dispatch_table = copyreg.dispatch_table.copy()

        for t in torch._storage_classes:
            self._dispatch_table[t] = TorchMpReductions.reduce_storage

        for t in torch._tensor_classes:
            self._dispatch_table[t] = TorchMpReductions.reduce_tensor
        self._dispatch_table[torch.Tensor] = TorchMpReductions.reduce_tensor
        self._dispatch_table[torch.nn.parameter.Parameter] = (
            TorchMpReductions.reduce_tensor
        )


def worker_loop(a):
    rpc.init_rpc("worker1", rank=1, world_size=2)
    rpc.shutdown()


def worker_fn(m):
    pass


class TestRPCPickler(TestCase):
    def test_case(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        with fs_sharing():
            r = multiprocessing.spawn(worker_loop, join=False)

            try:
                with _use_rpc_pickler(ShareMemoryRPCPickler()):
                    rpc.init_rpc("worker0", rank=0, world_size=2)
                    m = torch.nn.Linear(1, 2)
                    m.share_memory()
                    rref = rpc.remote("worker1", worker_fn, args=(m,))

                    rref.to_here()
            finally:
                rpc.shutdown()
                r.join()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ShareMemoryRPCPickler`, `TestRPCPickler`

**Functions defined**: `fs_sharing`, `__init__`, `worker_loop`, `worker_fn`, `test_case`

**Key imports**: contextlib, copyreg, os, sys, torch, torch.distributed as dist, torch.distributed.rpc as rpc, torch.multiprocessing.reductions as TorchMpReductions, multiprocessing, _use_rpc_pickler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/rpc`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copyreg`
- `os`
- `sys`
- `torch`
- `torch.distributed as dist`
- `torch.distributed.rpc as rpc`
- `torch.multiprocessing.reductions as TorchMpReductions`
- `torch.distributed.rpc.api`: _use_rpc_pickler
- `torch.distributed.rpc.internal`: _InternalRPCPickler
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/rpc/test_share_memory.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/rpc`):

- [`test_faulty_agent.py_docs.md`](./test_faulty_agent.py_docs.md)
- [`test_tensorpipe_agent.py_docs.md`](./test_tensorpipe_agent.py_docs.md)


## Cross-References

- **File Documentation**: `test_share_memory.py_docs.md`
- **Keyword Index**: `test_share_memory.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/distributed/checkpoint/examples/stateful_example.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/examples/stateful_example.py`
- **Size**: 2,817 bytes (2.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

# pyre-unsafe


import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


def _make_stateful(model, optim):
    _patch_model_state_dict(model)
    _patch_optimizer_state_dict(model, optimizers=optim)


def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


def _init_model(device, world_size):
    device_mesh = init_device_mesh(device, (world_size,))
    model = Model().cuda()
    model = FSDP(
        model,
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    _make_stateful(model, optim)

    return model, optim


def run(rank, world_size, device="cuda"):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model, optim = _init_model(device, world_size)
    _train(model, optim, train_steps=2)

    dcp.save(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )

    # presumably do something else
    model, optim = _init_model(device, world_size)
    dcp.load(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )
    _train(model, optim, train_steps=2)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running stateful checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Model`

**Functions defined**: `__init__`, `forward`, `get_input`, `_make_stateful`, `_train`, `_init_model`, `run`

**Key imports**: os, shutil, torch, torch.distributed as dist, torch.distributed.checkpoint as dcp, torch.multiprocessing as mp, torch.nn as nn, init_device_mesh, FullyShardedDataParallel as FSDP


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint/examples`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dcp`
- `torch.multiprocessing as mp`
- `torch.nn as nn`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/checkpoint/examples`):

- [`fsdp_checkpoint_example.py_docs.md`](./fsdp_checkpoint_example.py_docs.md)
- [`async_checkpointing_example.py_docs.md`](./async_checkpointing_example.py_docs.md)


## Cross-References

- **File Documentation**: `stateful_example.py_docs.md`
- **Keyword Index**: `stateful_example.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

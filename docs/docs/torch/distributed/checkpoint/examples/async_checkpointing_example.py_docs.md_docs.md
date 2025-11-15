# Documentation: `docs/torch/distributed/checkpoint/examples/async_checkpointing_example.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/examples/async_checkpointing_example.py_docs.md`
- **Size**: 6,829 bytes (6.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/examples/async_checkpointing_example.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/examples/async_checkpointing_example.py`
- **Size**: 3,970 bytes (3.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

import os
import shutil
import traceback
from concurrent.futures import Future

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.device_mesh import init_device_mesh


DEVICE = "cuda"
NUM_EPOCHS = 1000
SAVE_PERIOD = 10
FAULT_PERIOD = 25
CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"


class InjectedException(Exception):
    pass


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Linear(8, 32)
        self.net2 = nn.Linear(32, 128)
        self.net3 = nn.Linear(128, 64)
        self.net4 = nn.Linear(64, 8)
        self.net5 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        x = F.sigmoid(self.net5(x))
        return x


def _init_model(rank, world_size):
    device_mesh = init_device_mesh(DEVICE, (world_size,))

    # Create a dummy model and wrap it in FSDP
    model = Model().cuda()
    device_mesh = init_device_mesh(DEVICE, (world_size,))
    model = FSDP(model, device_mesh=device_mesh, use_orig_params=True)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    _patch_model_state_dict(model)
    # pyrefly: ignore [bad-argument-type]
    _patch_optimizer_state_dict(model, optimizers=optim)

    return model, optim


def _print(msg):
    if dist.get_rank() == 0:
        print(msg)


def _input():
    x = torch.rand(128, 8, device="cuda")
    y = torch.zeros(128, 1, device="cuda")

    y[torch.sum(x, dim=1) >= 4] = 1.0

    return x, y


def run(rank, world_size):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model, optim = _init_model(rank, world_size)
    state_dict = {"model": model, "optim": optim}
    loss_calc = torch.nn.BCELoss()

    f = None
    # pyrefly: ignore [bad-assignment]
    for epoch in range(NUM_EPOCHS):
        try:
            torch.manual_seed(epoch)
            x, y = _input()

            loss = loss_calc(model(x), y)

            _print(f"{epoch=} {loss=}")

            loss.backward()
            optim.step()
            optim.zero_grad()

            if epoch % SAVE_PERIOD == 0:
                if f is not None:
                    if not isinstance(f, Future):
                        raise AssertionError("f should be a Future instance")
                    f.result()
                f = dcp.state_dict_saver.async_save(
                    state_dict, checkpoint_id=CHECKPOINT_DIR
                )

            if FAULT_PERIOD > 0 and epoch % FAULT_PERIOD == 0:
                raise InjectedException("Fault injection!")

        except InjectedException as e:
            dist.barrier()

            _print("Trainer encountered exception:")
            traceback.print_tb(e.__traceback__)

            _print("Reloading model from last checkpoint!")
            if f is not None:
                if not isinstance(f, Future):
                    raise AssertionError("f should be a Future instance") from None
                f.result()
            dcp.load(state_dict)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running an example of Async Checkpointing on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `InjectedException`, `Model`

**Functions defined**: `__init__`, `forward`, `_init_model`, `_print`, `_input`, `run`

**Key imports**: os, shutil, traceback, Future, torch, torch.distributed as dist, torch.distributed.checkpoint as dcp, torch.multiprocessing as mp, torch.nn as nn, torch.nn.functional as F


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint/examples`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `traceback`
- `concurrent.futures`: Future
- `torch`
- `torch.distributed as dist`
- `torch.distributed.checkpoint as dcp`
- `torch.multiprocessing as mp`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.tensor.device_mesh`: init_device_mesh


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/checkpoint/examples`):

- [`fsdp_checkpoint_example.py_docs.md`](./fsdp_checkpoint_example.py_docs.md)
- [`stateful_example.py_docs.md`](./stateful_example.py_docs.md)


## Cross-References

- **File Documentation**: `async_checkpointing_example.py_docs.md`
- **Keyword Index**: `async_checkpointing_example.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/checkpoint/examples`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/checkpoint/examples`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/checkpoint/examples`):

- [`fsdp_checkpoint_example.py_docs.md_docs.md`](./fsdp_checkpoint_example.py_docs.md_docs.md)
- [`stateful_example.py_docs.md_docs.md`](./stateful_example.py_docs.md_docs.md)
- [`fsdp_checkpoint_example.py_kw.md_docs.md`](./fsdp_checkpoint_example.py_kw.md_docs.md)
- [`async_checkpointing_example.py_kw.md_docs.md`](./async_checkpointing_example.py_kw.md_docs.md)
- [`stateful_example.py_kw.md_docs.md`](./stateful_example.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `async_checkpointing_example.py_docs.md_docs.md`
- **Keyword Index**: `async_checkpointing_example.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/examples/fsdp_checkpoint_example.py`
- **Size**: 4,262 bytes (4.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

"""
The following example demonstrates how to use Pytorch Distributed Checkpoint to save a FSDP model.

This is the current recommended way to checkpoint FSDP.
torch.save() and torch.load() is not recommended when checkpointing sharded models.
"""

import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


CHECKPOINT_DIR = f"/scratch/{os.environ.get('LOGNAME', '')}/checkpoint"


def opt_at(opt, idx):
    return list(opt.state.values())[idx]


def init_model():
    model = FSDP(torch.nn.Linear(4, 4).cuda(dist.get_rank()))
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.rand(4, 4)).sum().backward()
    optim.step()

    return model, optim


def print_params(stage, model_1, model_2, optim_1, optim_2):
    with FSDP.summon_full_params(model_1), FSDP.summon_full_params(model_2):
        print(
            f"{stage} --- rank: {dist.get_rank()}\n"
            f"model.weight: {model_1.weight}\n"
            f"model_2.weight:{model_2.weight}\n"
            f"model.bias: {model_1.bias}\n"
            f"model_2.bias: {model_2.bias}\n"
        )

    print(
        f"{stage} --- rank: {dist.get_rank()}\n"
        f"optim exp_avg:{opt_at(optim_1, 0)['exp_avg']}\n"
        f"optim_2 exp_avg:{opt_at(optim_2, 0)['exp_avg']}\n"
        f"optim exp_avg_sq:{opt_at(optim_1, 0)['exp_avg_sq']}\n"
        f"optim_2 exp_avg_sq:{opt_at(optim_2, 0)['exp_avg_sq']}\n"
    )


def run_fsdp_checkpoint_example(rank, world_size):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create a model
    model_1, optim_1 = init_model()

    # Save the model to CHECKPOINT_DIR
    with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_1.state_dict(),
            "optim": FSDP.optim_state_dict(model_1, optim_1),
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        )

    # Create a second model
    model_2, optim_2 = init_model()

    # Print the model parameters for both models.
    # Before loading, the parameters should be different.
    print_params("Before loading", model_1, model_2, optim_1, optim_2)

    # Load model_2 with parameters saved in CHECKPOINT_DIR
    with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model_2.state_dict(),
            # cannot load the optimizer state_dict together with the model state_dict
        }

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )
        model_2.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        optim_2.load_state_dict(flattened_osd)

    # Print the model parameters for both models.
    # After loading, the parameters should be the same.
    print_params("After loading", model_1, model_2, optim_1, optim_2)

    # Shut down world pg
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(
        run_fsdp_checkpoint_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

```



## High-Level Overview

"""The following example demonstrates how to use Pytorch Distributed Checkpoint to save a FSDP model.This is the current recommended way to checkpoint FSDP.torch.save() and torch.load() is not recommended when checkpointing sharded models.

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `opt_at`, `init_model`, `print_params`, `run_fsdp_checkpoint_example`

**Key imports**: os, shutil, torch, torch.distributed as dist, torch.distributed.checkpoint as dist_cp, torch.multiprocessing as mp, load_sharded_optimizer_state_dict, FullyShardedDataParallel as FSDP, StateDictType


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
- `torch.distributed.checkpoint as dist_cp`
- `torch.multiprocessing as mp`
- `torch.distributed.checkpoint.optimizer`: load_sharded_optimizer_state_dict
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.fsdp.fully_sharded_data_parallel`: StateDictType


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/checkpoint/examples`):

- [`stateful_example.py_docs.md`](./stateful_example.py_docs.md)
- [`async_checkpointing_example.py_docs.md`](./async_checkpointing_example.py_docs.md)


## Cross-References

- **File Documentation**: `fsdp_checkpoint_example.py_docs.md`
- **Keyword Index**: `fsdp_checkpoint_example.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `benchmarks/dynamo/dist_util.py`

## File Metadata

- **Path**: `benchmarks/dynamo/dist_util.py`
- **Size**: 3,907 bytes (3.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import argparse
import functools
import importlib
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo.testing import reduce_to_scalar_loss
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


try:
    from .torchbench import setup_torchbench_cwd
except ImportError:
    from torchbench import setup_torchbench_cwd


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    os.environ["RANK"] = os.getenv("RANK", "0")
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", "1")
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


class CustomLinear(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(a, b))

    def forward(self, x):
        return torch.mm(x, self.weight)


class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(10, 10000), nn.ReLU()]
            + [nn.Linear(10000, 10000), nn.ReLU()]
            + [MyModule(10000, 10000)]
            + [MyModule(10000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [MyModule(1000, 1000)]
            + [nn.Linear(1000, 5)]
        )

    def forward(self, x):
        return self.net(x)


def model_iter_fn(model, example_inputs, collect_outputs=False):
    outputs = model(*example_inputs)
    loss = reduce_to_scalar_loss(outputs)
    loss.backward()
    if collect_outputs:
        return outputs


def get_model(args):
    if args.torchbench_model:
        setup_torchbench_cwd()
        module = importlib.import_module(
            f"torchbenchmark.models.{args.torchbench_model}"
        )
        benchmark_cls = getattr(module, "Model", None)
        bm = benchmark_cls(test="train", device=args.device, batch_size=args.batch_size)
        model, inputs = bm.get_module()
    elif args.toy_model:
        model = ToyModel()
        inputs = (torch.randn(20, 10),)
    else:
        raise argparse.ArgumentError(
            args.torchbench_model, message="Must specify a model"
        )

    return model, inputs


def fsdp_checkpointing_base(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule):
        return isinstance(submodule, blocks)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


MODEL_FSDP_WRAP = {
    "toy_model": (MyModule,),
}


def apply_fsdp(args, model, use_checkpointing=False, use_wrap_policy=True):
    wrap_policy = None
    blocks = MODEL_FSDP_WRAP[
        "toy_model" if model.__class__ is ToyModel else args.torchbench_model
    ]
    if use_wrap_policy:
        wrap_policy = ModuleWrapPolicy(blocks)

    model = FSDP(model, auto_wrap_policy=wrap_policy, use_orig_params=True)
    if use_checkpointing:
        fsdp_checkpointing_base(model, blocks)
    return model

```



## High-Level Overview


This Python file contains 3 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CustomLinear`, `MyModule`, `ToyModel`

**Functions defined**: `setup`, `cleanup`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `model_iter_fn`, `get_model`, `fsdp_checkpointing_base`, `check_fn`, `apply_fsdp`

**Key imports**: argparse, functools, importlib, os, torch, torch.distributed as dist, torch.nn as nn, reduce_to_scalar_loss, FullyShardedDataParallel as FSDP, ModuleWrapPolicy


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `functools`
- `importlib`
- `os`
- `torch`
- `torch.distributed as dist`
- `torch.nn as nn`
- `torch._dynamo.testing`: reduce_to_scalar_loss
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.distributed.fsdp.wrap`: ModuleWrapPolicy
- `.torchbench`: setup_torchbench_cwd
- `torchbench`: setup_torchbench_cwd


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

Files in the same folder (`benchmarks/dynamo`):

- [`timm_models_list_cpu.txt_docs.md`](./timm_models_list_cpu.txt_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`benchmarks.py_docs.md`](./benchmarks.py_docs.md)
- [`check_graph_breaks.py_docs.md`](./check_graph_breaks.py_docs.md)
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `dist_util.py_docs.md`
- **Keyword Index**: `dist_util.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

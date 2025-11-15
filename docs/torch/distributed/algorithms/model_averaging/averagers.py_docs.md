# Documentation: `torch/distributed/algorithms/model_averaging/averagers.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/model_averaging/averagers.py`
- **Size**: 5,486 bytes (5.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.utils as utils
from torch.utils._typing_utils import not_none as _not_none


__all__ = ["ModelAverager", "PeriodicModelAverager"]


class ModelAverager(ABC):
    r"""Base class for all model averagers.

    Args:
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None):
        self.process_group = (
            process_group if process_group is not None else _not_none(dist.group.WORLD)
        )
        self.step = 0

    @abstractmethod
    def average_parameters(self, params):
        raise NotImplementedError


class PeriodicModelAverager(ModelAverager):
    r"""
    Averages parameters periodically after the warm-up stage.

    This can be used for running `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    by running :class:`~torch.nn.DistributedDataParallel` (DDP)
    using the subgroups created by :meth:`~torch.distributed.new_subgroups`.

    Args:
        period (int): The number of steps per model averaging.
                      Usually the period should be greater than ``1`` to reduce the communication cost.
                      Otherwise, only DDP needs to be used.
        warmup_steps (int): The number of warm-up steps. During this stage,
                            model averaging is skipped.
        process_group: The process group to be used for all-reduce.
                       If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD
        >>> import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>> import torch.nn as nn
        >>>
        >>> dist.init_process_group("nccl", rank=rank, world_size=16)
        >>> torch.cuda.set_device(rank)
        >>> module = nn.Linear(1, 1, bias=False).cuda()
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>> # Register a post-localSGD communication hook.
        >>> state = PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # In the first 100 steps, run global gradient averaging like normal DDP at every step.
        >>> # After 100 steps, run model averaging every 4 steps.
        >>> # Note that ``warmup_steps`` must be the same as ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> averager = averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>> for step in range(0, 200):
        >>>    optimizer.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    optimizer.step()
        >>>    # Will average model parameters globally every 4 steps. Thus,
        >>>    # inter-node communication only occurs every 4 iterations after
        >>>    # the initial ``warmup_steps`` period.
        >>>    averager.average_parameters(model.parameters())
    """

    def __init__(
        self, period, warmup_steps=0, process_group: Optional[dist.ProcessGroup] = None
    ):
        super().__init__(process_group)
        if warmup_steps < 0:
            raise ValueError("Arg ``warmup_steps`` must be a non-negative number.")
        self.warmup_steps = warmup_steps
        if period < 1:
            raise ValueError("Arg ``period`` must be a positive value.")
        elif period == 1:
            warnings.warn(
                "When period is 1, no need to use model averaging because the communication cost "
                "of all-reducing parameters will be no less than the cost of all-reducing gradients "
                "by DistributedDataParallel in the backward pass. Therefore, only "
                "DistributedDataParallel should be used for this case.",
                stacklevel=2,
            )
        self.period = period

    def average_parameters(
        self,
        params: Union[
            Iterable[torch.nn.Parameter], Iterable[dict[str, torch.nn.Parameter]]
        ],
    ):
        """
        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``.

        Can be divided by ``period``, where ``step`` is increased by 1
        at each iteration in the training loop.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.

        """
        if (
            self.step >= self.warmup_steps
            and (self.step - self.warmup_steps) % self.period == 0
        ):
            utils.average_parameters_or_parameter_groups(
                params, _not_none(self.process_group)
            )
        self.step += 1

```



## High-Level Overview

r"""Base class for all model averagers.    Args:        process_group: The process group to be used for all-reduce.                       If ``None``, the default process group, which                       is created by :func:`torch.distributed.init_process_group`,                       will be used. (default: ``None``)

This Python file contains 3 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ModelAverager`, `PeriodicModelAverager`

**Functions defined**: `__init__`, `average_parameters`, `__init__`, `average_parameters`

**Key imports**: warnings, ABC, abstractmethod, Iterable, Optional, Union, torch, torch.distributed as dist, torch.distributed.algorithms.model_averaging.utils as utils, not_none as _not_none, torch, torch.distributed as dist


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/model_averaging`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `abc`: ABC, abstractmethod
- `collections.abc`: Iterable
- `typing`: Optional, Union
- `torch`
- `torch.distributed as dist`
- `torch.distributed.algorithms.model_averaging.utils as utils`
- `torch.utils._typing_utils`: not_none as _not_none
- `torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD`
- `torch.distributed.algorithms.model_averaging.averagers as averagers`
- `torch.nn as nn`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`torch/distributed/algorithms/model_averaging`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`hierarchical_model_averager.py_docs.md`](./hierarchical_model_averager.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)


## Cross-References

- **File Documentation**: `averagers.py_docs.md`
- **Keyword Index**: `averagers.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

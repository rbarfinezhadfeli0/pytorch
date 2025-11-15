# Documentation: `torch/distributed/algorithms/ddp_comm_hooks/post_localSGD_hook.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/ddp_comm_hooks/post_localSGD_hook.py`
- **Size**: 5,150 bytes (5.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging

import torch
import torch.distributed as dist

from . import default_hooks as default


logger = logging.getLogger(__name__)


class PostLocalSGDState:
    r"""
    Store state for all-reducing gradients globally until given step, then locally after.

    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.

    If ``process_group`` is ``None``, the global process group will be used.
    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.

    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,
    because both true and false may give a faster convergence.
    """

    __slots__ = [
        "process_group",
        "subgroup",
        "start_localSGD_iter",
        "post_local_gradient_allreduce",
        "iter",
    ]

    def __init__(
        self,
        process_group,
        subgroup,
        start_localSGD_iter,
        post_local_gradient_allreduce=True,
    ):
        """Initialize state object with given parameters and log when localSGD start."""
        logger.info(
            "Local SGD will be started after %s iterations", start_localSGD_iter
        )

        # The group used for all-reducing gradients globally.
        self.process_group = process_group
        # The group used for all-reducing gradients locally.
        self.subgroup = subgroup
        self.start_localSGD_iter = start_localSGD_iter
        # Allreduce gradients locally since iteration `start_localSGD_iter`.
        # This may help with the convergence efficiency at the cost of relatively cheap intra-subgroup communication.
        self.post_local_gradient_allreduce = post_local_gradient_allreduce
        # Iteration/step in the training loop.
        self.iter = 0

    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last():
            self.iter += 1

        if self.iter == self.start_localSGD_iter:
            logger.info("Start to apply local SGD after %s iterations.", self.iter)


def post_localSGD_hook(
    state: PostLocalSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Run post-localSGD algorithm.

    This DDP communication hook is used for running post-localSGD algorithm,
    by combining with a model averaging component (e.g.,
    :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`)
    that runs after the optimizer step.

    Args:
        state (PostLocalSGDState): State information to run post-localSGD.
            Users mainly need to tune ``start_localSGD_iter`` to determine when to start local SGD.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PostLocalSGDState(process_group=process_group, subgroup=subgroup,
                                  start_localSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, post_localSGD_hook)
        >>> # Also need to establish a model averaging module and run model averaging after ``optimizer.step()``.
        >>> # Please refer to the examples in ``torch.distributed.algorithms.model_averaging.averagers`` module.
    """
    global_group_to_use = (
        state.process_group if state.process_group is not None else dist.group.WORLD
    )

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run allreduce using `global_group_to_use` in the first `start_localSGD_iter` iterations.
    if state.iter < state.start_localSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(global_group_to_use, input_tensor)  # type: ignore[arg-type]

    # If `post_local_gradient_allreduce` is not set,
    # then no gradient synchronization after the first `start_localSGD_iter` iterations.
    if not state.post_local_gradient_allreduce:
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Run allreduce using `subgroup` after the first `start_localSGD_iter` iterations.
    # Note that by default, a separate subgroup for each node is created which
    # causes an intra-node allreduce to be done at each training step.
    # From this moment, model averaging should run after the optimizer step,
    # to globally allreduce all the parameters.
    if state.subgroup is None:
        state.subgroup, _ = dist.new_subgroups()
    return default._allreduce_fut(state.subgroup, input_tensor)

```



## High-Level Overview

r"""    Store state for all-reducing gradients globally until given step, then locally after.    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,    and all-reducing gradients locally using ``subgroup`` afterwards.    If ``process_group`` is ``None``, the global process group will be used.    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,    because both true and false may give a faster convergence.

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PostLocalSGDState`

**Functions defined**: `__init__`, `maybe_increase_iter`, `post_localSGD_hook`

**Key imports**: logging, torch, torch.distributed as dist, default_hooks as default


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/ddp_comm_hooks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `torch`
- `torch.distributed as dist`
- `.`: default_hooks as default


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/distributed/algorithms/ddp_comm_hooks`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`debugging_hooks.py_docs.md`](./debugging_hooks.py_docs.md)
- [`powerSGD_hook.py_docs.md`](./powerSGD_hook.py_docs.md)
- [`optimizer_overlap_hooks.py_docs.md`](./optimizer_overlap_hooks.py_docs.md)
- [`mixed_precision_hooks.py_docs.md`](./mixed_precision_hooks.py_docs.md)
- [`default_hooks.py_docs.md`](./default_hooks.py_docs.md)
- [`ddp_zero_hook.py_docs.md`](./ddp_zero_hook.py_docs.md)
- [`quantization_hooks.py_docs.md`](./quantization_hooks.py_docs.md)


## Cross-References

- **File Documentation**: `post_localSGD_hook.py_docs.md`
- **Keyword Index**: `post_localSGD_hook.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

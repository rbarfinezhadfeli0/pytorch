# Documentation: `docs/torch/distributed/algorithms/ddp_comm_hooks/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/algorithms/ddp_comm_hooks/__init__.py_docs.md`
- **Size**: 6,973 bytes (6.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/distributed/algorithms/ddp_comm_hooks/__init__.py`

## File Metadata

- **Path**: `torch/distributed/algorithms/ddp_comm_hooks/__init__.py`
- **Size**: 4,167 bytes (4.07 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import sys
from enum import Enum
from functools import partial


# To suppress FutureWarning from partial since 3.13
if sys.version_info >= (3, 13):
    from enum import member

    def _enum_member(x):
        return member(x)
else:

    def _enum_member(x):
        return x


import torch.distributed as dist

from . import (
    debugging_hooks as debugging,
    default_hooks as default,
    optimizer_overlap_hooks as optimizer_overlap,
    powerSGD_hook as powerSGD,
    quantization_hooks as quantization,
)


__all__ = ["DDPCommHookType", "register_ddp_comm_hook"]


def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)


def _powerSGD_comm_hook_wrapper(
    comm_hook,
    model,
    state,
    matrix_approximation_rank,
    start_powerSGD_iter=1_000,
):
    """
    Wrap PowerSGD communication hook.

    To be consistent with the wrappers of other DDP comm hooks, the input state only needs to be a process group,
    which will be wrapped up with other state info.
    """
    powerSGD_state = powerSGD.PowerSGDState(
        process_group=state,
        matrix_approximation_rank=matrix_approximation_rank,
        start_powerSGD_iter=start_powerSGD_iter,
    )
    model.register_comm_hook(powerSGD_state, comm_hook)


class DDPCommHookType(Enum):
    """
    Enumerate ``ddp_comm_hooks`` and ``ddp_comm_hook_wrapper`` communucation hook types.

    DDPCommHookType enumerates the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    as names and ``ddp_comm_hook_wrapper`` partials with hook specified. As an example,
    you can register allreduce hook by
    ``DDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
    """

    ALLREDUCE = _enum_member(
        partial(_ddp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
    )
    FP16_COMPRESS = _enum_member(
        partial(_ddp_comm_hook_wrapper, comm_hook=default.fp16_compress_hook)
    )
    BF16_COMPRESS = _enum_member(
        partial(_ddp_comm_hook_wrapper, comm_hook=default.bf16_compress_hook)
    )
    QUANTIZE_PER_TENSOR = _enum_member(
        partial(
            _ddp_comm_hook_wrapper, comm_hook=quantization.quantization_pertensor_hook
        )
    )
    QUANTIZE_PER_CHANNEL = _enum_member(
        partial(
            _ddp_comm_hook_wrapper, comm_hook=quantization.quantization_perchannel_hook
        )
    )
    POWER_SGD = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.powerSGD_hook,
            matrix_approximation_rank=1,
        )
    )
    # Rank-2 PowerSGD can give a higher accuracy than the default rank-1 version,
    # but it runs slower and consumes more memory.
    POWER_SGD_RANK2 = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.powerSGD_hook,
            matrix_approximation_rank=2,
        )
    )
    # Batching can lead to a faster training at the cost of accuracy.
    BATCHED_POWER_SGD = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.batched_powerSGD_hook,
            matrix_approximation_rank=1,
        )
    )
    BATCHED_POWER_SGD_RANK2 = _enum_member(
        partial(
            _powerSGD_comm_hook_wrapper,
            comm_hook=powerSGD.batched_powerSGD_hook,
            matrix_approximation_rank=2,
        )
    )
    NOOP = _enum_member(
        partial(
            _ddp_comm_hook_wrapper,
            comm_hook=debugging.noop_hook,
        )
    )


def register_ddp_comm_hook(comm_hook_type: DDPCommHookType, model, state=None):
    """
    Register ``ddp_comm_hooks`` to DDP model.

    Registers the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    to the DDP model. User can specify the type of hook as an enum
    ``DDPCommHookType`` type using ``comm_hook_type`` input. State input will
    be passed to the model.
    Uses Python comm hook implementations.

    Example::
        >>> # xdoctest: +SKIP
        >>> register_ddp_comm_hook(DDPCommHookType.FP16_COMPRESS, model, state)
    """
    comm_hook_type.value(model=model, state=state)

```



## High-Level Overview

"""    Wrap PowerSGD communication hook.    To be consistent with the wrappers of other DDP comm hooks, the input state only needs to be a process group,    which will be wrapped up with other state info.

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DDPCommHookType`

**Functions defined**: `_enum_member`, `_enum_member`, `_ddp_comm_hook_wrapper`, `_powerSGD_comm_hook_wrapper`, `register_ddp_comm_hook`

**Key imports**: sys, Enum, partial, member, torch.distributed as dist


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/algorithms/ddp_comm_hooks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `enum`: Enum
- `functools`: partial
- `torch.distributed as dist`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

- [`debugging_hooks.py_docs.md`](./debugging_hooks.py_docs.md)
- [`post_localSGD_hook.py_docs.md`](./post_localSGD_hook.py_docs.md)
- [`powerSGD_hook.py_docs.md`](./powerSGD_hook.py_docs.md)
- [`optimizer_overlap_hooks.py_docs.md`](./optimizer_overlap_hooks.py_docs.md)
- [`mixed_precision_hooks.py_docs.md`](./mixed_precision_hooks.py_docs.md)
- [`default_hooks.py_docs.md`](./default_hooks.py_docs.md)
- [`ddp_zero_hook.py_docs.md`](./ddp_zero_hook.py_docs.md)
- [`quantization_hooks.py_docs.md`](./quantization_hooks.py_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/algorithms/ddp_comm_hooks`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/algorithms/ddp_comm_hooks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/distributed/algorithms/ddp_comm_hooks`):

- [`quantization_hooks.py_kw.md_docs.md`](./quantization_hooks.py_kw.md_docs.md)
- [`default_hooks.py_docs.md_docs.md`](./default_hooks.py_docs.md_docs.md)
- [`quantization_hooks.py_docs.md_docs.md`](./quantization_hooks.py_docs.md_docs.md)
- [`optimizer_overlap_hooks.py_docs.md_docs.md`](./optimizer_overlap_hooks.py_docs.md_docs.md)
- [`debugging_hooks.py_docs.md_docs.md`](./debugging_hooks.py_docs.md_docs.md)
- [`optimizer_overlap_hooks.py_kw.md_docs.md`](./optimizer_overlap_hooks.py_kw.md_docs.md)
- [`powerSGD_hook.py_docs.md_docs.md`](./powerSGD_hook.py_docs.md_docs.md)
- [`powerSGD_hook.py_kw.md_docs.md`](./powerSGD_hook.py_kw.md_docs.md)
- [`post_localSGD_hook.py_docs.md_docs.md`](./post_localSGD_hook.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/xpu/random.py_docs.md`

## File Metadata

- **Path**: `docs/torch/xpu/random.py_docs.md`
- **Size**: 7,953 bytes (7.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/xpu/random.py`

## File Metadata

- **Path**: `torch/xpu/random.py`
- **Size**: 5,279 bytes (5.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Iterable
from typing import Union

import torch
from torch import Tensor

from . import _lazy_call, _lazy_init, current_device, device_count


def get_rng_state(device: Union[int, str, torch.device] = "xpu") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(
    new_state: Tensor, device: Union[int, str, torch.device] = "xpu"
) -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    with torch._C._DisableFuncTorch():
        new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.set_state(new_state_copy)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb() -> None:
        idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb() -> None:
        for i in range(device_count()):
            default_generator = torch.xpu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb() -> None:
        idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.
    """

    def cb() -> None:
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch.xpu.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed() -> int:
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.initial_seed()


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]

```



## High-Level Overview

r"""Return the random number generator state of the specified GPU as a ByteTensor.    Args:        device (torch.device or int, optional): The device to return the RNG state of.            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).    .. warning::        This function eagerly initializes XPU.

This Python file contains 0 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_rng_state`, `get_rng_state_all`, `set_rng_state`, `cb`, `set_rng_state_all`, `manual_seed`, `cb`, `manual_seed_all`, `cb`, `seed`, `cb`, `seed_all`, `cb`, `initial_seed`

**Key imports**: Iterable, Union, torch, Tensor, _lazy_call, _lazy_init, current_device, device_count


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/xpu`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable
- `typing`: Union
- `torch`
- `.`: _lazy_call, _lazy_init, current_device, device_count


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/xpu`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`memory.py_docs.md`](./memory.py_docs.md)
- [`_gpu_trace.py_docs.md`](./_gpu_trace.py_docs.md)


## Cross-References

- **File Documentation**: `random.py_docs.md`
- **Keyword Index**: `random.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/xpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/xpu`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/xpu`):

- [`memory.py_kw.md_docs.md`](./memory.py_kw.md_docs.md)
- [`memory.py_docs.md_docs.md`](./memory.py_docs.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`streams.py_kw.md_docs.md`](./streams.py_kw.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`random.py_kw.md_docs.md`](./random.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`streams.py_docs.md_docs.md`](./streams.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `random.py_docs.md_docs.md`
- **Keyword Index**: `random.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

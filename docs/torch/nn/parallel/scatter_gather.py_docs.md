# Documentation: `torch/nn/parallel/scatter_gather.py`

## File Metadata

- **Path**: `torch/nn/parallel/scatter_gather.py`
- **Size**: 5,702 bytes (5.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Sequence
from typing import Any, overload, TypeVar
from typing_extensions import deprecated

import torch
from torch.nn.parallel._functions import Gather, Scatter


__all__ = ["scatter", "scatter_kwargs", "gather"]


@deprecated(
    "`is_namedtuple` is deprecated, please use the python checks instead",
    category=FutureWarning,
)
def is_namedtuple(obj: Any) -> bool:
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return _is_namedtuple(obj)


def _is_namedtuple(obj: Any) -> bool:
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


T = TypeVar("T", dict, list, tuple)


# For some reason, 'scatter' returns a tuple when given a single Tensor input but a list otherwise.
@overload
def scatter(
    inputs: torch.Tensor,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> tuple[torch.Tensor, ...]: ...


@overload
def scatter(
    inputs: T,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> list[T]: ...


def scatter(inputs, target_gpus, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            # pyrefly: ignore [no-matching-overload]
            return [
                # pyrefly: ignore [no-matching-overload]
                type(obj)(*args)
                # pyrefly: ignore  # no-matching-overload
                for args in zip(*map(scatter_map, obj), strict=False)
            ]
        if isinstance(obj, tuple) and len(obj) > 0:
            # pyrefly: ignore [no-matching-overload]
            return list(zip(*map(scatter_map, obj), strict=False))
        if isinstance(obj, list) and len(obj) > 0:
            # pyrefly: ignore [no-matching-overload]
            return [list(i) for i in zip(*map(scatter_map, obj), strict=False)]
        if isinstance(obj, dict) and len(obj) > 0:
            # pyrefly: ignore [no-matching-overload]
            return [
                # pyrefly: ignore [no-matching-overload]
                type(obj)(i)
                # pyrefly: ignore  # no-matching-overload
                for i in zip(*map(scatter_map, obj.items()), strict=False)
            ]
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res


def scatter_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_gpus: Sequence[int | torch.device],
    dim: int = 0,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    r"""Scatter with support for kwargs dictionary."""
    scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(
            () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
        )
    elif len(scattered_kwargs) < len(inputs):
        scattered_kwargs.extend(
            {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
        )
    return tuple(scattered_inputs), tuple(scattered_kwargs)


def gather(outputs: Any, target_device: int | torch.device, dim: int = 0) -> Any:
    r"""Gather tensors from different GPUs on a specified device.

    This function is useful for gathering the results of a distributed computation.
    It takes a sequence of objects, one for each GPU, and returns a single object
    on the specified device.

    Args:
        outputs (Any): A sequence of objects (potentially tensors) to gather.
        target_device (Union[int, torch.device]): The device to gather the tensors to.
            Use 'cpu' for CPU to avoid a deprecation warning.
        dim (int, optional): The dimension along which to gather. Default: 0.

    Returns:
        Any: A gathered object (potentially tensor) on the specified device.
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError("All dicts must have the same number of keys")
            # pyrefly: ignore [not-callable]
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        if _is_namedtuple(out):
            # pyrefly: ignore [no-matching-overload]
            return type(out)._make(map(gather_map, zip(*outputs, strict=True)))
        # pyrefly: ignore [no-matching-overload]
        return type(out)(map(gather_map, zip(*outputs, strict=True)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None  # type: ignore[assignment]
    return res

```



## High-Level Overview

r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `is_namedtuple`, `_is_namedtuple`, `scatter`, `scatter`, `scatter`, `scatter_map`, `scatter_kwargs`, `gather`, `gather_map`

**Key imports**: Sequence, Any, overload, TypeVar, deprecated, torch, Gather, Scatter


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/parallel`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Sequence
- `typing`: Any, overload, TypeVar
- `typing_extensions`: deprecated
- `torch`
- `torch.nn.parallel._functions`: Gather, Scatter


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

Files in the same folder (`torch/nn/parallel`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`data_parallel.py_docs.md`](./data_parallel.py_docs.md)
- [`parallel_apply.py_docs.md`](./parallel_apply.py_docs.md)
- [`replicate.py_docs.md`](./replicate.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`_functions.py_docs.md`](./_functions.py_docs.md)
- [`comm.py_docs.md`](./comm.py_docs.md)


## Cross-References

- **File Documentation**: `scatter_gather.py_docs.md`
- **Keyword Index**: `scatter_gather.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

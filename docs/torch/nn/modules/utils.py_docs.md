# Documentation: `torch/nn/modules/utils.py`

## File Metadata

- **Path**: `torch/nn/modules/utils.py`
- **Size**: 2,640 bytes (2.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import collections
from itertools import repeat
from typing import Any


__all__ = ["consume_prefix_in_state_dict_if_present"]


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _list_with_default(out_size: list[int], defaults: list[int]) -> list[int]:
    import torch

    if isinstance(out_size, (int, torch.SymInt)):
        # pyrefly: ignore [bad-return]
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(f"Input dimension should be at least {len(out_size) + 1}")
    return [
        v if v is not None else d
        for v, d in zip(out_size, defaults[-len(out_size) :], strict=False)
    ]


def consume_prefix_in_state_dict_if_present(
    state_dict: dict[str, Any],
    prefix: str,
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    .. note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if hasattr(state_dict, "_metadata"):
        keys = list(state_dict._metadata.keys())
        for key in keys:
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.
            if len(key) == 0:
                continue
            # handling both, 'module' case and  'module.' cases
            if key == prefix.replace(".", "") or key.startswith(prefix):
                newkey = key[len(prefix) :]
                state_dict._metadata[newkey] = state_dict._metadata.pop(key)

```



## High-Level Overview

r"""Reverse the order of `t` and repeat each element for `n` times.    This can be used to translate padding arg used by Conv and Pooling modules    to the ones used by `F.pad`.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_ntuple`, `parse`, `_reverse_repeat_tuple`, `_list_with_default`, `consume_prefix_in_state_dict_if_present`

**Key imports**: collections, repeat, Any, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `itertools`: repeat
- `typing`: Any
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/nn/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fold.py_docs.md`](./fold.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`channelshuffle.py_docs.md`](./channelshuffle.py_docs.md)
- [`adaptive.py_docs.md`](./adaptive.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`distance.py_docs.md`](./distance.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

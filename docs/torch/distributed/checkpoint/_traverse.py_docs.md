# Documentation: `torch/distributed/checkpoint/_traverse.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_traverse.py`
- **Size**: 6,966 bytes (6.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable, Collection, Mapping, MutableMapping
from typing import cast, Optional, TypeVar, Union

import torch
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.tensor import DTensor


PATH_ITEM = Union[str, int]
OBJ_PATH = tuple[PATH_ITEM, ...]
T = TypeVar("T")

STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


def _keep_visiting_tensors(value: STATE_DICT_ITEM) -> bool:
    return isinstance(value, torch.Tensor)


# TODO: update docstring for traverse.py
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.
    ``visitor`` will only be applied to elements in a list or a tuple, if the
    container contains tensors or mappings.
    """

    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            return False
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def traverse_state_dict_v_2_3(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """
    Traversal is short-circuited when if finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """

    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def set_element(
    root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: STATE_DICT_ITEM
) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: list[STATE_DICT_ITEM], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val = cast(STATE_DICT_ITEM, {} if type(key) is str else [])

        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            # pyrefly: ignore [bad-argument-type]
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) is int:
        extend_list(cast(list[STATE_DICT_ITEM], cur_container), key)

    cur_container[key] = value


def get_element(
    root_dict: STATE_DICT_TYPE,
    path: OBJ_PATH,
    default_value: Optional[T] = None,
) -> Optional[T]:
    """Retrieve the value at ``path``from ``root_dict``, returning ``default_value`` if not found."""
    cur_value = cast(CONTAINER_TYPE, root_dict)
    for part in path:
        if type(part) is int:
            if not isinstance(cur_value, list) or len(cur_value) < part:
                return default_value
        elif not isinstance(cur_value, Mapping) or part not in cur_value:
            return default_value

        # pyrefly: ignore [index-error]
        cur_value = cast(CONTAINER_TYPE, cur_value[part])
    return cast(Optional[T], cur_value)


def _print_nested(
    value: STATE_DICT_ITEM,
    prefix: str = "",
    print_fun: Callable[[str], None] = print,
) -> None:
    if type(value) is ShardedTensor:
        print_fun(f"{prefix} ShardedTensor size: {value.size()}")
        for shard in value.local_shards():
            _print_nested(
                shard.tensor,
                f"{shard.metadata.shard_offsets} ",
                print_fun=print_fun,
            )
    elif type(value) is (DTensor):
        print_fun(f"{prefix} DistributedTensor size: {value.size()}")
        # TODO: add local offset for _local_tensor in print_nested.
        _print_nested(
            value._local_tensor,
            print_fun=print_fun,
        )
    elif isinstance(value, torch.Tensor):
        print_fun(f"{prefix} Tensor size: {value.size()}")
    else:
        print_fun(f"{prefix} Type: {type(value)}")


def print_tensor(
    path: OBJ_PATH,
    value: STATE_DICT_ITEM,
    print_fun: Callable[[str], None] = print,
) -> None:
    """
    Use this callback with traverse_state_dict to print its content.

    By default the content is printed using the builtin ``print`` but this can
    be change by passing a different ``print_fun` callable.
    """
    _print_nested(value, prefix=str(path), print_fun=print_fun)

```



## High-Level Overview

"""    Invoke ``visitor`` for each value recursively in ``state_dict``.    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.    ``visitor`` will only be applied to elements in a list or a tuple, if the    container contains tensors or mappings.

This Python file contains 0 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_keep_visiting_tensors`, `traverse_state_dict`, `_is_terminal`, `_traverse_obj`, `traverse_state_dict_v_2_3`, `_is_terminal`, `_traverse_obj`, `set_element`, `extend_list`, `get_element`, `_print_nested`, `print_tensor`

**Key imports**: Callable, Collection, Mapping, MutableMapping, cast, Optional, TypeVar, Union, torch, ShardedTensor, STATE_DICT_TYPE, DTensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable, Collection, Mapping, MutableMapping
- `typing`: cast, Optional, TypeVar, Union
- `torch`
- `torch.distributed._shard.sharded_tensor.api`: ShardedTensor
- `torch.distributed.checkpoint.metadata`: STATE_DICT_TYPE
- `torch.distributed.tensor`: DTensor


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

Files in the same folder (`torch/distributed/checkpoint`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`filesystem.py_docs.md`](./filesystem.py_docs.md)
- [`_consolidate_hf_safetensors.py_docs.md`](./_consolidate_hf_safetensors.py_docs.md)
- [`hf_storage.py_docs.md`](./hf_storage.py_docs.md)
- [`state_dict_loader.py_docs.md`](./state_dict_loader.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`_storage_utils.py_docs.md`](./_storage_utils.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_async_process_executor.py_docs.md`](./_async_process_executor.py_docs.md)
- [`resharding.py_docs.md`](./resharding.py_docs.md)


## Cross-References

- **File Documentation**: `_traverse.py_docs.md`
- **Keyword Index**: `_traverse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

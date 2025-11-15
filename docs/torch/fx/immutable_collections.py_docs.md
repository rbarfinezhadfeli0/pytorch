# Documentation: `torch/fx/immutable_collections.py`

## File Metadata

- **Path**: `torch/fx/immutable_collections.py`
- **Size**: 3,269 bytes (3.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterable
from typing import Any, NoReturn, TypeVar
from typing_extensions import Self

from torch.utils._pytree import (
    _dict_flatten,
    _dict_flatten_with_keys,
    _dict_unflatten,
    _list_flatten,
    _list_flatten_with_keys,
    _list_unflatten,
    Context,
    register_pytree_node,
)

from ._compatibility import compatibility


__all__ = ["immutable_list", "immutable_dict"]


_help_mutation = """
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:

    new_args = ...  # copy and mutate args
    node.args = new_args
""".strip()


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _no_mutation(self: Any, *args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError(
        f"{type(self).__name__!r} object does not support mutation. {_help_mutation}",
    )


@compatibility(is_backward_compatible=True)
class immutable_list(list[_T]):
    """An immutable version of :class:`list`."""

    __delitem__ = _no_mutation
    __iadd__ = _no_mutation
    __imul__ = _no_mutation
    __setitem__ = _no_mutation
    append = _no_mutation
    clear = _no_mutation
    extend = _no_mutation
    insert = _no_mutation
    pop = _no_mutation
    remove = _no_mutation
    reverse = _no_mutation
    sort = _no_mutation

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))

    def __reduce__(self) -> tuple[type[Self], tuple[tuple[_T, ...]]]:
        return (type(self), (tuple(self),))


@compatibility(is_backward_compatible=True)
class immutable_dict(dict[_KT, _VT]):
    """An immutable version of :class:`dict`."""

    __delitem__ = _no_mutation
    __ior__ = _no_mutation
    __setitem__ = _no_mutation
    clear = _no_mutation
    pop = _no_mutation
    popitem = _no_mutation
    setdefault = _no_mutation
    update = _no_mutation  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[Self], tuple[tuple[tuple[_KT, _VT], ...]]]:
        return (type(self), (tuple(self.items()),))


# Register immutable collections for PyTree operations
def _immutable_list_flatten(d: immutable_list[_T]) -> tuple[list[_T], Context]:
    return _list_flatten(d)


def _immutable_list_unflatten(
    values: Iterable[_T],
    context: Context,
) -> immutable_list[_T]:
    return immutable_list(_list_unflatten(values, context))


def _immutable_dict_flatten(d: immutable_dict[Any, _VT]) -> tuple[list[_VT], Context]:
    return _dict_flatten(d)


def _immutable_dict_unflatten(
    values: Iterable[_VT],
    context: Context,
) -> immutable_dict[Any, _VT]:
    return immutable_dict(_dict_unflatten(values, context))


register_pytree_node(
    immutable_list,
    _immutable_list_flatten,
    _immutable_list_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
register_pytree_node(
    immutable_dict,
    _immutable_dict_flatten,
    _immutable_dict_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)

```



## High-Level Overview

_help_mutation = """If you are attempting to modify the kwargs or args of a torch.fx.Node object,instead create a new copy of it and assign the copy to the node:    new_args = ...  # copy and mutate args    node.args = new_args

This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `immutable_list`, `immutable_dict`

**Functions defined**: `_no_mutation`, `__hash__`, `__reduce__`, `__hash__`, `__reduce__`, `_immutable_list_flatten`, `_immutable_list_unflatten`, `_immutable_dict_flatten`, `_immutable_dict_unflatten`

**Key imports**: Iterable, Any, NoReturn, TypeVar, Self, compatibility


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable
- `typing`: Any, NoReturn, TypeVar
- `typing_extensions`: Self
- `._compatibility`: compatibility


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

Files in the same folder (`torch/fx`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`tensor_type.py_docs.md`](./tensor_type.py_docs.md)
- [`traceback.py_docs.md`](./traceback.py_docs.md)
- [`_symbolic_trace.py_docs.md`](./_symbolic_trace.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`node.py_docs.md`](./node.py_docs.md)
- [`annotate.py_docs.md`](./annotate.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`subgraph_rewriter.py_docs.md`](./subgraph_rewriter.py_docs.md)


## Cross-References

- **File Documentation**: `immutable_collections.py_docs.md`
- **Keyword Index**: `immutable_collections.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

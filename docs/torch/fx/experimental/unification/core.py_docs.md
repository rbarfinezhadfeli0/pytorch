# Documentation: `torch/fx/experimental/unification/core.py`

## File Metadata

- **Path**: `torch/fx/experimental/unification/core.py`
- **Size**: 2,780 bytes (2.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Iterator  # type: ignore[import]
from functools import partial

from .dispatch import dispatch
from .unification_tools import assoc  # type: ignore[import]
from .utils import transitive_get as walk
from .variable import isvar


__all__ = ["reify", "unify"]

###############
# Reification #
###############


@dispatch(Iterator, dict)
def _reify(t, s):
    return map(partial(reify, s=s), t)
    # return (reify(arg, s) for arg in t)


_reify


@dispatch(tuple, dict)  # type: ignore[no-redef]
def _reify(t, s):
    return tuple(reify(iter(t), s))


_reify


@dispatch(list, dict)  # type: ignore[no-redef]
def _reify(t, s):
    return list(reify(iter(t), s))


_reify


@dispatch(dict, dict)  # type: ignore[no-redef]
def _reify(d, s):
    return {k: reify(v, s) for k, v in d.items()}


_reify


@dispatch(object, dict)  # type: ignore[no-redef]
def _reify(o, s):
    return o  # catch all, just return the object


def reify(e, s):
    """Replace variables of expression with substitution
    >>> # xdoctest: +SKIP
    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))
    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """
    if isvar(e):
        return reify(s[e], s) if e in s else e
    return _reify(e, s)


###############
# Unification #
###############

seq = tuple, list, Iterator


@dispatch(seq, seq, dict)  # type: ignore[arg-type]
def _unify(u, v, s):
    if len(u) != len(v):
        return False
    for uu, vv in zip(u, v):  # avoiding recursion
        s = unify(uu, vv, s)
        if s is False:
            return False
    return s


#
# @dispatch((set, frozenset), (set, frozenset), dict)
# def _unify(u, v, s):
#     i = u & v
#     u = u - i
#     v = v - i
#     return _unify(sorted(u), sorted(v), s)
#
#
# @dispatch(dict, dict, dict)
# def _unify(u, v, s):
#     if len(u) != len(v):
#         return False
#     for key, uval in iteritems(u):
#         if key not in v:
#             return False
#         s = unify(uval, v[key], s)
#         if s is False:
#             return False
#     return s
#
#
# @dispatch(object, object, dict)
# def _unify(u, v, s):
#     return False  # catch all


@dispatch(object, object, dict)
def unify(u, v, s):  # no check at the moment
    """Find substitution so that u == v while satisfying s
    >>> x = var("x")
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    """
    u = walk(u, s)
    v = walk(v, s)
    if u == v:
        return s
    if isvar(u):
        return assoc(s, u, v)
    if isvar(v):
        return assoc(s, v, u)
    return _unify(u, v, s)


unify


@dispatch(object, object)  # type: ignore[no-redef]
def unify(u, v):
    return unify(u, v, {})

```



## High-Level Overview


This Python file contains 0 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_reify`, `_reify`, `_reify`, `_reify`, `_reify`, `reify`, `_unify`, `_unify`, `_unify`, `_unify`, `unify`, `unify`

**Key imports**: Iterator  , partial, dispatch, assoc  , transitive_get as walk, isvar


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental/unification`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterator  
- `functools`: partial
- `.dispatch`: dispatch
- `.unification_tools`: assoc  
- `.utils`: transitive_get as walk
- `.variable`: isvar


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

Files in the same folder (`torch/fx/experimental/unification`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`dispatch.py_docs.md`](./dispatch.py_docs.md)
- [`variable.py_docs.md`](./variable.py_docs.md)
- [`more.py_docs.md`](./more.py_docs.md)
- [`LICENSE.txt_docs.md`](./LICENSE.txt_docs.md)
- [`match.py_docs.md`](./match.py_docs.md)
- [`unification_tools.py_docs.md`](./unification_tools.py_docs.md)


## Cross-References

- **File Documentation**: `core.py_docs.md`
- **Keyword Index**: `core.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/fx/experimental/unification/utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/experimental/unification/utils.py_docs.md`
- **Size**: 5,302 bytes (5.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/experimental/unification/utils.py`

## File Metadata

- **Path**: `torch/fx/experimental/unification/utils.py`
- **Size**: 2,983 bytes (2.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
__all__ = ["hashable", "transitive_get", "raises", "reverse_dict", "xfail", "freeze"]


def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False


def transitive_get(key, d):
    """Transitive dict.get
    >>> d = {1: 2, 2: 3, 3: 4}
    >>> d.get(1)
    2
    >>> transitive_get(1, d)
    4
    """
    while hashable(key) and key in d:
        key = d[key]
    return key


def raises(err, lamda):  # codespell:ignore lamda
    try:
        lamda()  # codespell:ignore lamda
        return False
    except err:
        return True


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges):
    """Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> # xdoctest: +SKIP
    >>> _toposort({1: (2, 3), 2: (3,)})
    [1, 2, 3]
    Closely follows the wikipedia page [2]
    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    S = {v for v in edges if v not in incoming_edges}
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S.add(m)
    if any(incoming_edges.get(v) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d):
    """Reverses direction of dependence dict
    >>> d = {"a": (1, 2), "b": (2, 3), "c": ()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}
    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.
    """
    result = {}  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, ()) + (key,)
    return result


def xfail(func):
    try:
        func()
        raise Exception("XFailed test passed")  # pragma:nocover  # noqa: TRY002
    except Exception:
        pass


def freeze(d):
    """Freeze container to hashable form
    >>> freeze(1)
    1
    >>> freeze([1, 2])
    (1, 2)
    >>> freeze({1: 2})  # doctest: +SKIP
    frozenset([(1, 2)])
    """
    if isinstance(d, dict):
        return frozenset(map(freeze, d.items()))
    if isinstance(d, set):
        return frozenset(map(freeze, d))
    if isinstance(d, (tuple, list)):
        return tuple(map(freeze, d))
    return d

```



## High-Level Overview

"""Transitive dict.get    >>> d = {1: 2, 2: 3, 3: 4}    >>> d.get(1)    2    >>> transitive_get(1, d)    4

This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `hashable`, `transitive_get`, `raises`, `_toposort`, `reverse_dict`, `xfail`, `freeze`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental/unification`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*No imports detected.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
- [`dispatch.py_docs.md`](./dispatch.py_docs.md)
- [`core.py_docs.md`](./core.py_docs.md)
- [`variable.py_docs.md`](./variable.py_docs.md)
- [`more.py_docs.md`](./more.py_docs.md)
- [`LICENSE.txt_docs.md`](./LICENSE.txt_docs.md)
- [`match.py_docs.md`](./match.py_docs.md)
- [`unification_tools.py_docs.md`](./unification_tools.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/experimental/unification`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/experimental/unification`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/torch/fx/experimental/unification`):

- [`LICENSE.txt_docs.md_docs.md`](./LICENSE.txt_docs.md_docs.md)
- [`core.py_docs.md_docs.md`](./core.py_docs.md_docs.md)
- [`variable.py_docs.md_docs.md`](./variable.py_docs.md_docs.md)
- [`dispatch.py_kw.md_docs.md`](./dispatch.py_kw.md_docs.md)
- [`unification_tools.py_kw.md_docs.md`](./unification_tools.py_kw.md_docs.md)
- [`LICENSE.txt_kw.md_docs.md`](./LICENSE.txt_kw.md_docs.md)
- [`more.py_kw.md_docs.md`](./more.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`dispatch.py_docs.md_docs.md`](./dispatch.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md_docs.md`
- **Keyword Index**: `utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

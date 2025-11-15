# Documentation: `docs/torch/fx/experimental/unification/multipledispatch/conflict.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/experimental/unification/multipledispatch/conflict.py_docs.md`
- **Size**: 7,437 bytes (7.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/experimental/unification/multipledispatch/conflict.py`

## File Metadata

- **Path**: `torch/fx/experimental/unification/multipledispatch/conflict.py`
- **Size**: 4,210 bytes (4.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import operator

from .utils import _toposort, groupby
from .variadic import isvariadic


__all__ = [
    "AmbiguityWarning",
    "supercedes",
    "consistent",
    "ambiguous",
    "ambiguities",
    "super_signature",
    "edge",
    "ordering",
]


class AmbiguityWarning(Warning):
    pass


def supercedes(a, b):
    """A is consistent and strictly more specific than B"""
    if len(a) < len(b):
        # only case is if a is empty and b is variadic
        return not a and len(b) == 1 and isvariadic(b[-1])
    elif len(a) == len(b):
        return all(map(issubclass, a, b))
    else:
        # len(a) > len(b)
        p1 = 0
        p2 = 0
        while p1 < len(a) and p2 < len(b):
            cur_a = a[p1]
            cur_b = b[p2]
            if not (isvariadic(cur_a) or isvariadic(cur_b)):
                if not issubclass(cur_a, cur_b):
                    return False
                p1 += 1
                p2 += 1
            elif isvariadic(cur_a):
                assert p1 == len(a) - 1
                return p2 == len(b) - 1 and issubclass(cur_a, cur_b)
            elif isvariadic(cur_b):
                assert p2 == len(b) - 1
                if not issubclass(cur_a, cur_b):
                    return False
                p1 += 1
        return p2 == len(b) - 1 and p1 == len(a)


def consistent(a, b):
    """It is possible for an argument list to satisfy both A and B"""

    # Need to check for empty args
    if not a:
        return not b or isvariadic(b[0])
    if not b:
        return not a or isvariadic(a[0])

    # Non-empty args check for mutual subclasses
    if len(a) == len(b):
        return all(issubclass(aa, bb) or issubclass(bb, aa) for aa, bb in zip(a, b))
    else:
        p1 = 0
        p2 = 0
        while p1 < len(a) and p2 < len(b):
            cur_a = a[p1]
            cur_b = b[p2]
            if not issubclass(cur_b, cur_a) and not issubclass(cur_a, cur_b):
                return False
            if not (isvariadic(cur_a) or isvariadic(cur_b)):
                p1 += 1
                p2 += 1
            elif isvariadic(cur_a):
                p2 += 1
            elif isvariadic(cur_b):
                p1 += 1
        # We only need to check for variadic ends
        # Variadic types are guaranteed to be the last element
        return (
            isvariadic(cur_a)  # type: ignore[possibly-undefined]
            and p2 == len(b)
            or isvariadic(cur_b)  # type: ignore[possibly-undefined]
            and p1 == len(a)
        )


def ambiguous(a, b):
    """A is consistent with B but neither is strictly more specific"""
    return consistent(a, b) and not (supercedes(a, b) or supercedes(b, a))


def ambiguities(signatures):
    """All signature pairs such that A is ambiguous with B"""
    signatures = list(map(tuple, signatures))
    return {
        (a, b)
        for a in signatures
        for b in signatures
        if hash(a) < hash(b)
        and ambiguous(a, b)
        and not any(supercedes(c, a) and supercedes(c, b) for c in signatures)
    }


def super_signature(signatures):
    """A signature that would break ambiguities"""
    n = len(signatures[0])
    assert all(len(s) == n for s in signatures)

    return [max((type.mro(sig[i]) for sig in signatures), key=len)[0] for i in range(n)]


def edge(a, b, tie_breaker=hash):
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    # A either supersedes B and B does not supersede A or if B does then call
    # tie_breaker
    return supercedes(a, b) and (
        not supercedes(b, a) or tie_breaker(a) > tie_breaker(b)
    )


def ordering(signatures):
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(operator.itemgetter(0), edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}  # type: ignore[assignment, attr-defined]
    return _toposort(edges)

```



## High-Level Overview

"""A is consistent and strictly more specific than B"""    if len(a) < len(b):        # only case is if a is empty and b is variadic        return not a and len(b) == 1 and isvariadic(b[-1])    elif len(a) == len(b):        return all(map(issubclass, a, b))    else:        # len(a) > len(b)        p1 = 0        p2 = 0        while p1 < len(a) and p2 < len(b):            cur_a = a[p1]            cur_b = b[p2]            if not (isvariadic(cur_a) or isvariadic(cur_b)):                if not issubclass(cur_a, cur_b):                    return False                p1 += 1                p2 += 1            elif isvariadic(cur_a):                assert p1 == len(a) - 1                return p2 == len(b) - 1 and issubclass(cur_a, cur_b)            elif isvariadic(cur_b):                assert p2 == len(b) - 1                if not issubclass(cur_a, cur_b):                    return False                p1 += 1

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AmbiguityWarning`

**Functions defined**: `supercedes`, `consistent`, `ambiguous`, `ambiguities`, `super_signature`, `edge`, `ordering`

**Key imports**: operator, _toposort, groupby, isvariadic


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental/unification/multipledispatch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `.utils`: _toposort, groupby
- `.variadic`: isvariadic


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

Files in the same folder (`torch/fx/experimental/unification/multipledispatch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`core.py_docs.md`](./core.py_docs.md)
- [`dispatcher.py_docs.md`](./dispatcher.py_docs.md)
- [`variadic.py_docs.md`](./variadic.py_docs.md)


## Cross-References

- **File Documentation**: `conflict.py_docs.md`
- **Keyword Index**: `conflict.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/experimental/unification/multipledispatch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/experimental/unification/multipledispatch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/fx/experimental/unification/multipledispatch`):

- [`variadic.py_docs.md_docs.md`](./variadic.py_docs.md_docs.md)
- [`core.py_docs.md_docs.md`](./core.py_docs.md_docs.md)
- [`conflict.py_kw.md_docs.md`](./conflict.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`dispatcher.py_docs.md_docs.md`](./dispatcher.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`core.py_kw.md_docs.md`](./core.py_kw.md_docs.md)
- [`dispatcher.py_kw.md_docs.md`](./dispatcher.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `conflict.py_docs.md_docs.md`
- **Keyword Index**: `conflict.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

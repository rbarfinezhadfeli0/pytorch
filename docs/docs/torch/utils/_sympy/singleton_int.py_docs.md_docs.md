# Documentation: `docs/torch/utils/_sympy/singleton_int.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_sympy/singleton_int.py_docs.md`
- **Size**: 5,429 bytes (5.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_sympy/singleton_int.py`

## File Metadata

- **Path**: `torch/utils/_sympy/singleton_int.py`
- **Size**: 2,975 bytes (2.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import sympy
from sympy.multipledispatch import dispatch


__all__ = ["SingletonInt"]


class SingletonInt(sympy.AtomicExpr):
    # This is probably not super important unless we are in multiple dispatch
    # situations with other more exotic Expr types.
    _op_priority = 99999

    def __new__(cls, *args, coeff=None, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    # The semantics of this class should match that of NestedIntSymNodeImpl in
    # c10/core/NestedIntSymNodeImpl.h
    def __init__(self, val, *, coeff=1) -> None:
        self._val = val
        self._coeff = coeff
        super().__init__()

    # See NOTE [ Inequalities with nested int ]
    def _eval_Eq(self, other):
        if (
            isinstance(other, SingletonInt)
            and other._val == self._val
            and self._coeff == other._coeff
        ):
            return sympy.true
        else:
            return sympy.false

    # This is necessary so that calling expr.free_symbols on exprs that contain
    # this Singleton does not error
    @property
    def free_symbols(self):
        return set()

    def __mul__(self, other):
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other):
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    # Make sure we promptly raise an error instead of falling back to building
    # an expression tree. There are probably more ops, how can we be exhaustive?
    def __add__(self, other):
        raise NotImplementedError("NYI")

    def __sub__(self, other):
        raise NotImplementedError("NYI")

    def __truediv__(self, other):
        raise NotImplementedError("NYI")

    def __floordiv__(self, other):
        raise NotImplementedError("NYI")

    def __mod__(self, other):
        raise NotImplementedError("NYI")


# See NOTE [ Inequalities with nested int ]
@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a, b):
    if a < 2:
        return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, sympy.Integer)  # type: ignore[no-redef]
def _eval_is_ge(a, b):  # noqa: F811
    if b <= 2:
        return sympy.true
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, SingletonInt)  # type: ignore[no-redef]
def _eval_is_ge(a, b):  # noqa: F811
    if a._val == b._val:
        if a._coeff >= b._coeff:
            return sympy.true
        else:
            return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")

```



## High-Level Overview


This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SingletonInt`

**Functions defined**: `__new__`, `__init__`, `_eval_Eq`, `free_symbols`, `__mul__`, `__rmul__`, `__add__`, `__sub__`, `__truediv__`, `__floordiv__`, `__mod__`, `_eval_is_ge`, `_eval_is_ge`, `_eval_is_ge`

**Key imports**: sympy, dispatch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/_sympy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `sympy`
- `sympy.multipledispatch`: dispatch


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

Files in the same folder (`torch/utils/_sympy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`solve.py_docs.md`](./solve.py_docs.md)
- [`value_ranges.py_docs.md`](./value_ranges.py_docs.md)
- [`numbers.py_docs.md`](./numbers.py_docs.md)
- [`reference.py_docs.md`](./reference.py_docs.md)
- [`functions.py_docs.md`](./functions.py_docs.md)
- [`interp.py_docs.md`](./interp.py_docs.md)
- [`symbol.py_docs.md`](./symbol.py_docs.md)
- [`printers.py_docs.md`](./printers.py_docs.md)


## Cross-References

- **File Documentation**: `singleton_int.py_docs.md`
- **Keyword Index**: `singleton_int.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/_sympy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/_sympy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/utils/_sympy`):

- [`numbers.py_docs.md_docs.md`](./numbers.py_docs.md_docs.md)
- [`interp.py_docs.md_docs.md`](./interp.py_docs.md_docs.md)
- [`singleton_int.py_kw.md_docs.md`](./singleton_int.py_kw.md_docs.md)
- [`value_ranges.py_kw.md_docs.md`](./value_ranges.py_kw.md_docs.md)
- [`solve.py_docs.md_docs.md`](./solve.py_docs.md_docs.md)
- [`reference.py_kw.md_docs.md`](./reference.py_kw.md_docs.md)
- [`functions.py_kw.md_docs.md`](./functions.py_kw.md_docs.md)
- [`interp.py_kw.md_docs.md`](./interp.py_kw.md_docs.md)
- [`solve.py_kw.md_docs.md`](./solve.py_kw.md_docs.md)
- [`printers.py_docs.md_docs.md`](./printers.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `singleton_int.py_docs.md_docs.md`
- **Keyword Index**: `singleton_int.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

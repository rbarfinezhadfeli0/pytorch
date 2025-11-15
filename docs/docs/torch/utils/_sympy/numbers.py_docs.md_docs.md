# Documentation: `docs/torch/utils/_sympy/numbers.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_sympy/numbers.py_docs.md`
- **Size**: 14,662 bytes (14.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_sympy/numbers.py`

## File Metadata

- **Path**: `torch/utils/_sympy/numbers.py`
- **Size**: 11,495 bytes (11.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import mpmath.libmp as mlib  # type: ignore[import-untyped]
import sympy
from sympy import Expr
from sympy.core.decorators import _sympifyit
from sympy.core.expr import AtomicExpr
from sympy.core.numbers import Number
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S, Singleton


# pyrefly: ignore [invalid-inheritance]
class IntInfinity(Number, metaclass=Singleton):
    r"""Positive integer infinite quantity.

    Integer infinity is a value in an extended integers which
    is greater than all other integers.  We distinguish it from
    sympy's existing notion of infinity in that it reports that
    it is_integer.

    Infinity is a singleton, and can be accessed by ``S.IntInfinity``,
    or can be imported as ``int_oo``.
    """

    # NB: We can't actually mark this as infinite, as integer and infinite are
    # inconsistent assumptions in sympy.  We also report that we are complex,
    # different from sympy.oo

    is_integer = True
    is_commutative = True
    is_number = True
    is_extended_real = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    # Ensure we get dispatched to before plain numbers
    _op_priority = 100.0

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _sympystr(self, printer) -> str:
        return "int_oo"

    def _eval_subs(self, old, new):
        if self == old:
            return new

    # We could do these, not sure about it
    """
    def _eval_evalf(self, prec=None):
        return Float('inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    @_sympifyit("other", NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NegativeInfinity):
                return other
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    __radd__ = __add__

    @_sympifyit("other", NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.NegativeInfinity
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    @_sympifyit("other", NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.NegativeIntInfinity
        return Number.__mul__(self, other)

    __rmul__ = __mul__

    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return S.Infinity  # truediv produces float
            return S.NegativeInfinity  # truediv produces float
        return Number.__truediv__(self, other)

    def __abs__(self):
        return S.IntInfinity

    def __neg__(self):
        return S.NegativeIntInfinity

    def _eval_power(self, expt):
        if expt.is_extended_positive:
            return S.IntInfinity
        if expt.is_extended_negative:
            return S.Zero
        if expt is S.NaN:
            return S.NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re

            expt_real = re(expt)
            if expt_real.is_positive:
                return S.ComplexInfinity
            if expt_real.is_negative:
                return S.Zero
            if expt_real.is_zero:
                return S.NaN

            return self ** expt.evalf()

    def _as_mpf_val(self, prec):
        return mlib.finf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.IntInfinity

    def __ne__(self, other):
        return other is not S.IntInfinity

    def __gt__(self, other):
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.true

    def __ge__(self, other):
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.true

    def __lt__(self, other):
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.false

    def __le__(self, other):
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.false

    @_sympifyit("other", NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self


int_oo = S.IntInfinity


# pyrefly: ignore [invalid-inheritance]
class NegativeIntInfinity(Number, metaclass=Singleton):
    """Negative integer infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    IntInfinity
    """

    # Ensure we get dispatched to before plain numbers
    _op_priority = 100.0

    is_integer = True
    is_extended_real = True
    is_commutative = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _eval_subs(self, old, new):
        if self == old:
            return new

    def _sympystr(self, printer) -> str:
        return "-int_oo"

    """
    def _eval_evalf(self, prec=None):
        return Float('-inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    @_sympifyit("other", NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    __radd__ = __add__

    @_sympifyit("other", NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    @_sympifyit("other", NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.IntInfinity
        return Number.__mul__(self, other)

    __rmul__ = __mul__

    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity  # truediv returns float
        return Number.__truediv__(self, other)

    def __abs__(self):
        return S.IntInfinity

    def __neg__(self):
        return S.IntInfinity

    def _eval_power(self, expt):
        if expt.is_number:
            if expt in (
                S.NaN,
                S.Infinity,
                S.NegativeInfinity,
                S.IntInfinity,
                S.NegativeIntInfinity,
            ):
                return S.NaN

            if isinstance(expt, sympy.Integer) and expt.is_extended_positive:
                if expt.is_odd:
                    return S.NegativeIntInfinity
                else:
                    return S.IntInfinity

            inf_part = S.IntInfinity**expt
            s_part = S.NegativeOne**expt
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            if (
                inf_part is S.ComplexInfinity
                and s_part.is_finite
                and not s_part.is_zero
            ):
                return S.ComplexInfinity
            return s_part * inf_part

    def _as_mpf_val(self, prec):
        return mlib.fninf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.NegativeIntInfinity

    def __ne__(self, other):
        return other is not S.NegativeIntInfinity

    def __gt__(self, other):
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.false

    def __ge__(self, other):
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.false

    def __lt__(self, other):
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.true

    def __le__(self, other):
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.true

    @_sympifyit("other", NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

    def as_powers_dict(self):
        return {S.NegativeOne: 1, S.IntInfinity: 1}

```



## High-Level Overview

r"""Positive integer infinite quantity.    Integer infinity is a value in an extended integers which    is greater than all other integers.  We distinguish it from    sympy's existing notion of infinity in that it reports that    it is_integer.    Infinity is a singleton, and can be accessed by ``S.IntInfinity``,    or can be imported as ``int_oo``.

This Python file contains 2 class(es) and 49 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IntInfinity`, `NegativeIntInfinity`

**Functions defined**: `__new__`, `_sympystr`, `_eval_subs`, `_eval_evalf`, `evalf`, `__add__`, `__sub__`, `__rsub__`, `__mul__`, `__truediv__`, `__abs__`, `__neg__`, `_eval_power`, `_as_mpf_val`, `__hash__`, `__eq__`, `__ne__`, `__gt__`, `__ge__`, `__lt__`

**Key imports**: mpmath.libmp as mlib  , sympy, Expr, _sympifyit, AtomicExpr, Number, global_parameters, S, Singleton, re


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/_sympy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `mpmath.libmp as mlib  `
- `sympy`
- `sympy.core.decorators`: _sympifyit
- `sympy.core.expr`: AtomicExpr
- `sympy.core.numbers`: Number
- `sympy.core.parameters`: global_parameters
- `sympy.core.singleton`: S, Singleton
- `sympy.functions.elementary.complexes`: re


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

Files in the same folder (`torch/utils/_sympy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`solve.py_docs.md`](./solve.py_docs.md)
- [`value_ranges.py_docs.md`](./value_ranges.py_docs.md)
- [`singleton_int.py_docs.md`](./singleton_int.py_docs.md)
- [`reference.py_docs.md`](./reference.py_docs.md)
- [`functions.py_docs.md`](./functions.py_docs.md)
- [`interp.py_docs.md`](./interp.py_docs.md)
- [`symbol.py_docs.md`](./symbol.py_docs.md)
- [`printers.py_docs.md`](./printers.py_docs.md)


## Cross-References

- **File Documentation**: `numbers.py_docs.md`
- **Keyword Index**: `numbers.py_kw.md`
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

Files in the same folder (`docs/torch/utils/_sympy`):

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

- **File Documentation**: `numbers.py_docs.md_docs.md`
- **Keyword Index**: `numbers.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/utils/_sympy/solve.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_sympy/solve.py_docs.md`
- **Size**: 8,658 bytes (8.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_sympy/solve.py`

## File Metadata

- **Path**: `torch/utils/_sympy/solve.py`
- **Size**: 6,377 bytes (6.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import logging

import sympy

from torch.utils._sympy.functions import FloorDiv


log = logging.getLogger(__name__)

_MIRROR_REL_OP: dict[type[sympy.Basic], type[sympy.Rel]] = {
    sympy.Eq: sympy.Eq,
    sympy.Ne: sympy.Ne,
    sympy.Ge: sympy.Le,
    sympy.Gt: sympy.Lt,
    sympy.Le: sympy.Ge,
    sympy.Lt: sympy.Gt,
}

INEQUALITY_TYPES = (sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le)


def mirror_rel_op(type: type) -> type[sympy.Rel] | None:
    return _MIRROR_REL_OP.get(type)


# Tries to simplify 'expr', so as to leave only 'thing' in the left-hand side.
#
# Returns a tuple of:
#   1. The simplified expression
#   2. The expression on the right-hand side
#
# Returns 'None' if it can't reach a state where the only thing in the left
# hand side is 'thing'.
#
# 'trials': number of times 'try_solve' will try to isolate 'thing' to the
# left-hand side.
#
# 'floordiv_inequality': flag to enable conversion of 'FloorDiv' into
# inequalities.
def try_solve(
    expr: sympy.Basic,
    thing: sympy.Basic,
    trials: int = 5,
    floordiv_inequality: bool = True,
) -> tuple[sympy.Rel, sympy.Expr] | None:
    mirror = mirror_rel_op(type(expr))

    # Ignore unsupported expressions:
    #   - Those that are not relational operations
    #   - Those that don't have a mirror (just avoiding unexpected classes)
    if not isinstance(expr, sympy.Rel) or mirror is None:
        log.debug("expression with unsupported type: %s", type(expr))
        return None

    lhs_has_thing = expr.lhs.has(thing)
    rhs_has_thing = expr.rhs.has(thing)

    # Give up when 'thing' appears on both sides of the relational expression.
    # That is because, as is, we assume the thing we are trying to isolate is
    # only on the right-hand side.
    if lhs_has_thing and rhs_has_thing:
        log.debug("thing (%s) found in both sides of expression: %s", thing, expr)
        return None

    # Try considering both LHS and RHS by mirroring the original expression:
    # a < b ==> b > a
    expressions = []

    # Add each version of 'expr' if 'thing' is in its left-hand side.
    if lhs_has_thing:
        expressions.append(expr)
    if rhs_has_thing:
        expressions.append(mirror(expr.rhs, expr.lhs))

    for e in expressions:
        if e is None:
            continue

        if not isinstance(e, sympy.Rel):
            raise AssertionError("expected sympy.Rel")

        for _ in range(trials):
            trial = _try_isolate_lhs(e, thing, floordiv_inequality=floordiv_inequality)
            # Stop if there was no change in this trial.
            if trial == e:
                break
            e = trial  # type: ignore[assignment]

        # Return if we were able to isolate 'thing' on the left-hand side.
        if isinstance(e, sympy.Rel) and e.lhs == thing:
            log.debug("solved: %s ---> %s", expr, e)
            return e, e.rhs

    return None


def _try_isolate_lhs(
    e: sympy.Basic, thing: sympy.Basic, floordiv_inequality: bool
) -> sympy.Basic:
    op = type(e)

    if isinstance(e, sympy.Rel):
        # Move any constants in the left-hand side to the right-hand side.
        lhs_not_thing = (
            sum(a for a in e.lhs.args if not a.has(thing))
            if isinstance(e.lhs, sympy.Add)
            else 0
        )
        e = op(e.lhs - lhs_not_thing, e.rhs - lhs_not_thing)  # type: ignore[attr-defined]

    # Divide both sides by the factors that don't contain thing.
    if isinstance(e, sympy.Rel) and isinstance(e.lhs, sympy.Mul):
        lhs, rhs = e.args
        other = sympy.Mul(*[a for a in lhs.args if not a.has(thing)])

        # If we can't tell whether 'other' is negative or positive, we do nothing.
        # That is because we don't know whether we have mirror the operation or not.
        # We also divide only when we know 'rhs' is not zero.
        if not (isinstance(e, INEQUALITY_TYPES) and other.is_negative is None) and not (
            not isinstance(e, INEQUALITY_TYPES) and rhs.is_zero
        ):
            # Divide both sides by 'other'.
            lhs = lhs / other
            rhs = rhs / other

            # If 'e' is an inequality and 'other' is negative, we have to
            # mirror the expression.
            if isinstance(e, INEQUALITY_TYPES) and other.is_negative:
                op = mirror_rel_op(op)  # type: ignore[assignment]

            if op is None:
                raise AssertionError("expected op to be not None")
            e = op(lhs, rhs)

    ################################################################################
    # left-hand side is FloorDiv
    ################################################################################
    #
    # Given the expression: a // b op c
    # where 'op' is a relational operation, these rules only work if:
    #   - b > 0
    #   - c is an integer
    if (
        floordiv_inequality
        and isinstance(e, sympy.Rel)
        and isinstance(e.lhs, FloorDiv)
        and e.lhs.divisor.is_positive
        and e.rhs.is_integer
    ):
        # a // b == expr
        # => a >= (b * expr) and a < (b * (expr + 1))
        if isinstance(e, sympy.Eq):
            numerator, denominator = e.lhs.args
            return sympy.And(
                sympy.Ge(numerator, (e.rhs * denominator)),
                sympy.Lt(numerator, ((e.rhs + 1) * denominator)),
            )
        # a // b != expr
        # => a < (b * expr) or a >= (b * (expr + 1))
        if isinstance(e, sympy.Ne):
            numerator, denominator = e.lhs.args
            return sympy.Or(
                sympy.Lt(numerator, (e.rhs * denominator)),
                sympy.Ge(numerator, ((e.rhs + 1) * denominator)),
            )
        # The transformations below only work if b is positive.
        # Note: we only have this information for constants.
        # a // b > expr  => a >= b * (expr + 1)
        # a // b >= expr => a >= b * expr
        if isinstance(e, (sympy.Gt, sympy.Ge)):
            quotient = e.rhs if isinstance(e, sympy.Ge) else (e.rhs + 1)
            return sympy.Ge(e.lhs.args[0], (quotient * e.lhs.args[1]))
        # a // b < expr  => a < b * expr
        # a // b <= expr => a < b * (expr + 1)
        if isinstance(e, (sympy.Lt, sympy.Le)):
            quotient = e.rhs if isinstance(e, sympy.Lt) else (e.rhs + 1)
            return sympy.Lt(e.lhs.args[0], (quotient * e.lhs.args[1]))

    return e

```



## High-Level Overview


This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `mirror_rel_op`, `try_solve`, `_try_isolate_lhs`

**Key imports**: logging, sympy, FloorDiv


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/_sympy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `sympy`
- `torch.utils._sympy.functions`: FloorDiv


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
- [`value_ranges.py_docs.md`](./value_ranges.py_docs.md)
- [`numbers.py_docs.md`](./numbers.py_docs.md)
- [`singleton_int.py_docs.md`](./singleton_int.py_docs.md)
- [`reference.py_docs.md`](./reference.py_docs.md)
- [`functions.py_docs.md`](./functions.py_docs.md)
- [`interp.py_docs.md`](./interp.py_docs.md)
- [`symbol.py_docs.md`](./symbol.py_docs.md)
- [`printers.py_docs.md`](./printers.py_docs.md)


## Cross-References

- **File Documentation**: `solve.py_docs.md`
- **Keyword Index**: `solve.py_kw.md`
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

- [`numbers.py_docs.md_docs.md`](./numbers.py_docs.md_docs.md)
- [`interp.py_docs.md_docs.md`](./interp.py_docs.md_docs.md)
- [`singleton_int.py_kw.md_docs.md`](./singleton_int.py_kw.md_docs.md)
- [`value_ranges.py_kw.md_docs.md`](./value_ranges.py_kw.md_docs.md)
- [`reference.py_kw.md_docs.md`](./reference.py_kw.md_docs.md)
- [`functions.py_kw.md_docs.md`](./functions.py_kw.md_docs.md)
- [`interp.py_kw.md_docs.md`](./interp.py_kw.md_docs.md)
- [`solve.py_kw.md_docs.md`](./solve.py_kw.md_docs.md)
- [`printers.py_docs.md_docs.md`](./printers.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `solve.py_docs.md_docs.md`
- **Keyword Index**: `solve.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

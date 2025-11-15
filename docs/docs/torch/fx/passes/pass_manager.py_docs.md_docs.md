# Documentation: `docs/torch/fx/passes/pass_manager.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/pass_manager.py_docs.md`
- **Size**: 10,170 bytes (9.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/passes/pass_manager.py`

## File Metadata

- **Path**: `torch/fx/passes/pass_manager.py`
- **Size**: 7,085 bytes (6.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
from collections.abc import Callable
from functools import wraps
from inspect import unwrap
from typing import Optional


logger = logging.getLogger(__name__)

__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "this_before_that_pass_constraint",
    "these_before_those_pass_constraint",
]


# for callables which modify object inplace and return something other than
# the object on which they act
def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        fn(gm)
        return gm

    return wrapped_fn


def log_hook(fn: Callable, level=logging.INFO) -> Callable:
    """
    Logs callable output.

    This is useful for logging output of passes. Note inplace_wrapper replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before inplace_wrapper.


    ```
    def my_pass(d: Dict) -> bool:
        changed = False
        if "foo" in d:
            d["foo"] = "bar"
            changed = True
        return changed


    pm = PassManager(passes=[inplace_wrapper(log_hook(my_pass))])
    ```

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        val = fn(gm)
        logger.log(level, "Ran pass %s\t Return value: %s", fn, val)
        return val

    return wrapped_fn


def loop_pass(
    base_pass: Callable,
    n_iter: Optional[int] = None,
    predicate: Optional[Callable] = None,
):
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
    assert (n_iter is not None) ^ (predicate is not None), (
        "Exactly one of `n_iter`or `predicate` must be specified."
    )

    @wraps(base_pass)
    def new_pass(source):
        output = source
        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)
        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)
        else:
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )
        return output

    return new_pass


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable, Callable], bool], passes: list[Callable]
):
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1 :]):
            if constraint(a, b):
                continue
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )


def this_before_that_pass_constraint(this: Callable, that: Callable):
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.
    """

    def depends_on(a: Callable, b: Callable):
        return a != that or b != this

    return depends_on


def these_before_those_pass_constraint(these: Callable, those: Callable):
    """
    Defines a partial order ('depends on' function) where `these` must occur
    before `those`. Where the inputs are 'unwrapped' before comparison.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [
        loop_pass(pass_b, 3),
        loop_pass(pass_a, 5),
    ]

    constraints = [these_before_those_pass_constraint(pass_a, pass_b)]
    ```

    Args:
        these (Callable): pass which should occur first
        those (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

    def depends_on(a: Callable, b: Callable):
        return unwrap(a) != those or unwrap(b) != these

    return depends_on


class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): list of passes. A pass is a
            callable which modifies an object and returns modified object
        constraint (Optional[List[Callable]]): list of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
    """

    passes: list[Callable]
    constraints: list[Callable]
    _validated: bool = False

    def __init__(
        self,
        passes=None,
        constraints=None,
    ):
        self.passes = passes or []
        self.constraints = constraints or []

    @classmethod
    def build_from_passlist(cls, passes):
        pm = PassManager(passes)
        # TODO(alexbeloi): add constraint management/validation
        return pm

    def add_pass(self, _pass: Callable):
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, _passes: list[str]):
        if _passes is None:
            return
        passes_left = [ps for ps in self.passes if ps.__name__ not in _passes]
        self.passes = passes_left
        self._validated = False

    def replace_pass(self, _target, _replacement):
        passes_left = []
        for ps in self.passes:
            if ps.__name__ == _target.__name__:
                passes_left.append(_replacement)
            else:
                passes_left.append(ps)
        self.passes = passes_left
        self._validated = False

    def validate(self):
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def __call__(self, source):
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out

```



## High-Level Overview

"""    Convenience wrapper for passes which modify an object inplace. This    wrapper makes them return the modified object instead.    Args:        fn (Callable[Object, Any])    Returns:        wrapped_fn (Callable[Object, Object])

This Python file contains 1 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PassManager`

**Functions defined**: `inplace_wrapper`, `wrapped_fn`, `log_hook`, `my_pass`, `wrapped_fn`, `loop_pass`, `new_pass`, `_validate_pass_schedule_constraint`, `this_before_that_pass_constraint`, `depends_on`, `these_before_those_pass_constraint`, `depends_on`, `__init__`, `build_from_passlist`, `add_pass`, `add_constraint`, `remove_pass`, `replace_pass`, `validate`, `__call__`

**Key imports**: logging, Callable, wraps, unwrap, Optional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `collections.abc`: Callable
- `functools`: wraps
- `inspect`: unwrap
- `typing`: Optional


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

Files in the same folder (`torch/fx/passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`operator_support.py_docs.md`](./operator_support.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_drawer.py_docs.md`](./graph_drawer.py_docs.md)
- [`shape_prop.py_docs.md`](./shape_prop.py_docs.md)
- [`split_utils.py_docs.md`](./split_utils.py_docs.md)
- [`runtime_assert.py_docs.md`](./runtime_assert.py_docs.md)
- [`splitter_base.py_docs.md`](./splitter_base.py_docs.md)
- [`graph_transform_observer.py_docs.md`](./graph_transform_observer.py_docs.md)
- [`fake_tensor_prop.py_docs.md`](./fake_tensor_prop.py_docs.md)


## Cross-References

- **File Documentation**: `pass_manager.py_docs.md`
- **Keyword Index**: `pass_manager.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/passes`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/fx/passes`):

- [`split_utils.py_kw.md_docs.md`](./split_utils.py_kw.md_docs.md)
- [`fake_tensor_prop.py_kw.md_docs.md`](./fake_tensor_prop.py_kw.md_docs.md)
- [`tools_common.py_kw.md_docs.md`](./tools_common.py_kw.md_docs.md)
- [`param_fetch.py_kw.md_docs.md`](./param_fetch.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_manipulation.py_docs.md_docs.md`](./graph_manipulation.py_docs.md_docs.md)
- [`annotate_getitem_nodes.py_docs.md_docs.md`](./annotate_getitem_nodes.py_docs.md_docs.md)
- [`split_module.py_docs.md_docs.md`](./split_module.py_docs.md_docs.md)
- [`pass_manager.py_kw.md_docs.md`](./pass_manager.py_kw.md_docs.md)
- [`tools_common.py_docs.md_docs.md`](./tools_common.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pass_manager.py_docs.md_docs.md`
- **Keyword Index**: `pass_manager.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

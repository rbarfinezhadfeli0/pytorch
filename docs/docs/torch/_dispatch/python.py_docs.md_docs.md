# Documentation: `docs/torch/_dispatch/python.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_dispatch/python.py_docs.md`
- **Size**: 10,305 bytes (10.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_dispatch/python.py`

## File Metadata

- **Path**: `torch/_dispatch/python.py`
- **Size**: 6,750 bytes (6.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: allow-untyped-defs
import itertools
import unittest.mock
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar, Union
from typing_extensions import ParamSpec

import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
from torch._C import DispatchKey


__all__ = ["enable_python_dispatcher", "no_python_dispatcher", "enable_pre_dispatch"]

no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch

CROSSREF_FUNCTIONALIZE = False

_P = ParamSpec("_P")
_T = TypeVar("_T")


def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]:
    """
    Warning: the set of overloads this will report is very subtle.  It is precisely
    the set of torch.ops functions that have actually been accessed from Python
    (e.g., we actually called torch.ops.aten.blah at some point.  This is DIFFERENT
    from the set of registered operators, which will in general be a larger set,
    as this would include all operators which we ran C++ static initializers or
    Python operator registration on.  This does not eagerly populate the list on
    torch.ops.aten; this list is lazy!

    In other words, this is good for traversing over everything that has an
    OpOverload object allocated in Python.  We use it for cache invalidation, but
    don't rely on this list being complete.

    Note that even if we did report all C++ registered overloads, this isn't guaranteed
    to be complete either, as a subsequent lazy load of a library which triggers more
    registrations could add more things to the set.
    """
    for ns in torch.ops:
        packets = getattr(torch.ops, ns)
        for op_name in packets:
            packet = getattr(packets, op_name)
            for overload in packet:
                yield getattr(packet, overload)


@contextmanager
def suspend_functionalization():
    f_tls = torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.Functionalize
    )
    f_rv = torch._C._functionalization_reapply_views_tls()
    if f_tls:
        torch._disable_functionalization()
    try:
        yield
    finally:
        if f_tls:
            torch._enable_functionalization(reapply_views=f_rv)


def check_tensor_metadata_matches(nv, rv, desc):
    assert callable(desc)
    assert nv.size() == rv.size(), f"{desc()}: sizes {nv.size()} != {rv.size()}"
    assert nv.dtype == rv.dtype, f"{desc()}: dtype {nv.dtype} != {rv.dtype}"
    same_strides, idx = torch._prims_common.check_significant_strides(
        nv, rv, only_cuda=False
    )
    assert same_strides, (
        f"{desc()}: strides {nv.stride()} != {rv.stride()} (mismatch at index {idx})"
    )


def check_metadata_matches(n, r, desc):
    assert callable(desc)
    n_vals, _n_spec = pytree.tree_flatten(n)
    r_vals, _r_spec = pytree.tree_flatten(r)
    # TODO: test the specs match; empirically  sometimes we have a tuple
    # on one side and a list on the other
    assert len(n_vals) == len(r_vals), f"{len(n_vals)} != {len(r_vals)}"
    for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
        if not isinstance(rv, torch.Tensor):
            continue
        check_tensor_metadata_matches(nv, rv, lambda: f"{desc()} output {i}")


class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def _fmt(a: object) -> object:
    if isinstance(a, torch.Tensor):
        return Lit(
            f"torch.empty_strided({tuple(a.size())}, {a.stride()}, dtype={a.dtype})"
        )
    else:
        return a


def make_crossref_functionalize(
    op: torch._ops.OpOverload[_P, _T], final_key: DispatchKey
) -> Union[Callable[_P, _T], DispatchKey]:
    from torch._subclasses.fake_tensor import FakeTensorMode

    # This case is pretty weird, suppress it for now
    if op is torch.ops.aten.lift_fresh.default:
        return final_key

    def handler(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        fake_mode = FakeTensorMode()

        def fakeify_defun(t):
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    r = torch._from_functional_tensor(t)
                    # NB: This assumes that the inner tensor sizes/strides match
                    # the outer tensor sizes/strides.  This doesn't necessarily have to
                    # be the case, see discussion at
                    # https://github.com/pytorch/pytorch/pull/87610/files/401ddeda1d769bedc88a12de332c7357b60e51a4#r1007264456
                    assert t.size() == r.size()
                    assert t.stride() == r.stride()
                else:
                    r = t
                # TODO: suppress guards
                return fake_mode.from_tensor(r)
            return t

        def maybe_detach(t):
            if isinstance(t, torch.Tensor):
                return t.detach()
            else:
                return t

        # TODO: This probably does the wrong thing if you're running other
        # substantive modes with the normal op outside here
        with (
            torch.utils._python_dispatch._disable_current_modes(),
            suspend_functionalization(),
        ):
            f_args, f_kwargs = pytree.tree_map(fakeify_defun, (args, kwargs))
            orig_f_args, orig_f_kwargs = pytree.tree_map(
                maybe_detach, (f_args, f_kwargs)
            )
            with fake_mode:
                f_r = op(*f_args, **f_kwargs)  # pyrefly: ignore [invalid-param-spec]
        r = op._op_dk(final_key, *args, **kwargs)

        def desc():
            fmt_args = ", ".join(
                itertools.chain(
                    (repr(pytree.tree_map(_fmt, a)) for a in orig_f_args),
                    (
                        f"{k}={pytree.tree_map(_fmt, v)}"
                        for k, v in orig_f_kwargs.items()
                    ),
                )
            )
            return f"{op}({fmt_args})"

        check_metadata_matches(f_r, r, desc)
        return r

    return handler


# NB: enabling this is slow, don't do it in a hot loop.  This is purely
# for debugging purposes.
@contextmanager
def enable_crossref_functionalize():
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
    try:
        with (
            enable_python_dispatcher(),
            unittest.mock.patch("torch._dispatch.python.CROSSREF_FUNCTIONALIZE", True),
        ):
            yield
    finally:
        for op in all_py_loaded_overloads():
            op._uncache_dispatch(torch._C.DispatchKey.Functionalize)

```



## High-Level Overview

"""    Warning: the set of overloads this will report is very subtle.  It is precisely    the set of torch.ops functions that have actually been accessed from Python    (e.g., we actually called torch.ops.aten.blah at some point.  This is DIFFERENT    from the set of registered operators, which will in general be a larger set,    as this would include all operators which we ran C++ static initializers or    Python operator registration on.  This does not eagerly populate the list on    torch.ops.aten; this list is lazy!    In other words, this is good for traversing over everything that has an    OpOverload object allocated in Python.  We use it for cache invalidation, but    don't rely on this list being complete.    Note that even if we did report all C++ registered overloads, this isn't guaranteed    to be complete either, as a subsequent lazy load of a library which triggers more    registrations could add more things to the set.

This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Lit`

**Functions defined**: `all_py_loaded_overloads`, `suspend_functionalization`, `check_tensor_metadata_matches`, `check_metadata_matches`, `__init__`, `__repr__`, `_fmt`, `make_crossref_functionalize`, `handler`, `fakeify_defun`, `maybe_detach`, `desc`, `enable_crossref_functionalize`

**Key imports**: itertools, unittest.mock, Callable, Iterator, contextmanager, TypeVar, Union, ParamSpec, torch, torch._C, torch._ops, torch.utils._python_dispatch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dispatch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `unittest.mock`
- `collections.abc`: Callable, Iterator
- `contextlib`: contextmanager
- `typing`: TypeVar, Union
- `typing_extensions`: ParamSpec
- `torch`
- `torch._C`
- `torch._ops`
- `torch.utils._python_dispatch`
- `torch.utils._pytree as pytree`
- `torch._subclasses.fake_tensor`: FakeTensorMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_dispatch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `python.py_docs.md`
- **Keyword Index**: `python.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_dispatch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_dispatch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_dispatch`):

- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`python.py_kw.md_docs.md`](./python.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `python.py_docs.md_docs.md`
- **Keyword Index**: `python.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

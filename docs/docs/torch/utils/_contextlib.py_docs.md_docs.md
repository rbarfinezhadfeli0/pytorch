# Documentation: `docs/torch/utils/_contextlib.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/_contextlib.py_docs.md`
- **Size**: 9,388 bytes (9.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/_contextlib.py`

## File Metadata

- **Path**: `torch/utils/_contextlib.py`
- **Size**: 6,402 bytes (6.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Extra utilities for working with context managers that should have been
# in the standard library but are not

import functools
import inspect
import sys
import warnings
from collections.abc import Callable
from typing import Any, cast, overload, TypeVar
from typing_extensions import Self


# Used for annotating the decorator usage of _DecoratorContextManager (e.g.,
# 'no_grad' and 'enable_grad').
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def _wrap_generator(ctx_factory, func):
    """
    Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)

        # Generators are suspended and unsuspended at `yield`, hence we
        # make sure the grad mode is properly set every time the execution
        # flow returns into the wrapped generator and restored when it
        # returns through our `yield` to our caller (see PR #49017).
        try:
            # Issuing `None` to a generator fires it up
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # Forward the response to our caller and get its next request
                    request = yield response

                except GeneratorExit:
                    # Inform the still active generator about its imminent closure
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:  # noqa: B036
                    # Propagate the exception thrown at us by the caller
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # Pass the last request to the generator and get its response
                    with ctx_factory():
                        response = gen.send(request)

        # We let the exceptions raised above by the generator's `.throw` or
        # `.send` methods bubble up to our caller, except for StopIteration
        except StopIteration as e:
            # The generator informed us that it is done: take whatever its
            # returned value (if any) was and indicate that we're done too
            # by returning it (see docs for python's return-statement).
            return e.value

    return generator_context


def context_decorator(ctx, func):
    """
    Like contextlib.ContextDecorator.

    But with the following differences:
    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    if callable(ctx) and hasattr(ctx, "__enter__"):
        raise AssertionError(
            f"Passed in {ctx} is both callable and also a valid context manager "
            "(has __enter__), making it ambiguous which interface to use.  If you "
            "intended to pass a context manager factory, rewrite your call as "
            "context_decorator(lambda: ctx()); if you intended to pass a context "
            "manager directly, rewrite your call as context_decorator(lambda: ctx)"
        )

    if not callable(ctx):

        def ctx_factory():
            return ctx

    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        # pyrefly: ignore [bad-context-manager]
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        if inspect.isclass(orig_func):
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else.",
                FutureWarning,
                stacklevel=2,
            )
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""

    @overload
    def __new__(cls, orig_func: F) -> F: ...  # type: ignore[misc]
    @overload
    def __new__(cls, orig_func: None = None) -> Self: ...

    def __new__(cls, orig_func: F | None = None) -> Self | F:  # type: ignore[misc]
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)

```



## High-Level Overview

"""    Wrap each generator invocation with the context manager factory.    The input should be a function that returns a context manager,    not a context manager itself, to handle one-shot context managers.

This Python file contains 5 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_DecoratorContextManager`, `_NoParamDecoratorContextManager`

**Functions defined**: `_wrap_generator`, `generator_context`, `context_decorator`, `ctx_factory`, `decorate_context`, `__call__`, `__enter__`, `__exit__`, `clone`, `__new__`, `__new__`, `__new__`

**Key imports**: functools, inspect, sys, warnings, Callable, Any, cast, overload, TypeVar, Self


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `inspect`
- `sys`
- `warnings`
- `collections.abc`: Callable
- `typing`: Any, cast, overload, TypeVar
- `typing_extensions`: Self


## Code Patterns & Idioms

### Common Patterns

- **Context Manager**: Implements context manager protocol
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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `_contextlib.py_docs.md`
- **Keyword Index**: `_contextlib.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Context Manager**: Implements context manager protocol
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

Files in the same folder (`docs/torch/utils`):

- [`show_pickle.py_docs.md_docs.md`](./show_pickle.py_docs.md_docs.md)
- [`file_baton.py_docs.md_docs.md`](./file_baton.py_docs.md_docs.md)
- [`_filelock.py_kw.md_docs.md`](./_filelock.py_kw.md_docs.md)
- [`_config_module.py_docs.md_docs.md`](./_config_module.py_docs.md_docs.md)
- [`cpp_extension.py_docs.md_docs.md`](./cpp_extension.py_docs.md_docs.md)
- [`checkpoint.py_docs.md_docs.md`](./checkpoint.py_docs.md_docs.md)
- [`module_tracker.py_kw.md_docs.md`](./module_tracker.py_kw.md_docs.md)
- [`dlpack.py_docs.md_docs.md`](./dlpack.py_docs.md_docs.md)
- [`_import_utils.py_kw.md_docs.md`](./_import_utils.py_kw.md_docs.md)
- [`_traceback.py_kw.md_docs.md`](./_traceback.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_contextlib.py_docs.md_docs.md`
- **Keyword Index**: `_contextlib.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/autograd/grad_mode.py`

## File Metadata

- **Path**: `torch/autograd/grad_mode.py`
- **Size**: 14,193 bytes (13.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any, Union

import torch
from torch.utils._contextlib import (
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
    F,
)


__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "set_multithreading_enabled",
]


class no_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
        If you want to disable forward AD for a computation, you can unpack
        your dual tensors.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @torch.no_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with torch.no_grad():
        ...     a = torch.nn.Parameter(torch.rand(10))
        >>> a.requires_grad
        True
    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        enable_grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     with torch.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        tensor([2.])
        >>> @torch.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with torch.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True
        >>> @torch.enable_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> with torch.no_grad():
        ...     z = tripler(x)
        >>> z.requires_grad
        True

    """

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)


class set_grad_enabled(_DecoratorContextManager):
    r"""Context-manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> _ = torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> _ = torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch.is_grad_enabled()
        self.mode = mode
        torch._C._set_grad_enabled(mode)

    def __call__(self, orig_func: F) -> F:
        torch._C._set_grad_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)

    def __str__(self) -> str:
        return f"{torch.typename(self)}(mode={self.mode})"

    def __repr__(self) -> str:
        return str(self)

    def clone(self) -> "set_grad_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class inference_mode(_DecoratorContextManager):
    r"""Context manager that enables or disables inference mode.

    InferenceMode is analogous to :class:`~no_grad` and should be used
    when you are certain your operations will not interact with autograd
    (e.g., during data loading or model evaluation). Compared to
    :class:`~no_grad`, it removes additional overhead by disabling view
    tracking and version counter bumps. It is also more restrictive, in
    that tensors created in this mode cannot be used in computations
    recorded by autograd.

    This context manager is thread-local; it does not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        Inference mode is one of several mechanisms that can locally enable
        or disable gradients. See :ref:`locally-disable-grad-doc` for a
        comparison. If avoiding the use of tensors created in inference mode
        in autograd-tracked regions is difficult, consider benchmarking your
        code with and without inference mode to weigh the performance benefits
        against the trade-offs. You can always use :class:`~no_grad` instead.

    .. note::
       Unlike some other mechanisms that locally enable or disable grad,
       entering inference_mode also disables :ref:`forward-mode AD <forward-mode-ad>`.

    .. warning::
        `inference_mode` does NOT automatically set the model to evaluation mode.
        For proper inference behavior (e.g., disabling dropout, using running statistics
        in batch normalization), you must explicitly set your model to evaluation mode using
        `model.eval()` in addition to using this context manager.

    Args:
        mode (bool or function): Either a boolean flag to enable or disable
            inference mode, or a Python function to decorate with inference
            mode enabled.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> # xdoctest: +SKIP("want string isn't quite right")
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...     return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False
        >>> @torch.inference_mode()
        ... def doubler(x):
        ...     return x * 2
        >>> out = doubler(x)
        >>> out.requires_grad
        False

    """

    def __init__(self, mode: bool = True) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.mode = mode

    def __new__(cls, mode=True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self._inference_mode_context = torch._C._InferenceMode(self.mode)
        self._inference_mode_context.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._inference_mode_context.__exit__(exc_type, exc_value, traceback)

    def clone(self) -> "inference_mode":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


def _enter_inference_mode(mode):
    mode_context = torch._C._InferenceMode(mode)
    mode_context.__enter__()
    return mode_context


def _exit_inference_mode(mode):
    mode.__exit__(None, None, None)


class set_multithreading_enabled(_DecoratorContextManager):
    r"""Context-manager that sets multithreaded backwards on or off.

    ``set_multithreading_enabled`` will enable or disable multithreaded backwards based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable multithreaded backwards (``True``), or disable
                     (``False``).

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_multithreading_enabled()
        torch._C._set_multithreading_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_multithreading_enabled(self.prev)

    def clone(self) -> "set_multithreading_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class _force_original_view_tracking(_DecoratorContextManager):
    r"""Context-manager that sets whether or not to always enable view-replay in autograd.

    ``set_view_replay_enabled`` will enable or disable view-replay based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    When a tensor view is mutated, the autograd engine needs to decide whether or not
    to regenerate the "updated view" by either replaying the chain of views from the updated base,
    or with a single call to as_strided.

    If set_view_replay_enabled is set to True, then autograd will always use view replay.
    Otherwise, it will fall back to its existing logic.

    Args:
        mode (bool): Flag whether to enable view-replay (``True``), or disable
                     (``False``).

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_view_replay_enabled()
        torch._C._set_view_replay_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_view_replay_enabled(self.prev)

    def clone(self):
        return self.__class__(self.mode)


class _unsafe_preserve_version_counter(_DecoratorContextManager):
    r"""DO NOT USE THIS UNLESS YOU KNOW EXACTLY WHAT YOU'RE DOING.

    This context manager can lead to arbitrary silent-correctness issues in any other part of your code
    (even the ones not touched directly by the context manager)!

    Ordinarily, autograd will track mutations to tensors by incrementing it's `._version` attribute.
    This is generally important for correctness, as for example, mutating a tensor that autograd has saved
    for the backwards pass can result in incorrect gradients, and autograd uses the version counter to detect
    and error out in this situation.

    However, there are rare instances where it might be useful to hide mutations from autograd. For example:
    if a tensor is very large, and you'd like to free its memory by storing it elsewhere, and re-populate
    the tensor right before it is needed by autograd.

    Args:
        tensor (torch.Tensor): the tensor in question, that you would like to preserve the version counter of.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, tensors: Union[torch.Tensor, tuple[torch.Tensor, ...]]) -> None:
        self.tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tensors
        if not isinstance(self.tensors, tuple):
            raise AssertionError("Expected tensors to be a tuple")
        self.prev_versions = tuple(t._version for t in self.tensors)

    def __enter__(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args) -> None:
        torch._C._autograd._unsafe_set_version_counter(self.tensors, self.prev_versions)

```



## High-Level Overview

r"""Context-manager that disables gradient calculation.    Disabling gradient calculation is useful for inference, when you are sure    that you will not call :meth:`Tensor.backward()`. It will reduce memory    consumption for computations that would otherwise have `requires_grad=True`.    In this mode, the result of every computation will have    `requires_grad=False`, even when the inputs have `requires_grad=True`.    There is an exception! All factory functions, or functions that create    a new Tensor and take a requires_grad kwarg, will NOT be affected by    this mode.    This context manager is thread local; it will not affect computation    in other threads.    Also functions as a decorator.    .. note::        No-grad is one of several mechanisms that can enable or        disable gradients locally see :ref:`locally-disable-grad-doc` for        more information on how they compare.    .. note::        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.        If you want to disable forward AD for a computation, you can unpack        your dual tensors.    Example::        >>> # xdoctest: +SKIP

This Python file contains 7 class(es) and 36 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `no_grad`, `enable_grad`, `set_grad_enabled`, `inference_mode`, `set_multithreading_enabled`, `_force_original_view_tracking`, `_unsafe_preserve_version_counter`

**Functions defined**: `doubler`, `tripler`, `__init__`, `__enter__`, `__exit__`, `doubler`, `tripler`, `__enter__`, `__exit__`, `__init__`, `__call__`, `__enter__`, `__exit__`, `__str__`, `__repr__`, `clone`, `func`, `doubler`, `__init__`, `__new__`

**Key imports**: Any, Union, torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, Union
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`function.py_docs.md`](./function.py_docs.md)
- [`variable.py_docs.md`](./variable.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`forward_ad.py_docs.md`](./forward_ad.py_docs.md)
- [`gradcheck.py_docs.md`](./gradcheck.py_docs.md)
- [`functional.py_docs.md`](./functional.py_docs.md)
- [`profiler.py_docs.md`](./profiler.py_docs.md)
- [`anomaly_mode.py_docs.md`](./anomaly_mode.py_docs.md)
- [`profiler_util.py_docs.md`](./profiler_util.py_docs.md)


## Cross-References

- **File Documentation**: `grad_mode.py_docs.md`
- **Keyword Index**: `grad_mode.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

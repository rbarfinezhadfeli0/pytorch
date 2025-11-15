# Documentation: `torch/autograd/anomaly_mode.py`

## File Metadata

- **Path**: `torch/autograd/anomaly_mode.py`
- **Size**: 4,964 bytes (4.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""Autograd anomaly mode."""

import warnings

import torch


__all__ = ["detect_anomaly", "set_detect_anomaly"]


class detect_anomaly:
    r"""Context-manager that enable anomaly detection for the autograd engine.

    This does two things:

    - Running the forward pass with detection enabled will allow the backward
      pass to print the traceback of the forward operation that created the failing
      backward function.
    - If ``check_nan`` is ``True``, any backward computation that generate "nan"
      value will raise an error. Default ``True``.

    .. warning::
        This mode should be enabled only for debugging as the different tests
        will slow down your program execution.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ANOMALY)
        >>> import torch
        >>> from torch import autograd
        >>> class MyFunc(autograd.Function):
        ...     @staticmethod
        ...     def forward(ctx, inp):
        ...         return inp.clone()
        ...
        ...     @staticmethod
        ...     def backward(ctx, gO):
        ...         # Error during the backward pass
        ...         raise RuntimeError("Some error in backward")
        ...         return gO.clone()
        >>> def run_fn(a):
        ...     out = MyFunc.apply(a)
        ...     return out.sum()
        >>> inp = torch.rand(10, 10, requires_grad=True)
        >>> out = run_fn(inp)
        >>> out.backward()
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward
        >>> with autograd.detect_anomaly():
        ...     inp = torch.rand(10, 10, requires_grad=True)
        ...     out = run_fn(inp)
        ...     out.backward()
            Traceback of forward call that caused the error:
              File "tmp.py", line 53, in <module>
                out = run_fn(inp)
              File "tmp.py", line 44, in run_fn
                out = MyFunc.apply(a)
            Traceback (most recent call last):
              File "<stdin>", line 4, in <module>
              File "/your/pytorch/install/torch/_tensor.py", line 93, in backward
                torch.autograd.backward(self, gradient, retain_graph, create_graph)
              File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
                allow_unreachable=True)  # allow_unreachable flag
              File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
                return self._forward_cls.backward(self, *args)
              File "<stdin>", line 8, in backward
            RuntimeError: Some error in backward

    """

    def __init__(self, check_nan=True) -> None:  # noqa: D107
        self.prev = torch.is_anomaly_enabled()
        self.check_nan = check_nan
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        warnings.warn(
            "Anomaly Detection has been enabled. "
            "This mode will increase the runtime "
            "and should only be enabled for debugging.",
            stacklevel=2,
        )

    def __enter__(self) -> None:  # noqa: D105
        torch.set_anomaly_enabled(True, self.check_nan)

    def __exit__(self, *args: object) -> None:  # noqa: D105
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)


class set_detect_anomaly:
    r"""Context-manager that sets the anomaly detection for the autograd engine on or off.

    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    See ``detect_anomaly`` above for details of the anomaly detection behaviour.

    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
        check_nan (bool): Flag whether to raise an error when the backward
                          generate "nan"

    """

    def __init__(self, mode: bool, check_nan: bool = True) -> None:  # noqa: D107
        self.prev = torch.is_anomaly_enabled()
        self.prev_check_nan = torch.is_anomaly_check_nan_enabled()
        torch.set_anomaly_enabled(mode, check_nan)

    def __enter__(self) -> None:  # noqa: D105
        pass

    def __exit__(self, *args: object) -> None:  # noqa: D105
        torch.set_anomaly_enabled(self.prev, self.prev_check_nan)

```



## High-Level Overview

r"""Autograd anomaly mode."""import warningsimport torch__all__ = ["detect_anomaly", "set_detect_anomaly"]class detect_anomaly:

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `detect_anomaly`, `MyFunc`, `set_detect_anomaly`

**Functions defined**: `forward`, `backward`, `run_fn`, `__init__`, `__enter__`, `__exit__`, `__init__`, `__enter__`, `__exit__`

**Key imports**: warnings, torch, torch, autograd


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`function.py_docs.md`](./function.py_docs.md)
- [`variable.py_docs.md`](./variable.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`forward_ad.py_docs.md`](./forward_ad.py_docs.md)
- [`gradcheck.py_docs.md`](./gradcheck.py_docs.md)
- [`functional.py_docs.md`](./functional.py_docs.md)
- [`profiler.py_docs.md`](./profiler.py_docs.md)
- [`profiler_util.py_docs.md`](./profiler_util.py_docs.md)


## Cross-References

- **File Documentation**: `anomaly_mode.py_docs.md`
- **Keyword Index**: `anomaly_mode.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/jit/_async.py_docs.md`

## File Metadata

- **Path**: `docs/torch/jit/_async.py_docs.md`
- **Size**: 6,740 bytes (6.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/jit/_async.py`

## File Metadata

- **Path**: `torch/jit/_async.py`
- **Size**: 3,827 bytes (3.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
"""Async API.

This module contains the API for parallelism in TorchScript, notably:
    * torch.jit.fork
    * torch.jit.wait

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import torch
from torch._jit_internal import Future
from torch.jit._builtins import _register_builtin
from torch.utils import set_module


set_module(Future, "torch.jit")


def fork(func, *args, **kwargs):
    r"""
    Create an asynchronous task executing `func` and a reference to the value of the result of this execution.

    `fork` will return immediately, so the return value of `func` may not have been computed yet. To force completion
    of the task and access the return value invoke `torch.jit.wait` on the Future. `fork` invoked
    with a `func` which returns `T` is typed as `torch.jit.Future[T]`. `fork` calls can be arbitrarily
    nested, and may be invoked with positional and keyword arguments.
    Asynchronous execution will only occur when run in TorchScript. If run in pure python,
    `fork` will not execute in parallel. `fork` will also not execute in parallel when invoked
    while tracing, however the `fork` and `wait` calls will be captured in the exported IR Graph.

    .. warning::
        `fork` tasks will execute non-deterministically. We recommend only spawning
        parallel fork tasks for pure functions that do not modify their inputs,
        module attributes, or global state.

    Args:
        func (callable or torch.nn.Module):  A Python function or `torch.nn.Module`
            that will be invoked. If executed in TorchScript, it will execute asynchronously,
            otherwise it will not. Traced invocations of fork will be captured in the IR.
        ``*args``, ``**kwargs``: arguments to invoke `func` with.
    Returns:
        `torch.jit.Future[T]`: a reference to the execution of `func`. The value `T`
        can only be accessed by forcing completion of `func` through `torch.jit.wait`.

    Example (fork a free function):

    .. code-block:: python

        import torch
        from torch import Tensor


        def foo(a: Tensor, b: int) -> Tensor:
            return a + b


        def bar(a):
            fut: torch.jit.Future[Tensor] = torch.jit.fork(foo, a, b=2)
            return torch.jit.wait(fut)


        script_bar = torch.jit.script(bar)
        input = torch.tensor(2)
        # only the scripted version executes asynchronously
        assert script_bar(input) == bar(input)
        # trace is not run asynchronously, but fork is captured in IR
        graph = torch.jit.trace(bar, (input,)).graph
        assert "fork" in str(graph)

    Example (fork a module method):

    .. code-block:: python

        import torch
        from torch import Tensor


        class AddMod(torch.nn.Module):
            def forward(self, a: Tensor, b: int):
                return a + b


        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super(self).__init__()
                self.mod = AddMod()

            def forward(self, input):
                fut = torch.jit.fork(self.mod, a, b=2)
                return torch.jit.wait(fut)


        input = torch.tensor(2)
        mod = Mod()
        assert mod(input) == torch.jit.script(mod).forward(input)
    """
    return torch._C.fork(func, *args, **kwargs)


def wait(future):
    r"""
    Force completion of a `torch.jit.Future[T]` asynchronous task, returning the result of the task.

    See :func:`~fork` for docs and examples.
    Args:
        future (torch.jit.Future[T]): an asynchronous task reference, created through `torch.jit.fork`
    Returns:
        `T`: the return value of the completed task
    """
    return torch._C.wait(future)


_register_builtin(wait, "aten::wait")

```



## High-Level Overview

"""Async API.This module contains the API for parallelism in TorchScript, notably:    * torch.jit.fork    * torch.jit.waitThis is not intended to be imported directly; please use the exposedfunctionalities in `torch.jit`.

This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AddMod`, `Mod`

**Functions defined**: `fork`, `foo`, `bar`, `forward`, `__init__`, `forward`, `wait`

**Key imports**: torch, Future, _register_builtin, set_module, torch, Tensor, torch, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._jit_internal`: Future
- `torch.jit._builtins`: _register_builtin
- `torch.utils`: set_module


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_decompositions.py_docs.md`](./_decompositions.py_docs.md)
- [`_dataclass_impls.py_docs.md`](./_dataclass_impls.py_docs.md)
- [`quantized.py_docs.md`](./quantized.py_docs.md)
- [`frontend.py_docs.md`](./frontend.py_docs.md)
- [`_builtins.py_docs.md`](./_builtins.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)
- [`_state.py_docs.md`](./_state.py_docs.md)
- [`_await.py_docs.md`](./_await.py_docs.md)


## Cross-References

- **File Documentation**: `_async.py_docs.md`
- **Keyword Index**: `_async.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/jit`):

- [`_check.py_kw.md_docs.md`](./_check.py_kw.md_docs.md)
- [`_shape_functions.py_docs.md_docs.md`](./_shape_functions.py_docs.md_docs.md)
- [`_trace.py_kw.md_docs.md`](./_trace.py_kw.md_docs.md)
- [`_logging.py_docs.md_docs.md`](./_logging.py_docs.md_docs.md)
- [`_async.py_kw.md_docs.md`](./_async.py_kw.md_docs.md)
- [`_state.py_docs.md_docs.md`](./_state.py_docs.md_docs.md)
- [`_decomposition_utils.py_kw.md_docs.md`](./_decomposition_utils.py_kw.md_docs.md)
- [`frontend.py_docs.md_docs.md`](./frontend.py_docs.md_docs.md)
- [`_check.py_docs.md_docs.md`](./_check.py_docs.md_docs.md)
- [`_script.pyi_docs.md_docs.md`](./_script.pyi_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_async.py_docs.md_docs.md`
- **Keyword Index**: `_async.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

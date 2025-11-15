# Documentation: `torch/_dynamo/backends/registry.py`

## File Metadata

- **Path**: `torch/_dynamo/backends/registry.py`
- **Size**: 5,555 bytes (5.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
This module implements TorchDynamo's backend registry system for managing compiler backends.

The registry provides a centralized way to register, discover and manage different compiler
backends that can be used with torch.compile(). It handles:

- Backend registration and discovery through decorators and entry points
- Lazy loading of backend implementations
- Lookup and validation of backend names
- Categorization of backends using tags (debug, experimental, etc.)

Key components:
- CompilerFn: Type for backend compiler functions that transform FX graphs
- _BACKENDS: Registry mapping backend names to entry points
- _COMPILER_FNS: Registry mapping backend names to loaded compiler functions

Example usage:
    @register_backend
    def my_compiler(fx_graph, example_inputs):
        # Transform FX graph into optimized implementation
        return compiled_fn

    # Use registered backend
    torch.compile(model, backend="my_compiler")

The registry also supports discovering backends through setuptools entry points
in the "torch_dynamo_backends" group. Example:
```
setup.py
---
from setuptools import setup

setup(
    name='my_torch_backend',
    version='0.1',
    packages=['my_torch_backend'],
    entry_points={
        'torch_dynamo_backends': [
            # name = path to entry point of backend implementation
            'my_compiler = my_torch_backend.compiler:my_compiler_function',
        ],
    },
)
```
```
my_torch_backend/compiler.py
---
def my_compiler_function(fx_graph, example_inputs):
    # Transform FX graph into optimized implementation
    return compiled_fn
```
Using `my_compiler` backend:
```
import torch

model = ...  # Your PyTorch model
optimized_model = torch.compile(model, backend="my_compiler")
```
"""

import functools
import logging
from collections.abc import Callable, Sequence
from importlib.metadata import EntryPoint
from typing import Any, Optional, Protocol, Union

import torch
from torch import fx


log = logging.getLogger(__name__)


class CompiledFn(Protocol):
    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


CompilerFn = Callable[[fx.GraphModule, list[torch.Tensor]], CompiledFn]

_BACKENDS: dict[str, Optional[EntryPoint]] = {}
_COMPILER_FNS: dict[str, CompilerFn] = {}


def register_backend(
    compiler_fn: Optional[CompilerFn] = None,
    name: Optional[str] = None,
    tags: Sequence[str] = (),
) -> Callable[..., Any]:
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if compiler_fn is None:
        # @register_backend(name="") syntax
        return functools.partial(register_backend, name=name, tags=tags)  # type: ignore[return-value]
    assert callable(compiler_fn)
    name = name or compiler_fn.__name__
    assert name not in _COMPILER_FNS, f"duplicate name: {name}"
    if compiler_fn not in _BACKENDS:
        _BACKENDS[name] = None
    _COMPILER_FNS[name] = compiler_fn
    compiler_fn._tags = tuple(tags)  # type: ignore[attr-defined]
    return compiler_fn


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def lookup_backend(compiler_fn: Union[str, CompilerFn]) -> CompilerFn:
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        if compiler_fn not in _BACKENDS:
            from ..exc import InvalidBackend

            raise InvalidBackend(name=compiler_fn)

        if compiler_fn not in _COMPILER_FNS:
            entry_point = _BACKENDS[compiler_fn]
            if entry_point is not None:
                register_backend(compiler_fn=entry_point.load(), name=compiler_fn)
        compiler_fn = _COMPILER_FNS[compiler_fn]
    return compiler_fn


# NOTE: can't type this due to public api mismatch; follow up with dev team
def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:  # type: ignore[no-untyped-def]
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags_set = set(exclude_tags or ())

    backends = [
        name
        for name in _BACKENDS
        if name not in _COMPILER_FNS
        or not exclude_tags_set.intersection(_COMPILER_FNS[name]._tags)  # type: ignore[attr-defined]
    ]
    return sorted(backends)


@functools.cache
def _lazy_import() -> None:
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    from ..repro.after_dynamo import dynamo_minifier_backend

    assert dynamo_minifier_backend is not None

    _discover_entrypoint_backends()


@functools.cache
def _discover_entrypoint_backends() -> None:
    # importing here so it will pick up the mocked version in test_backends.py
    from importlib.metadata import entry_points

    group_name = "torch_dynamo_backends"
    eps = entry_points(group=group_name)
    eps_dict = {name: eps[name] for name in eps.names}
    for backend_name in eps_dict:
        _BACKENDS[backend_name] = eps_dict[backend_name]

```



## High-Level Overview

"""This module implements TorchDynamo's backend registry system for managing compiler backends.The registry provides a centralized way to register, discover and manage different compilerbackends that can be used with torch.compile(). It handles:- Backend registration and discovery through decorators and entry points- Lazy loading of backend implementations- Lookup and validation of backend names- Categorization of backends using tags (debug, experimental, etc.)Key components:- CompilerFn: Type for backend compiler functions that transform FX graphs- _BACKENDS: Registry mapping backend names to entry points- _COMPILER_FNS: Registry mapping backend names to loaded compiler functionsExample usage:    @register_backend    def my_compiler(fx_graph, example_inputs):        # Transform FX graph into optimized implementation        return compiled_fn    # Use registered backend    torch.compile(model, backend="my_compiler")The registry also supports discovering backends through setuptools entry pointsin the "torch_dynamo_backends" group. Example:```setup.py---from setuptools import setupsetup(    name='my_torch_backend',    version='0.1',    packages=['my_torch_backend'],    entry_points={        'torch_dynamo_backends': [            # name = path to entry point of backend implementation            'my_compiler = my_torch_backend.compiler:my_compiler_function',        ],    },)``````my_torch_backend/compiler.py---def my_compiler_function(fx_graph, example_inputs):    # Transform FX graph into optimized implementation    return compiled_fn

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CompiledFn`

**Functions defined**: `my_compiler`, `my_compiler_function`, `__call__`, `register_backend`, `lookup_backend`, `list_backends`, `_lazy_import`, `_discover_entrypoint_backends`

**Key imports**: setup, torch, functools, logging, Callable, Sequence, EntryPoint, Any, Optional, Protocol, Union, torch, fx, InvalidBackend


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `setuptools`: setup
- `torch`
- `functools`
- `logging`
- `collections.abc`: Callable, Sequence
- `importlib.metadata`: EntryPoint
- `typing`: Any, Optional, Protocol, Union
- `..exc`: InvalidBackend
- `..`: backends
- `..utils`: import_submodule
- `..repro.after_dynamo`: dynamo_minifier_backend


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/_dynamo/backends`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`onnxrt.py_docs.md`](./onnxrt.py_docs.md)
- [`cudagraphs.py_docs.md`](./cudagraphs.py_docs.md)
- [`debugging.py_docs.md`](./debugging.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`torchxla.py_docs.md`](./torchxla.py_docs.md)
- [`tensorrt.py_docs.md`](./tensorrt.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)
- [`tvm.py_docs.md`](./tvm.py_docs.md)


## Cross-References

- **File Documentation**: `registry.py_docs.md`
- **Keyword Index**: `registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

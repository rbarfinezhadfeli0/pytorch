# Documentation: `torch/jit/__init__.py`

## File Metadata

- **Path**: `torch/jit/__init__.py`
- **Size**: 8,366 bytes (8.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch._C

# These are imported so users can access them from the `torch.jit` module
from torch._jit_internal import (
    _Await,
    _drop,
    _IgnoreContextManager,
    _isinstance,
    _overload,
    _overload_method,
    export,
    Final,
    Future,
    ignore,
    is_scripting,
    unused,
)
from torch.jit._async import fork, wait
from torch.jit._await import _awaitable, _awaitable_nowait, _awaitable_wait
from torch.jit._decomposition_utils import _register_decomposition
from torch.jit._freeze import freeze, optimize_for_inference, run_frozen_optimizations
from torch.jit._fuser import (
    fuser,
    last_executed_optimized_graph,
    optimized_execution,
    set_fusion_strategy,
)
from torch.jit._ir_utils import _InsertPoint
from torch.jit._script import (
    _ScriptProfile,
    _unwrap_optional,
    Attribute,
    CompilationUnit,
    interface,
    RecursiveScriptClass,
    RecursiveScriptModule,
    script,
    script_method,
    ScriptFunction,
    ScriptModule,
    ScriptWarning,
)
from torch.jit._serialization import (
    jit_module_from_flatbuffer,
    load,
    save,
    save_jit_module_to_flatbuffer,
)
from torch.jit._trace import (
    _flatten,
    _get_trace_graph,
    _script_if_tracing,
    _unique_state_dict,
    is_tracing,
    ONNXTracedModule,
    TopLevelTracedModule,
    trace,
    trace_module,
    TracedModule,
    TracerWarning,
    TracingCheckError,
)
from torch.utils import set_module


__all__ = [
    "Attribute",
    "CompilationUnit",
    "Error",
    "Future",
    "ScriptFunction",
    "ScriptModule",
    "annotate",
    "enable_onednn_fusion",
    "export",
    "export_opnames",
    "fork",
    "freeze",
    "interface",
    "ignore",
    "isinstance",
    "load",
    "onednn_fusion_enabled",
    "optimize_for_inference",
    "save",
    "script",
    "script_if_tracing",
    "set_fusion_strategy",
    "strict_fusion",
    "trace",
    "trace_module",
    "unused",
    "wait",
]

# For backwards compatibility
_fork = fork
_wait = wait
_set_fusion_strategy = set_fusion_strategy


def export_opnames(m):
    r"""
    Generate new bytecode for a Script module.

    Returns what the op list would be for a Script Module based off the current code base.

    If you have a LiteScriptModule and want to get the currently present
    list of ops call _export_operator_list instead.
    """
    return torch._C._export_opnames(m._c)


# torch.jit.Error
Error = torch._C.JITException
set_module(Error, "torch.jit")
# This is not perfect but works in common cases
Error.__name__ = "Error"
Error.__qualname__ = "Error"


# for use in python if using annotate
def annotate(the_type, the_value):
    """Use to give type of `the_value` in TorchScript compiler.

    This method is a pass-through function that returns `the_value`, used to hint TorchScript
    compiler the type of `the_value`. It is a no-op when running outside of TorchScript.

    Though TorchScript can infer correct type for most Python expressions, there are some cases where
    type inference can be wrong, including:

    - Empty containers like `[]` and `{}`, which TorchScript assumes to be container of `Tensor`
    - Optional types like `Optional[T]` but assigned a valid value of type `T`, TorchScript would assume
      it is type `T` rather than `Optional[T]`

    Note that `annotate()` does not help in `__init__` method of `torch.nn.Module` subclasses because it
    is executed in eager mode. To annotate types of `torch.nn.Module` attributes,
    use :meth:`~torch.jit.Attribute` instead.

    Example:

    .. testcode::

        import torch
        from typing import Dict

        @torch.jit.script
        def fn():
            # Telling TorchScript that this empty dictionary is a (str -> int) dictionary
            # instead of default dictionary type of (str -> Tensor).
            d = torch.jit.annotate(Dict[str, int], {})

            # Without `torch.jit.annotate` above, following statement would fail because of
            # type mismatch.
            d["name"] = 20

    .. testcleanup::

        del fn

    Args:
        the_type: Python type that should be passed to TorchScript compiler as type hint for `the_value`
        the_value: Value or expression to hint type for.

    Returns:
        `the_value` is passed back as return value.
    """
    return the_value


def script_if_tracing(fn):
    """
    Compiles ``fn`` when it is first called during tracing.

    ``torch.jit.script`` has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Args:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `torch.jit.script` is returned.
        Otherwise, the original function `fn` is returned.
    """
    return _script_if_tracing(fn)


# for torch.jit.isinstance
def isinstance(obj, target_type):
    """
    Provide container type refinement in TorchScript.

    It can refine parameterized containers of the List, Dict, Tuple, and Optional types. E.g. ``List[str]``,
    ``Dict[str, List[torch.Tensor]]``, ``Optional[Tuple[int,str,int]]``. It can also
    refine basic types such as bools and ints that are available in TorchScript.

    Args:
        obj: object to refine the type of
        target_type: type to try to refine obj to
    Returns:
        ``bool``: True if obj was successfully refined to the type of target_type,
            False otherwise with no new type refinement


    Example (using ``torch.jit.isinstance`` for type refinement):
    .. testcode::

        import torch
        from typing import Any, Dict, List

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input: Any): # note the Any type
                if torch.jit.isinstance(input, List[torch.Tensor]):
                    for t in input:
                        y = t.clamp(0, 0.5)
                elif torch.jit.isinstance(input, Dict[str, str]):
                    for val in input.values():
                        print(val)

        m = torch.jit.script(MyModule())
        x = [torch.rand(3,3), torch.rand(4,3)]
        m(x)
        y = {"key1":"val1","key2":"val2"}
        m(y)
    """
    return _isinstance(obj, target_type)


class strict_fusion:
    """
    Give errors if not all nodes have been fused in inference, or symbolically differentiated in training.

    Example:
    Forcing fusion of additions.

    .. code-block:: python

        @torch.jit.script
        def foo(x):
            with torch.jit.strict_fusion():
                return x + x + x

    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            warnings.warn("Only works in script mode", stacklevel=2)

    def __enter__(self):
        pass

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        pass


# Context manager for globally hiding source ranges when printing graphs.
# Note that these functions are exposed to Python as static members of the
# Graph class, so mypy checks need to be skipped.
@contextmanager
def _hide_source_ranges() -> Iterator[None]:
    old_enable_source_ranges = torch._C.Graph.global_print_source_ranges  # type: ignore[attr-defined]
    try:
        torch._C.Graph.set_global_print_source_ranges(False)  # type: ignore[attr-defined]
        yield
    finally:
        torch._C.Graph.set_global_print_source_ranges(old_enable_source_ranges)  # type: ignore[attr-defined]


def enable_onednn_fusion(enabled: bool) -> None:
    """Enable or disables onednn JIT fusion based on the parameter `enabled`."""
    torch._C._jit_set_llga_enabled(enabled)


def onednn_fusion_enabled():
    """Return whether onednn JIT fusion is enabled."""
    return torch._C._jit_llga_enabled()


del Any

if not torch._C._jit_init():
    raise RuntimeError("JIT initialization failed")

```



## High-Level Overview


This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModule`, `strict_fusion`

**Functions defined**: `export_opnames`, `annotate`, `fn`, `script_if_tracing`, `isinstance`, `__init__`, `forward`, `foo`, `__init__`, `__enter__`, `__exit__`, `_hide_source_ranges`, `enable_onednn_fusion`, `onednn_fusion_enabled`

**Key imports**: warnings, Iterator, contextmanager, Any, torch._C, fork, wait, _awaitable, _awaitable_nowait, _awaitable_wait, _register_decomposition, freeze, optimize_for_inference, run_frozen_optimizations, _InsertPoint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `warnings`
- `collections.abc`: Iterator
- `contextlib`: contextmanager
- `typing`: Any
- `torch._C`
- `torch.jit._async`: fork, wait
- `torch.jit._await`: _awaitable, _awaitable_nowait, _awaitable_wait
- `torch.jit._decomposition_utils`: _register_decomposition
- `torch.jit._freeze`: freeze, optimize_for_inference, run_frozen_optimizations
- `torch.jit._ir_utils`: _InsertPoint
- `torch.utils`: set_module
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Asynchronous Programming**: Uses async/await
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

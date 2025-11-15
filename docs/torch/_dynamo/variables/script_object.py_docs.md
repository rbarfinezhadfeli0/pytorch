# Documentation: `torch/_dynamo/variables/script_object.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/script_object.py`
- **Size**: 5,957 bytes (5.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
This module implements variable tracking for TorchScript objects during Dynamo tracing.

The TorchScriptObjectVariable class provides specialized handling for TorchScript
objects with strong safety guarantees by:
- Enforcing method-call-only access to prevent unsafe attribute manipulation
- Converting graph breaks into hard errors via _raise_hard_error_if_graph_break
- Proper proxy and source tracking for TorchScript method calls
- Integration with higher-order operators for method call handling

Key safety features:
- Strict validation that only method calls are allowed (no direct attribute access)
- Immediate error reporting for potentially unsafe operations
- Proper source tracking for debugging and guard installation
- Safe handling of TorchScript object method calls through torchbind

The module ensures that TorchScript objects are handled safely during tracing
by limiting operations to known-safe patterns and failing fast for unsafe usage.
"""

import functools
from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._guards import Source
from torch._library.opaque_object import is_opaque_type, OpaqueTypeStr
from torch.fx.proxy import Proxy

from .. import graph_break_hints
from ..exc import unimplemented, UnsafeScriptObjectError, Unsupported
from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _raise_hard_error_if_graph_break(
    reason: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def deco(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                raise UnsafeScriptObjectError(e.msg) from e

        return graph_break_as_hard_error

    return deco


class TorchScriptObjectVariable(UserDefinedObjectVariable):
    _fake_script_object_cache: dict[int, "TorchScriptObjectVariable"] = {}

    @classmethod
    def is_matching_cls(cls, user_cls: type) -> bool:
        return issubclass(user_cls, torch.ScriptObject) or is_opaque_type(user_cls)

    @staticmethod
    def create(proxy: Proxy, value: Any, **options: Any) -> "TorchScriptObjectVariable":
        return TorchScriptObjectVariable(proxy, value, **options)

    def __init__(self, proxy: Proxy, value: Any, source: Source, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.proxy = proxy
        self.proxy.node.meta["example_value"] = value
        self.source = source

    def as_proxy(self) -> Proxy:
        return self.proxy

    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if getattr(self.value, "script_class_name", "") == OpaqueTypeStr:
            unimplemented(
                gb_type="Attempted to access attributes/methods on an OpaqueObject",
                context=f"value={self.value}, attr={name}",
                explanation="Attribute/method access of OpaqueObjects is not supported.",
                hints=[
                    "Use custom operators instead of direct attribute/method access.",
                ],
            )

        from torch._higher_order_ops.torchbind import call_torchbind

        from ..source import AttrSource
        from .higher_order_ops import TorchHigherOrderOperatorVariable

        method = getattr(self.value, name, None)
        if method is None:
            unimplemented(
                gb_type="FakeScriptObject missing method implementation",
                context=f"value={self.value}, method={name}",
                explanation=f"TorchScript object {self.value} doesn't define the method {name}.",
                hints=[
                    f"Ensure the method {name} is implemented in {self.value}.",
                    *graph_break_hints.USER_ERROR,
                ],
            )

        if not callable(method):
            unimplemented(
                gb_type="Attempted to access non-callable attribute of TorchScript object",
                context=f"value={self.value}, method={name}",
                explanation="Attribute accesses of TorchScript objects to non-callable attributes are not supported.",
                hints=[
                    "Use method calls instead of attribute access.",
                ],
            )
        assert self.source is not None
        return TorchHigherOrderOperatorVariable.make(
            call_torchbind,
            source=AttrSource(self.source, name),
            script_obj_var=self,
            method_name=name,
        )

    # We only support method calls on script objects. Interpreting the bytecodes
    # should go through var_getattr then call_function instead of call_method.
    #
    # However, it's possible for call_method to be used directly e.g. for __setattr__.
    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        unimplemented(
            gb_type="Weird method call on TorchScript object",
            context=f"value={self.value}, method={name}",
            explanation=(
                f"This particular method call ({name}) is not supported (e.g. calling `__setattr__`). "
                "Most method calls to TorchScript objects should be supported."
            ),
            hints=[
                "Avoid calling this method.",
            ],
        )

```



## High-Level Overview

"""This module implements variable tracking for TorchScript objects during Dynamo tracing.The TorchScriptObjectVariable class provides specialized handling for TorchScriptobjects with strong safety guarantees by:- Enforcing method-call-only access to prevent unsafe attribute manipulation- Converting graph breaks into hard errors via _raise_hard_error_if_graph_break- Proper proxy and source tracking for TorchScript method calls- Integration with higher-order operators for method call handlingKey safety features:- Strict validation that only method calls are allowed (no direct attribute access)- Immediate error reporting for potentially unsafe operations- Proper source tracking for debugging and guard installation- Safe handling of TorchScript object method calls through torchbindThe module ensures that TorchScript objects are handled safely during tracingby limiting operations to known-safe patterns and failing fast for unsafe usage.

This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TorchScriptObjectVariable`

**Functions defined**: `_raise_hard_error_if_graph_break`, `deco`, `graph_break_as_hard_error`, `is_matching_cls`, `create`, `__init__`, `as_proxy`, `var_getattr`, `call_method`

**Key imports**: functools, Callable, Iterable, Any, TYPE_CHECKING, TypeVar, ParamSpec, torch, Source, is_opaque_type, OpaqueTypeStr, Proxy, graph_break_hints, unimplemented, UnsafeScriptObjectError, Unsupported


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `collections.abc`: Callable, Iterable
- `typing`: Any, TYPE_CHECKING, TypeVar
- `typing_extensions`: ParamSpec
- `torch`
- `torch._guards`: Source
- `torch._library.opaque_object`: is_opaque_type, OpaqueTypeStr
- `torch.fx.proxy`: Proxy
- `..`: graph_break_hints
- `..exc`: unimplemented, UnsafeScriptObjectError, Unsupported
- `.base`: VariableTracker
- `.user_defined`: UserDefinedObjectVariable
- `torch._dynamo.symbolic_convert`: InstructionTranslator
- `torch._higher_order_ops.torchbind`: call_torchbind
- `..source`: AttrSource
- `.higher_order_ops`: TorchHigherOrderOperatorVariable


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/_dynamo/variables`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`nn_module.py_docs.md`](./nn_module.py_docs.md)
- [`higher_order_ops.py_docs.md`](./higher_order_ops.py_docs.md)
- [`tensor.py_docs.md`](./tensor.py_docs.md)
- [`torch.py_docs.md`](./torch.py_docs.md)
- [`constant.py_docs.md`](./constant.py_docs.md)
- [`dicts.py_docs.md`](./dicts.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`lists.py_docs.md`](./lists.py_docs.md)


## Cross-References

- **File Documentation**: `script_object.py_docs.md`
- **Keyword Index**: `script_object.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

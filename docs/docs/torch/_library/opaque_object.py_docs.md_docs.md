# Documentation: `docs/torch/_library/opaque_object.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_library/opaque_object.py_docs.md`
- **Size**: 9,980 bytes (9.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_library/opaque_object.py`

## File Metadata

- **Path**: `torch/_library/opaque_object.py`
- **Size**: 6,542 bytes (6.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any, NewType, Optional

import torch

from .fake_class_registry import FakeScriptObject, register_fake_class


@register_fake_class("aten::OpaqueObject")
class FakeOpaqueObject:
    def __init__(self) -> None:
        pass

    @classmethod
    def __obj_unflatten__(cls, flattened_ctx: dict[str, Any]) -> None:
        raise RuntimeError(
            "FakeOpaqueObject should not be created through __obj_unflatten__ "
            "and should be special handled. Please file an issue to Github."
        )


OpaqueTypeStr = "__torch__.torch.classes.aten.OpaqueObject"

OpaqueType = NewType("OpaqueType", torch._C.ScriptObject)


def make_opaque(payload: Any = None) -> torch._C.ScriptObject:
    """
    Creates an opaque object which stores the given Python object.
    This opaque object can be passed to any custom operator as an argument.
    The Python object can then be accessed from the opaque object using the `get_payload()` API.
    The opaque object has `._type()`
    "__torch__.torch.classes.aten.OpaqueObject", which should be the type used
    when creating custom operator schemas.

    Args:
        payload (Any): The Python object to store in the opaque object. This can
        be empty, and can be set with `set_payload()` later.

    Returns:
        torch._C.ScriptObject: The opaque object that stores the given Python object.

    Example:

        >>> import random
        >>> import torch
        >>> from torch._library.opaque_object import (
        ...     make_opaque,
        ...     get_payload,
        ...     set_payload,
        ... )
        >>>
        >>> class RNGState:
        >>>     def __init__(self, seed):
        >>>         self.rng = random.Random(seed)
        >>>
        >>> rng = RNGState(0)
        >>> obj = make_opaque()
        >>> set_payload(obj, rng)
        >>>
        >>> assert get_payload(obj) == rng
        >>>
        >>> lib = torch.library.Library("mylib", "FRAGMENT")
        >>>
        >>> torch.library.define(
        >>>     "mylib::noisy_inject",
        >>>     "(Tensor x, __torch__.torch.classes.aten.OpaqueObject obj) -> Tensor",
        >>>     tags=torch.Tag.pt2_compliant_tag,
        >>>     lib=lib,
        >>> )
        >>>
        >>> @torch.library.impl(
        >>>     "mylib::noisy_inject", "CompositeExplicitAutograd", lib=lib
        >>> )
        >>> def noisy_inject(x: torch.Tensor, obj: torch._C.ScriptObject) -> torch.Tensor:
        >>>     rng_state = get_payload(obj)
        >>>     assert isinstance(rng_state, RNGState)
        >>>     out = x.clone()
        >>>     for i in range(out.numel()):
        >>>         out.view(-1)[i] += rng_state.rng.random()
        >>>     return out
        >>>
        >>> print(torch.ops.mylib.noisy_inject(torch.ones(3), obj))
    """
    return torch._C._make_opaque_object(payload)


def get_payload(opaque_object: torch._C.ScriptObject) -> Any:
    """
    Retrieves the Python object stored in the given opaque object.

    Args:
        torch._C.ScriptObject: The opaque object that stores the given Python object.

    Returns:
        payload (Any): The Python object stored in the opaque object. This can
        be set with `set_payload()`.
    """
    if isinstance(opaque_object, FakeScriptObject):
        raise ValueError(
            "get_payload: this function was called with a FakeScriptObject "
            "implying that you are calling get_payload inside of a fake kernel."
            "The fake kernel should not depend on the contents of the "
            "OpaqueObject at all, so we're erroring out. If you need this"
            "functionality, consider creating a custom TorchBind Object instead"
            "(but note that this is more difficult)."
        )
    if not (
        isinstance(opaque_object, torch._C.ScriptObject)
        and opaque_object._type().qualified_name() == OpaqueTypeStr
    ):
        type_ = (
            opaque_object._type().qualified_name()
            if isinstance(opaque_object, torch._C.ScriptObject)
            else type(opaque_object)
        )
        raise ValueError(
            f"Tried to get the payload from a non-OpaqueObject of type `{type_}`"
        )
    return torch._C._get_opaque_object_payload(opaque_object)


def set_payload(opaque_object: torch._C.ScriptObject, payload: Any) -> None:
    """
    Sets the Python object stored in the given opaque object.

    Args:
        torch._C.ScriptObject: The opaque object that stores the given Python object.
        payload (Any): The Python object to store in the opaque object.
    """
    if isinstance(opaque_object, FakeScriptObject):
        raise ValueError(
            "set_payload: this function was called with a FakeScriptObject "
            "implying that you are calling get_payload inside of a fake kernel."
            "The fake kernel should not depend on the contents of the "
            "OpaqueObject at all, so we're erroring out. If you need this"
            "functionality, consider creating a custom TorchBind Object instead"
            "(but note that this is more difficult)."
        )

    if not (
        isinstance(opaque_object, torch._C.ScriptObject)
        and opaque_object._type().qualified_name() == OpaqueTypeStr
    ):
        type_ = (
            opaque_object._type().qualified_name()
            if isinstance(opaque_object, torch._C.ScriptObject)
            else type(opaque_object)
        )
        raise ValueError(
            f"Tried to get the payload from a non-OpaqueObject of type `{type_}`"
        )
    torch._C._set_opaque_object_payload(opaque_object, payload)


_OPAQUE_TYPES: dict[Any, str] = {}


def register_opaque_type(cls: Any, name: Optional[str] = None) -> None:
    """
    Registers the given type as an opaque type which allows this to be consumed
    by a custom operator.

    Args:
        cls (type): The class to register as an opaque type.
        name (str): A unique qualified name of the type.
    """
    if name is None:
        name = cls.__name__

    if "." in name:
        # The schema_type_parser will break up types with periods
        raise ValueError(
            f"Unable to accept name, {name}, for this opaque type as it contains a '.'"
        )
    _OPAQUE_TYPES[cls] = name

    torch._C._register_opaque_type(name)


def is_opaque_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque type.
    """
    if cls not in _OPAQUE_TYPES:
        return False

    return torch._C._is_opaque_type_registered(_OPAQUE_TYPES[cls])

```



## High-Level Overview

"""    Creates an opaque object which stores the given Python object.    This opaque object can be passed to any custom operator as an argument.    The Python object can then be accessed from the opaque object using the `get_payload()` API.    The opaque object has `._type()`    "__torch__.torch.classes.aten.OpaqueObject", which should be the type used    when creating custom operator schemas.    Args:        payload (Any): The Python object to store in the opaque object. This can        be empty, and can be set with `set_payload()` later.    Returns:        torch._C.ScriptObject: The opaque object that stores the given Python object.    Example:        >>> import random        >>> import torch        >>> from torch._library.opaque_object import (        ...     make_opaque,        ...     get_payload,        ...     set_payload,        ... )

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FakeOpaqueObject`, `RNGState`

**Functions defined**: `__init__`, `__obj_unflatten__`, `make_opaque`, `__init__`, `noisy_inject`, `get_payload`, `set_payload`, `register_opaque_type`, `is_opaque_type`

**Key imports**: Any, NewType, Optional, torch, FakeScriptObject, register_fake_class, random, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_library`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, NewType, Optional
- `torch`
- `.fake_class_registry`: FakeScriptObject, register_fake_class
- `random`


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

Files in the same folder (`torch/_library`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`effects.py_docs.md`](./effects.py_docs.md)
- [`autograd.py_docs.md`](./autograd.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`fake_impl.py_docs.md`](./fake_impl.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`fake_profile.py_docs.md`](./fake_profile.py_docs.md)
- [`simple_registry.py_docs.md`](./simple_registry.py_docs.md)
- [`infer_schema.py_docs.md`](./infer_schema.py_docs.md)


## Cross-References

- **File Documentation**: `opaque_object.py_docs.md`
- **Keyword Index**: `opaque_object.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_library`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_library`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_library`):

- [`fake_impl.py_docs.md_docs.md`](./fake_impl.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`opaque_object.py_kw.md_docs.md`](./opaque_object.py_kw.md_docs.md)
- [`infer_schema.py_kw.md_docs.md`](./infer_schema.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`custom_ops.py_docs.md_docs.md`](./custom_ops.py_docs.md_docs.md)
- [`simple_registry.py_kw.md_docs.md`](./simple_registry.py_kw.md_docs.md)
- [`autograd.py_kw.md_docs.md`](./autograd.py_kw.md_docs.md)
- [`triton.py_kw.md_docs.md`](./triton.py_kw.md_docs.md)
- [`simple_registry.py_docs.md_docs.md`](./simple_registry.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `opaque_object.py_docs.md_docs.md`
- **Keyword Index**: `opaque_object.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

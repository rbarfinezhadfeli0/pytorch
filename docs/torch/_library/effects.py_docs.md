# Documentation: `torch/_library/effects.py`

## File Metadata

- **Path**: `torch/_library/effects.py`
- **Size**: 2,035 bytes (1.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from enum import Enum
from typing import Optional

import torch


class EffectType(Enum):
    ORDERED = "Ordered"


from torch._library.utils import RegistrationHandle


class EffectHolder:
    """A holder where one can register an effect impl to."""

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        self._set_default_effect()

    def _set_default_effect(self) -> None:
        self._effect: Optional[EffectType] = None

        # If the op contains a ScriptObject input, we want to mark it as having effects
        namespace, opname = torch._library.utils.parse_namespace(self.qualname)
        split = opname.split(".")
        if len(split) > 1:
            assert len(split) == 2, (
                f"Tried to split {opname} based on '.' but found more than 1 '.'"
            )
            opname, overload = split
        else:
            overload = ""

        if namespace == "higher_order":
            return

        opname = f"{namespace}::{opname}"
        if torch._C._get_operation_overload(opname, overload) is not None:
            # Since we call this when destroying the library, sometimes the
            # schema will be gone already at that time.
            schema = torch._C._get_schema(opname, overload)
            for arg in schema.arguments:
                if isinstance(arg.type, torch.ClassType):
                    self._effect = EffectType.ORDERED
                    return

    @property
    def effect(self) -> Optional[EffectType]:
        return self._effect

    @effect.setter
    def effect(self, _):
        raise RuntimeError("Unable to directly set kernel.")

    def register(self, effect: Optional[EffectType]) -> RegistrationHandle:
        """Register an effect

        Returns a RegistrationHandle that one can use to de-register this
        effect.
        """
        self._effect = effect

        def deregister_effect():
            self._set_default_effect()

        handle = RegistrationHandle(deregister_effect)
        return handle

```



## High-Level Overview

"""A holder where one can register an effect impl to."""    def __init__(self, qualname: str):        self.qualname: str = qualname        self._set_default_effect()    def _set_default_effect(self) -> None:        self._effect: Optional[EffectType] = None        # If the op contains a ScriptObject input, we want to mark it as having effects        namespace, opname = torch._library.utils.parse_namespace(self.qualname)        split = opname.split(".")        if len(split) > 1:            assert len(split) == 2, (                f"Tried to split {opname} based on '.' but found more than 1 '.'"            )            opname, overload = split        else:            overload = ""        if namespace == "higher_order":            return        opname = f"{namespace}::{opname}"        if torch._C._get_operation_overload(opname, overload) is not None:            # Since we call this when destroying the library, sometimes the            # schema will be gone already at that time.            schema = torch._C._get_schema(opname, overload)            for arg in schema.arguments:                if isinstance(arg.type, torch.ClassType):                    self._effect = EffectType.ORDERED                    return    @property    def effect(self) -> Optional[EffectType]:        return self._effect

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EffectType`, `EffectHolder`

**Functions defined**: `__init__`, `_set_default_effect`, `effect`, `effect`, `register`, `deregister_effect`

**Key imports**: Enum, Optional, torch, RegistrationHandle


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_library`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `enum`: Enum
- `typing`: Optional
- `torch`
- `torch._library.utils`: RegistrationHandle


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
- [`autograd.py_docs.md`](./autograd.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`fake_impl.py_docs.md`](./fake_impl.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`fake_profile.py_docs.md`](./fake_profile.py_docs.md)
- [`simple_registry.py_docs.md`](./simple_registry.py_docs.md)
- [`opaque_object.py_docs.md`](./opaque_object.py_docs.md)
- [`infer_schema.py_docs.md`](./infer_schema.py_docs.md)


## Cross-References

- **File Documentation**: `effects.py_docs.md`
- **Keyword Index**: `effects.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

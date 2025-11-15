# Documentation: `torch/nested/_internal/nested_int.py`

## File Metadata

- **Path**: `torch/nested/_internal/nested_int.py`
- **Size**: 3,198 bytes (3.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import *  # noqa: F403

import torch
from torch.fx.experimental._constant_symnode import ConstantIntNode


__all__ = ["NestedIntNode"]


# Python version of aten/src/ATen/core/NestedIntSymNodeImpl.cpp
def _eq(lhs: Any, rhs: Any) -> bool:
    return (
        isinstance(lhs, NestedIntNode)
        and isinstance(rhs, NestedIntNode)
        and lhs.t_id == rhs.t_id
        and lhs.coeff == rhs.coeff
    )


def _ge(lhs: Any, rhs: Any) -> bool:
    if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
        if lhs.t_id == rhs.t_id:
            return lhs.coeff >= rhs.coeff
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(lhs, NestedIntNode):
        if rhs.is_constant() and rhs.constant_int() <= 2:
            return True
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(rhs, NestedIntNode):
        if lhs.is_constant() and lhs.constant_int() < 2:
            return False
        raise ValueError("ge: relation is indeterminate")
    else:
        raise ValueError("inputs unsupported")


class NestedIntNode:
    def __init__(self, t_id: int, coeff: int) -> None:
        self.t_id = t_id
        self.coeff = coeff

    def nested_int_coeff(self) -> int:
        return self.coeff

    def maybe_as_int(self) -> Optional[int]:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return True

    def clone(self) -> "NestedIntNode":
        return self

    def _str(self) -> Any:
        if self.coeff == 1:
            return f"j{self.t_id}"
        return f"{self.coeff}*j{self.t_id}"

    def str(self) -> Any:
        return self._str()

    def __str__(self) -> Any:
        return self._str()

    def __repr__(self) -> Any:
        return self._str()

    def _graph_repr(self) -> Any:
        return self._str()

    def mul(self, other: Any) -> "NestedIntNode":
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.t_id, self.coeff * other)

    def eq(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_eq(self, other))

    def ne(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _eq(self, other))

    def gt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(other, self))

    def lt(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(not _ge(self, other))

    def le(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(other, self))

    def ge(self, other: Any) -> Any:
        return torch._C._get_constant_bool_symnode(_ge(self, other))

    def is_symbolic(self) -> bool:
        return False

    def nested_int(self) -> int:
        return self.t_id

    def is_constant(self) -> bool:
        return False

    def wrap_int(self, num: int) -> ConstantIntNode:
        assert type(num) is int
        return ConstantIntNode(num)

```



## High-Level Overview


This Python file contains 1 class(es) and 26 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `NestedIntNode`

**Functions defined**: `_eq`, `_ge`, `__init__`, `nested_int_coeff`, `maybe_as_int`, `is_int`, `is_float`, `is_bool`, `is_nested_int`, `clone`, `_str`, `str`, `__str__`, `__repr__`, `_graph_repr`, `mul`, `eq`, `ne`, `gt`, `lt`

**Key imports**: torch, ConstantIntNode


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nested/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.fx.experimental._constant_symnode`: ConstantIntNode


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

Files in the same folder (`torch/nested/_internal`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ops.py_docs.md`](./ops.py_docs.md)
- [`nested_tensor.py_docs.md`](./nested_tensor.py_docs.md)
- [`sdpa.py_docs.md`](./sdpa.py_docs.md)


## Cross-References

- **File Documentation**: `nested_int.py_docs.md`
- **Keyword Index**: `nested_int.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

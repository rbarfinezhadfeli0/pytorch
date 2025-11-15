# Documentation: `torch/types.py`

## File Metadata

- **Path**: `torch/types.py`
- **Size**: 3,621 bytes (3.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
import os
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from collections.abc import Sequence
from typing import Any, IO, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import Self

# `as` imports have better static analysis support than assignment `ExposedType: TypeAlias = HiddenType`
from torch import (  # noqa: F401
    device as _device,
    DispatchKey,
    dtype as _dtype,
    layout as _layout,
    qscheme as _qscheme,
    Size,
    SymBool,
    SymFloat,
    SymInt,
    Tensor,
)


if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge


__all__ = ["Number", "Device", "FileLike", "Storage"]

# Convenience aliases for common composite types that we need
# to talk about in PyTorch
_TensorOrTensors: TypeAlias = Union[Tensor, Sequence[Tensor]]  # noqa: PYI047
_TensorOrTensorsOrGradEdge: TypeAlias = Union[  # noqa: PYI047
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

_size: TypeAlias = Union[Size, list[int], tuple[int, ...]]  # noqa: PYI042,PYI047
_symsize: TypeAlias = Union[Size, Sequence[Union[int, SymInt]]]  # noqa: PYI042,PYI047
_dispatchkey: TypeAlias = Union[str, DispatchKey]  # noqa: PYI042,PYI047

# int or SymInt
IntLikeType: TypeAlias = Union[int, SymInt]
# float or SymFloat
FloatLikeType: TypeAlias = Union[float, SymFloat]
# bool or SymBool
BoolLikeType: TypeAlias = Union[bool, SymBool]

py_sym_types = (SymInt, SymFloat, SymBool)  # left un-annotated intentionally
PySymType: TypeAlias = Union[SymInt, SymFloat, SymBool]

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = Union[int, float, bool]
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

FileLike: TypeAlias = Union[str, os.PathLike[str], IO[bytes]]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device: TypeAlias = Union[_device, str, int, None]


# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        raise NotImplementedError

    def _new_shared(self, size: int) -> Self:
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> Self:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> Self:
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> Self:
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self:
        raise NotImplementedError

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Storage`

**Functions defined**: `__deepcopy__`, `_new_shared`, `_write_file`, `element_size`, `is_shared`, `share_memory_`, `nbytes`, `cpu`, `data_ptr`, `from_file`, `_new_with_file`

**Key imports**: os, Sequence, Any, IO, TYPE_CHECKING, TypeAlias, Union, Self, GradientEdge


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `collections.abc`: Sequence
- `typing`: Any, IO, TYPE_CHECKING, TypeAlias, Union
- `typing_extensions`: Self
- `torch.autograd.graph`: GradientEdge


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_docs.py_docs.md`](./_tensor_docs.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `types.py_docs.md`
- **Keyword Index**: `types.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

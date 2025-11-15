# Documentation: `torch/torch_version.py`

## File Metadata

- **Path**: `torch/torch_version.py`
- **Size**: 2,530 bytes (2.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterable
from typing import Any

from torch._vendor.packaging.version import InvalidVersion, Version
from torch.version import __version__ as internal_version


__all__ = ["TorchVersion"]


class TorchVersion(str):
    """A string with magic powers to compare to both Version and iterables!
    Prior to 1.10.0 torch.__version__ was stored as a str and so many did
    comparisons against torch.__version__ as if it were a str. In order to not
    break them we have TorchVersion which masquerades as a str while also
    having the ability to compare against both packaging.version.Version as
    well as tuples of values, eg. (1, 2, 1)
    Examples:
        Comparing a TorchVersion object to a Version object
            TorchVersion('1.10.0a') > Version('1.10.0a')
        Comparing a TorchVersion object to a Tuple object
            TorchVersion('1.10.0a') > (1, 2)    # 1.2
            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1
        Comparing a TorchVersion object against a string
            TorchVersion('1.10.0a') > '1.2'
            TorchVersion('1.10.0a') > '1.2.1'
    """

    __slots__ = ()

    # fully qualified type names here to appease mypy
    def _convert_to_version(self, inp: Any) -> Any:
        if isinstance(inp, Version):
            return inp
        elif isinstance(inp, str):
            return Version(inp)
        elif isinstance(inp, Iterable):
            # Ideally this should work for most cases by attempting to group
            # the version tuple, assuming the tuple looks (MAJOR, MINOR, ?PATCH)
            # Examples:
            #   * (1)         -> Version("1")
            #   * (1, 20)     -> Version("1.20")
            #   * (1, 20, 1)  -> Version("1.20.1")
            return Version(".".join(str(item) for item in inp))
        else:
            raise InvalidVersion(inp)

    def _cmp_wrapper(self, cmp: Any, method: str) -> bool:
        try:
            return getattr(Version(self), method)(self._convert_to_version(cmp))
        except BaseException as e:
            if not isinstance(e, InvalidVersion):
                raise
            # Fall back to regular string comparison if dealing with an invalid
            # version like 'parrot'
            return getattr(super(), method)(cmp)


for cmp_method in ["__gt__", "__lt__", "__eq__", "__ge__", "__le__"]:
    setattr(
        TorchVersion,
        cmp_method,
        lambda x, y, method=cmp_method: x._cmp_wrapper(y, method),
    )

__version__ = TorchVersion(internal_version)

```



## High-Level Overview

"""A string with magic powers to compare to both Version and iterables!    Prior to 1.10.0 torch.__version__ was stored as a str and so many did    comparisons against torch.__version__ as if it were a str. In order to not    break them we have TorchVersion which masquerades as a str while also    having the ability to compare against both packaging.version.Version as    well as tuples of values, eg. (1, 2, 1)    Examples:        Comparing a TorchVersion object to a Version object            TorchVersion('1.10.0a') > Version('1.10.0a')        Comparing a TorchVersion object to a Tuple object            TorchVersion('1.10.0a') > (1, 2)    # 1.2            TorchVersion('1.10.0a') > (1, 2, 1) # 1.2.1        Comparing a TorchVersion object against a string            TorchVersion('1.10.0a') > '1.2'            TorchVersion('1.10.0a') > '1.2.1'

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TorchVersion`

**Functions defined**: `_convert_to_version`, `_cmp_wrapper`

**Key imports**: Iterable, Any, InvalidVersion, Version, __version__ as internal_version


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable
- `typing`: Any
- `torch._vendor.packaging.version`: InvalidVersion, Version
- `torch.version`: __version__ as internal_version


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
- [`types.py_docs.md`](./types.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `torch_version.py_docs.md`
- **Keyword Index**: `torch_version.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

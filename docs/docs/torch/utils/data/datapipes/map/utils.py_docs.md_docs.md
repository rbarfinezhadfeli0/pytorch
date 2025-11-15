# Documentation: `docs/torch/utils/data/datapipes/map/utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/map/utils.py_docs.md`
- **Size**: 4,957 bytes (4.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/map/utils.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/map/utils.py`
- **Size**: 1,813 bytes (1.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from torch.utils.data.datapipes.datapipe import MapDataPipe


_T = TypeVar("_T")

__all__ = ["SequenceWrapperMapDataPipe"]


class SequenceWrapperMapDataPipe(MapDataPipe[_T]):
    r"""
    Wraps a sequence object into a MapDataPipe.

    Args:
        sequence: Sequence object to be wrapped into an MapDataPipe
        deepcopy: Option to deepcopy input sequence object

    .. note::
      If ``deepcopy`` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> dp = SequenceWrapper({"a": 100, "b": 200, "c": 300, "d": 400})
        >>> dp["a"]
        100
    """

    sequence: Sequence[_T] | Mapping[Any, _T]

    def __init__(
        self, sequence: Sequence[_T] | Mapping[Any, _T], deepcopy: bool = True
    ) -> None:
        if deepcopy:
            try:
                self.sequence = copy.deepcopy(sequence)
            except TypeError:
                warnings.warn(
                    "The input sequence can not be deepcopied, "
                    "please be aware of in-place modification would affect source data",
                    stacklevel=2,
                )
                self.sequence = sequence
        else:
            self.sequence = sequence

    def __getitem__(self, index: int) -> _T:
        return self.sequence[index]

    def __len__(self) -> int:
        return len(self.sequence)

```



## High-Level Overview

r"""    Wraps a sequence object into a MapDataPipe.    Args:        sequence: Sequence object to be wrapped into an MapDataPipe        deepcopy: Option to deepcopy input sequence object    .. note::      If ``deepcopy`` is set to False explicitly, users should ensure      that data pipeline doesn't contain any in-place operations over      the iterable instance, in order to prevent data inconsistency      across iterations.    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.map import SequenceWrapper        >>> dp = SequenceWrapper(range(10))        >>> list(dp)        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]        >>> dp = SequenceWrapper({"a": 100, "b": 200, "c": 300, "d": 400})        >>> dp["a"]        100

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SequenceWrapperMapDataPipe`

**Functions defined**: `__init__`, `__getitem__`, `__len__`

**Key imports**: copy, warnings, Mapping, Sequence, Any, TypeVar, MapDataPipe, SequenceWrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/map`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `warnings`
- `collections.abc`: Mapping, Sequence
- `typing`: Any, TypeVar
- `torch.utils.data.datapipes.datapipe`: MapDataPipe
- `torchdata.datapipes.map`: SequenceWrapper


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/utils/data/datapipes/map`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`combining.py_docs.md`](./combining.py_docs.md)
- [`callable.py_docs.md`](./callable.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`combinatorics.py_docs.md`](./combinatorics.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/data/datapipes/map`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/data/datapipes/map`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/torch/utils/data/datapipes/map`):

- [`combining.py_docs.md_docs.md`](./combining.py_docs.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`combinatorics.py_docs.md_docs.md`](./combinatorics.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`combining.py_kw.md_docs.md`](./combining.py_kw.md_docs.md)
- [`combinatorics.py_kw.md_docs.md`](./combinatorics.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`grouping.py_docs.md_docs.md`](./grouping.py_docs.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md_docs.md`
- **Keyword Index**: `utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

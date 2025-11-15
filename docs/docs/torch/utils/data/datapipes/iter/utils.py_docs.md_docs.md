# Documentation: `docs/torch/utils/data/datapipes/iter/utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/utils.py_docs.md`
- **Size**: 5,472 bytes (5.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/utils.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/utils.py`
- **Size**: 2,109 bytes (2.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import copy
import warnings
from collections.abc import Iterable, Iterator, Sized
from typing import TypeVar

from torch.utils.data.datapipes.datapipe import IterDataPipe


_T = TypeVar("_T")

__all__ = ["IterableWrapperIterDataPipe"]


class IterableWrapperIterDataPipe(IterDataPipe[_T]):
    r"""
    Wraps an iterable object to create an IterDataPipe.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in ``iter()``.

    .. note::
        If ``deepcopy`` is explicitly set to ``False``, users should ensure
        that the data pipeline doesn't contain any in-place operations over
        the iterable instance to prevent data inconsistency across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    def __init__(self, iterable: Iterable[_T], deepcopy: bool = True) -> None:
        self.iterable = iterable
        self.deepcopy = deepcopy

    def __iter__(self) -> Iterator[_T]:
        source_data = self.iterable
        if self.deepcopy:
            try:
                source_data = copy.deepcopy(self.iterable)
            # For the case that data cannot be deep-copied,
            # all in-place operations will affect iterable variable.
            # When this DataPipe is iterated second time, it will
            # yield modified items.
            except TypeError:
                warnings.warn(
                    "The input iterable can not be deepcopied, "
                    "please be aware of in-place modification would affect source data.",
                    stacklevel=2,
                )
        yield from source_data

    def __len__(self) -> int:
        if isinstance(self.iterable, Sized):
            return len(self.iterable)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

```



## High-Level Overview

r"""    Wraps an iterable object to create an IterDataPipe.    Args:        iterable: Iterable object to be wrapped into an IterDataPipe        deepcopy: Option to deepcopy input iterable object for each            iterator. The copy is made when the first element is read in ``iter()``.    .. note::        If ``deepcopy`` is explicitly set to ``False``, users should ensure        that the data pipeline doesn't contain any in-place operations over        the iterable instance to prevent data inconsistency across iterations.    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.iter import IterableWrapper        >>> dp = IterableWrapper(range(10))        >>> list(dp)        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IterableWrapperIterDataPipe`

**Functions defined**: `__init__`, `__iter__`, `__len__`

**Key imports**: copy, warnings, Iterable, Iterator, Sized, TypeVar, IterDataPipe, IterableWrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `warnings`
- `collections.abc`: Iterable, Iterator, Sized
- `typing`: TypeVar
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torchdata.datapipes.iter`: IterableWrapper


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

Files in the same folder (`torch/utils/data/datapipes/iter`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`combining.py_docs.md`](./combining.py_docs.md)
- [`callable.py_docs.md`](./callable.py_docs.md)
- [`filelister.py_docs.md`](./filelister.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`selecting.py_docs.md`](./selecting.py_docs.md)
- [`sharding.py_docs.md`](./sharding.py_docs.md)
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/data/datapipes/iter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/utils/data/datapipes/iter`):

- [`combining.py_docs.md_docs.md`](./combining.py_docs.md_docs.md)
- [`selecting.py_docs.md_docs.md`](./selecting.py_docs.md_docs.md)
- [`sharding.py_kw.md_docs.md`](./sharding.py_kw.md_docs.md)
- [`filelister.py_kw.md_docs.md`](./filelister.py_kw.md_docs.md)
- [`fileopener.py_kw.md_docs.md`](./fileopener.py_kw.md_docs.md)
- [`routeddecoder.py_docs.md_docs.md`](./routeddecoder.py_docs.md_docs.md)
- [`selecting.py_kw.md_docs.md`](./selecting.py_kw.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`filelister.py_docs.md_docs.md`](./filelister.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `utils.py_docs.md_docs.md`
- **Keyword Index**: `utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/utils/data/datapipes/iter/fileopener.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/fileopener.py_docs.md`
- **Size**: 6,734 bytes (6.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/fileopener.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/fileopener.py`
- **Size**: 2,889 bytes (2.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterable, Iterator
from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames


__all__ = [
    "FileOpenerIterDataPipe",
]


@functional_datapipe("open_files")
class FileOpenerIterDataPipe(IterDataPipe[tuple[str, IOBase]]):
    r"""
    Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``r``, other options are
            ``b`` for reading in binary mode and ``t`` for text mode.
        encoding: An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python's GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import (
        ...     FileLister,
        ...     FileOpener,
        ...     StreamReader,
        ... )
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith(".txt"))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [('./abc.txt', 'abc')]
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "r",
        encoding: str | None = None,
        length: int = -1,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.mode: str = mode
        self.encoding: str | None = encoding

        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError(f"Invalid mode {mode}")
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken

        if "b" in mode and encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")

        self.length: int = length

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self) -> Iterator[tuple[str, IOBase]]:
        yield from get_file_binaries_from_pathnames(
            self.datapipe, self.mode, self.encoding
        )

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length

```



## High-Level Overview

r"""    Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).    Args:        datapipe: Iterable datapipe that provides pathnames        mode: An optional string that specifies the mode in which            the file is opened by ``open()``. It defaults to ``r``, other options are            ``b`` for reading in binary mode and ``t`` for text mode.        encoding: An optional string that specifies the encoding of the            underlying file. It defaults to ``None`` to match the default encoding of ``open``.        length: Nominal length of the datapipe    Note:        The opened file handles will be closed by Python's GC periodically. Users can choose        to close them explicitly.    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.iter import (        ...     FileLister,        ...     FileOpener,        ...     StreamReader,        ... )        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith(".txt"))        >>> dp = FileOpener(dp)        >>> dp = StreamReader(dp)        >>> list(dp)        [('./abc.txt', 'abc')]

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FileOpenerIterDataPipe`

**Functions defined**: `__init__`, `__iter__`, `__len__`

**Key imports**: Iterable, Iterator, IOBase, functional_datapipe, IterDataPipe, get_file_binaries_from_pathnames


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable, Iterator
- `io`: IOBase
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torch.utils.data.datapipes.utils.common`: get_file_binaries_from_pathnames


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

Files in the same folder (`torch/utils/data/datapipes/iter`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`combining.py_docs.md`](./combining.py_docs.md)
- [`callable.py_docs.md`](./callable.py_docs.md)
- [`filelister.py_docs.md`](./filelister.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`selecting.py_docs.md`](./selecting.py_docs.md)
- [`sharding.py_docs.md`](./sharding.py_docs.md)
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `fileopener.py_docs.md`
- **Keyword Index**: `fileopener.py_kw.md`
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
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`routeddecoder.py_docs.md_docs.md`](./routeddecoder.py_docs.md_docs.md)
- [`selecting.py_kw.md_docs.md`](./selecting.py_kw.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`filelister.py_docs.md_docs.md`](./filelister.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `fileopener.py_docs.md_docs.md`
- **Keyword Index**: `fileopener.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

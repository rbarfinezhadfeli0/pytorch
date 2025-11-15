# Documentation: `docs/torch/utils/data/datapipes/iter/streamreader.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/streamreader.py_docs.md`
- **Size**: 4,865 bytes (4.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/streamreader.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/streamreader.py`
- **Size**: 1,537 bytes (1.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterator
from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = ["StreamReaderIterDataPipe"]


@functional_datapipe("read_from_stream")
class StreamReaderIterDataPipe(IterDataPipe[tuple[str, bytes]]):
    r"""
    Given IO streams and their label names, yield bytes with label name as tuple.

    (functional name: ``read_from_stream``).

    Args:
        datapipe: Iterable DataPipe provides label/URL and byte stream
        chunk: Number of bytes to be read from stream per iteration.
            If ``None``, all bytes will be read until the EOF.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """

    def __init__(
        self, datapipe: IterDataPipe[tuple[str, IOBase]], chunk: int | None = None
    ) -> None:
        self.datapipe = datapipe
        self.chunk = chunk

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        for furl, stream in self.datapipe:
            while True:
                d = stream.read(self.chunk)
                if not d:
                    stream.close()
                    break
                yield (furl, d)

```



## High-Level Overview

r"""    Given IO streams and their label names, yield bytes with label name as tuple.    (functional name: ``read_from_stream``).    Args:        datapipe: Iterable DataPipe provides label/URL and byte stream        chunk: Number of bytes to be read from stream per iteration.            If ``None``, all bytes will be read until the EOF.    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader        >>> from io import StringIO        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])        >>> list(StreamReader(dp, chunk=1))        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StreamReaderIterDataPipe`

**Functions defined**: `__init__`, `__iter__`

**Key imports**: Iterator, IOBase, functional_datapipe, IterDataPipe, IterableWrapper, StreamReader, StringIO


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterator
- `io`: IOBase
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torchdata.datapipes.iter`: IterableWrapper, StreamReader


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
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `streamreader.py_docs.md`
- **Keyword Index**: `streamreader.py_kw.md`
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

- **File Documentation**: `streamreader.py_docs.md_docs.md`
- **Keyword Index**: `streamreader.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

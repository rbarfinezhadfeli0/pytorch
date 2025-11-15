# Documentation: `docs/torch/utils/data/datapipes/iter/filelister.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/filelister.py_docs.md`
- **Size**: 6,260 bytes (6.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/filelister.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/filelister.py`
- **Size**: 2,554 bytes (2.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Iterator, Sequence

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter.utils import IterableWrapperIterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root


__all__ = ["FileListerIterDataPipe"]


@functional_datapipe("list_files")
class FileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.

    Multiple root directories can be provided (functional name: ``list_files``).

    Args:
        root: Root directory or a sequence of root directories
        masks: Unix style filter string or string list for filtering file name(s)
        recursive: Whether to return pathname from nested directories or not
        abspath: Whether to return relative pathname or absolute pathname
        non_deterministic: Whether to return pathname in sorted order or not.
            If ``False``, the results yielded from each root directory will be sorted
        length: Nominal length of the datapipe

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import FileLister
        >>> dp = FileLister(root=".", recursive=True)
        >>> list(dp)
        ['example.py', './data/data.tar']
    """

    def __init__(
        self,
        root: str | Sequence[str] | IterDataPipe = ".",
        masks: str | list[str] = "",
        *,
        recursive: bool = False,
        abspath: bool = False,
        non_deterministic: bool = False,
        length: int = -1,
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = [root]
        if not isinstance(root, IterDataPipe):
            root = IterableWrapperIterDataPipe(root)
        self.datapipe: IterDataPipe = root
        self.masks: str | list[str] = masks
        self.recursive: bool = recursive
        self.abspath: bool = abspath
        self.non_deterministic: bool = non_deterministic
        self.length: int = length

    def __iter__(self) -> Iterator[str]:
        for path in self.datapipe:
            yield from get_file_pathnames_from_root(
                path, self.masks, self.recursive, self.abspath, self.non_deterministic
            )

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length

```



## High-Level Overview

r"""    Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.    Multiple root directories can be provided (functional name: ``list_files``).    Args:        root: Root directory or a sequence of root directories        masks: Unix style filter string or string list for filtering file name(s)        recursive: Whether to return pathname from nested directories or not        abspath: Whether to return relative pathname or absolute pathname        non_deterministic: Whether to return pathname in sorted order or not.            If ``False``, the results yielded from each root directory will be sorted        length: Nominal length of the datapipe    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.iter import FileLister        >>> dp = FileLister(root=".", recursive=True)        >>> list(dp)        ['example.py', './data/data.tar']

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FileListerIterDataPipe`

**Functions defined**: `__init__`, `__iter__`, `__len__`

**Key imports**: Iterator, Sequence, functional_datapipe, IterDataPipe, IterableWrapperIterDataPipe, get_file_pathnames_from_root, FileLister


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterator, Sequence
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torch.utils.data.datapipes.iter.utils`: IterableWrapperIterDataPipe
- `torch.utils.data.datapipes.utils.common`: get_file_pathnames_from_root
- `torchdata.datapipes.iter`: FileLister


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
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`selecting.py_docs.md`](./selecting.py_docs.md)
- [`sharding.py_docs.md`](./sharding.py_docs.md)
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `filelister.py_docs.md`
- **Keyword Index**: `filelister.py_kw.md`
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


## Cross-References

- **File Documentation**: `filelister.py_docs.md_docs.md`
- **Keyword Index**: `filelister.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/utils/data/datapipes/map/grouping.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/map/grouping.py`
- **Size**: 2,488 bytes (2.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe


__all__ = ["BatcherMapDataPipe"]


_T = TypeVar("_T")


@functional_datapipe("batch")
class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    r"""
    Create mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> batch_dp = dp.batch(batch_size=2)
        >>> list(batch_dp)
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """

    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        datapipe: MapDataPipe[_T],
        batch_size: int,
        drop_last: bool = False,
        wrapper_class: type[DataChunk] = DataChunk,
    ) -> None:
        if batch_size <= 0:
            raise AssertionError("Batch size is required to be larger than 0!")
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __getitem__(self, index) -> DataChunk:
        batch: list = []
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        try:
            batch.extend(self.datapipe[i] for i in indices)
            return self.wrapper_class(batch)
        except IndexError as e:
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.") from e

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

```



## High-Level Overview

r"""    Create mini-batches of data (functional name: ``batch``).    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.    Args:        datapipe: Iterable DataPipe being batched        batch_size: The size of each batch        drop_last: Option to drop the last batch if it's not full    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.map import SequenceWrapper        >>> dp = SequenceWrapper(range(10))        >>> batch_dp = dp.batch(batch_size=2)        >>> list(batch_dp)        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BatcherMapDataPipe`

**Functions defined**: `__init__`, `__getitem__`, `__len__`

**Key imports**: Sized, TypeVar, functional_datapipe, DataChunk, MapDataPipe, SequenceWrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/map`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Sized
- `typing`: TypeVar
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: DataChunk, MapDataPipe
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
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`combinatorics.py_docs.md`](./combinatorics.py_docs.md)


## Cross-References

- **File Documentation**: `grouping.py_docs.md`
- **Keyword Index**: `grouping.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

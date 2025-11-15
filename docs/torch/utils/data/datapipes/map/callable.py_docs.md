# Documentation: `torch/utils/data/datapipes/map/callable.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/map/callable.py`
- **Size**: 1,933 bytes (1.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


__all__ = ["MapperMapDataPipe", "default_fn"]


_T_co = TypeVar("_T_co", covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


@functional_datapipe("map")
class MapperMapDataPipe(MapDataPipe[_T_co]):
    r"""
    Apply the input function over each item from the source DataPipe (functional name: ``map``).

    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source MapDataPipe
        fn: Function being applied to each item

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> def add_one(x):
        ...     return x + 1
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(add_one)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    datapipe: MapDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        _check_unpickable_fn(fn)
        self.fn = fn  # type: ignore[assignment]

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return len(self.datapipe)

    def __getitem__(self, index) -> _T_co:
        return self.fn(self.datapipe[index])

```



## High-Level Overview

r"""    Apply the input function over each item from the source DataPipe (functional name: ``map``).    The function can be any regular Python function or partial object. Lambda    function is not recommended as it is not supported by pickle.    Args:        datapipe: Source MapDataPipe        fn: Function being applied to each item    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper        >>> def add_one(x):        ...     return x + 1        >>> dp = SequenceWrapper(range(10))        >>> map_dp_1 = dp.map(add_one)        >>> list(map_dp_1)        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)        >>> list(map_dp_2)        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MapperMapDataPipe`

**Functions defined**: `default_fn`, `add_one`, `__init__`, `__len__`, `__getitem__`

**Key imports**: Callable, TypeVar, functional_datapipe, MapDataPipe, _check_unpickable_fn, SequenceWrapper, Mapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/map`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: TypeVar
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: MapDataPipe
- `torch.utils.data.datapipes.utils.common`: _check_unpickable_fn
- `torchdata.datapipes.map`: SequenceWrapper, Mapper


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`combinatorics.py_docs.md`](./combinatorics.py_docs.md)


## Cross-References

- **File Documentation**: `callable.py_docs.md`
- **Keyword Index**: `callable.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

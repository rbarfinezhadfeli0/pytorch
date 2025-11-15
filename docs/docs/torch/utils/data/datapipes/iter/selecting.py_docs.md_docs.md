# Documentation: `docs/torch/utils/data/datapipes/iter/selecting.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/selecting.py_docs.md`
- **Size**: 6,860 bytes (6.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/selecting.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/selecting.py`
- **Size**: 3,308 bytes (3.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Callable, Iterator
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    StreamWrapper,
    validate_input_col,
)


__all__ = ["FilterIterDataPipe"]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("filter")
class FilterIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        input_col: Index or indices of data which ``filter_fn`` is applied, such as:

            - ``None`` as default to apply ``filter_fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """

    datapipe: IterDataPipe[_T_co]
    filter_fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        filter_fn: Callable,
        input_col=None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(filter_fn)
        self.filter_fn = filter_fn  # type: ignore[assignment]

        self.input_col = input_col
        validate_input_col(filter_fn, input_col)

    def _apply_filter_fn(self, data) -> bool:
        if self.input_col is None:
            return self.filter_fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.filter_fn(*args)
        else:
            return self.filter_fn(data[self.input_col])

    def __iter__(self) -> Iterator[_T_co]:
        for data in self.datapipe:
            condition, filtered = self._returnIfTrue(data)
            if condition:
                yield filtered
            else:
                StreamWrapper.close_streams(data)

    def _returnIfTrue(self, data: _T) -> tuple[bool, _T]:
        condition = self._apply_filter_fn(data)

        if df_wrapper.is_column(condition):
            # We are operating on DataFrames filter here
            result = []
            for idx, mask in enumerate(df_wrapper.iterate(condition)):
                if mask:
                    result.append(df_wrapper.get_item(data, idx))
            if result:
                return True, df_wrapper.concat(result)
            else:
                return False, None  # type: ignore[return-value]

        if not isinstance(condition, bool):
            raise ValueError(
                "Boolean output is required for `filter_fn` of FilterIterDataPipe, got",
                type(condition),
            )

        return condition, data

```



## High-Level Overview

r"""    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).    Args:        datapipe: Iterable DataPipe being filtered        filter_fn: Customized function mapping an element to a boolean.        input_col: Index or indices of data which ``filter_fn`` is applied, such as:            - ``None`` as default to apply ``filter_fn`` to the data directly.            - Integer(s) is used for list/tuple.            - Key(s) is used for dict.    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.iter import IterableWrapper        >>> def is_even(n):        ...     return n % 2 == 0        >>> dp = IterableWrapper(range(5))        >>> filter_dp = dp.filter(filter_fn=is_even)        >>> list(filter_dp)        [0, 2, 4]

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FilterIterDataPipe`

**Functions defined**: `is_even`, `__init__`, `_apply_filter_fn`, `__iter__`, `_returnIfTrue`

**Key imports**: Callable, Iterator, TypeVar, functional_datapipe, dataframe_wrapper as df_wrapper, IterDataPipe, IterableWrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable, Iterator
- `typing`: TypeVar
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.dataframe`: dataframe_wrapper as df_wrapper
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torchdata.datapipes.iter`: IterableWrapper


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
- [`sharding.py_docs.md`](./sharding.py_docs.md)
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `selecting.py_docs.md`
- **Keyword Index**: `selecting.py_kw.md`
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
- [`sharding.py_kw.md_docs.md`](./sharding.py_kw.md_docs.md)
- [`filelister.py_kw.md_docs.md`](./filelister.py_kw.md_docs.md)
- [`fileopener.py_kw.md_docs.md`](./fileopener.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`routeddecoder.py_docs.md_docs.md`](./routeddecoder.py_docs.md_docs.md)
- [`selecting.py_kw.md_docs.md`](./selecting.py_kw.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`filelister.py_docs.md_docs.md`](./filelister.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `selecting.py_docs.md_docs.md`
- **Keyword Index**: `selecting.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

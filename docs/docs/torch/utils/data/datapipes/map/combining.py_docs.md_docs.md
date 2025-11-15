# Documentation: `docs/torch/utils/data/datapipes/map/combining.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/map/combining.py_docs.md`
- **Size**: 7,111 bytes (6.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/map/combining.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/map/combining.py`
- **Size**: 3,903 bytes (3.81 KB)
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
from torch.utils.data.datapipes.datapipe import MapDataPipe


__all__ = ["ConcaterMapDataPipe", "ZipperMapDataPipe"]

_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("concat")
class ConcaterMapDataPipe(MapDataPipe):
    r"""
    Concatenate multiple Map DataPipes (functional name: ``concat``).

    The new index of is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
        datapipes: Map DataPipes being concatenated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """

    datapipes: tuple[MapDataPipe]

    def __init__(self, *datapipes: MapDataPipe) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __getitem__(self, index) -> _T_co:  # type: ignore[type-var]
        offset = 0
        for dp in self.datapipes:
            # pyrefly: ignore [bad-argument-type]
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                # pyrefly: ignore [bad-argument-type]
                offset += len(dp)
        raise IndexError(f"Index {index} is out of range.")

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return sum(len(dp) for dp in self.datapipes)


@functional_datapipe("zip")
class ZipperMapDataPipe(MapDataPipe[tuple[_T_co, ...]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """

    datapipes: tuple[MapDataPipe[_T_co], ...]

    def __init__(self, *datapipes: MapDataPipe[_T_co]) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes

    def __getitem__(self, index) -> tuple[_T_co, ...]:
        res = []
        for dp in self.datapipes:
            try:
                res.append(dp[index])
            except IndexError as e:
                raise IndexError(
                    f"Index {index} is out of range for one of the input MapDataPipes {dp}."
                ) from e
        return tuple(res)

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return min(len(dp) for dp in self.datapipes)

```



## High-Level Overview

r"""    Concatenate multiple Map DataPipes (functional name: ``concat``).    The new index of is the cumulative sum of source DataPipes.    For example, if there are 2 source DataPipes both with length 5,    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to    elements of the first DataPipe, and 5 to 9 would refer to elements    of the second DataPipe.    Args:        datapipes: Map DataPipes being concatenated    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.map import SequenceWrapper        >>> dp1 = SequenceWrapper(range(3))        >>> dp2 = SequenceWrapper(range(3))        >>> concat_dp = dp1.concat(dp2)        >>> list(concat_dp)        [0, 1, 2, 0, 1, 2]

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConcaterMapDataPipe`, `ZipperMapDataPipe`

**Functions defined**: `__init__`, `__getitem__`, `__len__`, `__init__`, `__getitem__`, `__len__`

**Key imports**: Sized, TypeVar, functional_datapipe, MapDataPipe, SequenceWrapper, SequenceWrapper


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
- [`callable.py_docs.md`](./callable.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)
- [`combinatorics.py_docs.md`](./combinatorics.py_docs.md)


## Cross-References

- **File Documentation**: `combining.py_docs.md`
- **Keyword Index**: `combining.py_kw.md`
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

- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`combinatorics.py_docs.md_docs.md`](./combinatorics.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`combining.py_kw.md_docs.md`](./combining.py_kw.md_docs.md)
- [`combinatorics.py_kw.md_docs.md`](./combinatorics.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`grouping.py_docs.md_docs.md`](./grouping.py_docs.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `combining.py_docs.md_docs.md`
- **Keyword Index**: `combining.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

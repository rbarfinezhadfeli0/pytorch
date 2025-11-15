# Documentation: `docs/torch/utils/data/datapipes/map/combinatorics.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/map/combinatorics.py_docs.md`
- **Size**: 8,168 bytes (7.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/map/combinatorics.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/map/combinatorics.py`
- **Size**: 4,267 bytes (4.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import random
from collections.abc import Iterator
from typing import TypeVar

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


__all__ = ["ShufflerIterDataPipe"]


_T_co = TypeVar("_T_co", covariant=True)


# @functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle().set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        >>> list(shuffle_dp)
        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
        >>> # Reset seed for Shuffler
        >>> shuffle_dp = shuffle_dp.set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    Note:
        Even thought this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
        of data during data-processing.
    """

    datapipe: MapDataPipe[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random

    def __init__(
        self,
        datapipe: MapDataPipe[_T_co],
        *,
        indices: list | None = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # pyrefly: ignore [bad-argument-type]
        self.indices = list(range(len(datapipe))) if indices is None else indices
        self._enabled = True
        self._seed = None
        self._rng = random.Random()
        self._shuffled_indices: list = self.indices

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[_T_co]:
        if not self._enabled:
            for idx in self.indices:
                yield self.datapipe[idx]
        else:
            while self._shuffled_indices:
                idx = self._shuffled_indices.pop()
                yield self.datapipe[idx]

    def reset(self) -> None:
        if self._enabled and self._seed is None:
            self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._rng.seed(self._seed)
        self._seed = None
        self._shuffled_indices = self._rng.sample(self.indices, len(self.indices))

    def __len__(self) -> int:
        # pyrefly: ignore [bad-argument-type]
        return len(self.datapipe)

    def __getstate__(self):
        state = (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            rng_state,
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)


MapDataPipe.register_datapipe_as_function("shuffle", ShufflerIterDataPipe)

```



## High-Level Overview

r"""    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to    set up random seed are different based on :attr:`num_workers`.    For single-process mode (:attr:`num_workers == 0`), the random seed is set before    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed    for each worker process.    Args:        datapipe: MapDataPipe being shuffled        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing    Example:        >>> # xdoctest: +SKIP        >>> from torchdata.datapipes.map import SequenceWrapper        >>> dp = SequenceWrapper(range(10))        >>> shuffle_dp = dp.shuffle().set_seed(0)        >>> list(shuffle_dp)        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]        >>> list(shuffle_dp)        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]        >>> # Reset seed for Shuffler        >>> shuffle_dp = shuffle_dp.set_seed(0)        >>> list(shuffle_dp)        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]    Note:        Even thought this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ShufflerIterDataPipe`

**Functions defined**: `__init__`, `set_shuffle`, `set_seed`, `__iter__`, `reset`, `__len__`, `__getstate__`, `__setstate__`

**Key imports**: random, Iterator, TypeVar, torch, IterDataPipe, MapDataPipe, SequenceWrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/map`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `collections.abc`: Iterator
- `typing`: TypeVar
- `torch`
- `torch.utils.data.datapipes.datapipe`: IterDataPipe, MapDataPipe
- `torchdata.datapipes.map`: SequenceWrapper


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

Files in the same folder (`torch/utils/data/datapipes/map`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`combining.py_docs.md`](./combining.py_docs.md)
- [`callable.py_docs.md`](./callable.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`grouping.py_docs.md`](./grouping.py_docs.md)


## Cross-References

- **File Documentation**: `combinatorics.py_docs.md`
- **Keyword Index**: `combinatorics.py_kw.md`
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
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`grouping.py_kw.md_docs.md`](./grouping.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`combining.py_kw.md_docs.md`](./combining.py_kw.md_docs.md)
- [`combinatorics.py_kw.md_docs.md`](./combinatorics.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`grouping.py_docs.md_docs.md`](./grouping.py_docs.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `combinatorics.py_docs.md_docs.md`
- **Keyword Index**: `combinatorics.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

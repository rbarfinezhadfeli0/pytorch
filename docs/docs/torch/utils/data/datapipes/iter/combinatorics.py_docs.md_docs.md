# Documentation: `docs/torch/utils/data/datapipes/iter/combinatorics.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/combinatorics.py_docs.md`
- **Size**: 9,749 bytes (9.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/combinatorics.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/combinatorics.py`
- **Size**: 6,513 bytes (6.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import random
from collections.abc import Iterator, Sized
from typing import TypeVar

import torch
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.sampler import Sampler, SequentialSampler


__all__ = [
    "SamplerIterDataPipe",
    "ShufflerIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


class SamplerIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

    Args:
        datapipe: IterDataPipe to sample from
        sampler: Sampler class to generate sample elements from input DataPipe.
            Default is :class:`SequentialSampler` for IterDataPipe
    """

    datapipe: IterDataPipe
    sampler: Sampler

    def __init__(
        self,
        datapipe: IterDataPipe,
        sampler: type[Sampler] = SequentialSampler,
        sampler_args: tuple | None = None,
        sampler_kwargs: dict | None = None,
    ) -> None:
        if not isinstance(datapipe, Sized):
            raise AssertionError(
                "Sampler class requires input datapipe implemented `__len__`"
            )
        super().__init__()
        # pyrefly: ignore [bad-assignment]
        self.datapipe = datapipe
        self.sampler_args = () if sampler_args is None else sampler_args
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        self.sampler_kwargs["data_source"] = self.datapipe
        self.sampler = sampler(*self.sampler_args, **self.sampler_kwargs)

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized):
            return len(self.sampler)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("shuffle")
class ShufflerIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Shuffle the input DataPipe with a buffer (functional name: ``shuffle``).

    The buffer with ``buffer_size`` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.

    ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    ``buffer_size`` is required to be greater than or equal to the size of datapipe.

    When it is used with :class:`torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: The IterDataPipe being shuffled
        buffer_size: The buffer size for shuffling (default to ``10000``)
        unbatch_level: Specifies if it is necessary to unbatch source data before
            applying the shuffle

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> shuffle_dp = dp.shuffle()
        >>> list(shuffle_dp)
        [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
    """

    datapipe: IterDataPipe[_T_co]
    buffer_size: int
    _buffer: list[_T_co]
    _enabled: bool
    _seed: int | None
    _rng: random.Random

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        *,
        buffer_size: int = 10000,
        unbatch_level: int = 0,
    ) -> None:
        super().__init__()
        # TODO: Performance optimization
        #       buffer can be a fixed size and remove expensive `append()` and `len()` operations
        self._buffer: list[_T_co] = []
        if buffer_size <= 0:
            raise AssertionError("buffer_size should be larger than 0")
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.buffer_size = buffer_size
        self._enabled = True
        self._seed = None
        self._rng = random.Random()

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[_T_co]:
        if not self._enabled:
            yield from self.datapipe
        else:
            for x in self.datapipe:
                if len(self._buffer) == self.buffer_size:
                    idx = self._rng.randint(0, len(self._buffer) - 1)
                    val, self._buffer[idx] = self._buffer[idx], x
                    yield val
                else:
                    self._buffer.append(x)
            while self._buffer:
                idx = self._rng.randint(0, len(self._buffer) - 1)
                yield self._buffer.pop(idx)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def reset(self) -> None:
        self._buffer = []
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._rng.seed(self._seed)
            self._seed = None

    def __getstate__(self):
        state = (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            self._rng.getstate(),
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    def __del__(self) -> None:
        self._buffer.clear()

```



## High-Level Overview

r"""    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).    Args:        datapipe: IterDataPipe to sample from        sampler: Sampler class to generate sample elements from input DataPipe.            Default is :class:`SequentialSampler` for IterDataPipe

This Python file contains 4 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SamplerIterDataPipe`, `ShufflerIterDataPipe`

**Functions defined**: `__init__`, `__iter__`, `__len__`, `__init__`, `set_shuffle`, `set_seed`, `__iter__`, `__len__`, `reset`, `__getstate__`, `__setstate__`, `__del__`

**Key imports**: random, Iterator, Sized, TypeVar, torch, functional_datapipe, IterDataPipe, Sampler, SequentialSampler, IterableWrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `collections.abc`: Iterator, Sized
- `typing`: TypeVar
- `torch`
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: IterDataPipe
- `torch.utils.data.sampler`: Sampler, SequentialSampler
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
- [`selecting.py_docs.md`](./selecting.py_docs.md)
- [`sharding.py_docs.md`](./sharding.py_docs.md)
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `combinatorics.py_docs.md`
- **Keyword Index**: `combinatorics.py_kw.md`
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

- **File Documentation**: `combinatorics.py_docs.md_docs.md`
- **Keyword Index**: `combinatorics.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

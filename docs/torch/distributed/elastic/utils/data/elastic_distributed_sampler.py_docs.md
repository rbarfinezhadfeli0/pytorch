# Documentation: `torch/distributed/elastic/utils/data/elastic_distributed_sampler.py`

## File Metadata

- **Path**: `torch/distributed/elastic/utils/data/elastic_distributed_sampler.py`
- **Size**: 3,072 bytes (3.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Iterator, Sized
from typing import cast, Optional, TypeVar

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


T = TypeVar("T")

__all__ = ["ElasticDistributedSampler"]


class ElasticDistributedSampler(DistributedSampler[T]):
    """
    Sampler that restricts data loading to a subset of
    the dataset for elastic training.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        start_index (optional):  Which index of the dataset to start sampling from
    """

    def __init__(
        self,
        dataset: Dataset[T],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        start_index: int = 0,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        if not isinstance(dataset, Sized):
            raise TypeError("Dataset must be an instance of collections.abc.Sized")

        # Cast to Sized for mypy
        # pyrefly: ignore [redundant-cast]
        sized_dataset = cast(Sized, dataset)

        if start_index >= len(sized_dataset):
            raise ValueError(
                f"Start index {start_index} should be less than dataset size {len(sized_dataset)}"
            )

        self.start_index = start_index
        sized_dataset = cast(Sized, self.dataset)
        self.num_samples = math.ceil(
            float(len(sized_dataset) - self.start_index) / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        sized_dataset = cast(Sized, self.dataset)
        indices = (
            torch.randperm(len(sized_dataset) - self.start_index, generator=g)
            .add(self.start_index)
            .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

```



## High-Level Overview

"""    Sampler that restricts data loading to a subset of    the dataset for elastic training.    It is especially useful in conjunction with    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each    process can pass a DistributedSampler instance as a DataLoader sampler,    and load a subset of the original dataset that is exclusive to it.    .. note::        Dataset is assumed to be of constant size.    Args:        dataset: Dataset used for sampling.        num_replicas (optional): Number of processes participating in            distributed training.        rank (optional): Rank of the current process within num_replicas.        start_index (optional):  Which index of the dataset to start sampling from

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ElasticDistributedSampler`

**Functions defined**: `__init__`, `__iter__`, `__len__`

**Key imports**: math, Iterator, Sized, cast, Optional, TypeVar, torch, Dataset, DistributedSampler


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/elastic/utils/data`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `collections.abc`: Iterator, Sized
- `typing`: cast, Optional, TypeVar
- `torch`
- `torch.utils.data`: Dataset
- `torch.utils.data.distributed`: DistributedSampler


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/distributed/elastic/utils/data`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`cycling_iterator.py_docs.md`](./cycling_iterator.py_docs.md)


## Cross-References

- **File Documentation**: `elastic_distributed_sampler.py_docs.md`
- **Keyword Index**: `elastic_distributed_sampler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

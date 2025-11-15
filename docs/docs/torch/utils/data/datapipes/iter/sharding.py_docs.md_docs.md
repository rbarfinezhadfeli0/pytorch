# Documentation: `docs/torch/utils/data/datapipes/iter/sharding.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/data/datapipes/iter/sharding.py_docs.md`
- **Size**: 6,614 bytes (6.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/utils/data/datapipes/iter/sharding.py`

## File Metadata

- **Path**: `torch/utils/data/datapipes/iter/sharding.py`
- **Size**: 3,587 bytes (3.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections.abc import Sized
from enum import IntEnum
from typing import NoReturn

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = [
    "SHARDING_PRIORITIES",
    "ShardingFilterIterDataPipe",
]


class SHARDING_PRIORITIES(IntEnum):
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3


class _ShardingIterDataPipe(IterDataPipe):
    def apply_sharding(
        self,
        num_of_instances: int,
        instance_id: int,
        sharding_group: SHARDING_PRIORITIES,
    ) -> NoReturn:
        raise NotImplementedError


@functional_datapipe("sharding_filter")
class ShardingFilterIterDataPipe(_ShardingIterDataPipe):
    r"""
    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).

    After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
    """

    def __init__(
        self, source_datapipe: IterDataPipe, sharding_group_filter=None
    ) -> None:
        self.source_datapipe = source_datapipe
        self.sharding_group_filter = sharding_group_filter
        self.groups: dict[int, tuple[int, int]] = {}
        self.num_of_instances = 1
        self.instance_id = 0
        self._update_num_of_instances()

    def apply_sharding(
        self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT
    ):
        if instance_id >= num_of_instances:
            raise ValueError(
                f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})"
            )
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise RuntimeError(
                    "ShardingFilter cannot mix DEFAULT and non DEFAULT groups"
                )
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()

    def _update_num_of_instances(self) -> None:
        sorted_sharding_groups = [
            self.groups[key]
            for key in sorted(self.groups.keys())
            if self.sharding_group_filter is None or key == self.sharding_group_filter
        ]

        sorted_sharding_groups.reverse()

        self.num_of_instances = 1
        self.instance_id = 0

        for group_num_of_instances, group_instance_id in sorted_sharding_groups:
            self.instance_id += self.num_of_instances * group_instance_id
            self.num_of_instances *= group_num_of_instances

    def __iter__(self):
        for i, item in enumerate(self.source_datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self) -> int:
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe) // self.num_of_instances + (
                1
                if (
                    self.instance_id < len(self.source_datapipe) % self.num_of_instances
                )
                else 0
            )
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

```



## High-Level Overview

r"""    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``).    After ``apply_sharding`` is called, each instance of the DataPipe (on different workers) will have every `n`-th element of the    original DataPipe, where `n` equals to the number of instances.    Args:        source_datapipe: Iterable DataPipe that will be sharded

This Python file contains 3 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SHARDING_PRIORITIES`, `_ShardingIterDataPipe`, `ShardingFilterIterDataPipe`

**Functions defined**: `apply_sharding`, `__init__`, `apply_sharding`, `_update_num_of_instances`, `__iter__`, `__len__`

**Key imports**: Sized, IntEnum, NoReturn, functional_datapipe, IterDataPipe


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/data/datapipes/iter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Sized
- `enum`: IntEnum
- `typing`: NoReturn
- `torch.utils.data.datapipes._decorator`: functional_datapipe
- `torch.utils.data.datapipes.datapipe`: IterDataPipe


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
- [`streamreader.py_docs.md`](./streamreader.py_docs.md)
- [`routeddecoder.py_docs.md`](./routeddecoder.py_docs.md)


## Cross-References

- **File Documentation**: `sharding.py_docs.md`
- **Keyword Index**: `sharding.py_kw.md`
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

- **File Documentation**: `sharding.py_docs.md_docs.md`
- **Keyword Index**: `sharding.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

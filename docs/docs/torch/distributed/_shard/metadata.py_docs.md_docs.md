# Documentation: `docs/torch/distributed/_shard/metadata.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_shard/metadata.py_docs.md`
- **Size**: 5,091 bytes (4.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/_shard/metadata.py`

## File Metadata

- **Path**: `torch/distributed/_shard/metadata.py`
- **Size**: 2,215 bytes (2.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Union

from torch.distributed.remote_device import _remote_device


@dataclass
class ShardMetadata:
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the original tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_sizes(List[int]): Integers indicating the size of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(:class:`torch.distributed._remote_device`):
            Specifies the placement of this shard.
    """

    __slots__ = ["shard_offsets", "shard_sizes", "placement"]

    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: Optional[_remote_device]

    def __init__(
        self,
        shard_offsets: list[int],
        shard_sizes: list[int],
        placement: Optional[Union[str, _remote_device]] = None,
    ):
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        if isinstance(placement, str):
            self.placement = _remote_device(placement)
        else:
            self.placement = placement
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError(
                f"shard_offsets and shard_sizes should have "
                f"the same number of elements, found {len(self.shard_offsets)} "
                f"and {self.shard_sizes} respectively"
            )

        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError("shard_offsets should be >=0")
            if self.shard_sizes[i] < 0:
                raise ValueError("shard_sizes should be >= 0")

    def __hash__(self):
        def _hash_reduce(a, b):
            return (a << 8) + hash(b)

        res = reduce(_hash_reduce, self.shard_offsets, 37)
        res = reduce(_hash_reduce, self.shard_sizes, res)
        res = _hash_reduce(res, self.placement)
        return res

```



## High-Level Overview

"""    Represents a shard of the overall Tensor including its    offsets, lengths and device placement.    Args:        shard_offsets(List[int]): Offsets in the original tensor indicating            the start offsets for this shard. Should have the same rank as            the original tensor.        shard_sizes(List[int]): Integers indicating the size of each            dimension for this shard. Should have the same rank as the            original tensor.        placement(:class:`torch.distributed._remote_device`):            Specifies the placement of this shard.

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ShardMetadata`

**Functions defined**: `__init__`, `__hash__`, `_hash_reduce`

**Key imports**: dataclass, reduce, Optional, Union, _remote_device


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_shard`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass
- `functools`: reduce
- `typing`: Optional, Union
- `torch.distributed.remote_device`: _remote_device


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

Files in the same folder (`torch/distributed/_shard`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`op_registry_utils.py_docs.md`](./op_registry_utils.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`common_op_utils.py_docs.md`](./common_op_utils.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)
- [`sharder.py_docs.md`](./sharder.py_docs.md)


## Cross-References

- **File Documentation**: `metadata.py_docs.md`
- **Keyword Index**: `metadata.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/_shard`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_shard`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/_shard`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`metadata.py_kw.md_docs.md`](./metadata.py_kw.md_docs.md)
- [`common_op_utils.py_docs.md_docs.md`](./common_op_utils.py_docs.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`sharder.py_docs.md_docs.md`](./sharder.py_docs.md_docs.md)
- [`sharder.py_kw.md_docs.md`](./sharder.py_kw.md_docs.md)
- [`op_registry_utils.py_kw.md_docs.md`](./op_registry_utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `metadata.py_docs.md_docs.md`
- **Keyword Index**: `metadata.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

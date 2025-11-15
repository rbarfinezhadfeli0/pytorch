# Documentation: `docs/torch/distributed/_shard/sharded_tensor/shard.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_shard/sharded_tensor/shard.py_docs.md`
- **Size**: 5,083 bytes (4.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/_shard/sharded_tensor/shard.py`

## File Metadata

- **Path**: `torch/distributed/_shard/sharded_tensor/shard.py`
- **Size**: 2,361 bytes (2.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from dataclasses import dataclass

import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.remote_device import _remote_device


@dataclass
class Shard:
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.

    Args:
        tensor(torch.Tensor): Local tensor for the shard.
        metadata(:class `torch.distributed._shard.sharded_tensor.ShardMetadata`):
            The metadata for the shard, including offsets, lengths and device placement.
    """

    __slots__ = ["tensor", "metadata"]
    tensor: torch.Tensor
    metadata: ShardMetadata

    def __post_init__(self) -> None:
        # verification between local tensor and metadata
        if list(self.tensor.size()) != self.metadata.shard_sizes:
            raise ValueError(
                "Shard tensor size does not match with metadata.shard_lengths! "
                f"Found shard tensor size: {list(self.tensor.size())}, "
                f"metadata.shard_lengths: {self.metadata.shard_sizes}, "
            )
        placement_device = self.metadata.placement
        if (
            placement_device is not None
            and placement_device.device() != self.tensor.device
        ):
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {self.tensor.device}, "
                f"local shard metadata placement device: {placement_device.device()}"
            )

    @classmethod
    def from_tensor_and_offsets(
        cls, tensor: torch.Tensor, shard_offsets: list[int], rank: int
    ) -> "Shard":
        """
        Creates a Shard of a ShardedTensor from a local torch.Tensor, shard_offsets and rank.

        Args:
            tensor(torch.Tensor): Local tensor for the shard.
            shard_offsets(List[int]): List of integers specify the offset
                of the shard on each dimension.
            rank(int): Specify the rank for the shard.
        """
        shard_sizes = list(tensor.size())
        placement = _remote_device(f"rank:{rank}/{str(tensor.device)}")
        shard_meta = ShardMetadata(
            shard_offsets=shard_offsets, shard_sizes=shard_sizes, placement=placement
        )
        return Shard(tensor, shard_meta)

```



## High-Level Overview

"""    Container which holds the data for a shard as a Tensor and also    the associated metadata for that shard.    Args:        tensor(torch.Tensor): Local tensor for the shard.        metadata(:class `torch.distributed._shard.sharded_tensor.ShardMetadata`):            The metadata for the shard, including offsets, lengths and device placement.

This Python file contains 2 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Shard`

**Functions defined**: `__post_init__`, `from_tensor_and_offsets`

**Key imports**: dataclass, torch, ShardMetadata, _remote_device


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass
- `torch`
- `torch.distributed._shard.metadata`: ShardMetadata
- `torch.distributed.remote_device`: _remote_device


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed/_shard/sharded_tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`logger.py_docs.md`](./logger.py_docs.md)
- [`reshard.py_docs.md`](./reshard.py_docs.md)
- [`metadata.py_docs.md`](./metadata.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)


## Cross-References

- **File Documentation**: `shard.py_docs.md`
- **Keyword Index**: `shard.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/_shard/sharded_tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/distributed/_shard/sharded_tensor`):

- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`metadata.py_docs.md_docs.md`](./metadata.py_docs.md_docs.md)
- [`logger.py_kw.md_docs.md`](./logger.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`reshard.py_docs.md_docs.md`](./reshard.py_docs.md_docs.md)
- [`metadata.py_kw.md_docs.md`](./metadata.py_kw.md_docs.md)
- [`shard.py_kw.md_docs.md`](./shard.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`logger.py_docs.md_docs.md`](./logger.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `shard.py_docs.md_docs.md`
- **Keyword Index**: `shard.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

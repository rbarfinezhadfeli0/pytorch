# Documentation: `docs/torch/distributed/_shard/sharded_tensor/metadata.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_shard/sharded_tensor/metadata.py_docs.md`
- **Size**: 6,570 bytes (6.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/_shard/sharded_tensor/metadata.py`

## File Metadata

- **Path**: `torch/distributed/_shard/sharded_tensor/metadata.py`
- **Size**: 2,998 bytes (2.93 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch.distributed._shard.metadata import ShardMetadata


class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )


@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: list[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))

    tensor_properties: TensorProperties = field(default_factory=TensorProperties)

```



## High-Level Overview

"""Properties used to create :class:`Tensor`"""    # Regular tensor fields    dtype: torch.dtype = field(default=torch.get_default_dtype())    layout: torch.layout = field(default=torch.strided)    requires_grad: bool = False    memory_format: torch.memory_format = field(default=torch.contiguous_format)    pin_memory: bool = False    def __getstate__(self):        # Since torch.memory_format cannot be pickled!        memory_format = self.memory_format        if memory_format == torch.contiguous_format:            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT        elif memory_format == torch.channels_last:            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST        elif memory_format == torch.preserve_format:            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT        else:            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")        return (            self.dtype,            self.layout,            self.requires_grad,            mem_format_encoding,            self.pin_memory,        )    def __setstate__(        self,        state,    ):        (

This Python file contains 3 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MEM_FORMAT_ENCODING`, `TensorProperties`, `ShardedTensorMetadata`

**Functions defined**: `__getstate__`, `__setstate__`, `create_from_tensor`

**Key imports**: dataclass, field, Enum, torch, ShardMetadata


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass, field
- `enum`: Enum
- `torch`
- `torch.distributed._shard.metadata`: ShardMetadata


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/distributed/_shard/sharded_tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`logger.py_docs.md`](./logger.py_docs.md)
- [`reshard.py_docs.md`](./reshard.py_docs.md)
- [`shard.py_docs.md`](./shard.py_docs.md)
- [`api.py_docs.md`](./api.py_docs.md)


## Cross-References

- **File Documentation**: `metadata.py_docs.md`
- **Keyword Index**: `metadata.py_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data

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
- [`logger.py_kw.md_docs.md`](./logger.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`reshard.py_docs.md_docs.md`](./reshard.py_docs.md_docs.md)
- [`metadata.py_kw.md_docs.md`](./metadata.py_kw.md_docs.md)
- [`shard.py_docs.md_docs.md`](./shard.py_docs.md_docs.md)
- [`shard.py_kw.md_docs.md`](./shard.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`logger.py_docs.md_docs.md`](./logger.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `metadata.py_docs.md_docs.md`
- **Keyword Index**: `metadata.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

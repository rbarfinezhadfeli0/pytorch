# Documentation: `docs/torch/distributed/checkpoint/metadata.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/metadata.py_docs.md`
- **Size**: 8,614 bytes (8.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/metadata.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/metadata.py`
- **Size**: 5,631 bytes (5.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import torch
from torch.distributed.checkpoint.stateful import StatefulT


__all__ = [
    "ChunkStorageMetadata",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "Metadata",
    "MetadataIndex",
    "TensorProperties",
    "StorageMeta",
]


@dataclass
class ChunkStorageMetadata:
    """
    Each chunk is expected to have the same properties of the TensorStorageMetadata
    that includes it.
    """

    offsets: torch.Size
    sizes: torch.Size


class _MEM_FORMAT_ENCODING(Enum):
    """Describe the memory format of a tensor."""

    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default_factory=torch.get_default_dtype)
    # This field is deprecated.
    layout: torch.layout = field(default=torch.strided)
    # This field is deprecated.
    requires_grad: bool = False
    # This field is deprecated.
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    # This field is deprecated.
    pin_memory: bool = False

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
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

        if mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == _MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
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
class TensorStorageMetadata:
    properties: TensorProperties
    size: torch.Size
    chunks: list[ChunkStorageMetadata]


@dataclass
class BytesStorageMetadata:
    pass


STORAGE_TYPES = Union[TensorStorageMetadata, BytesStorageMetadata]
STATE_DICT_TYPE = dict[str, Union[StatefulT, Any]]


@dataclass
class StorageMeta:
    checkpoint_id: Union[str, os.PathLike, None] = None
    save_id: Optional[str] = None
    load_id: Optional[str] = None
    modules: list[str] = field(default_factory=list)


@dataclass
class Metadata:
    """This class represents the metadata of the checkpoint."""

    # Keys are the same from the `state_dict` used.
    state_dict_metadata: dict[str, STORAGE_TYPES]
    # It is the responsibility of the planner and storage plugins to ensure
    # backward compatibility of the planner_data and storage_data. DCP will
    # also ensure the backward compatibility of the metadata in this file and
    # the metadata of the built-in planner and storage plugins.
    planner_data: Any = None
    storage_data: Any = None
    storage_meta: Optional[StorageMeta] = None
    version: Optional[str] = None


@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""

    fqn: str
    """Fully Qualified Name of the object"""

    offset: Optional[torch.Size] = None
    """If the object is a tensor, offset into the tensor we're looking for"""

    index: Optional[int] = field(hash=False, compare=False, default=None)
    """
    Index hint when searching for tensor chunk to speedup lookups (optional)

    A common representation of a sharded tensor is as a list of chunks so to
    find the index in such a list you need to linear search it.

    When constructing an instance of MetadataIndex that points to that list,
    one can provide the index as a hint and it will be probed first before
    the linear search and thus making it significantly faster.
    """

    def __init__(
        self,
        fqn: str,
        offset: Optional[Sequence[int]] = None,
        index: Optional[int] = None,
    ):
        # We must use object.__setattr__ due to frozen=True
        object.__setattr__(self, "fqn", fqn)
        object.__setattr__(self, "index", index)
        if offset is not None:
            object.__setattr__(self, "offset", torch.Size(offset))

```



## High-Level Overview

"""    Each chunk is expected to have the same properties of the TensorStorageMetadata    that includes it.

This Python file contains 10 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ChunkStorageMetadata`, `_MEM_FORMAT_ENCODING`, `TensorProperties`, `TensorStorageMetadata`, `BytesStorageMetadata`, `StorageMeta`, `Metadata`, `MetadataIndex`

**Functions defined**: `__getstate__`, `__setstate__`, `create_from_tensor`, `__init__`

**Key imports**: os, Sequence, dataclass, field, Enum, Any, Optional, Union, torch, StatefulT


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `collections.abc`: Sequence
- `dataclasses`: dataclass, field
- `enum`: Enum
- `typing`: Any, Optional, Union
- `torch`
- `torch.distributed.checkpoint.stateful`: StatefulT


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

Files in the same folder (`torch/distributed/checkpoint`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`filesystem.py_docs.md`](./filesystem.py_docs.md)
- [`_consolidate_hf_safetensors.py_docs.md`](./_consolidate_hf_safetensors.py_docs.md)
- [`hf_storage.py_docs.md`](./hf_storage.py_docs.md)
- [`state_dict_loader.py_docs.md`](./state_dict_loader.py_docs.md)
- [`logging_handlers.py_docs.md`](./logging_handlers.py_docs.md)
- [`_storage_utils.py_docs.md`](./_storage_utils.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`_async_process_executor.py_docs.md`](./_async_process_executor.py_docs.md)
- [`resharding.py_docs.md`](./resharding.py_docs.md)


## Cross-References

- **File Documentation**: `metadata.py_docs.md`
- **Keyword Index**: `metadata.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/checkpoint`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/checkpoint`):

- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_async_process_executor.py_kw.md_docs.md`](./_async_process_executor.py_kw.md_docs.md)
- [`stateful.py_kw.md_docs.md`](./stateful.py_kw.md_docs.md)
- [`state_dict_loader.py_kw.md_docs.md`](./state_dict_loader.py_kw.md_docs.md)
- [`_async_executor.py_kw.md_docs.md`](./_async_executor.py_kw.md_docs.md)
- [`_state_dict_stager.py_kw.md_docs.md`](./_state_dict_stager.py_kw.md_docs.md)
- [`_extension.py_kw.md_docs.md`](./_extension.py_kw.md_docs.md)
- [`resharding.py_docs.md_docs.md`](./resharding.py_docs.md_docs.md)
- [`format_utils.py_docs.md_docs.md`](./format_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `metadata.py_docs.md_docs.md`
- **Keyword Index**: `metadata.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

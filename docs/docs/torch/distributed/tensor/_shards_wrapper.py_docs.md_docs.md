# Documentation: `docs/torch/distributed/tensor/_shards_wrapper.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_shards_wrapper.py_docs.md`
- **Size**: 16,312 bytes (15.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/_shards_wrapper.py`

## File Metadata

- **Path**: `torch/distributed/tensor/_shards_wrapper.py`
- **Size**: 13,478 bytes (13.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)


aten = torch.ops.aten


class LocalShardsWrapper(torch.Tensor):
    """
    A wrapper class to hold local shards of a DTensor.
    This class is used largely for checkpointing purposes and implicitly subtypes
    the _Checkpointable protocol.
    """

    __slots__ = ["_local_shards", "_storage_meta"]
    _local_shards: list[torch.Tensor]
    _storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: list[torch.Tensor], local_offsets: list[tuple[int, ...]]
    ) -> "LocalShardsWrapper":
        assert all(
            tensor.device == local_shards[0].device for tensor in local_shards[1:]
        )

        # if empty shard, we create a empty tensor
        if len(local_shards) == 0:
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                torch.Size([0, 0]),
            )
            r._local_shards = []
            r._storage_meta = TensorStorageMetadata(
                properties=TensorProperties(),
                size=torch.Size([0, 0]),
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size([0, 0]), sizes=torch.Size([0, 0])
                    )
                ],
            )
            return r

        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1 and local_shards[0].ndim == 2:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

        # in cases of sharding optimizer rowwise, we calculate total tensor size by "concat" on first tensor dimension
        if len(local_shards) > 1 and local_shards[0].ndim == 1:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[0] += shard.size()[0]

        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for shard, offset in zip(local_shards, local_offsets)
        ]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            torch.Size(cat_tensor_shape),
        )
        r._local_shards = local_shards
        r._storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}

        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather_into_tensor,
            torch.ops._c10d_functional.wait_tensor.default: cls.handle_wait_tensor,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
            aten.detach.default: cls.handle_detach,
            aten.clone.default: cls.handle_clone,
            aten.new_empty.default: cls.handle_new_empty,
        }

        if func in dispatcher:
            return dispatcher[func](args, kwargs)
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
    def handle_all_gather_into_tensor(args, kwargs) -> torch.Tensor:
        dim = args[0].local_sizes()[0][1]
        cat_tensor = torch.cat(
            [t.view(-1) for t in args[0].local_shards()], dim=0
        ).view(-1, dim)
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            cat_tensor, *args[1:], **kwargs
        )

    @staticmethod
    def handle_wait_tensor(args, kwargs) -> torch.Tensor:
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    def handle_to_copy(args, kwargs) -> torch.Tensor:
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_view(args, kwargs) -> "LocalShardsWrapper":
        view_shape = args[1]
        res_shards_list = []
        if len(args[0].local_shards()) > 1:
            if args[0].local_shards()[0].ndim == 2:
                assert (
                    args[0].storage_metadata().size[0] == view_shape[0]
                    and args[0].storage_metadata().size[1] == view_shape[1]
                )
                # This accounts for a DTensor quirk, when multiple shards are present on a rank, DTensor on
                # init calls view_as() on the global tensor shape
                # will fail because the view shape is not applicable to individual shards.
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            elif args[0].local_shards()[0].ndim == 1:
                assert args[0].storage_metadata().size[0] == view_shape[0]
                # This case is for optimizer sharding as regardless of sharding type, optimizer state is row wise sharded
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            else:
                raise NotImplementedError("No support for view on tensors ndim > 2")
        else:
            # view is called per shard
            res_shards_list = [
                aten.view.default(shard, args[1], **kwargs)
                for shard in args[0].local_shards()
            ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_equal(args, kwargs) -> bool:
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of shards
        """
        a, b = args[0], args[1]
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        if a.storage_metadata() != b.storage_metadata():
            return False
        return True

    @staticmethod
    def handle_detach(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        deatched_local_shards = [
            aten.detach.default(shard) for shard in self_ls.local_shards()
        ]
        self_ls._local_shards = deatched_local_shards
        self_ls._storage_meta.properties.requires_grad = False
        return self_ls

    @staticmethod
    def handle_clone(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        desired_memory_format = kwargs.get("memory_format", None)
        if desired_memory_format and desired_memory_format != torch.preserve_format:
            raise NotImplementedError(
                f"{desired_memory_format} is not supported for LocalShardsWrapper!"
            )
        cloned_local_shards = [
            shard.clone(memory_format=desired_memory_format)
            for shard in self_ls._local_shards
        ]
        return LocalShardsWrapper(cloned_local_shards, self_ls.local_offsets())

    @staticmethod
    def handle_new_empty(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        return LocalShardsWrapper(
            [torch.empty_like(shard) for shard in self_ls._local_shards],
            self_ls.local_offsets(),
        )

    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        return (
            self._local_shards[0].device if self._local_shards else torch.device("meta")
        )

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        return self._local_shards[0].is_meta if self._local_shards else True

    def is_pinned(self) -> bool:  # type: ignore[override]
        return self._storage_meta.properties.pin_memory

    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        self._storage_meta.properties.requires_grad = requires_grad
        [shard.requires_grad_(requires_grad) for shard in self._local_shards]
        return self

    def local_shards(self) -> list[torch.Tensor]:
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def local_sizes(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        """
        Returns a :class:`list[ChunkStorageMetadata]` object corresponding to the
        metadata for each tensor shard
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
        return self._storage_meta

    def is_empty_shard(self) -> bool:
        """
        Returns a :class:`bool` object indicating if the local tensor on current rank
        is an empty tensor
        """
        return self._storage_meta.size[0] == 0 and self._storage_meta.size[1] == 0

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=object.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """
        For compatibility with DCP, we support creation of chunk lists
        such that they can be saved properly.
        """
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """
        For compatibility with DCP, we support finding shard based on index
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        # Fast lookup path
        if index.index is not None:
            if (
                len(self._local_shards) > index.index
                and self._storage_meta.chunks[index.index].offsets == index.offset
            ):
                return self._local_shards[index.index]

        if index.offset is not None:
            for shard, chunk in zip(self._local_shards, self._storage_meta.chunks):
                if chunk.offsets == index.offset:
                    return shard

        # Empty shard case
        if len(self._local_shards) == 0 and self._storage_meta.chunks[
            0
        ].sizes == torch.Size([0, 0]):
            return torch.empty(0)

        raise ValueError(
            f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'"
        )

    def _get_tensor_size_bytes(self) -> int:
        object_size = 0
        for shard in self.local_shards():
            object_size += shard.nelement() * shard.element_size()
        return object_size

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:  # type: ignore[override]
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    def __str__(self) -> str:
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

```



## High-Level Overview

"""    A wrapper class to hold local shards of a DTensor.    This class is used largely for checkpointing purposes and implicitly subtypes    the _Checkpointable protocol.

This Python file contains 4 class(es) and 27 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LocalShardsWrapper`

**Functions defined**: `__new__`, `__torch_dispatch__`, `handle_all_gather_into_tensor`, `handle_wait_tensor`, `handle_to_copy`, `handle_view`, `handle_equal`, `handle_detach`, `handle_clone`, `handle_new_empty`, `device`, `is_meta`, `is_pinned`, `requires_grad_`, `local_shards`, `local_sizes`, `local_offsets`, `local_chunks`, `storage_metadata`, `is_empty_shard`

**Key imports**: Any, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `torch`


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

Files in the same folder (`torch/distributed/tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tp_conv.py_docs.md`](./_tp_conv.py_docs.md)
- [`_dispatch.py_docs.md`](./_dispatch.py_docs.md)
- [`_random.py_docs.md`](./_random.py_docs.md)
- [`_collective_utils.py_docs.md`](./_collective_utils.py_docs.md)
- [`device_mesh.py_docs.md`](./device_mesh.py_docs.md)
- [`_redistribute.py_docs.md`](./_redistribute.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`_sharding_prop.py_docs.md`](./_sharding_prop.py_docs.md)


## Cross-References

- **File Documentation**: `_shards_wrapper.py_docs.md`
- **Keyword Index**: `_shards_wrapper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/tensor`):

- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`placement_types.py_kw.md_docs.md`](./placement_types.py_kw.md_docs.md)
- [`device_mesh.py_kw.md_docs.md`](./device_mesh.py_kw.md_docs.md)
- [`_sharding_prop.py_kw.md_docs.md`](./_sharding_prop.py_kw.md_docs.md)
- [`_random.py_kw.md_docs.md`](./_random.py_kw.md_docs.md)
- [`_collective_utils.py_kw.md_docs.md`](./_collective_utils.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_collective_utils.py_docs.md_docs.md`](./_collective_utils.py_docs.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_shards_wrapper.py_docs.md_docs.md`
- **Keyword Index**: `_shards_wrapper.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

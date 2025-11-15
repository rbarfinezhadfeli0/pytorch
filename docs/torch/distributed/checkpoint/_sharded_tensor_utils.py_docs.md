# Documentation: `torch/distributed/checkpoint/_sharded_tensor_utils.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_sharded_tensor_utils.py`
- **Size**: 4,144 bytes (4.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor, ShardMetadata
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.remote_device import _remote_device

from ._traverse import OBJ_PATH, set_element, STATE_DICT_ITEM, traverse_state_dict
from .utils import _element_wise_add, _normalize_device_info


if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata


# TODO: We need to refactor this code.
def _flatten_sharded_tensors(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    r"""
    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.

    The resulting ShardedTensor instances are only correct regarding the local shard and
    MUST not be used for any other purpose but checkpointing, as no operator will work with them.

    This function should be used in conjunction with a state_dict produced by FSDP's
    StateDictType.SHARDED_STATE_DICT methods.
    """
    new_state_dict: STATE_DICT_TYPE = {}

    def rewrite_dict(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if not isinstance(value, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        shards = value.local_shards()

        if len(shards) == 0:
            return
        if len(shards) != 1:
            set_element(new_state_dict, path, value)
            return

        outer_shard = shards[0]

        inner_st = outer_shard.tensor
        if not isinstance(inner_st, ShardedTensor):
            set_element(new_state_dict, path, value)
            return

        if len(inner_st.local_shards()) != 1:
            raise ValueError("Cannot handle inner tensor with more than 1 shard")
        inner_shard = inner_st.local_shards()[0]

        local_shards = [
            Shard(
                tensor=inner_shard.tensor,
                metadata=ShardMetadata(
                    shard_offsets=_element_wise_add(
                        outer_shard.metadata.shard_offsets,
                        inner_shard.metadata.shard_offsets,
                    ),
                    shard_sizes=inner_shard.metadata.shard_sizes,
                    placement=f"rank:{dist.get_rank()}/{inner_shard.tensor.device}",
                ),
            )
        ]

        st_meta: ShardedTensorMetadata = copy.deepcopy(value.metadata())
        other_rank = 0 if dist.get_rank() > 0 else 1
        device_info = _normalize_device_info(inner_shard.tensor.device.type, 0)

        # Remove the outer ST shard the inner ST covers
        for i, shard_md in enumerate(st_meta.shards_metadata):
            if shard_md.shard_offsets == outer_shard.metadata.shard_offsets:
                st_meta.shards_metadata.pop(i)
                break

        # Attribute other rank for the other shards
        for shard_md in st_meta.shards_metadata:
            shard_md.placement = _remote_device(f"rank:{other_rank}/{device_info}")

        # Add other inner shards from the inner tensor
        for inner_md in inner_st.metadata().shards_metadata:
            if inner_md.shard_offsets != inner_shard.metadata.shard_offsets:
                st_meta.shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=_element_wise_add(
                            outer_shard.metadata.shard_offsets,
                            inner_md.shard_offsets,
                        ),
                        shard_sizes=inner_md.shard_sizes,
                        placement=f"rank:{other_rank}/{device_info}",
                    )
                )

        # Finally add this shard
        st_meta.shards_metadata.append(local_shards[0].metadata)

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards=local_shards,
            sharded_tensor_metadata=st_meta,
        )
        set_element(new_state_dict, path, st)

    traverse_state_dict(state_dict, rewrite_dict)
    return new_state_dict

```



## High-Level Overview

r"""    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.    The resulting ShardedTensor instances are only correct regarding the local shard and    MUST not be used for any other purpose but checkpointing, as no operator will work with them.    This function should be used in conjunction with a state_dict produced by FSDP's    StateDictType.SHARDED_STATE_DICT methods.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_flatten_sharded_tensors`, `rewrite_dict`

**Key imports**: copy, TYPE_CHECKING, torch.distributed as dist, Shard, ShardedTensor, ShardMetadata, STATE_DICT_TYPE, _remote_device, OBJ_PATH, set_element, STATE_DICT_ITEM, traverse_state_dict, _element_wise_add, _normalize_device_info, ShardedTensorMetadata


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `typing`: TYPE_CHECKING
- `torch.distributed as dist`
- `torch.distributed._shard.sharded_tensor`: Shard, ShardedTensor, ShardMetadata
- `torch.distributed.checkpoint.metadata`: STATE_DICT_TYPE
- `torch.distributed.remote_device`: _remote_device
- `._traverse`: OBJ_PATH, set_element, STATE_DICT_ITEM, traverse_state_dict
- `.utils`: _element_wise_add, _normalize_device_info
- `torch.distributed._shard.sharded_tensor.metadata`: ShardedTensorMetadata


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

- **File Documentation**: `_sharded_tensor_utils.py_docs.md`
- **Keyword Index**: `_sharded_tensor_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

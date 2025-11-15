# Documentation: `docs/torch/distributed/checkpoint/resharding.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/resharding.py_docs.md`
- **Size**: 5,436 bytes (5.31 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/resharding.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/resharding.py`
- **Size**: 2,275 bytes (2.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata


__all__: list[str] = []


def _check_shard_metadata_pair_overlap(
    shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata
) -> bool:
    """Check if two shards overlap."""
    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.offsets)
    for i in range(ndims):
        if shard1.offsets[i] >= shard2.offsets[i] + shard2.sizes[i]:
            return False
        if shard2.offsets[i] >= shard1.offsets[i] + shard1.sizes[i]:
            return False

    return True


def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata
) -> list[tuple[int, int, int, int]]:
    """
    Return the overlapping region between saved_shard and current_shard.

    There returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
    narrows = []
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.offsets,
            current_shard.offsets,
            saved_shard.sizes,
            current_shard.sizes,
        )
    ):
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows

```



## High-Level Overview

"""Check if two shards overlap."""    # For each dim of each shard, check if one shard resides on the other    # end of second shard with respect to that dim. As an example for a 2D    # shard, we would check if one shard is above or on the left of the    # other shard.    ndims = len(shard1.offsets)    for i in range(ndims):        if shard1.offsets[i] >= shard2.offsets[i] + shard2.sizes[i]:            return False        if shard2.offsets[i] >= shard1.offsets[i] + shard1.sizes[i]:            return False    return Truedef _shards_get_overlap_region_wrt_saved_tensor(    saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata) -> list[tuple[int, int, int, int]]:

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_check_shard_metadata_pair_overlap`, `_shards_get_overlap_region_wrt_saved_tensor`

**Key imports**: ChunkStorageMetadata


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.distributed.checkpoint.metadata`: ChunkStorageMetadata


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


## Cross-References

- **File Documentation**: `resharding.py_docs.md`
- **Keyword Index**: `resharding.py_kw.md`
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

Files in the same folder (`docs/torch/distributed/checkpoint`):

- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_async_process_executor.py_kw.md_docs.md`](./_async_process_executor.py_kw.md_docs.md)
- [`stateful.py_kw.md_docs.md`](./stateful.py_kw.md_docs.md)
- [`state_dict_loader.py_kw.md_docs.md`](./state_dict_loader.py_kw.md_docs.md)
- [`_async_executor.py_kw.md_docs.md`](./_async_executor.py_kw.md_docs.md)
- [`_state_dict_stager.py_kw.md_docs.md`](./_state_dict_stager.py_kw.md_docs.md)
- [`_extension.py_kw.md_docs.md`](./_extension.py_kw.md_docs.md)
- [`format_utils.py_docs.md_docs.md`](./format_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `resharding.py_docs.md_docs.md`
- **Keyword Index**: `resharding.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/distributed/checkpoint/_dedup_save_plans.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_dedup_save_plans.py_docs.md`
- **Size**: 5,711 bytes (5.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_dedup_save_plans.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_dedup_save_plans.py`
- **Size**: 2,754 bytes (2.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
import dataclasses
from collections import defaultdict
from typing import TYPE_CHECKING

from torch.distributed.checkpoint.planner import SavePlan, WriteItem


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import MetadataIndex

__all__ = ["dedup_save_plans"]


def dedup_save_plans(
    all_plans: list[SavePlan],
    save_to_lowest_rank: bool = False,
) -> list[SavePlan]:
    """
    Removes duplicate entries from appearing on multiple SavePlans. For each duplicate across
    a set of SavePlans, only the smallest SavePlan in terms of planned storage keeps the entry.

    Please note that this function does not modify the original SavePlans, but rather returns
    """

    # Map to query the plan indices that a write item is duplicated in
    write_item_to_plan_indices: dict[MetadataIndex, set[int]] = defaultdict(set)
    # Map to query the write item from its index
    write_item_idx_to_write_item: dict[MetadataIndex, WriteItem] = {}
    # Set of write item indices that are present in each plan
    # After deduplication, this will be the set of write item indices that are present in the final plans
    plan_to_item_indices: list[set[MetadataIndex]] = [
        {item.index for item in plan.items} for plan in all_plans
    ]

    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # map each write item to its plan
            write_item_to_plan_indices[write_item.index].add(plan_idx)
            write_item_idx_to_write_item[write_item.index] = write_item
    plan_to_size = [0] * len(all_plans)
    for write_item_idx, plan_indices in write_item_to_plan_indices.items():
        if save_to_lowest_rank:
            select_plan_idx = min(plan_indices)
        else:
            select_plan_idx = min(
                plan_indices, key=lambda plan_idx: plan_to_size[plan_idx]
            )

        write_item = write_item_idx_to_write_item[write_item_idx]
        # Ignore the storage size of anything that is not a tensor, since
        # we don't know how much storage they represent
        plan_to_size[select_plan_idx] += write_item.tensor_storage_size() or 1
        for plan_idx in plan_indices - {select_plan_idx}:
            plan_to_item_indices[plan_idx].discard(write_item_idx)
    # Sanity check
    if len(all_plans) != len(plan_to_item_indices):
        raise AssertionError("len(all_plans) != len(plan_to_item_indices)")
    # Create new plans with the updated write items post deduplication
    return [
        dataclasses.replace(
            plan, items=[item for item in plan.items if item.index in item_indexes]
        )
        for plan, item_indexes in zip(all_plans, plan_to_item_indices)
    ]

```



## High-Level Overview

"""    Removes duplicate entries from appearing on multiple SavePlans. For each duplicate across    a set of SavePlans, only the smallest SavePlan in terms of planned storage keeps the entry.    Please note that this function does not modify the original SavePlans, but rather returns

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `dedup_save_plans`

**Key imports**: dataclasses, defaultdict, TYPE_CHECKING, SavePlan, WriteItem, MetadataIndex


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `collections`: defaultdict
- `typing`: TYPE_CHECKING
- `torch.distributed.checkpoint.planner`: SavePlan, WriteItem
- `torch.distributed.checkpoint.metadata`: MetadataIndex


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

- **File Documentation**: `_dedup_save_plans.py_docs.md`
- **Keyword Index**: `_dedup_save_plans.py_kw.md`
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
- [`resharding.py_docs.md_docs.md`](./resharding.py_docs.md_docs.md)
- [`format_utils.py_docs.md_docs.md`](./format_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_dedup_save_plans.py_docs.md_docs.md`
- **Keyword Index**: `_dedup_save_plans.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

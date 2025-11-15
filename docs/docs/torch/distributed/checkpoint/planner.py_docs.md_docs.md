# Documentation: `docs/torch/distributed/checkpoint/planner.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/planner.py_docs.md`
- **Size**: 19,682 bytes (19.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/planner.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/planner.py`
- **Size**: 16,263 bytes (15.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import abc
import io
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import reduce
from typing import Any, Optional, Union

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    StorageMeta,
    TensorProperties,
)


__all__ = [
    "WriteItemType",
    "LoadItemType",
    "BytesIOWriteData",
    "TensorWriteData",
    "WriteItem",
    "ReadItem",
    "SavePlan",
    "LoadPlan",
    "SavePlanner",
    "LoadPlanner",
]


class WriteItemType(Enum):
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()


class LoadItemType(Enum):
    TENSOR = auto()
    BYTE_IO = auto()


@dataclass(frozen=True)
class BytesIOWriteData:
    nbytes: int


@dataclass(frozen=True)
class TensorWriteData:
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: torch.Size


@dataclass(frozen=True)
class WriteItem:
    """Dataclass which holds information about what needs to be written to storage."""

    index: MetadataIndex
    type: WriteItemType

    # Size of bytesIO data to be written.
    bytes_io_data: Optional[BytesIOWriteData] = None

    # Value present if it's a tensor write
    tensor_data: Optional[TensorWriteData] = None

    def tensor_storage_size(self) -> Optional[int]:
        """
        Calculates the storage size of the underlying tensor, or None if this is not a tensor write.

        Returns:
            Optional[int] storage size, in bytes of underlying tensor if any.
        """
        if self.tensor_data is None:
            return None

        numels = reduce(operator.mul, self.tensor_data.size, 1)
        dtype_size = torch._utils._element_size(self.tensor_data.properties.dtype)
        return numels * dtype_size


@dataclass(frozen=True)
class ReadItem:
    # Read Item
    type: LoadItemType

    # Index into the state_dict
    dest_index: MetadataIndex
    # Offsets into destination tensor
    dest_offsets: torch.Size

    # Index into the checkpoint
    storage_index: MetadataIndex
    # Offset into the checkpoint data
    storage_offsets: torch.Size

    # Size of the hypercube to copy
    lengths: torch.Size


@dataclass(frozen=True)
class SavePlan:
    items: list[WriteItem]
    storage_data: Any = None
    planner_data: Any = None
    # This is used to indicate that the ranks should
    # use the cached plans to write data instead.
    usable: bool = True


@dataclass
class LoadPlan:
    items: list[ReadItem]
    storage_data: Any = None
    planner_data: Any = None


class SavePlanner(abc.ABC):
    """
    Abstract class defining the protocol used by save_state_dict to plan the save process.

    SavePlanners are stateful objects that can be used to customize the whole save process.

    SavePlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during save_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of a checkpoint save.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `SavePlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the SavePlan from all ranks and make any global decision.

    4) finish_plan - called on all ranks.
        This gives each rank a chance to adjust to global planning decisions.

    5) resolve_data - called multiple times on each rank
        Lookups a value on the `state_dict` for the storage layer to write.

    Users are recommended to extend DefaultSavePlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are 3 usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the save process as it
    doesn't requite understanding the intrincacies of how SavePlan works:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultSavePlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         storage_meta: Optional[StorageMeta],
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>> # prefix all keys with `foo_``
    >>>         super().set_up_planner({"foo_" + k: v for k, v in state_dict.items()}, storage_meta, is_coordinator)

    Modifying local plan and lookup in tandem. This is useful when fine control of how data is persisted

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class FP16Planner(DefaultSavePlanner):
    >>>     def create_local_plan(self):
    >>>         plan = super().create_local_plan()
    >>>         for p in plan:
    >>>             if p.tensor_data is not None:
    >>>                 p.tensor_data.properties.dtype = torch.float16
    >>>         return plan
    >>>
    >>>     def resolve_data(self, write_item):
    >>>         item = super().resolve_data(write_item)
    >>>         return item if write_item.type == WriteItemType.BYTE_IO else item.to(torch.float16)

    Using the global planning step to make central decisions that can't be made individually by each rank

    >>> # xdoctest: +SKIP("undefined vars")
    >>> from itertools import zip_longest
    >>> from dataclasses import replace
    >>> class DDPLoadBalancingPlanner(DefaultSavePlanner):
    >>> # This uses the default local plan behavior of having all non-sharded writes in rank 0
    >>> # This sample doesn't handle ShardedTensors
    >>>     def create_global_plan(self, all_plans):
    >>>         iters = [iter(all_plans[0].items)] * len(all_plans)
    >>>         items_per_rank = [
    >>>             [item for item in items if item is not None]
    >>>             for items in zip(*zip_longest(*iters), strict=True)
    >>>         ]
    >>>         all_plans = [
    >>>             replace(plan, items=items)
    >>>             for plan, items in zip(all_plans, items_per_rank, strict=True)
    >>>         ]
    >>>         return super().create_global_plan(all_plans)

    Finally, some planners need to save additional metadata in the checkpoint, this is
    accomplished by having each rank contribute their data items in the local plan and
    the global planner aggregate them:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class SaveExtraDataPlanner(DefaultSavePlanner):
    >>>     def create_local_plan(self) -> SavePlan:
    >>>         plan = super().create_local_plan()
    >>>         return replace(plan, planner_data="per-rank-data")
    >>>
    >>>     def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    >>>         global_plan, metadata = super().create_global_plan(all_plans)
    >>>         merged_data = [p.planner_data for p in global_plan]
    >>>         metadata = replace(metadata, planner_data=merged_data)
    >>>         return global_plan, metadata
    """

    # Save plan for the current rank as computed by `create_local_plan` API
    # Cached on the local rank.
    _cached_save_plan: dict[str, SavePlan] = {}
    # Final save plan for the current rank.
    # This is created by merging the plan created by `create_local_plan` API
    # and the result of `create_global_plan` for the given rank.
    # This is the final plan computed by the `finish_plan` API that gets
    # sent to the `write_data`.
    # Cached on the local rank.
    _cached_final_save_plan: dict[str, SavePlan] = {}
    # Collection of all the local plans from all the ranks.
    # This is the input to the `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_all_plans: dict[str, list[SavePlan]] = {}
    # Global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_global_plan: dict[str, list[SavePlan]] = {}
    # Metadata for the global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_metadata: dict[str, Metadata] = {}

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: Optional[StorageMeta] = None,
        is_coordinator: bool = False,
    ) -> None:
        """
        Initialize this planner to save ``state_dict``.

        Implementations should save those values as they won't be provided lated in the save process.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan:
        """
        Compute the save plan for the current rank.

        This will be aggregated and passed to create_global_plan.
        Planner specific data can be passed through SavePlan::planner_data.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """

    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        """
        Merge the plan created by `create_local_plan` and the result of `create_global_plan`.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        """
        Transform and prepare ``write_item`` from ``state_dict`` for storage, ensuring idempotency and thread-safety.

        Lookup the object associated with ``write_item`` in ``state_dict`` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idempotent and thread-save. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsibility to figure out how to save them.
        """


class LoadPlanner:
    """
    Abstract class defining the protocol used by load_state_dict to plan the load process.

    LoadPlanner are stateful objects that can be used to customize the whole load process.

    LoadPlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during load_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of loading a checkpoint.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `LoadPlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the LoadPlan from all ranks and make any global decision.

    4) load_bytes - called multiple times on each rank
        This is called once per non-tensor value in state_dict.

    5) resolve_tensor and commit_tensor - called multiple times on each rank
        They are called in pair for each Tensor value in state_dict.

    Users are recommended to extend DefaultLoadPlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are two usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the load process as it
    doesn't requite understanding the intrincacies of how LoadPlan works. We need
    to keep a reference to the original state_dict as load happens in place so
    we need to be able to perform it in place

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultLoadPlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         metadata: Metadata,
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>>         self.original_state_dict = state_dict
    >>>         state_dict = {"foo_" + k: v for k, v in state_dict.items()}
    >>>
    >>>         if self.flatten_sharded_tensors:
    >>>             state_dict = _flatten_sharded_tensors(state_dict)
    >>>
    >>>         if self.flatten_state_dict:
    >>>             state_dict, self.mappings = flatten_state_dict(state_dict)
    >>>
    >>>         self.state_dict = state_dict
    >>>         self.metadata = metadata
    >>>         self.is_coordinator = is_coordinator
    >>>
    >>>     def load_bytes(self, read_item, value):
    >>> # Remove the "foo_" prefix
    >>>         self.original_state_dict[read_item.dest_index.fqn[4:]] = torch.load(value, weights_only=False)


    Modifying resolve_tensor and commit_tensor to handle load time transformation.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class MetaModelMaterialize(DefaultSavePlanner):
    >>>     def resolve_tensor(self, read_item):
    >>>         tensor = super().resolve_tensor(read_item)
    >>>         return torch.empty_like(tensor, device="cpu")
    >>>
    >>>     def commit_tensor(self, read_item, tensor):
    >>>         self.state_dict[read_item.dest_index.fqn] = tensor
    """

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        """
        Initialize this instance to load data into ``state_dict``.

        . N.B. This is called on every rank.
        """

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        """
        Create a LoadPlan based on state_dict and metadata provided by set_up_planner.

        . N.B. This is called on every rank.
        """

    @abc.abstractmethod
    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        """
        Compute the global load plan and return plans for each rank.

        . N.B. This is called on the coordinator rank only
        """

    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
        """Accept the plan from coordinator and return final LoadPlan."""

    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        """
        Load the item described by ``read_item``and ``value``.

        This method is expected to modify in-place the underlying state_dict.

        The contents of ``value`` are defined by the SavePlanner used to produce
        the checkpoint being loaded.
        """

    def resolve_bytes(self, read_item: ReadItem) -> io.BytesIO:
        """
        Return the BytesIO to be used by the StorageReader to load `read_item`.

        The BytesIO should alias with one on the underlying state_dict as StorageReader will replace its contents.
        """
        raise NotImplementedError("LoadPlanner.resolve_bytes is not implemented")

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        """
        Return the tensor described by ``read_item`` to be used by the StorageReader to load `read_item`.

        The tensor should alias with one on the underlying state_dict as StorageReader will replace its contents.
        If, for any reason, that's not possible, the planner can use the ``commit_tensor`` method to copy the data
        back to the one in state_dict.
        """

    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        """
        Call once the StorageReader finished loading data into ``tensor``.

        The provided tensor is the same one returned by the call to ``resolve_tensor``.
        This method is only needed if this LoadPlanner needs to post process ``tensor`` prior to
        copying it back to the one in the state_dict.

        The contents of tensor will follow its device synchronization model.
        """

```



## High-Level Overview


This Python file contains 22 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WriteItemType`, `LoadItemType`, `BytesIOWriteData`, `TensorWriteData`, `WriteItem`, `ReadItem`, `SavePlan`, `LoadPlan`, `SavePlanner`, `RenamePlanner`, `FP16Planner`, `DDPLoadBalancingPlanner`, `SaveExtraDataPlanner`, `LoadPlanner`, `RenamePlanner`, `MetaModelMaterialize`

**Functions defined**: `tensor_storage_size`, `set_up_planner`, `create_local_plan`, `resolve_data`, `create_global_plan`, `create_local_plan`, `create_global_plan`, `set_up_planner`, `create_local_plan`, `create_global_plan`, `finish_plan`, `resolve_data`, `set_up_planner`, `load_bytes`, `resolve_tensor`, `commit_tensor`, `set_up_planner`, `create_local_plan`, `create_global_plan`, `finish_plan`

**Key imports**: abc, io, operator, dataclass, auto, Enum, reduce, Any, Optional, Union, torch, zip_longest, replace


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `io`
- `operator`
- `dataclasses`: dataclass
- `enum`: auto, Enum
- `functools`: reduce
- `typing`: Any, Optional, Union
- `torch`
- `itertools`: zip_longest


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `planner.py_docs.md`
- **Keyword Index**: `planner.py_kw.md`
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

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `planner.py_docs.md_docs.md`
- **Keyword Index**: `planner.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

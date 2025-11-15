# Documentation: `torch/distributed/checkpoint/storage.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/storage.py`
- **Size**: 9,834 bytes (9.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import abc
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
)
from torch.futures import Future


__all__ = ["WriteResult", "StorageWriter", "StorageReader"]


@dataclass(frozen=True)
class WriteResult:
    index: MetadataIndex

    size_in_bytes: int
    storage_data: Any


class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) set_up_storage_writer()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """
        Calls to indicates a brand new checkpoint write is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint write. The meaning of the checkpiont_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plan (SavePlan): The local plan from the ``SavePlanner`` in use.

        Returns:
            A transformed ``SavePlan`` after storage local planning
        """

    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        """
        Perform centralized planning of storage.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plans: A list of ``SavePlan`` instances, one for each rank.

        Returns:
            A list of transformed ``SavePlan`` after storage global planning
        """

    @abc.abstractmethod
    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        """
        Write all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``SavePlanner::resolve_data`` on each item
        from the plan to get access to the underlying object to write.

        Subclasses should lazily call `resolve_data` as it can allocate memory.
        In case of tensors, make following assumptions:

        - They might be on any device, including not matching the one on ``WriteItem::tensor_data``
        - They might be views or not contiguous. Only the projection needs to be saved.

        Args:
            plan (SavePlan): The save plan to execute.
            planner (SavePlanner): Planner object to be used to resolve items to data.

        Returns:
            A future that completes to a list of WriteResult
        """

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        """
        Write the metadata and marks the current checkpoint as successful.

        The actual format/schema used for serializing `metadata` is an
        implementation detail. The only requirement is that it's recoverable
        in to the same object graph.

        Args:
            metadata (Metadata): metadata for the new checkpoint
            results: A list of WriteResults from all ranks.

        Returns:
            None
        """

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Check if the given checkpoint_id is supported by the storage. This allow
        us to enable automatic storage selection.
        """
        ...

    def storage_meta(self) -> Optional[StorageMeta]:
        """
        Return the storage-specific metadata. This is used to store additional information
        in a checkpoint that can be useful for providing request-level observability. StorageMeta
        is passed to the ``SavePlanner`` during save calls. Returns None by default.

        TODO: provide an example
        """
        return None


class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    One StorageReader instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expected the following sequence of calls by ``load_state_dict``:

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) read_metadata()
    2) (all ranks) set_up_storage_reader()
    3) (all ranks) prepare_local_plan()
    4) (coordinator) prepare_global_plan()
    5) (all ranks) read_data()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """
        Calls to indicates a brand new checkpoint read is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint read. The meaning of the checkpiont_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is more like a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        """
        Read the checkpoint metadata.

        Returns:
            The metadata object associated with the checkpoint being loaded.

        """

    @abc.abstractmethod
    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        """
        Initialize this instance.

        Args:
            metadata (Metadata): The metadata schema to use.
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plan (LoadPlan): The local plan from the ``LoadPlan`` in use.

        Returns:
            A transformed ``LoadPlan`` after storage local planning
        """

    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        """
        Perform centralized planning of storage loading.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plans: A list of ``LoadPlan`` instances, one for each rank.

        Returns:
            A list of transformed ``LoadPlan`` after storage global planning
        """

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Read all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``LoadPlanner::load_bytes`` to deserialize a BytesIO
        object into the right place.

        A subclass should call ``LoadPlanner::resolve_tensor`` to get access to the
        tensors that in should load data into.

        It's the StorageLayer responsibility to properly schedule any cross device copies
        required.

        Args:
            plan (LoadPlan): The local plan to execute on
            planner (LoadPlanner): The planner object to use to resolve items.

        Returns:
            A future that completes once all reads are finished.
        """

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Check if the given checkpoint_id is supported by the storage. This allow
        us to enable automatic storage selection.
        """
        ...

```



## High-Level Overview

"""    Interface used by ``save_state_dict`` to write to storage.    One StorageWriter instance acts as both the coordinator and the follower    in a distributed checkpoint. As part of initialization, each instance    is told its role.    A subclass should expect the following sequence of calls.    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.    1) (all ranks) set_up_storage_writer()    2) (all ranks) prepare_local_plan()    3) (coordinator) prepare_global_plan()    4) (all ranks) write_data()    5) (coordinator) finish()

This Python file contains 9 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WriteResult`, `StorageWriter`, `StorageReader`

**Functions defined**: `reset`, `set_up_storage_writer`, `prepare_local_plan`, `prepare_global_plan`, `write_data`, `finish`, `validate_checkpoint_id`, `storage_meta`, `reset`, `read_metadata`, `set_up_storage_reader`, `prepare_local_plan`, `prepare_global_plan`, `read_data`, `validate_checkpoint_id`

**Key imports**: abc, os, dataclass, Any, Optional, Union, Metadata, MetadataIndex, StorageMeta, Future


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `os`
- `dataclasses`: dataclass
- `typing`: Any, Optional, Union
- `torch.distributed.checkpoint.metadata`: Metadata, MetadataIndex, StorageMeta
- `torch.futures`: Future


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

- **File Documentation**: `storage.py_docs.md`
- **Keyword Index**: `storage.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/distributed/checkpoint/_checkpointer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_checkpointer.py_docs.md`
- **Size**: 7,214 bytes (7.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_checkpointer.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_checkpointer.py`
- **Size**: 3,802 bytes (3.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from concurrent.futures import Future
from typing import Any, Optional

import torch.distributed as dist
import torch.distributed.checkpoint.state_dict_loader as loader
import torch.distributed.checkpoint.state_dict_saver as saver
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import (
    LoadPlanner,
    SavePlanner,
    StorageReader,
    StorageWriter,
)


__all__: list[str] = []


class _Checkpointer:
    """This base class specifies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    def __init__(
        self,
        storage_writer: StorageWriter,
        storage_reader: StorageReader,
        *,
        process_group: Optional[dist.ProcessGroup] = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        load_planner: Optional[LoadPlanner] = None,
        save_planner: Optional[SavePlanner] = None,
    ):
        """Initializes the Checkpointer instance.

        Args:
            storage_writer: Instance of StorageWrite use to perform writes.
            storage_reader: StorageReader used to load data from.
            process_group: ProcessGroup to be used for cross-rank synchronization.
            coordinator_rank: Rank to use to coordinate the checkpoint. rank0 is used by default.
            no_dist: If ``True``, distributed checkpoint will not load in SPMD style. (Default: ``False``)
            loader_planner: Instance of LoadPlanner to use when loading.
            save_planner: Instance of SavePlanner to use when saving.
        """
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.load_planner = load_planner
        self.save_planner = save_planner

    def save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Metadata:
        """Calls :py:meth: `torch.distributed.state_dict_saver.save`. Utilizing values passed during initialization."""
        return saver.save(
            state_dict,
            self.storage_writer,
            process_group=self.process_group,
            coordinator_rank=self.coordinator_rank,
            no_dist=self.no_dist,
            planner=self.save_planner,
        )

    def async_save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Future:
        """
        Calls :py:meth: `torch.distributed.state_dict_saver._async_save`. Utilizing values passed during initialization.

        Returns:
            Future: A future holding the resultant Metadata object from `save`.
        """
        response = saver.async_save(
            state_dict,
            storage_writer=self.storage_writer,
            process_group=self.process_group,
            planner=self.save_planner,
        )
        if not isinstance(response, Future):
            raise AssertionError("response should be a Future instance")
        return response

    def load(self, state_dict: dict[str, Any]) -> None:
        """Calls :py:meth: `torch.distributed.state_dict_loader.load`. Utilizing values passed during initialization."""
        loader.load(
            state_dict,
            storage_reader=self.storage_reader,
            process_group=self.process_group,
            planner=self.load_planner,
        )

```



## High-Level Overview

"""This base class specifies a high level API for saving and loading    distributed `state_dict` 's. It provides an abstraction over the low-level APIs    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling    :py:meth: `torch.distributed.state_dict_saver.save` and    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage    readers and writers.    .. warning::        This feature is experimental and subject to removal/change.

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_Checkpointer`

**Functions defined**: `__init__`, `save`, `async_save`, `load`

**Key imports**: Future, Any, Optional, torch.distributed as dist, torch.distributed.checkpoint.state_dict_loader as loader, torch.distributed.checkpoint.state_dict_saver as saver, Metadata, STATE_DICT_TYPE


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `concurrent.futures`: Future
- `typing`: Any, Optional
- `torch.distributed as dist`
- `torch.distributed.checkpoint.state_dict_loader as loader`
- `torch.distributed.checkpoint.state_dict_saver as saver`
- `torch.distributed.checkpoint.metadata`: Metadata, STATE_DICT_TYPE


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

- **File Documentation**: `_checkpointer.py_docs.md`
- **Keyword Index**: `_checkpointer.py_kw.md`
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

- **File Documentation**: `_checkpointer.py_docs.md_docs.md`
- **Keyword Index**: `_checkpointer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/distributed/checkpoint/_async_thread_executor.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_async_thread_executor.py_docs.md`
- **Size**: 5,639 bytes (5.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_async_thread_executor.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_async_thread_executor.py`
- **Size**: 2,476 bytes (2.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# pyre-strict
# mypy: allow-untyped-defs
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

import torch.distributed as dist
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


def save_wrapper(
    staging_future_or_state_dict: Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE],
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Future:
    from torch.distributed.checkpoint.state_dict_saver import save

    staged_dict = (
        staging_future_or_state_dict.result()
        if isinstance(staging_future_or_state_dict, Future)
        else staging_future_or_state_dict
    )
    return save(
        staged_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="AsyncCheckpointExecutor"
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE],
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        f: Future = self._executor.submit(
            save_wrapper,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_ThreadBasedAsyncCheckpointExecutor`

**Functions defined**: `save_wrapper`, `__init__`, `execute_save`

**Key imports**: os, Future, ThreadPoolExecutor, Optional, Union, torch.distributed as dist, _AsyncCheckpointExecutor, STATE_DICT_TYPE, SavePlanner, StorageWriter, save


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `concurrent.futures`: Future, ThreadPoolExecutor
- `typing`: Optional, Union
- `torch.distributed as dist`
- `torch.distributed.checkpoint._async_executor`: _AsyncCheckpointExecutor
- `torch.distributed.checkpoint.metadata`: STATE_DICT_TYPE
- `torch.distributed.checkpoint.planner`: SavePlanner
- `torch.distributed.checkpoint.storage`: StorageWriter
- `torch.distributed.checkpoint.state_dict_saver`: save


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `_async_thread_executor.py_docs.md`
- **Keyword Index**: `_async_thread_executor.py_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `_async_thread_executor.py_docs.md_docs.md`
- **Keyword Index**: `_async_thread_executor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

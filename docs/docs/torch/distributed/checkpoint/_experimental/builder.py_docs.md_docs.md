# Documentation: `docs/torch/distributed/checkpoint/_experimental/builder.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_experimental/builder.py_docs.md`
- **Size**: 9,467 bytes (9.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_experimental/builder.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_experimental/builder.py`
- **Size**: 6,123 bytes (5.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Factory functions for creating checkpointer instances with sensible defaults.

This module provides high-level factory functions that simplify the creation
of checkpointer instances by automatically handling component initialization
and configuration with reasonable defaults.
"""

from collections.abc import Callable
from typing import Any, Optional

import torch.distributed as dist

from .barriers import create_barrier_from_config
from .checkpoint_process import CheckpointProcess
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig, WriterHook
from .checkpointer import AsyncCheckpointer, SyncCheckpointer
from .config import CheckpointerConfig
from .staging import DefaultStager
from .types import RankInfo


def _get_default_rank_info() -> RankInfo:
    """
    Get default rank information from the current distributed environment.

    Returns:
        RankInfo: Rank information from the default process group if initialized,
                 otherwise single-rank fallback.
    """
    if dist.is_initialized():
        return RankInfo(
            global_world_size=dist.get_world_size(),
            global_rank=dist.get_rank(),
        )
    else:
        # Single-rank fallback
        return RankInfo(global_world_size=1, global_rank=0)


def default_subprocess_init_fn(*_: Any) -> None:
    """Default subprocess initialization function (no-op)."""


def default_writer_init_fn(rank_info: RankInfo) -> CheckpointWriter:
    """Default checkpoint writer initialization function."""
    return CheckpointWriter(
        config=CheckpointWriterConfig(),
        rank_info=rank_info,
    )


def make_sync_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: Optional[RankInfo] = None,
    commit_hook: Optional[WriterHook] = None,
) -> SyncCheckpointer:
    """
    Factory function to create a SyncCheckpointer instance with sensible defaults.

    This function creates a synchronous checkpointer with default components, automatically
    detecting rank information from the default process group if available, and using the
    provided component configurations.

    Args:
        config: CheckpointerConfig containing component-specific configurations
               (writer_config, staging_config, process_config). Defaults to CheckpointerConfig().
        rank_info: RankInfo for distributed training. Defaults to auto-detection from
                  the default PyTorch distributed process group if initialized, otherwise
                  falls back to single-rank (world_size=1, rank=0).
        commit_hook: Optional hook for custom actions before and after checkpoint commits.

    Returns:
        SyncCheckpointer: A configured synchronous checkpointer instance.

    Examples:
        # Simplest usage - auto-detect rank, default config
        checkpointer = make_sync_checkpointer()

        # Explicit rank configuration
        checkpointer = make_sync_checkpointer(
            rank_info=RankInfo(global_world_size=4, global_rank=0)
        )

        # Disable barrier
        from .barriers import BarrierConfig
        config = CheckpointerConfig(barrier_config=BarrierConfig(barrier_type=None))
        checkpointer = make_sync_checkpointer(config=config)
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    barrier = create_barrier_from_config(config.barrier_config)

    writer = CheckpointWriter(
        config=config.writer_config,
        rank_info=rank_info,
        barrier=barrier,
        commit_hook=commit_hook,
    )

    return SyncCheckpointer(
        writer=writer,
        reader=reader,
    )


def make_async_checkpointer(
    config: CheckpointerConfig = CheckpointerConfig(),
    rank_info: Optional[RankInfo] = None,
    subprocess_init_fn: Callable[..., None] = default_subprocess_init_fn,
    subprocess_init_args: tuple[Any, ...] = (),
    checkpoint_writer_init_fn: Callable[..., CheckpointWriter] = default_writer_init_fn,
    checkpoint_writer_init_args: Optional[dict[str, Any]] = None,
) -> AsyncCheckpointer:
    """
    Factory function to create an AsyncCheckpointer instance with sensible defaults.

    This function creates an asynchronous checkpointer using the provided configuration,
    automatically detecting rank information if not provided.

    Args:
        config: CheckpointerConfig containing component-specific configurations.
        rank_info: RankInfo for distributed training. Defaults to auto-detection.
        subprocess_init_fn: Function to initialize the subprocess. Defaults to no-op.
        subprocess_init_args: Arguments to pass to subprocess_init_fn.
        checkpoint_writer_init_fn: Function to create CheckpointWriter instance.
        checkpoint_writer_init_args: Arguments to pass to checkpoint_writer_init_fn.

    Returns:
        AsyncCheckpointer: A configured asynchronous checkpointer instance.

    Examples:
        # Create with default config
        checkpointer = make_async_checkpointer()

        # Create with custom init functions
        checkpointer = make_async_checkpointer(
            subprocess_init_fn=my_subprocess_init_fn,
            checkpoint_writer_init_fn=my_writer_init_fn
        )
    """
    if rank_info is None:
        rank_info = _get_default_rank_info()

    reader = CheckpointReader(
        rank_info=rank_info,
    )

    checkpoint_stager = DefaultStager(
        config=config.staging_config,
    )

    checkpoint_writer_init_args = checkpoint_writer_init_args or {}

    checkpoint_process = CheckpointProcess(
        rank_info=rank_info,
        config=config.process_config,
        subprocess_init_fn=subprocess_init_fn,
        subprocess_init_args=subprocess_init_args,
        checkpoint_writer_init_fn=checkpoint_writer_init_fn,
        checkpoint_writer_init_args=checkpoint_writer_init_args,
    )

    return AsyncCheckpointer(
        checkpoint_stager=checkpoint_stager,
        checkpoint_process=checkpoint_process,
        reader=reader,
    )

```



## High-Level Overview

"""Factory functions for creating checkpointer instances with sensible defaults.This module provides high-level factory functions that simplify the creationof checkpointer instances by automatically handling component initializationand configuration with reasonable defaults.

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_default_rank_info`, `default_subprocess_init_fn`, `default_writer_init_fn`, `make_sync_checkpointer`, `make_async_checkpointer`

**Key imports**: Callable, Any, Optional, torch.distributed as dist, create_barrier_from_config, CheckpointProcess, CheckpointReader, CheckpointWriter, CheckpointWriterConfig, WriterHook, AsyncCheckpointer, SyncCheckpointer, CheckpointerConfig, DefaultStager


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint/_experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Any, Optional
- `torch.distributed as dist`
- `.barriers`: create_barrier_from_config
- `.checkpoint_process`: CheckpointProcess
- `.checkpoint_reader`: CheckpointReader
- `.checkpoint_writer`: CheckpointWriter, CheckpointWriterConfig, WriterHook
- `.checkpointer`: AsyncCheckpointer, SyncCheckpointer
- `.config`: CheckpointerConfig
- `.staging`: DefaultStager
- `.types`: RankInfo


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/checkpoint/_experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`barriers.py_docs.md`](./barriers.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`checkpointer.py_docs.md`](./checkpointer.py_docs.md)
- [`checkpoint_writer.py_docs.md`](./checkpoint_writer.py_docs.md)
- [`checkpoint_process.py_docs.md`](./checkpoint_process.py_docs.md)
- [`staging.py_docs.md`](./staging.py_docs.md)


## Cross-References

- **File Documentation**: `builder.py_docs.md`
- **Keyword Index**: `builder.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/checkpoint/_experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/checkpoint/_experimental`, which is part of the **core PyTorch library**.



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

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/checkpoint/_experimental`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`barriers.py_docs.md_docs.md`](./barriers.py_docs.md_docs.md)
- [`checkpoint_process.py_kw.md_docs.md`](./checkpoint_process.py_kw.md_docs.md)
- [`checkpoint_reader.py_kw.md_docs.md`](./checkpoint_reader.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`config.py_kw.md_docs.md`](./config.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`checkpointer.py_docs.md_docs.md`](./checkpointer.py_docs.md_docs.md)
- [`staging.py_kw.md_docs.md`](./staging.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `builder.py_docs.md_docs.md`
- **Keyword Index**: `builder.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

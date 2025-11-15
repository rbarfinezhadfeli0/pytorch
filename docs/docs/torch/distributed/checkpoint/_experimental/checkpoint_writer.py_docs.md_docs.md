# Documentation: `docs/torch/distributed/checkpoint/_experimental/checkpoint_writer.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_experimental/checkpoint_writer.py_docs.md`
- **Size**: 8,319 bytes (8.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_experimental/checkpoint_writer.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_experimental/checkpoint_writer.py`
- **Size**: 5,270 bytes (5.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Checkpoint writer functionality for machine learning models.

This module provides classes for writing checkpoints to storage, including
determining checkpoint layout, configuring the writer, and defining hooks
for custom actions during the checkpoint writing process.
"""

import abc
import logging
import os
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .barriers import Barrier
from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


class WriterHook(abc.ABC):
    """
    Abstract base class for checkpoint commit hooks.

    A commit hook provides callbacks that are executed before and after a checkpoint
    is committed to storage. This allows for custom actions to be performed at specific
    points in the checkpoint writing process, such as metadata updates, cleanup operations,
    or notifications.
    """

    @abc.abstractmethod
    def pre_commit(self, path: str, **kwargs: dict[str, Any]) -> None:
        """
        Performs actions before committing the checkpoint.
        """

    @abc.abstractmethod
    def post_commit(self, path: str, **kwargs: dict[str, Any]) -> None:
        """
        Performs actions after committing the checkpoint.
        """


@dataclass
class CheckpointWriterConfig:
    """
    Configuration options for the CheckpointWriter.

    Attributes:
        write_barrier_timeout_secs: Maximum time in seconds to wait for all ranks
            to reach the checkpoint barrier before timing out. Default is 600 seconds.
    """

    write_barrier_timeout_secs: int = 600


class CheckpointWriter:
    """
    Handles writing state dictionaries to storage.

    This class is responsible for writing model state dictionaries to storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        config: CheckpointWriterConfig,
        rank_info: RankInfo,
        barrier: Optional[Barrier] = None,
        commit_hook: Optional[WriterHook] = None,
    ):
        """
        Initialize a CheckpointWriter.

        Args:
            config: Configuration options for the checkpoint writer.
            rank_info: Information about the current rank in a distributed setting.
            barrier: Optional synchronization barrier for distributed checkpointing.
                    Note: The barrier should be initialized with the appropriate barrier_prefix
                    and timeout_secs parameters.
            commit_hook: Optional hook for custom actions before and after checkpoint commits.
        """

        self._config = config
        self._rank_info = rank_info
        self._commit_hook = commit_hook
        self._barrier = barrier

    def write(
        self,
        path: str,
        state_dict: STATE_DICT,
        **kwargs: dict[str, Any],
    ) -> Optional[Future[None]]:
        """
        Writes the state_dict to storage.

        Args:
            path (str): The path to write the checkpoint to.
            state_dict (STATE_DICT): The state_dict to write.
            **kwargs: Additional keyword arguments passed to hooks.

        Returns:
            Optional[Future[None]]: A future for tracking the write operation, if applicable.
        """
        logger.debug(
            "Writing checkpoint to %s for rank %s",
            path,
            self._rank_info.global_rank,
        )
        dir_path = Path(path)
        full_path = dir_path / f"checkpoint_{self._rank_info.global_rank}.pt"
        os.makedirs(
            os.path.dirname(full_path),
            exist_ok=True,
        )
        torch.save(state_dict, full_path)
        logger.debug("Successfully saved checkpoint file to %s", full_path)

        # Execute pre-commit hook if available
        commit_hook = self._commit_hook
        if commit_hook is not None:
            logger.debug("Executing pre-commit hook for %s", path)
            commit_hook.pre_commit(path, **kwargs)

        # Wait for all ranks to finish writing if barrier is available
        barrier = self._barrier
        if barrier is not None:
            logger.info(
                "Waiting for all ranks at barrier with timeout %ss",
                self._config.write_barrier_timeout_secs,
            )
            barrier.execute_barrier()
            logger.info("All ranks passed barrier")
        else:
            logger.info("No barrier configured, skipping synchronization")

        # Execute commit hook if available
        if commit_hook is not None:
            logger.debug("Executing commit hook for %s", path)
            commit_hook.post_commit(path, **kwargs)

        logger.info(
            "Successfully wrote checkpoint to %s for rank %s",
            path,
            self._rank_info.global_rank,
        )
        return None

    def close(self) -> None:
        """
        Close the writer and release any resources.

        This is a no-op for the base CheckpointWriter but may be overridden
        by subclasses that need to perform cleanup.
        """
        logger.debug("Closing checkpoint writer")

```



## High-Level Overview

"""Checkpoint writer functionality for machine learning models.This module provides classes for writing checkpoints to storage, includingdetermining checkpoint layout, configuring the writer, and defining hooksfor custom actions during the checkpoint writing process.

This Python file contains 6 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WriterHook`, `CheckpointWriterConfig`, `CheckpointWriter`

**Functions defined**: `pre_commit`, `post_commit`, `__init__`, `write`, `close`

**Key imports**: abc, logging, os, Future, dataclass, Path, Any, Optional, torch, Barrier, RankInfo, STATE_DICT


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint/_experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `logging`
- `os`
- `concurrent.futures`: Future
- `dataclasses`: dataclass
- `pathlib`: Path
- `typing`: Any, Optional
- `torch`
- `.barriers`: Barrier
- `.types`: RankInfo, STATE_DICT


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

Files in the same folder (`torch/distributed/checkpoint/_experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`barriers.py_docs.md`](./barriers.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`checkpointer.py_docs.md`](./checkpointer.py_docs.md)
- [`checkpoint_process.py_docs.md`](./checkpoint_process.py_docs.md)
- [`staging.py_docs.md`](./staging.py_docs.md)
- [`builder.py_docs.md`](./builder.py_docs.md)


## Cross-References

- **File Documentation**: `checkpoint_writer.py_docs.md`
- **Keyword Index**: `checkpoint_writer.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`docs/torch/distributed/checkpoint/_experimental`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`barriers.py_docs.md_docs.md`](./barriers.py_docs.md_docs.md)
- [`checkpoint_process.py_kw.md_docs.md`](./checkpoint_process.py_kw.md_docs.md)
- [`checkpoint_reader.py_kw.md_docs.md`](./checkpoint_reader.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`config.py_kw.md_docs.md`](./config.py_kw.md_docs.md)
- [`builder.py_docs.md_docs.md`](./builder.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`checkpointer.py_docs.md_docs.md`](./checkpointer.py_docs.md_docs.md)
- [`staging.py_kw.md_docs.md`](./staging.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `checkpoint_writer.py_docs.md_docs.md`
- **Keyword Index**: `checkpoint_writer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

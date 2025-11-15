# Documentation: `docs/torch/distributed/checkpoint/_experimental/checkpoint_reader.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/_experimental/checkpoint_reader.py_docs.md`
- **Size**: 11,637 bytes (11.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/checkpoint/_experimental/checkpoint_reader.py`

## File Metadata

- **Path**: `torch/distributed/checkpoint/_experimental/checkpoint_reader.py`
- **Size**: 8,695 bytes (8.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Checkpoint reader functionality for machine learning models.

This module provides classes for reading checkpoints from storage, including
determining checkpoint layout and configuring the reader.
"""

import logging
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from .types import RankInfo, STATE_DICT


logger = logging.getLogger(__name__)


class CheckpointReader:
    """
    Handles reading state dictionaries from storage.

    This class is responsible for reading model state dictionaries from storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        rank_info: RankInfo,
    ):
        """
        Initialize a CheckpointReader.

        Args:
            rank_info: Information about the current rank in a distributed setting.
        """

        self._rank_info = rank_info

    def read(
        self,
        path: str,
        state_dict: Optional[STATE_DICT] = None,
        *,
        map_location: Any = None,
        **kwargs: dict[str, Any],
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads a state dictionary from storage.

        Args:
            path (str): The path from which to read the checkpoint.
            map_location (Any): Device mapping function or device name for relocating tensors.
            **kwargs: Additional keyword arguments passed to torch.load.

        Returns:
            STATE_DICT: The loaded state dictionary.
            list[str]: List of missing keys.
        """
        logger.debug(
            "Reading checkpoint from %s for rank %s",
            path,
            self._rank_info.global_rank,
        )

        dir_path = Path(path)
        file_path = dir_path / f"checkpoint_{self._rank_info.global_rank}.pt"

        # Check if the file exists
        if not os.path.exists(file_path):
            logger.error("Checkpoint file not found at %s", file_path)
            raise FileNotFoundError(f"Checkpoint file not found at {file_path}")

        if state_dict is None:
            result: tuple[STATE_DICT, list[str]] = (
                torch.load(file_path, map_location=map_location),
                [],
            )
        else:
            result = self._partial_read(
                file_path, state_dict, map_location=map_location, **kwargs
            )
        logger.debug("Successfully read checkpoint file from %s", file_path)
        return result

    def _partial_read(
        self,
        file_path: Path,
        state_dict: STATE_DICT,
        *,
        map_location: Any = None,
        **kwargs: dict[str, Any],
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads only the keys present in state_dict from the checkpoint file.

        This method optimizes checkpoint loading by only loading the tensors that
        are actually needed, based on the keys present in the input state_dict.
        This can significantly reduce memory usage and loading time for large checkpoints
        when only a subset of the model needs to be loaded.

        Args:
            file_path (str): The path to the checkpoint file.
            state_dict (STATE_DICT): The state dictionary containing keys to load.
            map_location (Any): Device mapping function or device name for relocating tensors.
            **kwargs: Additional keyword arguments passed to torch.load.

        Returns:
            tuple[STATE_DICT, list[str]]: The updated state dictionary with loaded values and a list of missing keys.
        """

        with FakeTensorMode():
            metadata_dict = torch.load(file_path, map_location=map_location)

        missing_keys = []

        with open(file_path, "rb") as file:
            # Helper function to load tensor data from file
            def load_tensor(
                target: Optional[torch.Tensor], source: torch.Tensor, full_key: str
            ) -> torch.Tensor:
                if target is not None and (
                    target.size() != source.size() or target.dtype != source.dtype
                ):
                    raise RuntimeError(
                        f"Target tensor size={target.size()} dtype={target.dtype} does not match "
                        f"source tensor size={source.size()} dtype={source.dtype} for key {full_key}"
                    )

                tensor_offset = source.untyped_storage()._checkpoint_offset

                if tensor_offset is None:
                    raise AssertionError(
                        "checkpoint_offset for tensor in torch serialized file is not set. This could "
                        "happen if the checkpoint was saved with a older version of Pytorch. "
                        "Please make sure that the checkpoint was saved with Pytorch 2.7 or later."
                    )

                tensor_len = source.nelement() * source.element_size()
                file.seek(
                    tensor_offset + source.element_size() * int(source.storage_offset())
                )
                if target is None:
                    target = torch.empty(
                        source.size(), dtype=source.dtype, device=source.device
                    )

                buffer = file.read(tensor_len)
                cpu_tensor = torch.frombuffer(buffer, dtype=source.dtype)
                tensor = cpu_tensor.view(source.size())
                target.copy_(tensor)
                return target

            # Helper function to recursively process nested structures
            def process_value(
                target_value: Any, source_value: Any, key_path: str
            ) -> Any:
                source_type = type(source_value)
                if source_type is torch._subclasses.fake_tensor.FakeTensor:
                    source_type = torch.Tensor
                if target_value is not None and not isinstance(
                    target_value, source_type
                ):
                    raise RuntimeError(
                        f"Target value {key_path} is set to {type(target_value)}, but source value is {type(source_value)}"
                    )
                if isinstance(source_value, torch.Tensor):
                    return load_tensor(target_value, source_value, key_path)
                elif isinstance(source_value, dict):
                    if target_value is None:
                        # create a new map with all the keys present in source_value
                        target_value = dict.fromkeys(source_value.keys())

                    # pyrefly: ignore [missing-attribute]
                    for key in list(target_value.keys()):
                        current_path = f"{key_path}.{key}" if key_path else key
                        if key in source_value:
                            target_value[key] = process_value(
                                target_value[key], source_value[key], current_path
                            )
                        else:
                            missing_keys.append(current_path)

                    return target_value
                elif isinstance(source_value, list):
                    if target_value is None:
                        target_value = [None] * len(source_value)
                    result = []
                    for i, (target_item, source_item) in enumerate(
                        zip_longest(target_value, source_value, fillvalue=None)
                    ):
                        current_path = f"{key_path}[{i}]" if key_path else f"[{i}]"
                        result.append(
                            process_value(target_item, source_item, current_path)
                        )
                    return result
                else:
                    return source_value

            # Start recursive processing from the root of the state dictionary
            updated_state_dict = process_value(state_dict, metadata_dict, "")

        if missing_keys:
            if len(missing_keys) > 10:
                logger.warning(
                    "Missing %s keys from checkpoint: %s... (and %s more)",
                    len(missing_keys),
                    missing_keys[:10],
                    len(missing_keys) - 10,
                )
            else:
                logger.warning(
                    "Missing %s keys from checkpoint: %s",
                    len(missing_keys),
                    missing_keys,
                )

        return updated_state_dict, missing_keys

```



## High-Level Overview

"""Checkpoint reader functionality for machine learning models.This module provides classes for reading checkpoints from storage, includingdetermining checkpoint layout and configuring the reader.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CheckpointReader`

**Functions defined**: `__init__`, `read`, `_partial_read`, `load_tensor`, `process_value`

**Key imports**: logging, os, zip_longest, Path, Any, Optional, torch, FakeTensorMode, RankInfo, STATE_DICT


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/checkpoint/_experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `os`
- `itertools`: zip_longest
- `pathlib`: Path
- `typing`: Any, Optional
- `torch`
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `.types`: RankInfo, STATE_DICT


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
- [`builder.py_docs.md`](./builder.py_docs.md)


## Cross-References

- **File Documentation**: `checkpoint_reader.py_docs.md`
- **Keyword Index**: `checkpoint_reader.py_kw.md`
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

- **File Documentation**: `checkpoint_reader.py_docs.md_docs.md`
- **Keyword Index**: `checkpoint_reader.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

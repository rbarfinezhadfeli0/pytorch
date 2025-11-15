# Documentation: `docs/test/distributed/checkpoint/_experimental/test_checkpoint_writer.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/_experimental/test_checkpoint_writer.py_docs.md`
- **Size**: 11,051 bytes (10.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/_experimental/test_checkpoint_writer.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/_experimental/test_checkpoint_writer.py`
- **Size**: 7,106 bytes (6.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed checkpointing"]

import os
import shutil
import tempfile
from typing import Any, Optional
from unittest.mock import MagicMock

import torch
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
    WriterHook,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class MockWriterHook(WriterHook):
    """Mock implementation of WriterHook for testing."""

    def __init__(self):
        self.pre_commit_called = False
        self.commit_called = False
        self.pre_commit_path: Optional[str] = None
        self.commit_path: Optional[str] = None
        self.pre_commit_kwargs: Optional[dict[str, Any]] = None
        self.commit_kwargs: Optional[dict[str, Any]] = None

    def pre_commit(self, path: str, **kwargs: Any):
        self.pre_commit_called = True
        self.pre_commit_path = path
        self.pre_commit_kwargs = kwargs

    def post_commit(self, path: str, **kwargs: Any):
        self.commit_called = True
        self.commit_path = path
        self.commit_kwargs = kwargs


class TestCheckpointWriterConfig(TestCase):
    def test_default_values(self):
        """Test that CheckpointWriterConfig has the correct default values."""
        options = CheckpointWriterConfig()
        self.assertEqual(options.write_barrier_timeout_secs, 600)

    def test_custom_values(self):
        """Test that CheckpointWriterConfig can be initialized with custom values."""
        options = CheckpointWriterConfig(write_barrier_timeout_secs=300)
        self.assertEqual(options.write_barrier_timeout_secs, 300)


class TestCheckpointWriter(TestCase):
    def setUp(self):
        super().setUp()
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test objects
        self.rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
        )
        self.options = CheckpointWriterConfig()
        self.mock_barrier = MagicMock()
        self.mock_hook = MockWriterHook()

        # Create the checkpoint writer
        self.writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=self.mock_barrier,
            commit_hook=self.mock_hook,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_write_creates_checkpoint_file(self):
        """Test that write creates a checkpoint file with the correct content."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write
        self.writer.write(checkpoint_path, self.state_dict)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

        # Load the checkpoint and verify its contents
        loaded_state_dict = torch.load(expected_file_path)
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)

    def test_write_calls_barrier(self):
        """Test that write calls the barrier with the correct parameters."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write
        self.writer.write(checkpoint_path, self.state_dict)

        # Verify that the barrier was called
        self.mock_barrier.execute_barrier.assert_called_once()

    def test_write_calls_commit_hooks(self):
        """Test that write calls the commit hooks with the correct parameters."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write with additional kwargs
        kwargs = {"extra": "value"}
        self.writer.write(checkpoint_path, self.state_dict, **kwargs)

        # Verify that the pre_commit hook was called with the correct parameters
        self.assertTrue(self.mock_hook.pre_commit_called)
        self.assertEqual(self.mock_hook.pre_commit_path, checkpoint_path)
        self.assertEqual(
            self.mock_hook.pre_commit_kwargs is not None
            and self.mock_hook.pre_commit_kwargs["extra"],
            "value",
        )

        # Verify that the commit hook was called with the correct parameters
        self.assertTrue(self.mock_hook.commit_called)
        self.assertEqual(self.mock_hook.commit_path, checkpoint_path)
        self.assertEqual(
            self.mock_hook.commit_kwargs is not None
            and self.mock_hook.commit_kwargs["extra"],
            "value",
        )

    def test_write_without_barrier(self):
        """Test that write works correctly without a barrier."""
        # Create a writer without a barrier
        writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=None,
            commit_hook=self.mock_hook,
        )

        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_barrier")

        # Call write
        writer.write(checkpoint_path, self.state_dict)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

    def test_write_without_commit_hook(self):
        """Test that write works correctly without a commit hook."""
        # Create a writer without a commit hook
        writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=self.mock_barrier,
            commit_hook=None,
        )

        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_hook")

        # Call write
        writer.write(checkpoint_path, self.state_dict)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

        # Verify that the barrier was still called
        self.mock_barrier.execute_barrier.assert_called_once()

    def test_close(self):
        """Test that close doesn't raise any exceptions."""
        # This is a no-op in the base class, so just verify it doesn't raise
        self.writer.close()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Mock implementation of WriterHook for testing."""    def __init__(self):        self.pre_commit_called = False        self.commit_called = False        self.pre_commit_path: Optional[str] = None        self.commit_path: Optional[str] = None        self.pre_commit_kwargs: Optional[dict[str, Any]] = None        self.commit_kwargs: Optional[dict[str, Any]] = None    def pre_commit(self, path: str, **kwargs: Any):        self.pre_commit_called = True        self.pre_commit_path = path        self.pre_commit_kwargs = kwargs    def post_commit(self, path: str, **kwargs: Any):        self.commit_called = True        self.commit_path = path        self.commit_kwargs = kwargsclass TestCheckpointWriterConfig(TestCase):    def test_default_values(self):

This Python file contains 3 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MockWriterHook`, `TestCheckpointWriterConfig`, `TestCheckpointWriter`

**Functions defined**: `__init__`, `pre_commit`, `post_commit`, `test_default_values`, `test_custom_values`, `setUp`, `tearDown`, `test_write_creates_checkpoint_file`, `test_write_calls_barrier`, `test_write_calls_commit_hooks`, `test_write_without_barrier`, `test_write_without_commit_hook`, `test_close`

**Key imports**: os, shutil, tempfile, Any, Optional, MagicMock, torch, RankInfo, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint/_experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `tempfile`
- `typing`: Any, Optional
- `unittest.mock`: MagicMock
- `torch`
- `torch.distributed.checkpoint._experimental.types`: RankInfo
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/checkpoint/_experimental/test_checkpoint_writer.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint/_experimental`):

- [`test_staging.py_docs.md`](./test_staging.py_docs.md)
- [`test_barriers.py_docs.md`](./test_barriers.py_docs.md)
- [`test_checkpoint_process.py_docs.md`](./test_checkpoint_process.py_docs.md)
- [`test_builder.py_docs.md`](./test_builder.py_docs.md)
- [`test_checkpoint_reader.py_docs.md`](./test_checkpoint_reader.py_docs.md)
- [`test_checkpointer.py_docs.md`](./test_checkpointer.py_docs.md)
- [`test_types.py_docs.md`](./test_types.py_docs.md)


## Cross-References

- **File Documentation**: `test_checkpoint_writer.py_docs.md`
- **Keyword Index**: `test_checkpoint_writer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/checkpoint/_experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/checkpoint/_experimental`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/test/distributed/checkpoint/_experimental/test_checkpoint_writer.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/checkpoint/_experimental`):

- [`test_staging.py_docs.md_docs.md`](./test_staging.py_docs.md_docs.md)
- [`test_builder.py_docs.md_docs.md`](./test_builder.py_docs.md_docs.md)
- [`test_staging.py_kw.md_docs.md`](./test_staging.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)
- [`test_checkpointer.py_docs.md_docs.md`](./test_checkpointer.py_docs.md_docs.md)
- [`test_checkpoint_process.py_kw.md_docs.md`](./test_checkpoint_process.py_kw.md_docs.md)
- [`test_checkpointer.py_kw.md_docs.md`](./test_checkpointer.py_kw.md_docs.md)
- [`test_checkpoint_process.py_docs.md_docs.md`](./test_checkpoint_process.py_docs.md_docs.md)
- [`test_checkpoint_writer.py_kw.md_docs.md`](./test_checkpoint_writer.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_checkpoint_writer.py_docs.md_docs.md`
- **Keyword Index**: `test_checkpoint_writer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

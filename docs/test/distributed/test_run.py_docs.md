# Documentation: `test/distributed/test_run.py`

## File Metadata

- **Path**: `test/distributed/test_run.py`
- **Size**: 3,525 bytes (3.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import MagicMock, patch

import torch.distributed.run as run
from torch.distributed.launcher.api import launch_agent, LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


class RunTest(TestCase):
    def setUp(self):
        super().setUp()
        # Save original environment variable if it exists
        self.original_signals_env = os.environ.get(
            "TORCHELASTIC_SIGNALS_TO_HANDLE", None
        )

    def tearDown(self):
        # Restore original environment variable
        if self.original_signals_env is not None:
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"] = self.original_signals_env
        elif "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

    def test_signals_to_handle_default(self):
        """Test that the default value for signals_to_handle is correctly set."""
        parser = run.get_args_parser()
        args = parser.parse_args(["dummy_script.py"])
        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGINT,SIGHUP,SIGQUIT")

    def test_signals_to_handle_custom(self):
        """Test that a custom value for signals_to_handle is correctly parsed."""
        parser = run.get_args_parser()
        args = parser.parse_args(
            ["--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2", "dummy_script.py"]
        )
        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGUSR1,SIGUSR2")

    def test_config_from_args_signals_to_handle(self):
        """Test that the signals_to_handle argument is correctly passed to LaunchConfig."""
        parser = run.get_args_parser()
        args = parser.parse_args(
            ["--signals-to-handle=SIGTERM,SIGUSR1,SIGUSR2", "dummy_script.py"]
        )
        config, _, _ = run.config_from_args(args)
        self.assertEqual(config.signals_to_handle, "SIGTERM,SIGUSR1,SIGUSR2")

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_sets_environment_variable(self, mock_get_handler, mock_agent):
        """Test that launch_agent sets the TORCHELASTIC_SIGNALS_TO_HANDLE environment variable."""
        # Setup
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",
        )
        entrypoint = "dummy_script.py"
        args = []

        # Make sure the environment variable doesn't exist before the test
        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:
            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]

        # Mock agent.run() to return a MagicMock
        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = MagicMock(
            is_failed=lambda: False, return_values={}
        )
        mock_agent.return_value = mock_agent_instance

        # Call launch_agent
        launch_agent(config, entrypoint, args)

        # Verify that the environment variable was set correctly
        self.assertEqual(
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"], "SIGTERM,SIGUSR1,SIGUSR2"
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test that the default value for signals_to_handle is correctly set."""        parser = run.get_args_parser()        args = parser.parse_args(["dummy_script.py"])        self.assertEqual(args.signals_to_handle, "SIGTERM,SIGINT,SIGHUP,SIGQUIT")    def test_signals_to_handle_custom(self):

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RunTest`

**Functions defined**: `setUp`, `tearDown`, `test_signals_to_handle_default`, `test_signals_to_handle_custom`, `test_config_from_args_signals_to_handle`, `test_launch_agent_sets_environment_variable`

**Key imports**: os, MagicMock, patch, torch.distributed.run as run, launch_agent, LaunchConfig, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest.mock`: MagicMock, patch
- `torch.distributed.run as run`
- `torch.distributed.launcher.api`: launch_agent, LaunchConfig
- `torch.testing._internal.common_utils`: run_tests, TestCase


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

This is a test file. Run it with:

```bash
python test/distributed/test_run.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_c10d_logger.py_docs.md`](./test_c10d_logger.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_run.py_docs.md`
- **Keyword Index**: `test_run.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

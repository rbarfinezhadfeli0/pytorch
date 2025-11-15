# Documentation: `docs/test/distributed/launcher/test_api.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/launcher/test_api.py_docs.md`
- **Size**: 6,743 bytes (6.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/launcher/test_api.py`

## File Metadata

- **Path**: `test/distributed/launcher/test_api.py`
- **Size**: 3,706 bytes (3.62 KB)
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

from torch.distributed.launcher.api import launch_agent, LaunchConfig
from torch.testing._internal.common_utils import run_tests, TestCase


class LauncherApiTest(TestCase):
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

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_sets_signals_env_var(self, mock_get_handler, mock_agent):
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

    @patch("torch.distributed.launcher.api.LocalElasticAgent")
    @patch("torch.distributed.launcher.api.rdzv_registry.get_rendezvous_handler")
    def test_launch_agent_default_signals(self, mock_get_handler, mock_agent):
        """Test that launch_agent uses the default signals if not specified."""
        # Setup
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=1,
            # Not specifying signals_to_handle, should use default
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

        # Verify that the environment variable was set to the default value
        self.assertEqual(
            os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"],
            "SIGTERM,SIGINT,SIGHUP,SIGQUIT",
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test that launch_agent sets the TORCHELASTIC_SIGNALS_TO_HANDLE environment variable."""        # Setup        config = LaunchConfig(            min_nodes=1,            max_nodes=1,            nproc_per_node=1,            signals_to_handle="SIGTERM,SIGUSR1,SIGUSR2",        )        entrypoint = "dummy_script.py"        args = []        # Make sure the environment variable doesn't exist before the test        if "TORCHELASTIC_SIGNALS_TO_HANDLE" in os.environ:            del os.environ["TORCHELASTIC_SIGNALS_TO_HANDLE"]        # Mock agent.run() to return a MagicMock

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LauncherApiTest`

**Functions defined**: `setUp`, `tearDown`, `test_launch_agent_sets_signals_env_var`, `test_launch_agent_default_signals`

**Key imports**: os, MagicMock, patch, launch_agent, LaunchConfig, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/launcher`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest.mock`: MagicMock, patch
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
python test/distributed/launcher/test_api.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/launcher`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`api_test.py_docs.md`](./api_test.py_docs.md)
- [`launch_test.py_docs.md`](./launch_test.py_docs.md)
- [`script_deviceid.py_docs.md`](./script_deviceid.py_docs.md)


## Cross-References

- **File Documentation**: `test_api.py_docs.md`
- **Keyword Index**: `test_api.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/launcher`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/launcher`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/distributed/launcher/test_api.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/launcher`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`launch_test.py_kw.md_docs.md`](./launch_test.py_kw.md_docs.md)
- [`test_api.py_kw.md_docs.md`](./test_api.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`script_deviceid.py_kw.md_docs.md`](./script_deviceid.py_kw.md_docs.md)
- [`test_run.py_docs.md_docs.md`](./test_run.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`launch_test.py_docs.md_docs.md`](./launch_test.py_docs.md_docs.md)
- [`api_test.py_kw.md_docs.md`](./api_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_api.py_docs.md_docs.md`
- **Keyword Index**: `test_api.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `.ci/lumen_cli/tests/test_docker_helper.py`

## File Metadata

- **Path**: `.ci/lumen_cli/tests/test_docker_helper.py`
- **Size**: 2,973 bytes (2.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
import unittest
from unittest import mock
from unittest.mock import MagicMock

import docker.errors as derr
from cli.lib.common.docker_helper import _get_client, local_image_exists


class TestDockerImageHelpers(unittest.TestCase):
    def setUp(self):
        # Reset the singleton in the target module
        patcher = mock.patch("cli.lib.common.docker_helper._docker_client", None)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_local_image_exists_true(self):
        # Mock a docker client whose images.get returns an object (no exception)
        mock_client = MagicMock()
        mock_client.images.get.return_value = object()
        ok = local_image_exists("repo:tag", client=mock_client)
        self.assertTrue(ok)

    def test_local_image_exists_not_found_false(self):
        mock_client = MagicMock()
        # Raise docker.errors.NotFound
        mock_client.images.get.side_effect = derr.NotFound("nope")
        ok = local_image_exists("missing:latest", client=mock_client)
        self.assertFalse(ok)

    def test_local_image_exists_api_error_false(self):
        mock_client = MagicMock()
        mock_client.images.get.side_effect = derr.APIError("boom", None)

        ok = local_image_exists("broken:tag", client=mock_client)
        self.assertFalse(ok)

    def test_local_image_exists_uses_lazy_singleton(self):
        # Patch docker.from_env used by _get_client()
        with mock.patch(
            "cli.lib.common.docker_helper.docker.from_env"
        ) as mock_from_env:
            mock_docker_client = MagicMock()
            mock_from_env.return_value = mock_docker_client

            # First call should create and cache the client
            c1 = _get_client()
            self.assertIs(c1, mock_docker_client)
            mock_from_env.assert_called_once()

            # Second call should reuse cached client (no extra from_env calls)
            c2 = _get_client()
            self.assertIs(c2, mock_docker_client)
            mock_from_env.assert_called_once()  # still once

    def test_local_image_exists_without_client_param_calls_get_client_once(self):
        # Ensure _get_client is called and cached; local_image_exists should reuse it
        with mock.patch("cli.lib.common.docker_helper._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # 1st call
            local_image_exists("repo:tag")
            # 2nd call
            local_image_exists("repo:tag2")

            # local_image_exists should call _get_client each time,
            # but your _get_client itself caches docker.from_env.
            self.assertEqual(mock_get_client.call_count, 2)
            self.assertEqual(mock_client.images.get.call_count, 2)
            mock_client.images.get.assert_any_call("repo:tag")
            mock_client.images.get.assert_any_call("repo:tag2")


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDockerImageHelpers`

**Functions defined**: `setUp`, `test_local_image_exists_true`, `test_local_image_exists_not_found_false`, `test_local_image_exists_api_error_false`, `test_local_image_exists_uses_lazy_singleton`, `test_local_image_exists_without_client_param_calls_get_client_once`

**Key imports**: unittest, mock, MagicMock, docker.errors as derr, _get_client, local_image_exists


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/lumen_cli/tests`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `unittest.mock`: MagicMock
- `docker.errors as derr`
- `cli.lib.common.docker_helper`: _get_client, local_image_exists


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .ci/lumen_cli/tests/test_docker_helper.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.ci/lumen_cli/tests`):

- [`test_cli_helper.py_docs.md`](./test_cli_helper.py_docs.md)
- [`test_app.py_docs.md`](./test_app.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_vllm.py_docs.md`](./test_vllm.py_docs.md)
- [`test_run_plan.py_docs.md`](./test_run_plan.py_docs.md)
- [`test_envs_helper.py_docs.md`](./test_envs_helper.py_docs.md)
- [`test_path_helper.py_docs.md`](./test_path_helper.py_docs.md)


## Cross-References

- **File Documentation**: `test_docker_helper.py_docs.md`
- **Keyword Index**: `test_docker_helper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

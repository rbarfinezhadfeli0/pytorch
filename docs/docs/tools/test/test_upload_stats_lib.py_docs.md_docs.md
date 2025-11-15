# Documentation: `docs/tools/test/test_upload_stats_lib.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_upload_stats_lib.py_docs.md`
- **Size**: 11,957 bytes (11.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_upload_stats_lib.py`

## File Metadata

- **Path**: `tools/test/test_upload_stats_lib.py`
- **Size**: 8,433 bytes (8.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import gzip
import inspect
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.upload_metrics import add_global_metric, emit_metric, global_metrics
from tools.stats.upload_stats_lib import get_s3_resource, remove_nan_inf


sys.path.remove(str(REPO_ROOT))

# default values
REPO = "some/repo"
BUILD_ENV = "cuda-10.2"
TEST_CONFIG = "test-config"
WORKFLOW = "some-workflow"
JOB = "some-job"
RUN_ID = 56
RUN_NUMBER = 123
RUN_ATTEMPT = 3
PR_NUMBER = 6789
JOB_ID = 234
JOB_NAME = "some-job-name"


@mock.patch("boto3.resource")
class TestUploadStats(unittest.TestCase):
    emitted_metric: dict[str, Any] = {"did_not_emit": True}

    def mock_put_item(self, **kwargs: Any) -> None:
        # Utility for mocking putting items into s3.  THis will save the emitted
        # metric so tests can check it
        self.emitted_metric = json.loads(
            gzip.decompress(kwargs["Body"]).decode("utf-8")
        )

    # Before each test, set the env vars to their default values
    def setUp(self) -> None:
        get_s3_resource.cache_clear()
        global_metrics.clear()

        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
                "TEST_CONFIG": TEST_CONFIG,
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_WORKFLOW": WORKFLOW,
                "GITHUB_JOB": JOB,
                "GITHUB_RUN_ID": str(RUN_ID),
                "GITHUB_RUN_NUMBER": str(RUN_NUMBER),
                "GITHUB_RUN_ATTEMPT": str(RUN_ATTEMPT),
                "JOB_ID": str(JOB_ID),
                "JOB_NAME": str(JOB_NAME),
            },
            clear=True,  # Don't read any preset env vars
        ).start()

    def test_emits_default_and_given_metrics(self, mock_resource: Any) -> None:
        metric = {
            "some_number": 123,
            "float_number": 32.34,
        }

        # Querying for this instead of hard coding it b/c this will change
        # based on whether we run this test directly from python or from
        # pytest
        current_module = inspect.getmodule(inspect.currentframe()).__name__  # type: ignore[union-attr]

        emit_should_include = {
            "metric_name": "metric_name",
            "calling_file": "test_upload_stats_lib.py",
            "calling_module": current_module,
            "calling_function": "test_emits_default_and_given_metrics",
            "repo": REPO,
            "workflow": WORKFLOW,
            "build_environment": BUILD_ENV,
            "job": JOB,
            "test_config": TEST_CONFIG,
            "run_id": RUN_ID,
            "run_number": RUN_NUMBER,
            "run_attempt": RUN_ATTEMPT,
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
            "info": metric,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, **emit_should_include},
        )

    def test_when_global_metric_specified_then_it_emits_it(
        self, mock_resource: Any
    ) -> None:
        metric = {
            "some_number": 123,
        }

        global_metric_name = "global_metric"
        global_metric_value = "global_value"

        add_global_metric(global_metric_name, global_metric_value)

        emit_should_include = {
            **metric,
            global_metric_name: global_metric_value,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
        )

    def test_when_local_and_global_metric_specified_then_global_is_overridden(
        self, mock_resource: Any
    ) -> None:
        global_metric_name = "global_metric"
        global_metric_value = "global_value"
        local_override = "local_override"

        add_global_metric(global_metric_name, global_metric_value)

        metric = {
            "some_number": 123,
            global_metric_name: local_override,
        }

        emit_should_include = {
            **metric,
            global_metric_name: local_override,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
        )

    def test_when_optional_envvar_set_to_actual_value_then_emit_vars_emits_it(
        self, mock_resource: Any
    ) -> None:
        metric = {
            "some_number": 123,
        }

        emit_should_include = {
            "info": {**metric},
            "pr_number": PR_NUMBER,
        }

        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": str(PR_NUMBER),
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, **emit_should_include},
        )

    def test_when_optional_envvar_set_to_a_empty_str_then_emit_vars_ignores_it(
        self, mock_resource: Any
    ) -> None:
        metric = {"some_number": 123}

        emit_should_include: dict[str, Any] = metric.copy()

        # Github Actions defaults some env vars to an empty string
        default_val = ""
        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": default_val,
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
            f"Metrics should be emitted when an option parameter is set to '{default_val}'",
        )
        self.assertFalse(
            self.emitted_metric.get("pr_number"),
            f"Metrics should not include optional item 'pr_number' when it's envvar is set to '{default_val}'",
        )

    def test_no_metrics_emitted_if_required_env_var_not_set(
        self, mock_resource: Any
    ) -> None:
        metric = {"some_number": 123}

        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
            },
            clear=True,
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertTrue(self.emitted_metric["did_not_emit"])

    def test_no_metrics_emitted_if_required_env_var_set_to_empty_string(
        self, mock_resource: Any
    ) -> None:
        metric = {"some_number": 123}

        mock.patch.dict(
            "os.environ",
            {
                "GITHUB_JOB": "",
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertTrue(self.emitted_metric["did_not_emit"])

    def test_remove_nan_inf(self, _mocked_resource: Any) -> None:
        checks = [
            (float("inf"), '"inf"', "Infinity"),
            (float("nan"), '"nan"', "NaN"),
            ({1: float("inf")}, '{"1": "inf"}', '{"1": Infinity}'),
            ([float("nan")], '["nan"]', "[NaN]"),
            ({1: [float("nan")]}, '{"1": ["nan"]}', '{"1": [NaN]}'),
        ]

        for input, clean, unclean in checks:
            clean_output = json.dumps(remove_nan_inf(input))
            unclean_output = json.dumps(input)
            self.assertEqual(
                clean_output,
                clean,
                f"Expected {clean} when input is {unclean}, got {clean_output}",
            )
            self.assertEqual(
                unclean_output,
                unclean,
                f"Expected {unclean} when input is {unclean}, got {unclean_output}",
            )


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview


This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestUploadStats`

**Functions defined**: `mock_put_item`, `setUp`, `test_emits_default_and_given_metrics`, `test_when_global_metric_specified_then_it_emits_it`, `test_when_local_and_global_metric_specified_then_global_is_overridden`, `test_when_optional_envvar_set_to_actual_value_then_emit_vars_emits_it`, `test_when_optional_envvar_set_to_a_empty_str_then_emit_vars_ignores_it`, `test_no_metrics_emitted_if_required_env_var_not_set`, `test_no_metrics_emitted_if_required_env_var_set_to_empty_string`, `test_remove_nan_inf`

**Key imports**: annotations, gzip, inspect, json, sys, unittest, Path, Any, mock, add_global_metric, emit_metric, global_metrics


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `gzip`
- `inspect`
- `json`
- `sys`
- `unittest`
- `pathlib`: Path
- `typing`: Any
- `tools.stats.upload_metrics`: add_global_metric, emit_metric, global_metrics
- `tools.stats.upload_stats_lib`: get_s3_resource, remove_nan_inf


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python tools/test/test_upload_stats_lib.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test`):

- [`test_codegen.py_docs.md`](./test_codegen.py_docs.md)
- [`linter_test_case.py_docs.md`](./linter_test_case.py_docs.md)
- [`test_upload_gate.py_docs.md`](./test_upload_gate.py_docs.md)
- [`test_gen_backend_stubs.py_docs.md`](./test_gen_backend_stubs.py_docs.md)
- [`test_gb_registry_linter.py_docs.md`](./test_gb_registry_linter.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_set_linter.py_docs.md`](./test_set_linter.py_docs.md)
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `test_upload_stats_lib.py_docs.md`
- **Keyword Index**: `test_upload_stats_lib.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python docs/tools/test/test_upload_stats_lib.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/test`):

- [`test_gen_backend_stubs.py_kw.md_docs.md`](./test_gen_backend_stubs.py_kw.md_docs.md)
- [`test_upload_stats_lib.py_kw.md_docs.md`](./test_upload_stats_lib.py_kw.md_docs.md)
- [`test_cmake.py_kw.md_docs.md`](./test_cmake.py_kw.md_docs.md)
- [`test_upload_test_stats.py_docs.md_docs.md`](./test_upload_test_stats.py_docs.md_docs.md)
- [`test_codegen_model.py_docs.md_docs.md`](./test_codegen_model.py_docs.md_docs.md)
- [`test_codegen.py_docs.md_docs.md`](./test_codegen.py_docs.md_docs.md)
- [`test_vulkan_codegen.py_kw.md_docs.md`](./test_vulkan_codegen.py_kw.md_docs.md)
- [`test_set_linter.py_docs.md_docs.md`](./test_set_linter.py_docs.md_docs.md)
- [`test_gb_registry_linter.py_kw.md_docs.md`](./test_gb_registry_linter.py_kw.md_docs.md)
- [`test_upload_test_stats.py_kw.md_docs.md`](./test_upload_test_stats.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_upload_stats_lib.py_docs.md_docs.md`
- **Keyword Index**: `test_upload_stats_lib.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

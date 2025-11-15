# Documentation: `tools/test/test_create_alerts.py`

## File Metadata

- **Path**: `tools/test/test_create_alerts.py`
- **Size**: 2,772 bytes (2.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

from typing import Any
from unittest import main, TestCase

from tools.alerts.create_alerts import filter_job_names, JobStatus


JOB_NAME = "periodic / linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck / test (default, 2, 2, linux.4xlarge.nvidia.gpu)"
MOCK_TEST_DATA = [
    {
        "sha": "f02f3046571d21b48af3067e308a1e0f29b43af9",
        "id": 7819529276,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7819529276?check_suite_focus=true",  # @lint-ignore
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7819529276",
        "durationS": 14876,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 83818,
        "repo": "pytorch/pytorch",
    },
    {
        "sha": "d0d6b1f2222bf90f478796d84a525869898f55b6",
        "id": 7818399623,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7818399623?check_suite_focus=true",  # @lint-ignore
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7818399623",
        "durationS": 14882,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 72821,
        "repo": "pytorch/pytorch",
    },
]


class TestGitHubPR(TestCase):
    # Should fail when jobs are ? ? Fail Fail
    def test_alert(self) -> None:
        modified_data: list[Any] = [{}]
        modified_data.append({})
        modified_data.extend(MOCK_TEST_DATA)
        status = JobStatus(JOB_NAME, modified_data)
        self.assertTrue(status.should_alert())

    # test filter job names
    def test_job_filter(self) -> None:
        job_names = [
            "pytorch_linux_xenial_py3_6_gcc5_4_test",
            "pytorch_linux_xenial_py3_6_gcc5_4_test2",
        ]
        self.assertListEqual(
            filter_job_names(job_names, ""),
            job_names,
            "empty regex should match all jobs",
        )
        self.assertListEqual(filter_job_names(job_names, ".*"), job_names)
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*"), job_names)
        self.assertListEqual(
            filter_job_names(job_names, ".*xenial.*test2"),
            ["pytorch_linux_xenial_py3_6_gcc5_4_test2"],
        )
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*test3"), [])
        self.assertRaises(
            Exception,
            lambda: filter_job_names(job_names, "["),
            msg="malformed regex should throw exception",
        )


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGitHubPR`

**Functions defined**: `test_alert`, `test_job_filter`

**Key imports**: annotations, Any, main, TestCase, filter_job_names, JobStatus


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: Any
- `unittest`: main, TestCase
- `tools.alerts.create_alerts`: filter_job_names, JobStatus


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python tools/test/test_create_alerts.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test`):

- [`test_upload_stats_lib.py_docs.md`](./test_upload_stats_lib.py_docs.md)
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

- **File Documentation**: `test_create_alerts.py_docs.md`
- **Keyword Index**: `test_create_alerts.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

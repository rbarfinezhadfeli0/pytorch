# Documentation: `docs/tools/stats/upload_test_stats_running_jobs.py_docs.md`

## File Metadata

- **Path**: `docs/tools/stats/upload_test_stats_running_jobs.py_docs.md`
- **Size**: 5,167 bytes (5.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/stats/upload_test_stats_running_jobs.py`

## File Metadata

- **Path**: `tools/stats/upload_test_stats_running_jobs.py`
- **Size**: 2,189 bytes (2.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import sys
import time
from functools import cache
from typing import Any

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_stats_lib import get_s3_resource
from tools.stats.upload_test_stats import get_tests


BUCKET_PREFIX = "workflows_failing_pending_upload"


@cache
def get_bucket() -> Any:
    return get_s3_resource().Bucket("gha-artifacts")


def delete_obj(key: str) -> None:
    # Does not raise error if key does not exist
    get_bucket().delete_objects(
        Delete={
            "Objects": [{"Key": key}],
            "Quiet": True,
        }
    )


def put_object(key: str) -> None:
    get_bucket().put_object(
        Key=key,
        Body=b"",
    )


def do_upload(workflow_id: int) -> None:
    workflow_attempt = 1
    test_cases = get_tests(workflow_id, workflow_attempt)
    # Flush stdout so that any errors in upload show up last in the logs.
    sys.stdout.flush()
    upload_additional_info(workflow_id, workflow_attempt, test_cases)


def get_workflow_ids(pending: bool = False) -> list[int]:
    prefix = f"{BUCKET_PREFIX}/{'pending/' if pending else ''}"
    objs = get_bucket().objects.filter(Prefix=prefix)
    return [int(obj.key.split("/")[-1].split(".")[0]) for obj in objs]


def read_s3(pending: bool = False) -> None:
    while True:
        workflows = get_workflow_ids(pending)
        if not workflows:
            if pending:
                break
            # Wait for more stuff to show up
            print("Sleeping for 60 seconds")
            time.sleep(60)
        for workflow_id in workflows:
            print(f"Processing {workflow_id}")
            put_object(f"{BUCKET_PREFIX}/pending/{workflow_id}.txt")
            delete_obj(f"{BUCKET_PREFIX}/{workflow_id}.txt")
            try:
                do_upload(workflow_id)
            except Exception as e:
                print(f"Failed to upload {workflow_id}: {e}")
            delete_obj(f"{BUCKET_PREFIX}/pending/{workflow_id}.txt")


if __name__ == "__main__":
    # Workflows in the pending folder were previously in progress of uploading
    # but failed to complete, so we need to retry them.
    read_s3(pending=True)
    read_s3()

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_bucket`, `delete_obj`, `put_object`, `do_upload`, `get_workflow_ids`, `read_s3`

**Key imports**: sys, time, cache, Any, upload_additional_info, get_s3_resource, get_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/stats`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `time`
- `functools`: cache
- `typing`: Any
- `tools.stats.test_dashboard`: upload_additional_info
- `tools.stats.upload_stats_lib`: get_s3_resource
- `tools.stats.upload_test_stats`: get_tests


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python tools/stats/upload_test_stats_running_jobs.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/stats`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`upload_sccache_stats.py_docs.md`](./upload_sccache_stats.py_docs.md)
- [`upload_external_contrib_stats.py_docs.md`](./upload_external_contrib_stats.py_docs.md)
- [`check_disabled_tests.py_docs.md`](./check_disabled_tests.py_docs.md)
- [`upload_metrics.py_docs.md`](./upload_metrics.py_docs.md)
- [`import_test_stats.py_docs.md`](./import_test_stats.py_docs.md)
- [`utilization_stats_lib.py_docs.md`](./utilization_stats_lib.py_docs.md)
- [`upload_artifacts.py_docs.md`](./upload_artifacts.py_docs.md)
- [`upload_test_stats_intermediate.py_docs.md`](./upload_test_stats_intermediate.py_docs.md)
- [`export_test_times.py_docs.md`](./export_test_times.py_docs.md)


## Cross-References

- **File Documentation**: `upload_test_stats_running_jobs.py_docs.md`
- **Keyword Index**: `upload_test_stats_running_jobs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/stats`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/stats`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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
python docs/tools/stats/upload_test_stats_running_jobs.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/stats`):

- [`upload_dynamo_perf_stats.py_docs.md_docs.md`](./upload_dynamo_perf_stats.py_docs.md_docs.md)
- [`check_disabled_tests.py_kw.md_docs.md`](./check_disabled_tests.py_kw.md_docs.md)
- [`import_test_stats.py_docs.md_docs.md`](./import_test_stats.py_docs.md_docs.md)
- [`upload_artifacts.py_docs.md_docs.md`](./upload_artifacts.py_docs.md_docs.md)
- [`upload_stats_lib.py_kw.md_docs.md`](./upload_stats_lib.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_dashboard.py_docs.md_docs.md`](./test_dashboard.py_docs.md_docs.md)
- [`upload_test_stats.py_kw.md_docs.md`](./upload_test_stats.py_kw.md_docs.md)
- [`utilization_stats_lib.py_docs.md_docs.md`](./utilization_stats_lib.py_docs.md_docs.md)
- [`upload_test_stats_running_jobs.py_kw.md_docs.md`](./upload_test_stats_running_jobs.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `upload_test_stats_running_jobs.py_docs.md_docs.md`
- **Keyword Index**: `upload_test_stats_running_jobs.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

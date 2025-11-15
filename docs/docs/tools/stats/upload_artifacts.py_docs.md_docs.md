# Documentation: `docs/tools/stats/upload_artifacts.py_docs.md`

## File Metadata

- **Path**: `docs/tools/stats/upload_artifacts.py_docs.md`
- **Size**: 4,719 bytes (4.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/stats/upload_artifacts.py`

## File Metadata

- **Path**: `tools/stats/upload_artifacts.py`
- **Size**: 2,064 bytes (2.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import os
import re
from tempfile import TemporaryDirectory

from tools.stats.upload_stats_lib import download_gha_artifacts, upload_file_to_s3


ARTIFACTS = [
    "sccache-stats",
    "test-jsons",
    "test-reports",
    "usage-log",
]
BUCKET_NAME = "gha-artifacts"
FILENAME_REGEX = r"-runattempt\d+"


def get_artifacts(repo: str, workflow_run_id: int, workflow_run_attempt: int) -> None:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        for artifact in ARTIFACTS:
            artifact_paths = download_gha_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )

            for artifact_path in artifact_paths:
                # GHA artifact is named as follows: NAME-runattempt${{ github.run_attempt }}-SUFFIX.zip
                # and we want remove the run_attempt to conform with the naming convention on S3, i.e.
                # pytorch/pytorch/WORKFLOW_ID/RUN_ATTEMPT/artifact/NAME-SUFFIX.zip
                s3_filename = re.sub(FILENAME_REGEX, "", artifact_path.name)
                upload_file_to_s3(
                    file_name=str(artifact_path.resolve()),
                    bucket=BUCKET_NAME,
                    key=f"{repo}/{workflow_run_id}/{workflow_run_attempt}/artifact/{s3_filename}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test artifacts from GHA to S3")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    args = parser.parse_args()
    get_artifacts(args.repo, args.workflow_run_id, args.workflow_run_attempt)

```



## High-Level Overview


This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_artifacts`

**Key imports**: argparse, os, re, TemporaryDirectory, download_gha_artifacts, upload_file_to_s3


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/stats`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `os`
- `re`
- `tempfile`: TemporaryDirectory
- `tools.stats.upload_stats_lib`: download_gha_artifacts, upload_file_to_s3


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

Test files for this module may be located in the `test/` directory.

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
- [`upload_test_stats_intermediate.py_docs.md`](./upload_test_stats_intermediate.py_docs.md)
- [`export_test_times.py_docs.md`](./export_test_times.py_docs.md)


## Cross-References

- **File Documentation**: `upload_artifacts.py_docs.md`
- **Keyword Index**: `upload_artifacts.py_kw.md`
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

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/stats`):

- [`upload_dynamo_perf_stats.py_docs.md_docs.md`](./upload_dynamo_perf_stats.py_docs.md_docs.md)
- [`check_disabled_tests.py_kw.md_docs.md`](./check_disabled_tests.py_kw.md_docs.md)
- [`import_test_stats.py_docs.md_docs.md`](./import_test_stats.py_docs.md_docs.md)
- [`upload_stats_lib.py_kw.md_docs.md`](./upload_stats_lib.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_dashboard.py_docs.md_docs.md`](./test_dashboard.py_docs.md_docs.md)
- [`upload_test_stats.py_kw.md_docs.md`](./upload_test_stats.py_kw.md_docs.md)
- [`utilization_stats_lib.py_docs.md_docs.md`](./utilization_stats_lib.py_docs.md_docs.md)
- [`upload_test_stats_running_jobs.py_kw.md_docs.md`](./upload_test_stats_running_jobs.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `upload_artifacts.py_docs.md_docs.md`
- **Keyword Index**: `upload_artifacts.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

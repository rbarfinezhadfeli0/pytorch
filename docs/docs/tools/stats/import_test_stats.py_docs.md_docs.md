# Documentation: `docs/tools/stats/import_test_stats.py_docs.md`

## File Metadata

- **Path**: `docs/tools/stats/import_test_stats.py_docs.md`
- **Size**: 9,376 bytes (9.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/stats/import_test_stats.py`

## File Metadata

- **Path**: `tools/stats/import_test_stats.py`
- **Size**: 6,203 bytes (6.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```python
#!/usr/bin/env python3

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING
from urllib.request import urlopen


if TYPE_CHECKING:
    from collections.abc import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_disabled_issues() -> list[str]:
    reenabled_issues = os.getenv("REENABLED_ISSUES", "")
    issue_numbers = reenabled_issues.split(",")
    print("Ignoring disabled issues: ", issue_numbers)
    return issue_numbers


DISABLED_TESTS_FILE = ".pytorch-disabled-tests.json"
ADDITIONAL_CI_FILES_FOLDER = Path(".additional_ci_files")
TEST_TIMES_FILE = "test-times.json"
TEST_CLASS_TIMES_FILE = "test-class-times.json"
TEST_FILE_RATINGS_FILE = "test-file-ratings.json"
TEST_CLASS_RATINGS_FILE = "test-class-ratings.json"
TD_HEURISTIC_PROFILING_FILE = "td_heuristic_profiling.json"
TD_HEURISTIC_HISTORICAL_EDITED_FILES = "td_heuristic_historical_edited_files.json"
TD_HEURISTIC_PREVIOUSLY_FAILED = "previous_failures.json"
TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL = "previous_failures_additional.json"

FILE_CACHE_LIFESPAN_SECONDS = datetime.timedelta(hours=3).seconds


def fetch_and_cache(
    dirpath: str | Path,
    name: str,
    url: str,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """
    This fetch and cache utils allows sharing between different process.
    """
    Path(dirpath).mkdir(exist_ok=True)

    path = os.path.join(dirpath, name)
    print(f"Downloading {url} to {path}")

    def is_cached_file_valid() -> bool:
        # Check if the file is new enough (see: FILE_CACHE_LIFESPAN_SECONDS). A real check
        # could make a HEAD request and check/store the file's ETag
        fname = Path(path)
        now = datetime.datetime.now()
        mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
        diff = now - mtime
        return diff.total_seconds() < FILE_CACHE_LIFESPAN_SECONDS

    if os.path.exists(path) and is_cached_file_valid():
        # Another test process already download the file, so don't re-do it
        with open(path) as f:
            return cast(dict[str, Any], json.load(f))

    for _ in range(3):
        try:
            contents = urlopen(url, timeout=5).read().decode("utf-8")
            processed_contents = process_fn(json.loads(contents))
            with open(path, "w") as f:
                f.write(json.dumps(processed_contents))
            return processed_contents
        except Exception as e:
            print(f"Could not download {url} because: {e}.")
    print(f"All retries exhausted, downloading {url} failed.")
    return {}


def get_test_times() -> dict[str, dict[str, float]]:
    return get_from_test_infra_generated_stats(
        "test-times.json",
        TEST_TIMES_FILE,
        "Couldn't download test times...",
    )


def get_test_class_times() -> dict[str, dict[str, float]]:
    return get_from_test_infra_generated_stats(
        "test-class-times.json",
        TEST_CLASS_TIMES_FILE,
        "Couldn't download test times...",
    )


def get_disabled_tests(
    dirpath: str, filename: str = DISABLED_TESTS_FILE
) -> dict[str, Any] | None:
    def process_disabled_test(the_response: dict[str, Any]) -> dict[str, Any]:
        # remove re-enabled tests and condense even further by getting rid of pr_num
        disabled_issues = get_disabled_issues()
        disabled_test_from_issues = {}
        for test_name, (pr_num, link, platforms) in the_response.items():
            if pr_num not in disabled_issues:
                disabled_test_from_issues[test_name] = (
                    link,
                    platforms,
                )
        return disabled_test_from_issues

    try:
        url = "https://ossci-metrics.s3.amazonaws.com/disabled-tests-condensed.json"
        return fetch_and_cache(dirpath, filename, url, process_disabled_test)
    except Exception:
        print("Couldn't download test skip set, leaving all tests enabled...")
        return {}


def get_test_file_ratings() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "file_test_rating.json",
        TEST_FILE_RATINGS_FILE,
        "Couldn't download test file ratings file, not reordering...",
    )


def get_test_class_ratings() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "file_test_class_rating.json",
        TEST_CLASS_RATINGS_FILE,
        "Couldn't download test class ratings file, not reordering...",
    )


def get_td_heuristic_historial_edited_files_json() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "td_heuristic_historical_edited_files.json",
        TD_HEURISTIC_HISTORICAL_EDITED_FILES,
        "Couldn't download td_heuristic_historical_edited_files.json, not reordering...",
    )


def get_td_heuristic_profiling_json() -> dict[str, Any]:
    return get_from_test_infra_generated_stats(
        "td_heuristic_profiling.json",
        TD_HEURISTIC_PROFILING_FILE,
        "Couldn't download td_heuristic_profiling.json not reordering...",
    )


def copy_pytest_cache() -> None:
    original_path = REPO_ROOT / ".pytest_cache/v/cache/lastfailed"
    if not original_path.exists():
        return
    shutil.copyfile(
        original_path,
        REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PREVIOUSLY_FAILED,
    )


def copy_additional_previous_failures() -> None:
    original_path = (
        REPO_ROOT / ".pytest_cache" / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    )
    if not original_path.exists():
        return
    shutil.copyfile(
        original_path,
        REPO_ROOT
        / ADDITIONAL_CI_FILES_FOLDER
        / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL,
    )


def get_from_test_infra_generated_stats(
    from_file: str, to_file: str, failure_explanation: str
) -> dict[str, Any]:
    url = f"https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/{from_file}"
    try:
        return fetch_and_cache(
            REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER, to_file, url, lambda x: x
        )
    except Exception:
        print(failure_explanation)
        return {}

```



## High-Level Overview

"""    This fetch and cache utils allows sharing between different process.

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_disabled_issues`, `fetch_and_cache`, `is_cached_file_valid`, `get_test_times`, `get_test_class_times`, `get_disabled_tests`, `process_disabled_test`, `get_test_file_ratings`, `get_test_class_ratings`, `get_td_heuristic_historial_edited_files_json`, `get_td_heuristic_profiling_json`, `copy_pytest_cache`, `copy_additional_previous_failures`, `get_from_test_infra_generated_stats`

**Key imports**: annotations, datetime, json, os, shutil, Path, Any, cast, TYPE_CHECKING, urlopen, Callable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/stats`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `datetime`
- `json`
- `os`
- `shutil`
- `pathlib`: Path
- `typing`: Any, cast, TYPE_CHECKING
- `urllib.request`: urlopen
- `collections.abc`: Callable


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
python tools/stats/import_test_stats.py
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
- [`utilization_stats_lib.py_docs.md`](./utilization_stats_lib.py_docs.md)
- [`upload_artifacts.py_docs.md`](./upload_artifacts.py_docs.md)
- [`upload_test_stats_intermediate.py_docs.md`](./upload_test_stats_intermediate.py_docs.md)
- [`export_test_times.py_docs.md`](./export_test_times.py_docs.md)


## Cross-References

- **File Documentation**: `import_test_stats.py_docs.md`
- **Keyword Index**: `import_test_stats.py_kw.md`
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
python docs/tools/stats/import_test_stats.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/stats`):

- [`upload_dynamo_perf_stats.py_docs.md_docs.md`](./upload_dynamo_perf_stats.py_docs.md_docs.md)
- [`check_disabled_tests.py_kw.md_docs.md`](./check_disabled_tests.py_kw.md_docs.md)
- [`upload_artifacts.py_docs.md_docs.md`](./upload_artifacts.py_docs.md_docs.md)
- [`upload_stats_lib.py_kw.md_docs.md`](./upload_stats_lib.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_dashboard.py_docs.md_docs.md`](./test_dashboard.py_docs.md_docs.md)
- [`upload_test_stats.py_kw.md_docs.md`](./upload_test_stats.py_kw.md_docs.md)
- [`utilization_stats_lib.py_docs.md_docs.md`](./utilization_stats_lib.py_docs.md_docs.md)
- [`upload_test_stats_running_jobs.py_kw.md_docs.md`](./upload_test_stats_running_jobs.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `import_test_stats.py_docs.md_docs.md`
- **Keyword Index**: `import_test_stats.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

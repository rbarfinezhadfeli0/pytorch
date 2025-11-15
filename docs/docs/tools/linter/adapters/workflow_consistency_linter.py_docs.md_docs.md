# Documentation: `docs/tools/linter/adapters/workflow_consistency_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/workflow_consistency_linter.py_docs.md`
- **Size**: 7,671 bytes (7.49 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/workflow_consistency_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/workflow_consistency_linter.py`
- **Size**: 4,576 bytes (4.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""Checks for consistency of jobs between different GitHub workflows.

Any job with a specific `sync-tag` must match all other jobs with the same `sync-tag`.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from yaml import dump, load


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


if TYPE_CHECKING:
    from collections.abc import Iterable


# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def glob_yamls(path: Path) -> Iterable[Path]:
    return itertools.chain(path.glob("**/*.yml"), path.glob("**/*.yaml"))


def load_yaml(path: Path) -> Any:
    with open(path) as f:
        return load(f, Loader)


def is_workflow(yaml: Any) -> bool:
    return yaml.get("jobs") is not None


def print_lint_message(
    path: Path,
    job: dict[str, Any],
    sync_tag: str,
    baseline_path: Path,
    baseline_job_id: str,
) -> None:
    job_id = next(iter(job.keys()))
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if f"{job_id}:" in line:
            line_number = i + 1

    lint_message = LintMessage(
        path=str(path),
        # pyrefly: ignore [unbound-name]
        line=line_number,
        char=None,
        code="WORKFLOWSYNC",
        severity=LintSeverity.ERROR,
        name="workflow-inconsistency",
        original=None,
        replacement=None,
        description=f"Job doesn't match other job {baseline_job_id} in file {baseline_path} with sync-tag: '{sync_tag}'",
    )
    print(json.dumps(lint_message._asdict()), flush=True)


def get_jobs_with_sync_tag(
    job: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    sync_tag = job.get("with", {}).get("sync-tag")
    if sync_tag is None:
        return None

    # remove the "if" field, which we allow to be different between jobs
    # (since you might have different triggering conditions on pull vs.
    # trunk, say.)
    if "if" in job:
        del job["if"]

    # same is true for ['with']['test-matrix']
    if "test-matrix" in job.get("with", {}):
        del job["with"]["test-matrix"]
    return (sync_tag, job_id, job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="workflow consistency linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    # Go through all files, aggregating jobs with the same sync tag
    tag_to_jobs = defaultdict(list)
    for path in REPO_ROOT.glob(".github/workflows/*"):
        if not path.is_file() or path.suffix not in {".yml", ".yaml"}:
            continue
        workflow = load_yaml(path)
        if not is_workflow(workflow):
            continue
        clean_path = path.relative_to(REPO_ROOT)
        jobs = workflow.get("jobs", {})
        for job_id, job in jobs.items():
            res = get_jobs_with_sync_tag(job)
            if res is None:
                continue
            sync_tag, job_id, job_dict = res
            tag_to_jobs[sync_tag].append((clean_path, job_id, job_dict))

    # Check the files passed as arguments
    for path in args.filenames:
        workflow = load_yaml(Path(path))
        jobs = workflow["jobs"]
        for job_id, job in jobs.items():
            res = get_jobs_with_sync_tag(job)
            if res is None:
                continue
            sync_tag, job_id, job_dict = res
            job_str = dump(job_dict)

            # For each sync tag, check that all the jobs have the same code.
            for baseline_path, baseline_job_id, baseline_dict in tag_to_jobs[sync_tag]:
                baseline_str = dump(baseline_dict)

                if job_id != baseline_job_id or job_str != baseline_str:
                    print_lint_message(
                        path, job_dict, sync_tag, baseline_path, baseline_job_id
                    )

```



## High-Level Overview

"""Checks for consistency of jobs between different GitHub workflows.Any job with a specific `sync-tag` must match all other jobs with the same `sync-tag`.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `glob_yamls`, `load_yaml`, `is_workflow`, `print_lint_message`, `get_jobs_with_sync_tag`

**Key imports**: annotations, argparse, itertools, json, defaultdict, Enum, Path, Any, NamedTuple, TYPE_CHECKING, dump, load, Iterable


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `itertools`
- `json`
- `collections`: defaultdict
- `enum`: Enum
- `pathlib`: Path
- `typing`: Any, NamedTuple, TYPE_CHECKING
- `yaml`: dump, load
- `collections.abc`: Iterable


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`tools/linter/adapters`):

- [`grep_linter.py_docs.md`](./grep_linter.py_docs.md)
- [`import_linter.py_docs.md`](./import_linter.py_docs.md)
- [`gha_linter.py_docs.md`](./gha_linter.py_docs.md)
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `workflow_consistency_linter.py_docs.md`
- **Keyword Index**: `workflow_consistency_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/linter/adapters`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/tools/linter/adapters`):

- [`pyrefly_linter.py_kw.md_docs.md`](./pyrefly_linter.py_kw.md_docs.md)
- [`codespell_linter.py_kw.md_docs.md`](./codespell_linter.py_kw.md_docs.md)
- [`no_workflows_on_fork.py_kw.md_docs.md`](./no_workflows_on_fork.py_kw.md_docs.md)
- [`bazel_linter.py_kw.md_docs.md`](./bazel_linter.py_kw.md_docs.md)
- [`mypy_linter.py_docs.md_docs.md`](./mypy_linter.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`exec_linter.py_kw.md_docs.md`](./exec_linter.py_kw.md_docs.md)
- [`clangformat_linter.py_docs.md_docs.md`](./clangformat_linter.py_docs.md_docs.md)
- [`pip_init.py_kw.md_docs.md`](./pip_init.py_kw.md_docs.md)
- [`testowners_linter.py_docs.md_docs.md`](./testowners_linter.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `workflow_consistency_linter.py_docs.md_docs.md`
- **Keyword Index**: `workflow_consistency_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

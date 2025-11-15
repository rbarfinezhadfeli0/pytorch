# Documentation: `tools/linter/adapters/no_workflows_on_fork.py`

## File Metadata

- **Path**: `tools/linter/adapters/no_workflows_on_fork.py`
- **Size**: 7,079 bytes (6.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""
This a linter that ensures that jobs that can be triggered by push,
pull_request, or schedule will check if the repository owner is 'pytorch'.  This
ensures that forks will not run jobs.

There are some edge cases that might be caught, and this prevents workflows from
being reused in other organizations, but as of right now, there are no workflows
with both push/pull_request/etc and workflow_call triggers simultaneously, so
this is.

There is also a setting in Github repos that can disable all workflows for that
repo.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Optional, TYPE_CHECKING

from yaml import load


if TYPE_CHECKING:
    from collections.abc import Callable


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


def load_yaml(path: Path) -> Any:
    with open(path) as f:
        return load(f, Loader)


def gen_lint_message(
    filename: Optional[str] = None,
    original: Optional[str] = None,
    replacement: Optional[str] = None,
    description: Optional[str] = None,
) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="NO_WORKFLOWS_ON_FORK",
        severity=LintSeverity.ERROR,
        name="format",
        original=original,
        replacement=replacement,
        description=description,
    )


def check_file(filename: str) -> list[LintMessage]:
    logging.debug("Checking file %s", filename)

    workflow = load_yaml(Path(filename))
    bad_jobs: dict[str, Optional[str]] = {}
    if type(workflow) is not dict:
        return []

    # yaml parses "on" as True
    triggers = workflow.get(True, {})
    triggers_to_check = ["push", "schedule", "pull_request", "pull_request_target"]
    if not any(trigger in triggers_to_check for trigger in triggers):
        return []

    jobs = workflow.get("jobs", {})
    for job, definition in jobs.items():
        if definition.get("needs"):
            # The parent job will have the if statement
            continue

        if_statement = definition.get("if")

        if if_statement is None:
            bad_jobs[job] = None
        elif type(if_statement) is bool and not if_statement:
            # if: false
            pass
        else:
            if_statement = str(if_statement)
            valid_checks: list[Callable[[str], bool]] = [
                lambda x: "github.repository == 'pytorch/pytorch'" in x
                and "github.event_name != 'schedule' || github.repository == 'pytorch/pytorch'"
                not in x,
                lambda x: "github.repository_owner == 'pytorch'" in x,
            ]
            if not any(f(if_statement) for f in valid_checks):
                bad_jobs[job] = if_statement

    with open(filename) as f:
        lines = f.readlines()

    smart_enough = True
    original = "".join(lines)
    iterator = iter(range(len(lines)))
    replacement = ""
    for i in iterator:
        line = lines[i]
        # Search for job name
        re_match = re.match(r"( +)([-_\w]*):", line)
        if not re_match or re_match.group(2) not in bad_jobs:
            replacement += line
            continue
        job_name = re_match.group(2)

        failure_type = bad_jobs[job_name]
        if failure_type is None:
            # Just need to add an if statement
            replacement += (
                f"{line}{re_match.group(1)}  if: github.repository_owner == 'pytorch'\n"
            )
            continue

        # Search for if statement
        while re.match(r"^ +if:", line) is None:
            replacement += line
            i = next(iterator)
            line = lines[i]
        if i + 1 < len(lines) and not re.match(r"^ +(.*):", lines[i + 1]):
            # This is a multi line if statement
            smart_enough = False
            break

        if_statement_match = re.match(r"^ +if: ([^#]*)(#.*)?$", line)
        # Get ... in if: ... # comments
        if not if_statement_match:
            return [
                gen_lint_message(
                    description=f"Something went wrong when looking at {job_name}.",
                )
            ]

        if_statement = if_statement_match.group(1).strip()

        # Handle comment in if: ... # comments
        comments = if_statement_match.group(2) or ""
        if comments:
            comments = " " + comments

        # Too broad of a check, but should catch everything
        needs_parens = "||" in if_statement

        # Handle ${{ ... }}
        has_brackets = re.match(r"\$\{\{(.*)\}\}", if_statement)
        internal_statement = (
            has_brackets.group(1).strip() if has_brackets else if_statement
        )

        if needs_parens:
            internal_statement = f"({internal_statement})"
        new_line = f"{internal_statement} && github.repository_owner == 'pytorch'"

        # I don't actually know if we need the ${{ }} but do it just in case
        new_line = "${{ " + new_line + " }}" + comments

        replacement += f"{re_match.group(1)}  if: {new_line}\n"

    description = (
        "Please add checks for if: github.repository_owner == 'pytorch' in the following jobs in this file: "
        + ", ".join(job for job in bad_jobs)
    )

    if not smart_enough:
        return [
            gen_lint_message(
                filename=filename,
                description=description,
            )
        ]

    if replacement == original:
        return []

    return [
        gen_lint_message(
            filename=filename,
            original=original,
            replacement=replacement,
            description=description,
        )
    ]


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

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise

```



## High-Level Overview

"""This a linter that ensures that jobs that can be triggered by push,pull_request, or schedule will check if the repository owner is 'pytorch'.  Thisensures that forks will not run jobs.There are some edge cases that might be caught, and this prevents workflows frombeing reused in other organizations, but as of right now, there are no workflowswith both push/pull_request/etc and workflow_call triggers simultaneously, sothis is.There is also a setting in Github repos that can disable all workflows for thatrepo.

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `load_yaml`, `gen_lint_message`, `check_file`

**Key imports**: annotations, argparse, concurrent.futures, json, logging, os, re, Enum, Path, Any, NamedTuple, Optional, TYPE_CHECKING


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `concurrent.futures`
- `json`
- `logging`
- `os`
- `re`
- `enum`: Enum
- `pathlib`: Path
- `typing`: Any, NamedTuple, Optional, TYPE_CHECKING
- `yaml`: load
- `collections.abc`: Callable


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
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `no_workflows_on_fork.py_docs.md`
- **Keyword Index**: `no_workflows_on_fork.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `tools/linter/adapters/actionlint_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/actionlint_linter.py`
- **Size**: 4,125 bytes (4.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "ACTIONLINT"


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


RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?P<char>\d+):
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def check_file(
    binary: str,
    file: str,
) -> list[LintMessage]:
    try:
        proc = run_command(
            [
                binary,
                "-ignore",
                '"runs-on" section must be sequence node but got mapping node with "!!map" tag',
                "-ignore",
                'input "freethreaded" is not defined in action "actions/setup-python@v',
                file,
            ]
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["char"]),
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="actionlint runner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="actionlint binary path",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    if not os.path.exists(args.binary):
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Could not find actionlint binary at {args.binary},"
                " you may need to run `lintrunner init`."
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                args.binary,
                filename,
            ): filename
            for filename in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise

```



## High-Level Overview

r"""(?mx)    ^    (?P<file>.*?):    (?P<line>\d+):    (?P<char>\d+):    \s(?P<message>.*)    \s(?P<code>\[.*\])    $

This Python file contains 2 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `run_command`, `check_file`

**Key imports**: annotations, argparse, concurrent.futures, json, logging, os, re, subprocess, sys, time


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
- `subprocess`
- `sys`
- `time`
- `enum`: Enum
- `typing`: NamedTuple


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

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
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `actionlint_linter.py_docs.md`
- **Keyword Index**: `actionlint_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

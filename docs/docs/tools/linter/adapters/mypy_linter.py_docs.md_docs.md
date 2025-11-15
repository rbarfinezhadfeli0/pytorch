# Documentation: `docs/tools/linter/adapters/mypy_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/mypy_linter.py_docs.md`
- **Size**: 10,041 bytes (9.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/mypy_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/mypy_linter.py`
- **Size**: 7,035 bytes (6.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple


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


# tools/linter/flake8_linter.py:15:13: error: Incompatibl...int")  [assignment]
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)

# torch/_dynamo/variables/tensor.py:363: error: INTERNAL ERROR
INTERNAL_ERROR_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<severity>\S+?):?
    \s(?P<message>INTERNAL\sERROR.*)
    $
    """
)


def run_command(
    args: list[str],
    *,
    extra_env: dict[str, str] | None,
    retries: int,
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


# Severity is either "error" or "note":
# https://github.com/python/mypy/blob/8b47a032e1317fb8e3f9a818005a6b63e9bf0311/mypy/errors.py#L46-L47
severities = {
    "error": LintSeverity.ERROR,
    "note": LintSeverity.ADVICE,
}


def check_mypy_installed(code: str) -> list[LintMessage]:
    cmd = [sys.executable, "-mmypy", "-V"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return []
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"Could not run '{' '.join(cmd)}': {msg}",
            )
        ]


def in_github_actions() -> bool:
    return bool(os.getenv("GITHUB_ACTIONS"))


def check_files(
    filenames: list[str],
    config: str,
    retries: int,
    code: str,
) -> list[LintMessage]:
    # dmypy has a bug where it won't pick up changes if you pass it absolute
    # file names, see https://github.com/python/mypy/issues/16768
    filenames = [os.path.relpath(f) for f in filenames]
    try:
        mypy_commands = ["dmypy", "run", "--"]
        if in_github_actions():
            mypy_commands = ["mypy"]
        proc = run_command(
            [*mypy_commands, f"--config={config}"] + filenames,
            extra_env={},
            retries=retries,
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    stderr = str(proc.stderr, "utf-8").strip()
    if proc.returncode not in (0, 1):
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=stderr,
            )
        ]

    rc = [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ] + [
        LintMessage(
            path=match["file"],
            name="INTERNAL ERROR",
            description=match["message"],
            line=int(match["line"]),
            char=None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in INTERNAL_ERROR_RE.finditer(stderr)
    ]
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mypy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out mypy",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an mypy .ini config file",
    )
    parser.add_argument(
        "--code",
        default="MYPY",
        help="the code this lint should report as",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # Use a dictionary here to preserve order. mypy cares about order,
    # tragically, e.g. https://github.com/python/mypy/issues/2015
    filenames: dict[str, bool] = {}

    # If a stub file exists, have mypy check it instead of the original file, in
    # accordance with PEP-484 (see https://www.python.org/dev/peps/pep-0484/#stub-files)
    for filename in args.filenames:
        if filename.endswith(".pyi"):
            filenames[filename] = True
            continue

        stub_filename = filename.replace(".py", ".pyi")
        if Path(stub_filename).exists():
            filenames[stub_filename] = True
        else:
            filenames[filename] = True

    lint_messages = check_mypy_installed(args.code) + check_files(
        list(filenames), args.config, args.retries, args.code
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()

```



## High-Level Overview

r"""(?mx)    ^    (?P<file>.*?):    (?P<line>\d+):    (?:(?P<column>-?\d+):)?    \s(?P<severity>\S+?):?    \s(?P<message>.*)    \s(?P<code>\[.*\])    $

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `run_command`, `check_mypy_installed`, `in_github_actions`, `check_files`, `main`

**Key imports**: annotations, argparse, json, logging, os, re, subprocess, sys, time, Enum


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `json`
- `logging`
- `os`
- `re`
- `subprocess`
- `sys`
- `time`
- `enum`: Enum
- `pathlib`: Path
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
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `mypy_linter.py_docs.md`
- **Keyword Index**: `mypy_linter.py_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/tools/linter/adapters`):

- [`pyrefly_linter.py_kw.md_docs.md`](./pyrefly_linter.py_kw.md_docs.md)
- [`codespell_linter.py_kw.md_docs.md`](./codespell_linter.py_kw.md_docs.md)
- [`no_workflows_on_fork.py_kw.md_docs.md`](./no_workflows_on_fork.py_kw.md_docs.md)
- [`bazel_linter.py_kw.md_docs.md`](./bazel_linter.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`exec_linter.py_kw.md_docs.md`](./exec_linter.py_kw.md_docs.md)
- [`clangformat_linter.py_docs.md_docs.md`](./clangformat_linter.py_docs.md_docs.md)
- [`pip_init.py_kw.md_docs.md`](./pip_init.py_kw.md_docs.md)
- [`testowners_linter.py_docs.md_docs.md`](./testowners_linter.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `mypy_linter.py_docs.md_docs.md`
- **Keyword Index**: `mypy_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

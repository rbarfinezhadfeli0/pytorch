# Documentation: `docs/tools/linter/adapters/pyfmt_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/pyfmt_linter.py_docs.md`
- **Size**: 7,420 bytes (7.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/pyfmt_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/pyfmt_linter.py`
- **Size**: 4,632 bytes (4.52 KB)
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
from enum import Enum
from pathlib import Path
from typing import NamedTuple

# pyrefly: ignore [import-error]
import isort
import usort


IS_WINDOWS: bool = os.name == "nt"
REPO_ROOT = Path(__file__).absolute().parents[3]


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


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def format_error_message(filename: str, err: Exception) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="PYFMT",
        severity=LintSeverity.ADVICE,
        name="command-failed",
        original=None,
        replacement=None,
        description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
    )


def run_isort(content: str, path: Path) -> str:
    isort_config = isort.Config(settings_path=str(REPO_ROOT))

    is_this_file = path.samefile(__file__)
    if not is_this_file:
        content = re.sub(r"(#.*\b)usort:\s*skip\b", r"\g<1>isort: split", content)

    content = isort.code(content, config=isort_config, file_path=path)

    if not is_this_file:
        content = re.sub(r"(#.*\b)isort: split\b", r"\g<1>usort: skip", content)

    return content


def run_usort(content: str, path: Path) -> str:
    usort_config = usort.Config.find(path)

    return usort.usort_string(content, path=path, config=usort_config)


def run_ruff_format(content: str, path: Path) -> str:
    try:
        return subprocess.check_output(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--config",
                str(REPO_ROOT / "pyproject.toml"),
                "--stdin-filename",
                str(path),
                "-",
            ],
            input=content,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.output) from exc


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    original = replacement = path.read_text(encoding="utf-8")

    try:
        # NB: run isort first to enforce style for blank lines
        replacement = run_isort(replacement, path=path)
        replacement = run_usort(replacement, path=path)
        replacement = run_ruff_format(replacement, path=path)

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="PYFMT",
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except Exception as err:
        return [format_error_message(filename, err)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with usort + ruff-format.",
        fromfile_prefix_chars="@",
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
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

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


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `as_posix`, `format_error_message`, `run_isort`, `run_usort`, `run_ruff_format`, `check_file`, `main`

**Key imports**: annotations, argparse, concurrent.futures, json, logging, os, re, subprocess, sys, Enum


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
- `enum`: Enum
- `pathlib`: Path
- `typing`: NamedTuple
- `isort`
- `usort`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


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
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `pyfmt_linter.py_docs.md`
- **Keyword Index**: `pyfmt_linter.py_kw.md`
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
- [`mypy_linter.py_docs.md_docs.md`](./mypy_linter.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`exec_linter.py_kw.md_docs.md`](./exec_linter.py_kw.md_docs.md)
- [`clangformat_linter.py_docs.md_docs.md`](./clangformat_linter.py_docs.md_docs.md)
- [`pip_init.py_kw.md_docs.md`](./pip_init.py_kw.md_docs.md)
- [`testowners_linter.py_docs.md_docs.md`](./testowners_linter.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pyfmt_linter.py_docs.md_docs.md`
- **Keyword Index**: `pyfmt_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

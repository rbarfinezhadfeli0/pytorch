# Documentation: `docs/tools/linter/adapters/newlines_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/newlines_linter.py_docs.md`
- **Size**: 8,544 bytes (8.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/newlines_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/newlines_linter.py`
- **Size**: 5,746 bytes (5.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""
NEWLINE: Checks files to make sure there are no trailing newlines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from enum import Enum
from typing import NamedTuple


NEWLINE = 10  # ASCII "\n"
CARRIAGE_RETURN = 13  # ASCII "\r"
LINTER_CODE = "NEWLINE"
MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB in bytes


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


def check_file(filename: str) -> LintMessage | None:
    logging.debug("Checking file %s", filename)

    # Check if file is too large
    try:
        file_size = os.path.getsize(filename)
        if file_size > MAX_FILE_SIZE:
            return LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="file-too-large",
                original=None,
                replacement=None,
                description=f"File size ({file_size} bytes) exceeds {MAX_FILE_SIZE} bytes limit, skipping",
            )
    except OSError as err:
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="file-access-error",
            original=None,
            replacement=None,
            description=f"Failed to get file size: {err}",
        )

    with open(filename, "rb") as f:
        lines = f.readlines()

    if len(lines) == 0:
        # File is empty, just leave it alone.
        return None

    if len(lines) == 1 and len(lines[0]) == 1:
        # file is wrong whether or not the only byte is a newline
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="testestTrailing newline",
            original=None,
            replacement=None,
            description="Trailing newline found. Run `lintrunner --take NEWLINE -a` to apply changes.",
        )

    if len(lines[-1]) == 1 and lines[-1][0] == NEWLINE:
        try:
            original = b"".join(lines).decode("utf-8")
        except Exception as err:
            return LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Decoding failure",
                original=None,
                replacement=None,
                description=f"utf-8 decoding failed due to {err.__class__.__name__}:\n{err}",
            )

        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="Trailing newline",
            original=original,
            replacement=original.rstrip("\n") + "\n",
            description="Trailing newline found. Run `lintrunner --take NEWLINE -a` to apply changes.",
        )
    has_changes = False
    original_lines: list[bytes] | None = None
    for idx, line in enumerate(lines):
        if len(line) >= 2 and line[-1] == NEWLINE and line[-2] == CARRIAGE_RETURN:
            if not has_changes:
                original_lines = list(lines)
                has_changes = True
            lines[idx] = line[:-2] + b"\n"

    if has_changes:
        try:
            assert original_lines is not None
            original = b"".join(original_lines).decode("utf-8")
            replacement = b"".join(lines).decode("utf-8")
        except Exception as err:
            return LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Decoding failure",
                original=None,
                replacement=None,
                description=f"utf-8 decoding failed due to {err.__class__.__name__}:\n{err}",
            )
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="DOS newline",
            original=original,
            replacement=replacement,
            description="DOS newline found. Run `lintrunner --take NEWLINE -a` to apply changes.",
        )

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="location of native_functions.yaml",
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

    lint_messages = []
    for filename in args.filenames:
        lint_message = check_file(filename)
        if lint_message is not None:
            lint_messages.append(lint_message)

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)

```



## High-Level Overview

"""NEWLINE: Checks files to make sure there are no trailing newlines.

This Python file contains 2 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `check_file`

**Key imports**: annotations, argparse, json, logging, os, sys, Enum, NamedTuple


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
- `sys`
- `enum`: Enum
- `typing`: NamedTuple


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

- **File Documentation**: `newlines_linter.py_docs.md`
- **Keyword Index**: `newlines_linter.py_kw.md`
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

- **File Documentation**: `newlines_linter.py_docs.md_docs.md`
- **Keyword Index**: `newlines_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

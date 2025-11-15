# Documentation: `docs/tools/linter/adapters/exec_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/exec_linter.py_docs.md`
- **Size**: 4,793 bytes (4.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/exec_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/exec_linter.py`
- **Size**: 2,030 bytes (1.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""
EXEC: Ensure that source files are not executable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "EXEC"


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
    is_executable = os.access(filename, os.X_OK)
    if is_executable:
        return LintMessage(
            path=filename,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="executable-permissions",
            original=None,
            replacement=None,
            description="This file has executable permission; please remove it by using `chmod -x`.",
        )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="exec linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
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

"""EXEC: Ensure that source files are not executable.

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

- **File Documentation**: `exec_linter.py_docs.md`
- **Keyword Index**: `exec_linter.py_kw.md`
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

*No specific patterns automatically detected.*


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

- **File Documentation**: `exec_linter.py_docs.md_docs.md`
- **Keyword Index**: `exec_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

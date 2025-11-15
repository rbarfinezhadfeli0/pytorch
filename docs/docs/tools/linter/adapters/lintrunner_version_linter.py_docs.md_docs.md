# Documentation: `docs/tools/linter/adapters/lintrunner_version_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/lintrunner_version_linter.py_docs.md`
- **Size**: 4,910 bytes (4.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/lintrunner_version_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/lintrunner_version_linter.py`
- **Size**: 2,160 bytes (2.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import json
import subprocess
import sys
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "LINTRUNNER_VERSION"


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


def toVersionString(version_tuple: tuple[int, int, int]) -> str:
    return ".".join(str(x) for x in version_tuple)


if __name__ == "__main__":
    version_str = (
        subprocess.run(["lintrunner", "-V"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    import re

    version_match = re.compile(r"lintrunner (\d+)\.(\d+)\.(\d+)").match(version_str)

    if not version_match:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="Lintrunner is not installed, did you forget to run `make setup-lint && make lint`?",
        )
        sys.exit(0)

    curr_version = int(version_match[1]), int(version_match[2]), int(version_match[3])
    min_version = (0, 10, 7)

    if curr_version < min_version:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ADVICE,
            name="command-failed",
            original=None,
            replacement=None,
            description="".join(
                (
                    f"Lintrunner is out of date (you have v{toVersionString(curr_version)} ",
                    f"instead of v{toVersionString(min_version)}). ",
                    "Please run `pip install lintrunner -U` to update it",
                )
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)

```



## High-Level Overview


This Python file contains 2 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `toVersionString`

**Key imports**: annotations, json, subprocess, sys, Enum, NamedTuple, re


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `json`
- `subprocess`
- `sys`
- `enum`: Enum
- `typing`: NamedTuple
- `re`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `lintrunner_version_linter.py_docs.md`
- **Keyword Index**: `lintrunner_version_linter.py_kw.md`
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
- [`mypy_linter.py_docs.md_docs.md`](./mypy_linter.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`exec_linter.py_kw.md_docs.md`](./exec_linter.py_kw.md_docs.md)
- [`clangformat_linter.py_docs.md_docs.md`](./clangformat_linter.py_docs.md_docs.md)
- [`pip_init.py_kw.md_docs.md`](./pip_init.py_kw.md_docs.md)
- [`testowners_linter.py_docs.md_docs.md`](./testowners_linter.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `lintrunner_version_linter.py_docs.md_docs.md`
- **Keyword Index**: `lintrunner_version_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

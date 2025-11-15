# Documentation: `docs/tools/linter/adapters/codespell_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/codespell_linter.py_docs.md`
- **Size**: 8,640 bytes (8.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/codespell_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/codespell_linter.py`
- **Size**: 5,824 bytes (5.69 KB)
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
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).absolute().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DICTIONARY = REPO_ROOT / "tools" / "linter" / "dictionary.txt"

FORBIDDEN_WORDS = {
    "multipy",  # project pytorch/multipy is dead  # codespell:ignore multipy
}

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


def format_error_message(
    filename: str,
    error: Exception | None = None,
    *,
    message: str | None = None,
) -> LintMessage:
    if message is None and error is not None:
        message = (
            f"Failed due to {error.__class__.__name__}:\n{error}\n"
            "Please either fix the error or add the word(s) to the dictionary file.\n"
            "HINT: all-lowercase words in the dictionary can cover all case variations."
        )
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="CODESPELL",
        severity=LintSeverity.ERROR,
        name="spelling error",
        original=None,
        replacement=None,
        description=message,
    )


def run_codespell(path: Path) -> str:
    try:
        return subprocess.check_output(
            [
                sys.executable,
                "-m",
                "codespell_lib",
                "--toml",
                str(PYPROJECT),
                str(path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.output) from exc


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()

    # Check if file is too large
    try:
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            return [
                LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code="CODESPELL",
                    severity=LintSeverity.WARNING,
                    name="file-too-large",
                    original=None,
                    replacement=None,
                    description=f"File size ({file_size} bytes) exceeds {MAX_FILE_SIZE} bytes limit, skipping",
                )
            ]
    except OSError as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CODESPELL",
                severity=LintSeverity.ERROR,
                name="file-access-error",
                original=None,
                replacement=None,
                description=f"Failed to get file size: {err}",
            )
        ]

    try:
        run_codespell(path)
    except Exception as err:
        return [format_error_message(filename, err)]
    return []


def check_dictionary(filename: str) -> list[LintMessage]:
    """Check the dictionary file for duplicates."""
    path = Path(filename).absolute()
    try:
        words = path.read_text(encoding="utf-8").splitlines()
        words_set = set(words)
        if len(words) != len(words_set):
            raise ValueError("The dictionary file contains duplicate entries.")
        # pyrefly: ignore [no-matching-overload]
        uncased_words = list(map(str.lower, words))
        if uncased_words != sorted(uncased_words):
            raise ValueError(
                "The dictionary file is not sorted alphabetically (case-insensitive)."
            )
        for forbidden_word in sorted(
            FORBIDDEN_WORDS & (words_set | set(uncased_words))
        ):
            raise ValueError(
                f"The dictionary file contains a forbidden word: {forbidden_word!r}. "
                "Please remove it from the dictionary file and use 'codespell:ignore' "
                "inline comment instead."
            )
    except Exception as err:
        return [format_error_message(str(filename), err)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check files for spelling mistakes using codespell.",
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
        futures[executor.submit(check_dictionary, str(DICTIONARY))] = str(DICTIONARY)
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


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `format_error_message`, `run_codespell`, `check_file`, `check_dictionary`, `main`

**Key imports**: annotations, argparse, concurrent.futures, json, logging, os, subprocess, sys, Enum, Path


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
- `subprocess`
- `sys`
- `enum`: Enum
- `pathlib`: Path
- `typing`: NamedTuple


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
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `codespell_linter.py_docs.md`
- **Keyword Index**: `codespell_linter.py_kw.md`
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

- **File Documentation**: `codespell_linter.py_docs.md_docs.md`
- **Keyword Index**: `codespell_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

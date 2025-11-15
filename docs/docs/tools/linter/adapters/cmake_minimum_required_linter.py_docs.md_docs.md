# Documentation: `docs/tools/linter/adapters/cmake_minimum_required_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/cmake_minimum_required_linter.py_docs.md`
- **Size**: 10,095 bytes (9.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/cmake_minimum_required_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/cmake_minimum_required_linter.py`
- **Size**: 6,965 bytes (6.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import logging
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from packaging.requirements import Requirement
from packaging.version import Version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.setup_helpers.env import CMAKE_MINIMUM_VERSION_STRING


sys.path.remove(str(REPO_ROOT))


LINTER_CODE = "CMAKE_MINIMUM_REQUIRED"
CMAKE_MINIMUM_VERSION = Version(CMAKE_MINIMUM_VERSION_STRING)


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
    line: int | None = None,
    message: str | None = None,
) -> LintMessage:
    if message is None and error is not None:
        message = f"Failed due to {error.__class__.__name__}:\n{error}"
    return LintMessage(
        path=filename,
        line=line,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name="CMake minimum version",
        original=None,
        replacement=None,
        description=message,
    )


CMAKE_MINIMUM_REQUIRED_PATTERN = re.compile(
    r"cmake_minimum_required\(VERSION\s+(?P<version>\d+\.\d+(\.\d+)?)\b.*\)",
    flags=re.IGNORECASE,
)


def check_cmake(path: Path) -> list[LintMessage]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if match := CMAKE_MINIMUM_REQUIRED_PATTERN.search(line):
                version = match.group("version")
                if path.samefile(REPO_ROOT / "CMakeLists.txt"):
                    if Version(version) != CMAKE_MINIMUM_VERSION:
                        return [
                            format_error_message(
                                str(path),
                                line=i,
                                message=(
                                    f"CMake minimum version must be {CMAKE_MINIMUM_VERSION}, "
                                    f"but found {version}."
                                ),
                            )
                        ]
                elif Version(version) > CMAKE_MINIMUM_VERSION:
                    return [
                        format_error_message(
                            str(path),
                            line=i,
                            message=(
                                f"The environment can only provide CMake {CMAKE_MINIMUM_VERSION}, "
                                f"but found requiring {version}."
                            ),
                        )
                    ]
    return []


def check_requirement(
    requirement: Requirement,
    path: Path,
    *,
    line: int | None = None,
) -> LintMessage | None:
    if requirement.name.lower() != "cmake":
        return None

    for spec in requirement.specifier:
        if (
            spec.operator in ("==", ">=")
            and Version(spec.version.removesuffix(".*")) < CMAKE_MINIMUM_VERSION
        ):
            return format_error_message(
                str(path),
                line=line,
                message=(
                    f"CMake minimum version must be at least {CMAKE_MINIMUM_VERSION}, "
                    f"but found {spec}."
                ),
            )

    return None


def check_pyproject(path: Path) -> list[LintMessage]:
    try:
        pyproject = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as err:
        return [format_error_message(str(path), err)]

    if not isinstance(pyproject, dict):
        return []
    if not isinstance(pyproject.get("build-system"), dict):
        return []

    build_system = pyproject["build-system"]
    requires = build_system.get("requires")
    if not isinstance(requires, list):
        return []
    return list(
        filter(
            None,
            (check_requirement(Requirement(req), path=path) for req in requires),
        )
    )


def check_requirements(path: Path) -> list[LintMessage]:
    try:
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as err:
        return [format_error_message(str(path), err)]

    lint_messages = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(line)
        except Exception:
            continue
        lint_message = check_requirement(requirement, path=path, line=i)
        if lint_message is not None:
            lint_messages.append(lint_message)

    return lint_messages


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    basename = path.name.lower()
    if basename in ("cmakelists.txt", "cmakelists.txt.in") or basename.endswith(
        (".cmake", ".cmake.in")
    ):
        return check_cmake(path)
    if basename == "pyproject.toml":
        return check_pyproject(path)
    if fnmatch.fnmatch(basename, "*requirements*.txt") or fnmatch.fnmatch(
        basename, "*requirements*.in"
    ):
        return check_requirements(path)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check consistency of cmake minimum version in requirement files.",
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

**Functions defined**: `format_error_message`, `check_cmake`, `check_requirement`, `check_pyproject`, `check_requirements`, `check_file`, `main`

**Key imports**: annotations, argparse, concurrent.futures, fnmatch, json, logging, os, re, sys, Enum


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
- `fnmatch`
- `json`
- `logging`
- `os`
- `re`
- `sys`
- `enum`: Enum
- `pathlib`: Path
- `typing`: NamedTuple
- `packaging.requirements`: Requirement
- `packaging.version`: Version
- `tomllib`
- `tomli as tomllib  `
- `tools.setup_helpers.env`: CMAKE_MINIMUM_VERSION_STRING


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `cmake_minimum_required_linter.py_docs.md`
- **Keyword Index**: `cmake_minimum_required_linter.py_kw.md`
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

- May involve **JIT compilation** or compilation optimizations.
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

- **File Documentation**: `cmake_minimum_required_linter.py_docs.md_docs.md`
- **Keyword Index**: `cmake_minimum_required_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

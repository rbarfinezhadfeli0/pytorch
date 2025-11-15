# Documentation: `tools/linter/adapters/pyproject_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/pyproject_linter.py`
- **Size**: 8,081 bytes (7.89 KB)
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
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from packaging.specifiers import SpecifierSet
from packaging.version import Version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


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
        message = f"Failed due to {error.__class__.__name__}:\n{error}"
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="PYPROJECT",
        severity=LintSeverity.ERROR,
        name="pyproject.toml consistency",
        original=None,
        replacement=None,
        description=message,
    )


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    try:
        pyproject = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as err:
        return [format_error_message(filename, err)]

    if not (isinstance(pyproject, dict) and isinstance(pyproject.get("project"), dict)):
        return [
            format_error_message(
                filename,
                message=(
                    "'project' section in pyproject.toml must present and be a table."
                ),
            )
        ]

    project = pyproject["project"]
    requires_python = project.get("requires-python")
    if requires_python is not None:
        if not isinstance(requires_python, str):
            return [
                format_error_message(
                    filename,
                    message="'project.requires-python' must be a string.",
                )
            ]

        python_major = 3
        specifier_set = SpecifierSet(requires_python)
        for specifier in specifier_set:
            if Version(specifier.version).major != python_major:
                return [
                    format_error_message(
                        filename,
                        message=(
                            "'project.requires-python' must only specify "
                            f"Python {python_major} versions, but found {specifier.version}."
                        ),
                    )
                ]

        large_minor = 1000
        supported_python_versions = list(
            specifier_set.filter(
                f"{python_major}.{minor}" for minor in range(large_minor + 1)
            )
        )
        if not supported_python_versions:
            return [
                format_error_message(
                    filename,
                    message=(
                        "'project.requires-python' must specify at least one "
                        f"Python {python_major} version, but found {requires_python!r}."
                    ),
                )
            ]
        if f"{python_major}.0" in supported_python_versions:
            return [
                format_error_message(
                    filename,
                    message=(
                        "'project.requires-python' must specify a minimum version, "
                        f"but found {requires_python!r}."
                    ),
                )
            ]
        # if f"{python_major}.{large_minor}" in supported_python_versions:
        #     return [
        #         format_error_message(
        #             filename,
        #             message=(
        #                 "'project.requires-python' must specify a maximum version, "
        #                 f"but found {requires_python!r}."
        #             ),
        #         )
        #     ]

        classifiers = project.get("classifiers")
        if not (
            isinstance(classifiers, list)
            and all(isinstance(c, str) for c in classifiers)
        ):
            return [
                format_error_message(
                    filename,
                    message="'project.classifiers' must be an array of strings.",
                )
            ]
        if len(set(classifiers)) != len(classifiers):
            return [
                format_error_message(
                    filename,
                    message="'project.classifiers' must not contain duplicates.",
                )
            ]

        # python_version_classifiers = [
        #     c
        #     for c in classifiers
        #     if (
        #         c.startswith("Programming Language :: Python :: ")
        #         and not c.endswith((f":: {python_major}", f":: {python_major} :: Only"))
        #     )
        # ]
        # if python_version_classifiers:
        #     python_version_classifier_set = set(python_version_classifiers)
        #     supported_python_version_classifier_set = {
        #         f"Programming Language :: Python :: {v}"
        #         for v in supported_python_versions
        #     }
        #     if python_version_classifier_set != supported_python_version_classifier_set:
        #         missing_classifiers = sorted(
        #             supported_python_version_classifier_set
        #             - python_version_classifier_set
        #         )
        #         extra_classifiers = sorted(
        #             python_version_classifier_set
        #             - supported_python_version_classifier_set
        #         )
        #         if missing_classifiers:
        #             return [
        #                 format_error_message(
        #                     filename,
        #                     message=(
        #                         "'project.classifiers' is missing the following classifier(s):\n"
        #                         + "\n".join(f"  {c!r}" for c in missing_classifiers)
        #                     ),
        #                 )
        #             ]
        #         if extra_classifiers:
        #             return [
        #                 format_error_message(
        #                     filename,
        #                     message=(
        #                         "'project.classifiers' contains extra classifier(s):\n"
        #                         + "\n".join(f"  {c!r}" for c in extra_classifiers)
        #                     ),
        #                 )
        #             ]

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check consistency of pyproject.toml files.",
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


This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `format_error_message`, `check_file`, `main`

**Key imports**: annotations, argparse, concurrent.futures, json, logging, os, sys, Enum, Path, NamedTuple


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
- `sys`
- `enum`: Enum
- `pathlib`: Path
- `typing`: NamedTuple
- `packaging.specifiers`: SpecifierSet
- `packaging.version`: Version
- `tomllib`
- `tomli as tomllib  `


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

- **File Documentation**: `pyproject_linter.py_docs.md`
- **Keyword Index**: `pyproject_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

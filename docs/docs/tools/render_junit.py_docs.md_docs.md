# Documentation: `docs/tools/render_junit.py_docs.md`

## File Metadata

- **Path**: `docs/tools/render_junit.py_docs.md`
- **Size**: 5,746 bytes (5.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/render_junit.py`

## File Metadata

- **Path**: `tools/render_junit.py`
- **Size**: 3,301 bytes (3.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import Any


try:
    from junitparser import (  # type: ignore[import]
        Error,
        Failure,
        JUnitXml,
        TestCase,
        TestSuite,
    )
except ImportError as e:
    raise ImportError(
        "junitparser not found, please install with 'pip install junitparser'"
    ) from e

try:
    import rich
except ImportError:
    print("rich not found, for color output use 'pip install rich'")


def parse_junit_reports(path_to_reports: str) -> list[TestCase]:  # type: ignore[no-any-unimported]
    def parse_file(path: str) -> list[TestCase]:  # type: ignore[no-any-unimported]
        try:
            return convert_junit_to_testcases(JUnitXml.fromfile(path))
        except Exception as err:
            rich.print(
                f":Warning: [yellow]Warning[/yellow]: Failed to read {path}: {err}"
            )
            return []

    if not os.path.exists(path_to_reports):
        raise FileNotFoundError(f"Path '{path_to_reports}', not found")
    # Return early if the path provided is just a file
    if os.path.isfile(path_to_reports):
        return parse_file(path_to_reports)
    ret_xml = []
    if os.path.isdir(path_to_reports):
        for root, _, files in os.walk(path_to_reports):
            for fname in [f for f in files if f.endswith("xml")]:
                ret_xml += parse_file(os.path.join(root, fname))
    return ret_xml


def convert_junit_to_testcases(xml: JUnitXml | TestSuite) -> list[TestCase]:  # type: ignore[no-any-unimported]
    testcases = []
    for item in xml:
        if isinstance(item, TestSuite):
            testcases.extend(convert_junit_to_testcases(item))
        else:
            testcases.append(item)
    return testcases


def render_tests(testcases: list[TestCase]) -> None:  # type: ignore[no-any-unimported]
    num_passed = 0
    num_skipped = 0
    num_failed = 0
    for testcase in testcases:
        if not testcase.result:
            num_passed += 1
            continue
        for result in testcase.result:
            if isinstance(result, Error):
                icon = ":rotating_light: [white on red]ERROR[/white on red]:"
                num_failed += 1
            elif isinstance(result, Failure):
                icon = ":x: [white on red]Failure[/white on red]:"
                num_failed += 1
            else:
                num_skipped += 1
                continue
            rich.print(
                f"{icon} [bold red]{testcase.classname}.{testcase.name}[/bold red]"
            )
            print(f"{result.text}")
    rich.print(f":white_check_mark: {num_passed} [green]Passed[green]")
    rich.print(f":dash: {num_skipped} [grey]Skipped[grey]")
    rich.print(f":rotating_light: {num_failed} [grey]Failed[grey]")


def parse_args() -> Any:
    parser = argparse.ArgumentParser(
        description="Render xunit output for failed tests",
    )
    parser.add_argument(
        "report_path",
        help="Base xunit reports (single file or directory) to compare to",
    )
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    testcases = parse_junit_reports(options.report_path)
    render_tests(testcases)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parse_junit_reports`, `parse_file`, `convert_junit_to_testcases`, `render_tests`, `parse_args`, `main`

**Key imports**: annotations, argparse, os, Any, rich


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `os`
- `typing`: Any
- `rich`


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

Files in the same folder (`tools`):

- [`BUCK.bzl_docs.md`](./BUCK.bzl_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`extract_scripts.py_docs.md`](./extract_scripts.py_docs.md)
- [`nvcc_fix_deps.py_docs.md`](./nvcc_fix_deps.py_docs.md)
- [`update_masked_docs.py_docs.md`](./update_masked_docs.py_docs.md)
- [`optional_submodules.py_docs.md`](./optional_submodules.py_docs.md)
- [`gen_vulkan_spv.py_docs.md`](./gen_vulkan_spv.py_docs.md)
- [`generated_dirs.txt_docs.md`](./generated_dirs.txt_docs.md)
- [`build_libtorch.py_docs.md`](./build_libtorch.py_docs.md)


## Cross-References

- **File Documentation**: `render_junit.py_docs.md`
- **Keyword Index**: `render_junit.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools`, which contains **development tools and scripts**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools`):

- [`git_add_generated_dirs.sh_docs.md_docs.md`](./git_add_generated_dirs.sh_docs.md_docs.md)
- [`update_masked_docs.py_docs.md_docs.md`](./update_masked_docs.py_docs.md_docs.md)
- [`bazel.bzl_docs.md_docs.md`](./bazel.bzl_docs.md_docs.md)
- [`nightly_hotpatch.py_docs.md_docs.md`](./nightly_hotpatch.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`build_with_debinfo.py_docs.md_docs.md`](./build_with_debinfo.py_docs.md_docs.md)
- [`extract_scripts.py_docs.md_docs.md`](./extract_scripts.py_docs.md_docs.md)
- [`bazel.bzl_kw.md_docs.md`](./bazel.bzl_kw.md_docs.md)
- [`build_with_debinfo.py_kw.md_docs.md`](./build_with_debinfo.py_kw.md_docs.md)
- [`gen_flatbuffers.sh_kw.md_docs.md`](./gen_flatbuffers.sh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `render_junit.py_docs.md_docs.md`
- **Keyword Index**: `render_junit.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/tools/linter/adapters/nativefunctions_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/linter/adapters/nativefunctions_linter.py_docs.md`
- **Size**: 7,222 bytes (7.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/linter/adapters/nativefunctions_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/nativefunctions_linter.py`
- **Size**: 3,641 bytes (3.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""
Verify that it is possible to round-trip native_functions.yaml via ruamel under some
configuration.  Keeping native_functions.yaml consistent in this way allows us to
run codemods on the file using ruamel without introducing line noise.  Note that we don't
want to normalize the YAML file, as that would to lots of spurious lint failures.  Anything
that ruamel understands how to roundtrip, e.g., whitespace and comments, is OK!

ruamel is a bit picky about inconsistent indentation, so you will have to indent your
file properly.  Also, if you are working on changing the syntax of native_functions.yaml,
you may find that you want to use some format that is not what ruamel prefers.  If so,
it is OK to modify this script (instead of reformatting native_functions.yaml)--the point
is simply to make sure that there is *some* configuration of ruamel that can round trip
the YAML, not to be prescriptive about it.
"""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from io import StringIO
from typing import NamedTuple

import ruamel.yaml  # type: ignore[import]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--native-functions-yml",
        required=True,
        help="location of native_functions.yaml",
    )

    args = parser.parse_args()

    with open(args.native_functions_yml) as f:
        contents = f.read()

    yaml = ruamel.yaml.YAML()  # type: ignore[attr-defined]
    yaml.preserve_quotes = True  # type: ignore[assignment]
    yaml.width = 1000  # type: ignore[assignment]
    yaml.boolean_representation = ["False", "True"]  # type: ignore[attr-defined]
    try:
        r = yaml.load(contents)
    except Exception as err:
        msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code="NATIVEFUNCTIONS",
            severity=LintSeverity.ERROR,
            name="YAML load failure",
            original=None,
            replacement=None,
            description=f"Failed due to {err.__class__.__name__}:\n{err}",
        )

        print(json.dumps(msg._asdict()), flush=True)
        sys.exit(0)

    # Cuz ruamel's author intentionally didn't include conversion to string
    # https://stackoverflow.com/questions/47614862/best-way-to-use-ruamel-yaml-to-dump-to-string-not-to-stream
    string_stream = StringIO()
    yaml.dump(r, string_stream)
    new_contents = string_stream.getvalue()
    string_stream.close()

    if contents != new_contents:
        msg = LintMessage(
            path=args.native_functions_yml,
            line=None,
            char=None,
            code="NATIVEFUNCTIONS",
            severity=LintSeverity.ERROR,
            name="roundtrip inconsistency",
            original=contents,
            replacement=new_contents,
            description=(
                "YAML roundtrip failed; run `lintrunner --take NATIVEFUNCTIONS -a` to apply the suggested changes. "
                "If you think this is in error, please see tools/linter/adapters/nativefunctions_linter.py"
            ),
        )

        print(json.dumps(msg._asdict()), flush=True)

```



## High-Level Overview

"""Verify that it is possible to round-trip native_functions.yaml via ruamel under someconfiguration.  Keeping native_functions.yaml consistent in this way allows us torun codemods on the file using ruamel without introducing line noise.  Note that we don'twant to normalize the YAML file, as that would to lots of spurious lint failures.  Anythingthat ruamel understands how to roundtrip, e.g., whitespace and comments, is OK!ruamel is a bit picky about inconsistent indentation, so you will have to indent yourfile properly.  Also, if you are working on changing the syntax of native_functions.yaml,you may find that you want to use some format that is not what ruamel prefers.  If so,it is OK to modify this script (instead of reformatting native_functions.yaml)--the pointis simply to make sure that there is *some* configuration of ruamel that can round tripthe YAML, not to be prescriptive about it.

This Python file contains 2 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Key imports**: annotations, argparse, json, sys, Enum, StringIO, NamedTuple, ruamel.yaml  


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
- `sys`
- `enum`: Enum
- `io`: StringIO
- `typing`: NamedTuple
- `ruamel.yaml  `


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

- **File Documentation**: `nativefunctions_linter.py_docs.md`
- **Keyword Index**: `nativefunctions_linter.py_kw.md`
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

- **File Documentation**: `nativefunctions_linter.py_docs.md_docs.md`
- **Keyword Index**: `nativefunctions_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

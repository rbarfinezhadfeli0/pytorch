# Documentation: `tools/linter/adapters/import_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/import_linter.py`
- **Size**: 3,800 bytes (3.71 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
"""
Checks files to make sure there are no imports from disallowed third party
libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import token
from enum import Enum
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
    from . import _linter
else:
    import _linter


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


LINTER_CODE = "IMPORT_LINTER"
CURRENT_FILE_NAME = os.path.basename(__file__)
_MODULE_NAME_ALLOW_LIST: set[str] = set()

# Add builtin modules of python.
_MODULE_NAME_ALLOW_LIST.update(sys.stdlib_module_names)

# Add the allowed third party libraries. Please avoid updating this unless you
# understand the risks -- see `_ERROR_MESSAGE` for why.
_MODULE_NAME_ALLOW_LIST.update(
    [
        "sympy",
        "einops",
        "libfb",
        "torch",
        "tvm",
        "_pytest",
        "tabulate",
        "optree",
        "typing_extensions",
        "triton",
        "functorch",
        "torchrec",
        "numpy",
        "torch_xla",
    ]
)

_ERROR_MESSAGE = """
Please do not import third-party modules in PyTorch unless they're explicit
requirements of PyTorch. Imports of a third-party library may have side effects
and other unintentional behavior. If you're just checking if a module exists,
use sys.modules.get("torchrec") or the like.
"""


def check_file(filepath: str) -> list[LintMessage]:
    path = Path(filepath)
    file = _linter.PythonFile("import_linter", path)
    lint_messages = []
    for line_number, line_of_tokens in enumerate(file.token_lines):
        # Skip indents
        idx = 0
        for tok in line_of_tokens:
            if tok.type == token.INDENT:
                idx += 1
            else:
                break

        # Look for either "import foo..." or "from foo..."
        if idx + 1 < len(line_of_tokens):
            tok0 = line_of_tokens[idx]
            tok1 = line_of_tokens[idx + 1]
            if tok0.type == token.NAME and tok0.string in {"import", "from"}:
                if tok1.type == token.NAME:
                    module_name = tok1.string
                    if module_name not in _MODULE_NAME_ALLOW_LIST:
                        msg = LintMessage(
                            path=filepath,
                            line=line_number,
                            char=None,
                            code="IMPORT",
                            severity=LintSeverity.ERROR,
                            name="Disallowed import",
                            original=None,
                            replacement=None,
                            description=_ERROR_MESSAGE,
                        )
                        lint_messages.append(msg)
    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="native functions linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filepaths",
        nargs="+",
        help="paths of files to lint",
    )
    args = parser.parse_args()

    # Check all files.
    all_lint_messages = []
    for filepath in args.filepaths:
        lint_messages = check_file(filepath)
        all_lint_messages.extend(lint_messages)

    # Print out lint messages.
    for lint_message in all_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)

```



## High-Level Overview

"""Checks files to make sure there are no imports from disallowed third partylibraries.

This Python file contains 2 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `check_file`

**Key imports**: annotations, argparse, json, os, sys, token, Enum, Path, NamedTuple, TYPE_CHECKING, _linter


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
- `os`
- `sys`
- `token`
- `enum`: Enum
- `pathlib`: Path
- `typing`: NamedTuple, TYPE_CHECKING
- `.`: _linter
- `_linter`
- `third`
- `foo...`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
- [`gha_linter.py_docs.md`](./gha_linter.py_docs.md)
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `import_linter.py_docs.md`
- **Keyword Index**: `import_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

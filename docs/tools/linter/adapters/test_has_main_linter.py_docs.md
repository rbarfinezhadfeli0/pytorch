# Documentation: `tools/linter/adapters/test_has_main_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/test_has_main_linter.py`
- **Size**: 3,811 bytes (3.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""
This lint verifies that every Python test file (file that matches test_*.py or
*_test.py in the test folder) has a main block which raises an exception or
calls run_tests to ensure that the test will be run in OSS CI.

Takes ~2 minuters to run without the multiprocessing, probably overkill.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from enum import Enum
from typing import NamedTuple

# pyrefly: ignore [import-error]
import libcst as cst

# pyrefly: ignore [import-error]
import libcst.matchers as m


LINTER_CODE = "TEST_HAS_MAIN"


class HasMainVisiter(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.found = False

    def visit_Module(self, node: cst.Module) -> bool:
        name = m.Name("__name__")
        main = m.SimpleString('"__main__"') | m.SimpleString("'__main__'")
        run_test_call = m.Call(
            func=m.Name("run_tests") | m.Attribute(attr=m.Name("run_tests"))
        )
        # Distributed tests (i.e. MultiProcContinuousTest) calls `run_rank`
        # instead of `run_tests` in main
        run_rank_call = m.Call(
            func=m.Name("run_rank") | m.Attribute(attr=m.Name("run_rank"))
        )
        raise_block = m.Raise()

        # name == main or main == name
        if_main1 = m.Comparison(
            name,
            [m.ComparisonTarget(m.Equal(), main)],
        )
        if_main2 = m.Comparison(
            main,
            [m.ComparisonTarget(m.Equal(), name)],
        )
        for child in node.children:
            if m.matches(child, m.If(test=if_main1 | if_main2)):
                if m.findall(child, raise_block | run_test_call | run_rank_call):
                    self.found = True
                    break

        return False


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


def check_file(filename: str) -> list[LintMessage]:
    lint_messages = []

    with open(filename) as f:
        file = f.read()
        v = HasMainVisiter()
        cst.parse_module(file).visit(v)
        if not v.found:
            message = (
                "Test files need to have a main block which either calls run_tests "
                + "(to ensure that the tests are run during OSS CI) or raises an exception "
                + "and added to the blocklist in test/run_test.py"
            )
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[no-main]",
                    original=None,
                    replacement=None,
                    description=message,
                )
            )
    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="test files should have main block linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    pool = mp.Pool(8)
    lint_messages = pool.map(check_file, args.filenames)
    pool.close()
    pool.join()

    flat_lint_messages = []
    for sublist in lint_messages:
        flat_lint_messages.extend(sublist)

    for lint_message in flat_lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""This lint verifies that every Python test file (file that matches test_*.py or*_test.py in the test folder) has a main block which raises an exception orcalls run_tests to ensure that the test will be run in OSS CI.Takes ~2 minuters to run without the multiprocessing, probably overkill.

This Python file contains 3 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HasMainVisiter`, `LintSeverity`, `LintMessage`

**Functions defined**: `__init__`, `visit_Module`, `check_file`, `main`

**Key imports**: annotations, argparse, json, multiprocessing as mp, Enum, NamedTuple, libcst as cst, libcst.matchers as m


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
- `multiprocessing as mp`
- `enum`: Enum
- `typing`: NamedTuple
- `libcst as cst`
- `libcst.matchers as m`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python tools/linter/adapters/test_has_main_linter.py
```

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

- **File Documentation**: `test_has_main_linter.py_docs.md`
- **Keyword Index**: `test_has_main_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

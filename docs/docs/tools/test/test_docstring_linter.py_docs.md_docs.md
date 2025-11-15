# Documentation: `docs/tools/test/test_docstring_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_docstring_linter.py_docs.md`
- **Size**: 9,255 bytes (9.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_docstring_linter.py`

## File Metadata

- **Path**: `tools/test/test_docstring_linter.py`
- **Size**: 6,139 bytes (6.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```python
# mypy: ignore-errors

import io
import itertools
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

from tools.linter.adapters._linter.block import _get_decorators
from tools.linter.adapters.docstring_linter import (
    DocstringLinter,
    file_summary,
    make_recursive,
    make_terse,
)


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase

TEST_FILE = Path("tools/test/docstring_linter_testdata/python_code.py.txt")
TEST_FILE2 = Path("tools/test/docstring_linter_testdata/more_python_code.py.txt")
TEST_BLOCK_NAMES = Path("tools/test/docstring_linter_testdata/block_names.py.txt")
ARGS = "--max-class=5", "--max-def=6", "--min-docstring=16"


class TestDocstringLinter(LinterTestCase):
    LinterClass = DocstringLinter
    maxDiff = 10_240

    def test_python_code(self):
        self.lint_test(TEST_FILE, ARGS)

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_end_to_end(self, mock_stdout):
        argv_base = *ARGS, str(TEST_FILE), str(TEST_FILE2)
        report = "--report"
        write = "--write-grandfather"

        out = _next_stdout(mock_stdout)

        def run(name, *argv):
            DocstringLinter(argv_base + argv).lint_all()
            self.assertExpected(TEST_FILE2, next(out), name)

        with tempfile.TemporaryDirectory() as td:
            grandfather_file = f"{td}/grandfather.json"
            grandfather = f"--grandfather={grandfather_file}"

            # Find some failures
            run("before.txt", grandfather)

            # Rewrite grandfather file
            run("before.json", grandfather, report, write)
            actual = Path(grandfather_file).read_text()
            self.assertExpected(TEST_FILE2, actual, "grandfather.json")

            # Now there are no failures
            run("after.txt", grandfather)
            run("after.json", grandfather, report)

    def test_report(self):
        actual = _dumps(_data())
        self.assertExpected(TEST_FILE, actual, "report.json")

    def test_terse(self):
        terse = make_terse(_data(), index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.json")

    def test_terse_line(self):
        terse = make_terse(_data(), index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "terse.line.json")

    def test_recursive(self):
        recursive = make_recursive(_data())
        actual = _dumps(recursive)
        self.assertExpected(TEST_FILE, actual, "recursive.json")

    def test_terse_recursive(self):
        recursive = make_recursive(_data())
        terse = make_terse(recursive, index_by_line=False)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "recursive.terse.json")

    def test_terse_line_recursive(self):
        recursive = make_recursive(_data())
        terse = make_terse(recursive, index_by_line=True)
        actual = _dumps(terse)
        self.assertExpected(TEST_FILE, actual, "recursive.terse.line.json")

    def test_file_summary(self):
        actual = _dumps(file_summary(_data(), report_all=True))
        self.assertExpected(TEST_FILE, actual, "single.line.json")

    def test_file_names(self):
        f = DocstringLinter.make_file(TEST_BLOCK_NAMES)
        actual = [b.full_name for b in f.blocks]
        expected = [
            "top",
            "top.fun[1]",
            "top.fun[1].sab",
            "top.fun[1].sub",
            "top.fun[2]",
            "top.fun[2].sub[1]",
            "top.fun[2].sub[2]",
            "top.fun[3]",
            "top.fun[3].sub",
            "top.fun[3].sab",
            "top.run",
            "top.run.sub[1]",
            "top.run.sub[2]",
        ]
        self.assertEqual(actual, expected)

    def test_decorators(self):
        tests = itertools.product(INDENTS, DECORATORS.items())
        for indent, (name, (expected, test_inputs)) in tests:
            ind = indent * " "
            for data in test_inputs:
                prog = "".join(ind + d + "\n" for d in data)
                pf = DocstringLinter.make_file(prog)
                it = (i for i, t in enumerate(pf.tokens) if t.string == "def")
                def_t = next(it, 0)
                with self.subTest("Decorator", indent=indent, name=name, data=data):
                    actual = list(_get_decorators(pf.tokens, def_t))
                    self.assertEqual(actual, expected)


def _dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, indent=2) + "\n"


def _data(file=TEST_FILE):
    docstring_file = DocstringLinter.make_file(file)
    return [b.as_data() for b in docstring_file.blocks]


def _next_stdout(mock_stdout):
    length = 0
    while True:
        s = mock_stdout.getvalue()
        yield s[length:]
        length = len(s)


CONSTANT = "A = 10"
COMMENT = "# a simple function"
OVER = "@override"
WRAPS = "@functools.wraps(fn)"
MASSIVE = (
    "@some.long.path.very_long_function_name(",
    "    adjust_something_fiddly=1231232,",
    "    disable_something_critical=True,)",
)
MASSIVE_FLAT = (
    "@some.long.path.very_long_function_name("
    "adjust_something_fiddly=1231232,"
    "disable_something_critical=True,)"
)
DEF = "def function():", "    pass"

INDENTS = 0, 4, 8
DECORATORS = {
    "none": (
        [],
        (
            [],
            [*DEF],
            [COMMENT, *DEF],
            [CONSTANT, "", COMMENT, *DEF],
            [OVER, CONSTANT, *DEF],  # Probably not even Python. :-)
        ),
    ),
    "one": (
        [OVER],
        (
            [OVER, *DEF],
            [OVER, COMMENT, *DEF],
            [OVER, COMMENT, "", *DEF],
            [COMMENT, OVER, "", COMMENT, "", *DEF],
        ),
    ),
    "two": (
        [OVER, WRAPS],
        (
            [OVER, WRAPS, *DEF],
            [COMMENT, OVER, COMMENT, WRAPS, COMMENT, *DEF],
        ),
    ),
    "massive": (
        [MASSIVE_FLAT, OVER],
        ([*MASSIVE, OVER, *DEF],),
    ),
}

```



## High-Level Overview


This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDocstringLinter`

**Functions defined**: `test_python_code`, `test_end_to_end`, `run`, `test_report`, `test_terse`, `test_terse_line`, `test_recursive`, `test_terse_recursive`, `test_terse_line_recursive`, `test_file_summary`, `test_file_names`, `test_decorators`, `_dumps`, `_data`, `_next_stdout`, `function`

**Key imports**: io, itertools, json, sys, tempfile, Path, mock, _get_decorators, LinterTestCase, LinterTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `itertools`
- `json`
- `sys`
- `tempfile`
- `pathlib`: Path
- `unittest`: mock
- `tools.linter.adapters._linter.block`: _get_decorators
- `linter_test_case`: LinterTestCase
- `.linter_test_case`: LinterTestCase


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

This is a test file. Run it with:

```bash
python tools/test/test_docstring_linter.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test`):

- [`test_upload_stats_lib.py_docs.md`](./test_upload_stats_lib.py_docs.md)
- [`test_codegen.py_docs.md`](./test_codegen.py_docs.md)
- [`linter_test_case.py_docs.md`](./linter_test_case.py_docs.md)
- [`test_upload_gate.py_docs.md`](./test_upload_gate.py_docs.md)
- [`test_gen_backend_stubs.py_docs.md`](./test_gen_backend_stubs.py_docs.md)
- [`test_gb_registry_linter.py_docs.md`](./test_gb_registry_linter.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_set_linter.py_docs.md`](./test_set_linter.py_docs.md)
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `test_docstring_linter.py_docs.md`
- **Keyword Index**: `test_docstring_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python docs/tools/test/test_docstring_linter.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/test`):

- [`test_gen_backend_stubs.py_kw.md_docs.md`](./test_gen_backend_stubs.py_kw.md_docs.md)
- [`test_upload_stats_lib.py_kw.md_docs.md`](./test_upload_stats_lib.py_kw.md_docs.md)
- [`test_cmake.py_kw.md_docs.md`](./test_cmake.py_kw.md_docs.md)
- [`test_upload_test_stats.py_docs.md_docs.md`](./test_upload_test_stats.py_docs.md_docs.md)
- [`test_codegen_model.py_docs.md_docs.md`](./test_codegen_model.py_docs.md_docs.md)
- [`test_codegen.py_docs.md_docs.md`](./test_codegen.py_docs.md_docs.md)
- [`test_vulkan_codegen.py_kw.md_docs.md`](./test_vulkan_codegen.py_kw.md_docs.md)
- [`test_set_linter.py_docs.md_docs.md`](./test_set_linter.py_docs.md_docs.md)
- [`test_gb_registry_linter.py_kw.md_docs.md`](./test_gb_registry_linter.py_kw.md_docs.md)
- [`test_upload_test_stats.py_kw.md_docs.md`](./test_upload_test_stats.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_docstring_linter.py_docs.md_docs.md`
- **Keyword Index**: `test_docstring_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

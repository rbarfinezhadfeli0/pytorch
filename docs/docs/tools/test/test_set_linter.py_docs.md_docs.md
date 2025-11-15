# Documentation: `docs/tools/test/test_set_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_set_linter.py_docs.md`
- **Size**: 5,776 bytes (5.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_set_linter.py`

## File Metadata

- **Path**: `tools/test/test_set_linter.py`
- **Size**: 2,938 bytes (2.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```python
# mypy: ignore-errors
from __future__ import annotations

import sys
from pathlib import Path
from token import NAME
from tokenize import TokenInfo

from tools.linter.adapters.set_linter import SetLinter


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase


TESTDATA = Path("tools/test/set_linter_testdata")

TESTFILE = TESTDATA / "python_code.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes_doesnt_change.py.txt"
FILES = TESTFILE, INCLUDES_FILE, INCLUDES_FILE2


class TestSetLinter(LinterTestCase):
    maxDiff = 10000000
    LinterClass = SetLinter

    def test_get_all_tokens(self) -> None:
        self.assertEqual(EXPECTED_SETS, SetLinter.make_file(TESTFILE).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(SetLinter.make_file(TESTFILE).omitted.omitted)
        expected = [6, 16]
        self.assertEqual(expected, actual)

    def test_linting(self) -> None:
        for path in (TESTFILE, INCLUDES_FILE, INCLUDES_FILE2):
            with self.subTest(path):
                r = self.lint_fix_test(path, [])
                self.assertEqual(r.name, "Suggested fixes for set_linter")

    def test_bracket_pairs(self) -> None:
        TESTS: tuple[tuple[str, dict[int, int]], ...] = (
            ("", {}),
            ("{}", {0: 1}),
            ("{1}", {0: 2}),
            ("{1, 2}", {0: 4}),
            ("{1: 2}", {0: 4}),
            ("{One()}", {0: 4, 2: 3}),
            (
                "{One({1: [2], 2: {3}, 3: {4: 5}})}",
                {0: 25, 2: 24, 3: 23, 6: 8, 12: 14, 18: 22},
            ),
            ("f'{a}'", {}),
        )
        for s, expected in TESTS:
            pf = SetLinter.make_file(s)
            if s:
                actual = pf._lines_with_sets[0].bracket_pairs
            else:
                self.assertEqual(pf._lines_with_sets, [])
                actual = {}
            self.assertEqual(actual, expected)

    def test_match_braced_sets(self) -> None:
        TESTS: tuple[tuple[str, int], ...] = (
            ("{cast(int, inst.offset): inst for inst in instructions}", 0),
            ("", 0),
            ("{}", 0),
            ("{1: 0}", 0),
            ("{1}", 1),
            ("{i for i in range(2, 3)}", 1),
            ("{1, 2}", 1),
            ("{One({'a': 1}), Two([{}, {2}, {1, 2}])}", 3),
            ('f" {h:{w}} "', 0),
        )
        for s, expected in TESTS:
            pf = SetLinter.make_file(s)
            actual = pf._lines_with_sets and pf._lines_with_sets[0].braced_sets
            self.assertEqual(len(actual), expected)


EXPECTED_SETS = [
    TokenInfo(NAME, "set", (7, 4), (7, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (9, 4), (9, 7), "c = set\n"),
    TokenInfo(NAME, "set", (12, 3), (12, 6), "   set(\n"),
]

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSetLinter`

**Functions defined**: `test_get_all_tokens`, `test_omitted_lines`, `test_linting`, `test_bracket_pairs`, `test_match_braced_sets`

**Key imports**: annotations, sys, Path, NAME, TokenInfo, SetLinter, LinterTestCase, LinterTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `sys`
- `pathlib`: Path
- `token`: NAME
- `tokenize`: TokenInfo
- `tools.linter.adapters.set_linter`: SetLinter
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
python tools/test/test_set_linter.py
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
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `test_set_linter.py_docs.md`
- **Keyword Index**: `test_set_linter.py_kw.md`
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
python docs/tools/test/test_set_linter.py_docs.md
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
- [`test_gb_registry_linter.py_kw.md_docs.md`](./test_gb_registry_linter.py_kw.md_docs.md)
- [`test_upload_test_stats.py_kw.md_docs.md`](./test_upload_test_stats.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_set_linter.py_docs.md_docs.md`
- **Keyword Index**: `test_set_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

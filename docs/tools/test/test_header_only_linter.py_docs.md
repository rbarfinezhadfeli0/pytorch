# Documentation: `tools/test/test_header_only_linter.py`

## File Metadata

- **Path**: `tools/test/test_header_only_linter.py`
- **Size**: 3,777 bytes (3.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
import re
import unittest

from tools.linter.adapters.header_only_linter import (
    check_file,
    CPP_TEST_GLOBS,
    find_matched_symbols,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
    REPO_ROOT,
)


class TestHeaderOnlyLinter(unittest.TestCase):
    """
    Test the header only linter functionality
    """

    def test_find_matched_symbols(self) -> None:
        sample_regex = re.compile("symDef|symD|symC|bbb|a")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_matches = {"symDef", "symC", "a"}
        self.assertEqual(
            find_matched_symbols(sample_regex, test_globs), expected_matches
        )

    def test_find_matched_symbols_empty_regex(self) -> None:
        sample_regex = re.compile("")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_matches: set[str] = set()
        self.assertEqual(
            find_matched_symbols(sample_regex, test_globs), expected_matches
        )

    def test_check_file_no_issues(self) -> None:
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/good.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]
        self.assertEqual(len(check_file(sample_txt, test_globs)), 0)

    def test_check_empty_file(self) -> None:
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/empty.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]
        self.assertEqual(len(check_file(sample_txt, test_globs)), 0)

    def test_check_file_with_untested_symbols(self) -> None:
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/bad.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_msgs = [
            LintMessage(
                path=sample_txt,
                line=7,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[untested-symbol]",
                original=None,
                replacement=None,
                description=(
                    f"bbb has been included as a header-only API "
                    "but is not tested in any of CPP_TEST_GLOBS, which "
                    f"contains {CPP_TEST_GLOBS}.\n"
                    "Please add a .cpp test using the symbol without "
                    "linking anything to verify that the symbol is in "
                    "fact header-only. If you already have a test but it's"
                    " not found, please add the .cpp file to CPP_TEST_GLOBS"
                    " in tools/linters/adapters/header_only_linter.py."
                ),
            ),
            LintMessage(
                path=sample_txt,
                line=8,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[untested-symbol]",
                original=None,
                replacement=None,
                description=(
                    f"symD has been included as a header-only API "
                    "but is not tested in any of CPP_TEST_GLOBS, which "
                    f"contains {CPP_TEST_GLOBS}.\n"
                    "Please add a .cpp test using the symbol without "
                    "linking anything to verify that the symbol is in "
                    "fact header-only. If you already have a test but it's"
                    " not found, please add the .cpp file to CPP_TEST_GLOBS"
                    " in tools/linters/adapters/header_only_linter.py."
                ),
            ),
        ]
        self.assertEqual(set(check_file(sample_txt, test_globs)), set(expected_msgs))


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview

"""    Test the header only linter functionality

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHeaderOnlyLinter`

**Functions defined**: `test_find_matched_symbols`, `test_find_matched_symbols_empty_regex`, `test_check_file_no_issues`, `test_check_empty_file`, `test_check_file_with_untested_symbols`

**Key imports**: re, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `re`
- `unittest`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

This is a test file. Run it with:

```bash
python tools/test/test_header_only_linter.py
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

- **File Documentation**: `test_header_only_linter.py_docs.md`
- **Keyword Index**: `test_header_only_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/tools/test/test_gb_registry_linter.py_docs.md`

## File Metadata

- **Path**: `docs/tools/test/test_gb_registry_linter.py_docs.md`
- **Size**: 17,412 bytes (17.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/test/test_gb_registry_linter.py`

## File Metadata

- **Path**: `tools/test/test_gb_registry_linter.py`
- **Size**: 14,284 bytes (13.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: ignore-errors
import json
import shutil
import unittest
from pathlib import Path

from tools.linter.adapters.gb_registry_linter import (
    check_registry_sync,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
)


class TestGraphBreakRegistryLinter(unittest.TestCase):
    """
    Test the graph break registry linter functionality
    """

    def setUp(self):
        script_dir = Path(__file__).resolve()
        self.test_data_dir = script_dir.parent / "graph_break_registry_linter_testdata"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.test_data_dir / "graph_break_test_registry.json"
        with open(self.registry_path, "w") as f:
            json.dump({}, f)

        self.callsite_file = self.test_data_dir / "callsite_test.py"
        callsite_content = """from torch._dynamo.exc import unimplemented

def test(self):
    unimplemented(
        gb_type="testing",
        context="testing",
        explanation="testing",
        hints=["testing"],
    )
"""
        with open(self.callsite_file, "w") as f:
            f.write(callsite_content)

    def tearDown(self):
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def test_case1_new_gb_type(self):
        """Test Case 1: Adding a completely new gb_type to an empty registry."""
        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)

        expected_registry = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                }
            ]
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (added 1 new gb_types). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case2_rename_gb_type(self):
        """Test Case 2: Renaming a gb_type while keeping other content the same."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        renamed_callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(gb_type="renamed_testing", context="testing", explanation="testing", hints=["testing"])
"""
        with open(self.callsite_file, "w") as f:
            f.write(renamed_callsite_content)

        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        expected_registry = {
            "GB0000": [
                {
                    "Gb_type": "renamed_testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                },
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                },
            ]
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (renamed 'testing' → 'renamed_testing'). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case3_content_change(self):
        """Test Case 3: Changing the content of an existing gb_type."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "old_context",
                    "Explanation": "old_explanation",
                    "Hints": ["old_hint"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        updated_callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(gb_type="testing", context="new_context", explanation="new_explanation", hints=["new_hint"])
"""
        with open(self.callsite_file, "w") as f:
            f.write(updated_callsite_content)

        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        expected_registry = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "new_context",
                    "Explanation": "new_explanation",
                    "Hints": ["new_hint"],
                },
                {
                    "Gb_type": "testing",
                    "Context": "old_context",
                    "Explanation": "old_explanation",
                    "Hints": ["old_hint"],
                },
            ]
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case4_no_changes(self):
        """Test Case 4: Ensuring no message is produced when the registry is in sync."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "testing",
                    "Context": "testing",
                    "Explanation": "testing",
                    "Hints": ["testing"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages), 0, "Should have no messages when registry is already in sync"
        )

    def test_case5_new_gbid_on_full_change(self):
        """Test Case 5: A completely new entry should get a new GB ID."""
        registry_data = {
            "GB0000": [
                {
                    "Gb_type": "original_testing",
                    "Context": "original_context",
                    "Explanation": "original_explanation",
                    "Hints": ["original_hint"],
                }
            ]
        }
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2)

        new_callsite_content = """from torch._dynamo.exc import unimplemented
def test(self):
    unimplemented(
        gb_type="completely_new_testing",
        context="completely_new_context",
        explanation="completely_new_explanation",
        hints=["completely_new_hint"],
    )
"""
        with open(self.callsite_file, "w") as f:
            f.write(new_callsite_content)

        with open(self.registry_path) as f:
            original_content = f.read()

        messages = check_registry_sync(self.test_data_dir, self.registry_path)
        expected_registry = {
            "GB0000": [
                {
                    "Gb_type": "original_testing",
                    "Context": "original_context",
                    "Explanation": "original_explanation",
                    "Hints": ["original_hint"],
                }
            ],
            "GB0001": [
                {
                    "Gb_type": "completely_new_testing",
                    "Context": "completely_new_context",
                    "Explanation": "completely_new_explanation",
                    "Hints": ["completely_new_hint"],
                }
            ],
        }
        expected_replacement = (
            json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
        )
        expected_msg = LintMessage(
            path=str(self.registry_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="Registry sync needed",
            original=original_content,
            replacement=expected_replacement,
            description="Registry sync needed (added 1 new gb_types). Run `lintrunner -a` to apply changes.",
        )
        self.assertEqual(messages, [expected_msg])

        # Apply the fix and verify the file's final state
        if messages and messages[0].replacement:
            with open(self.registry_path, "w") as f:
                f.write(messages[0].replacement)

        messages_after_fix = check_registry_sync(self.test_data_dir, self.registry_path)
        self.assertEqual(
            len(messages_after_fix), 0, "Should have no messages after applying the fix"
        )

    def test_case6_dynamic_hints_from_variable(self):
        """Test Case 6: Verifies hints can be unpacked from an imported variable."""
        mock_hints_file = self.test_data_dir / "graph_break_hints.py"
        init_py = self.test_data_dir / "__init__.py"
        try:
            supportable_string = (
                "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you "
                "encounter this graph break often and it is causing performance issues."
            )
            mock_hints_content = f'SUPPORTABLE = ["{supportable_string}"]'
            with open(mock_hints_file, "w") as f:
                f.write(mock_hints_content)

            init_py.touch()

            dynamic_hints_callsite = """from torch._dynamo.exc import unimplemented
from torch._dynamo import graph_break_hints

def test(self):
    unimplemented(
        gb_type="testing_with_graph_break_hints",
        context="testing_with_graph_break_hints",
        explanation="testing_with_graph_break_hints",
        hints=[*graph_break_hints.SUPPORTABLE],
    )
    """
            with open(self.callsite_file, "w") as f:
                f.write(dynamic_hints_callsite)

            with open(self.registry_path) as f:
                original_content = f.read()

            messages = check_registry_sync(self.test_data_dir, self.registry_path)

            expected_registry = {
                "GB0000": [
                    {
                        "Gb_type": "testing_with_graph_break_hints",
                        "Context": "testing_with_graph_break_hints",
                        "Explanation": "testing_with_graph_break_hints",
                        "Hints": [supportable_string],
                    }
                ]
            }
            expected_replacement = (
                json.dumps(expected_registry, indent=2, ensure_ascii=False) + "\n"
            )
            expected_msg = LintMessage(
                path=str(self.registry_path),
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="Registry sync needed",
                original=original_content,
                replacement=expected_replacement,
                description="Registry sync needed (added 1 new gb_types). Run `lintrunner -a` to apply changes.",
            )

            self.assertEqual(messages, [expected_msg])

            if messages and messages[0].replacement:
                with open(self.registry_path, "w") as f:
                    f.write(messages[0].replacement)

            messages_after_fix = check_registry_sync(
                self.test_data_dir, self.registry_path
            )
            self.assertEqual(
                len(messages_after_fix),
                0,
                "Should have no messages after applying the fix",
            )
        finally:
            mock_hints_file.unlink()
            init_py.unlink()


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview

"""    Test the graph break registry linter functionality

This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGraphBreakRegistryLinter`

**Functions defined**: `setUp`, `test`, `tearDown`, `test_case1_new_gb_type`, `test_case2_rename_gb_type`, `test`, `test_case3_content_change`, `test`, `test_case4_no_changes`, `test_case5_new_gbid_on_full_change`, `test`, `test_case6_dynamic_hints_from_variable`, `test`

**Key imports**: json, shutil, unittest, Path, unimplemented, unimplemented, unimplemented, unimplemented, unimplemented, graph_break_hints


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `shutil`
- `unittest`
- `pathlib`: Path
- `torch._dynamo.exc`: unimplemented
- `torch._dynamo`: graph_break_hints


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
python tools/test/test_gb_registry_linter.py
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
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_set_linter.py_docs.md`](./test_set_linter.py_docs.md)
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `test_gb_registry_linter.py_docs.md`
- **Keyword Index**: `test_gb_registry_linter.py_kw.md`
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
python docs/tools/test/test_gb_registry_linter.py_docs.md
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

- **File Documentation**: `test_gb_registry_linter.py_docs.md_docs.md`
- **Keyword Index**: `test_gb_registry_linter.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

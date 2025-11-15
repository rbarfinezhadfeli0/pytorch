# Documentation: `docs/tools/testing/target_determination/heuristics/edited_by_pr.py_docs.md`

## File Metadata

- **Path**: `docs/tools/testing/target_determination/heuristics/edited_by_pr.py_docs.md`
- **Size**: 4,920 bytes (4.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/testing/target_determination/heuristics/edited_by_pr.py`

## File Metadata

- **Path**: `tools/testing/target_determination/heuristics/edited_by_pr.py`
- **Size**: 2,038 bytes (1.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import re
from typing import Any
from warnings import warn

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
    query_changed_files,
)
from tools.testing.test_run import TestRun


# Some files run tests in other test files, so we map them to each other here.
# This is a map from file that runs the test to regex that matches the file that
# contains the test. Test file with path test/a/b.py should of the form a/b.
# Regexes should be based on repo root.
ADDITIONAL_MAPPINGS = {
    # Not files that are tracked by git but rather functions defined in
    # run_test.py that generate test files which run tests in test/cpp_extensions.
    "test_cpp_extensions_aot_ninja": [r"test\/cpp_extensions.*"],
    "test_cpp_extensions_aot_no_ninja": [r"test\/cpp_extensions.*"],
}


class EditedByPR(HeuristicInterface):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        critical_tests = _get_modified_tests()
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )


def _get_modified_tests() -> set[str]:
    try:
        changed_files = query_changed_files()
        should_run = python_test_file_to_test_name(set(changed_files))
        for test_file, regexes in ADDITIONAL_MAPPINGS.items():
            if any(
                re.search(regex, changed_file) is not None
                for regex in regexes
                for changed_file in changed_files
            ):
                should_run.add(test_file)
        return should_run
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        # If unable to get changed files from git, quit without doing any sorting
    return set()

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EditedByPR`

**Functions defined**: `__init__`, `get_prediction_confidence`, `_get_modified_tests`

**Key imports**: annotations, re, Any, warn, TestRun


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/testing/target_determination/heuristics`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `re`
- `typing`: Any
- `warnings`: warn
- `tools.testing.test_run`: TestRun


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

This is a test file. Run it with:

```bash
python tools/testing/target_determination/heuristics/edited_by_pr.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing/target_determination/heuristics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`previously_failed_in_pr.py_docs.md`](./previously_failed_in_pr.py_docs.md)
- [`mentioned_in_pr.py_docs.md`](./mentioned_in_pr.py_docs.md)
- [`historical_class_failure_correlation.py_docs.md`](./historical_class_failure_correlation.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`llm.py_docs.md`](./llm.py_docs.md)
- [`correlated_with_historical_failures.py_docs.md`](./correlated_with_historical_failures.py_docs.md)
- [`public_bindings.py_docs.md`](./public_bindings.py_docs.md)
- [`interface.py_docs.md`](./interface.py_docs.md)
- [`historical_edited_files.py_docs.md`](./historical_edited_files.py_docs.md)


## Cross-References

- **File Documentation**: `edited_by_pr.py_docs.md`
- **Keyword Index**: `edited_by_pr.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/testing/target_determination/heuristics`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/testing/target_determination/heuristics`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

This is a test file. Run it with:

```bash
python docs/tools/testing/target_determination/heuristics/edited_by_pr.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/testing/target_determination/heuristics`):

- [`interface.py_kw.md_docs.md`](./interface.py_kw.md_docs.md)
- [`correlated_with_historical_failures.py_docs.md_docs.md`](./correlated_with_historical_failures.py_docs.md_docs.md)
- [`public_bindings.py_docs.md_docs.md`](./public_bindings.py_docs.md_docs.md)
- [`filepath.py_docs.md_docs.md`](./filepath.py_docs.md_docs.md)
- [`previously_failed_in_pr.py_docs.md_docs.md`](./previously_failed_in_pr.py_docs.md_docs.md)
- [`profiling.py_docs.md_docs.md`](./profiling.py_docs.md_docs.md)
- [`profiling.py_kw.md_docs.md`](./profiling.py_kw.md_docs.md)
- [`llm.py_docs.md_docs.md`](./llm.py_docs.md_docs.md)
- [`filepath.py_kw.md_docs.md`](./filepath.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `edited_by_pr.py_docs.md_docs.md`
- **Keyword Index**: `edited_by_pr.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

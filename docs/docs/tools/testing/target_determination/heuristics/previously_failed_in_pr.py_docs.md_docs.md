# Documentation: `docs/tools/testing/target_determination/heuristics/previously_failed_in_pr.py_docs.md`

## File Metadata

- **Path**: `docs/tools/testing/target_determination/heuristics/previously_failed_in_pr.py_docs.md`
- **Size**: 5,810 bytes (5.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/testing/target_determination/heuristics/previously_failed_in_pr.py`

## File Metadata

- **Path**: `tools/testing/target_determination/heuristics/previously_failed_in_pr.py`
- **Size**: 2,822 bytes (2.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TD_HEURISTIC_PREVIOUSLY_FAILED,
    TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL,
)
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    python_test_file_to_test_name,
)
from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]


class PreviouslyFailedInPR(HeuristicInterface):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        critical_tests = get_previous_failures() | read_additional_test_failures_file()
        return TestPrioritizations(
            tests, {TestRun(test): 1 for test in critical_tests if test in tests}
        )


def get_previous_failures() -> set[str]:
    path = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PREVIOUSLY_FAILED
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return set()
    with open(path) as f:
        return python_test_file_to_test_name(
            _parse_prev_failing_test_files(json.load(f))
        )


def _parse_prev_failing_test_files(last_failed_tests: dict[str, bool]) -> set[str]:
    prioritized_tests = set()

    # The keys are formatted as "test_file.py::test_class::test_method[params]"
    # We just need the test_file part
    for test in last_failed_tests:
        parts = test.split("::")
        if len(parts) > 1:
            test_file = parts[0]
            prioritized_tests.add(test_file)

    return prioritized_tests


def gen_additional_test_failures_file(tests: list[str]) -> None:
    # Segfaults usually result in no xml and some tests don't run through pytest
    # (ex doctests).  In these cases, there will be no entry in the pytest
    # cache, so we should generate a separate file for them and upload it to s3
    # along with the pytest cache
    pytest_cache_dir = REPO_ROOT / ".pytest_cache"
    if not os.path.exists(pytest_cache_dir):
        os.makedirs(pytest_cache_dir)
    with open(pytest_cache_dir / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL, "w") as f:
        json.dump(tests, f, indent=2)


def read_additional_test_failures_file() -> set[str]:
    path = (
        REPO_ROOT
        / ADDITIONAL_CI_FILES_FOLDER
        / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    )
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return set()
    with open(path) as f:
        s = set(json.load(f))
        print(f"additional failures: {s}")
        return s

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PreviouslyFailedInPR`

**Functions defined**: `__init__`, `get_prediction_confidence`, `get_previous_failures`, `_parse_prev_failing_test_files`, `gen_additional_test_failures_file`, `read_additional_test_failures_file`

**Key imports**: annotations, json, os, Path, Any, TestRun


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/testing/target_determination/heuristics`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `json`
- `os`
- `pathlib`: Path
- `typing`: Any
- `tools.testing.test_run`: TestRun


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python tools/testing/target_determination/heuristics/previously_failed_in_pr.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing/target_determination/heuristics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mentioned_in_pr.py_docs.md`](./mentioned_in_pr.py_docs.md)
- [`historical_class_failure_correlation.py_docs.md`](./historical_class_failure_correlation.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`llm.py_docs.md`](./llm.py_docs.md)
- [`correlated_with_historical_failures.py_docs.md`](./correlated_with_historical_failures.py_docs.md)
- [`public_bindings.py_docs.md`](./public_bindings.py_docs.md)
- [`interface.py_docs.md`](./interface.py_docs.md)
- [`historical_edited_files.py_docs.md`](./historical_edited_files.py_docs.md)


## Cross-References

- **File Documentation**: `previously_failed_in_pr.py_docs.md`
- **Keyword Index**: `previously_failed_in_pr.py_kw.md`
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


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
python docs/tools/testing/target_determination/heuristics/previously_failed_in_pr.py_docs.md
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
- [`edited_by_pr.py_docs.md_docs.md`](./edited_by_pr.py_docs.md_docs.md)
- [`profiling.py_docs.md_docs.md`](./profiling.py_docs.md_docs.md)
- [`profiling.py_kw.md_docs.md`](./profiling.py_kw.md_docs.md)
- [`llm.py_docs.md_docs.md`](./llm.py_docs.md_docs.md)
- [`filepath.py_kw.md_docs.md`](./filepath.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `previously_failed_in_pr.py_docs.md_docs.md`
- **Keyword Index**: `previously_failed_in_pr.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `tools/testing/target_determination/heuristics/historical_class_failure_correlation.py`

## File Metadata

- **Path**: `tools/testing/target_determination/heuristics/historical_class_failure_correlation.py`
- **Size**: 3,027 bytes (2.96 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, cast
from warnings import warn

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_CLASS_RATINGS_FILE,
)
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    normalize_ratings,
    query_changed_files,
    REPO_ROOT,
)
from tools.testing.test_run import TestRun


class HistoricalClassFailurCorrelation(HeuristicInterface):
    """
    This heuristic prioritizes test classes that have historically tended to fail
    when the files edited by current PR were modified.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        ratings = _get_ratings_for_tests(set(tests))
        test_ratings = {
            TestRun(k): v for (k, v) in ratings.items() if TestRun(k).test_file in tests
        }
        return TestPrioritizations(tests, normalize_ratings(test_ratings, 0.25))


def _get_historical_test_class_correlations() -> dict[str, dict[str, float]]:
    path = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TEST_CLASS_RATINGS_FILE
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return {}
    with open(path) as f:
        test_class_correlations = cast(dict[str, dict[str, float]], json.load(f))
        return test_class_correlations


def _get_ratings_for_tests(
    tests_to_run: set[str],
) -> dict[str, float]:
    # Get the files edited
    try:
        changed_files = query_changed_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        return {}

    test_class_correlations = _get_historical_test_class_correlations()
    if not test_class_correlations:
        return {}

    # Find the tests failures that are correlated with the edited files.
    # Filter the list to only include tests we want to run.
    ratings: dict[str, float] = defaultdict(float)
    for file in changed_files:
        for qualified_test_class, score in test_class_correlations.get(
            file, {}
        ).items():
            # qualified_test_class looks like "test_file::test_class"
            test_file, test_class = qualified_test_class.split("::")
            if test_file in tests_to_run:
                ratings[qualified_test_class] += score

    return ratings


def _rank_correlated_tests(
    tests_to_run: list[str],
) -> list[str]:
    # Find the tests failures that are correlated with the edited files.
    # Filter the list to only include tests we want to run.
    # pyrefly: ignore [bad-assignment]
    tests_to_run = set(tests_to_run)
    # pyrefly: ignore [bad-argument-type]
    ratings = _get_ratings_for_tests(tests_to_run)
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize

```



## High-Level Overview

"""    This heuristic prioritizes test classes that have historically tended to fail    when the files edited by current PR were modified.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HistoricalClassFailurCorrelation`

**Functions defined**: `__init__`, `get_prediction_confidence`, `_get_historical_test_class_correlations`, `_get_ratings_for_tests`, `_rank_correlated_tests`

**Key imports**: annotations, json, os, defaultdict, Any, cast, warn, TestRun


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
- `collections`: defaultdict
- `typing`: Any, cast
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
python tools/testing/target_determination/heuristics/historical_class_failure_correlation.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing/target_determination/heuristics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`previously_failed_in_pr.py_docs.md`](./previously_failed_in_pr.py_docs.md)
- [`mentioned_in_pr.py_docs.md`](./mentioned_in_pr.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`llm.py_docs.md`](./llm.py_docs.md)
- [`correlated_with_historical_failures.py_docs.md`](./correlated_with_historical_failures.py_docs.md)
- [`public_bindings.py_docs.md`](./public_bindings.py_docs.md)
- [`interface.py_docs.md`](./interface.py_docs.md)
- [`historical_edited_files.py_docs.md`](./historical_edited_files.py_docs.md)


## Cross-References

- **File Documentation**: `historical_class_failure_correlation.py_docs.md`
- **Keyword Index**: `historical_class_failure_correlation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

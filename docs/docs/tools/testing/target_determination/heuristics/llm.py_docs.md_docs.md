# Documentation: `docs/tools/testing/target_determination/heuristics/llm.py_docs.md`

## File Metadata

- **Path**: `docs/tools/testing/target_determination/heuristics/llm.py_docs.md`
- **Size**: 4,825 bytes (4.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/testing/target_determination/heuristics/llm.py`

## File Metadata

- **Path**: `tools/testing/target_determination/heuristics/llm.py`
- **Size**: 1,840 bytes (1.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import normalize_ratings
from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]


class LLM(HeuristicInterface):
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        critical_tests = self.get_mappings()
        filter_valid_tests = {
            TestRun(test): score
            for test, score in critical_tests.items()
            if test in tests
        }
        normalized_scores = normalize_ratings(filter_valid_tests, 0.25)
        return TestPrioritizations(tests, normalized_scores)

    def get_mappings(self) -> dict[str, float]:
        path = (
            REPO_ROOT
            / ADDITIONAL_CI_FILES_FOLDER
            / "llm_results/mappings/indexer-files-gitdiff-output.json"
        )
        if not os.path.exists(path):
            print(f"could not find path {path}")
            return {}
        with open(path) as f:
            # Group by file
            r = defaultdict(list)
            for key, value in json.load(f).items():
                re_match = re.match("(.*).py", key)
                if re_match:
                    file = re_match.group(1)
                    r[file].append(value)
            # Average the scores for each file
            r = {file: sum(scores) / len(scores) for file, scores in r.items()}
            return r

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LLM`

**Functions defined**: `__init__`, `get_prediction_confidence`, `get_mappings`

**Key imports**: annotations, json, os, re, defaultdict, Path, Any, ADDITIONAL_CI_FILES_FOLDER, normalize_ratings, TestRun


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
- `re`
- `collections`: defaultdict
- `pathlib`: Path
- `typing`: Any
- `tools.stats.import_test_stats`: ADDITIONAL_CI_FILES_FOLDER
- `tools.testing.target_determination.heuristics.utils`: normalize_ratings
- `tools.testing.test_run`: TestRun


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
python tools/testing/target_determination/heuristics/llm.py
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
- [`correlated_with_historical_failures.py_docs.md`](./correlated_with_historical_failures.py_docs.md)
- [`public_bindings.py_docs.md`](./public_bindings.py_docs.md)
- [`interface.py_docs.md`](./interface.py_docs.md)
- [`historical_edited_files.py_docs.md`](./historical_edited_files.py_docs.md)


## Cross-References

- **File Documentation**: `llm.py_docs.md`
- **Keyword Index**: `llm.py_kw.md`
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
python docs/tools/testing/target_determination/heuristics/llm.py_docs.md
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
- [`edited_by_pr.py_docs.md_docs.md`](./edited_by_pr.py_docs.md_docs.md)
- [`profiling.py_docs.md_docs.md`](./profiling.py_docs.md_docs.md)
- [`profiling.py_kw.md_docs.md`](./profiling.py_kw.md_docs.md)
- [`filepath.py_kw.md_docs.md`](./filepath.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `llm.py_docs.md_docs.md`
- **Keyword Index**: `llm.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

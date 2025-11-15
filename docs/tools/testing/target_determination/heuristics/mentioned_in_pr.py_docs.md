# Documentation: `tools/testing/target_determination/heuristics/mentioned_in_pr.py`

## File Metadata

- **Path**: `tools/testing/target_determination/heuristics/mentioned_in_pr.py`
- **Size**: 2,298 bytes (2.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import re
from typing import Any

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    get_git_commit_info,
    get_issue_or_pr_body,
    get_pr_number,
)
from tools.testing.test_run import TestRun


# This heuristic searches the PR body and commit titles, as well as issues/PRs
# mentioned in the PR body/commit title for test names (search depth of 1) and
# gives the test a rating of 1.  For example, if I mention "test_foo" in the PR
# body, test_foo will be rated 1.  If I mention #123 in the PR body, and #123
# mentions "test_foo", test_foo will be rated 1.
class MentionedInPR(HeuristicInterface):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _search_for_linked_issues(self, s: str) -> list[str]:
        return re.findall(r"#(\d+)", s) + re.findall(r"/pytorch/pytorch/.*/(\d+)", s)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        try:
            commit_messages = get_git_commit_info()
        except Exception as e:
            print(f"Can't get commit info due to {e}")
            commit_messages = ""
        try:
            pr_number = get_pr_number()
            if pr_number is not None:
                pr_body = get_issue_or_pr_body(pr_number)
            else:
                pr_body = ""
        except Exception as e:
            print(f"Can't get PR body due to {e}")
            pr_body = ""

        # Search for linked issues or PRs
        linked_issue_bodies: list[str] = []
        for issue in self._search_for_linked_issues(
            commit_messages
        ) + self._search_for_linked_issues(pr_body):
            try:
                linked_issue_bodies.append(get_issue_or_pr_body(int(issue)))
            except Exception:
                pass

        mentioned = []
        for test in tests:
            if (
                test in commit_messages
                or test in pr_body
                or any(test in body for body in linked_issue_bodies)
            ):
                mentioned.append(test)

        return TestPrioritizations(tests, {TestRun(test): 1 for test in mentioned})

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MentionedInPR`

**Functions defined**: `__init__`, `_search_for_linked_issues`, `get_prediction_confidence`

**Key imports**: annotations, re, Any, TestRun


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
python tools/testing/target_determination/heuristics/mentioned_in_pr.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing/target_determination/heuristics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`previously_failed_in_pr.py_docs.md`](./previously_failed_in_pr.py_docs.md)
- [`historical_class_failure_correlation.py_docs.md`](./historical_class_failure_correlation.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`llm.py_docs.md`](./llm.py_docs.md)
- [`correlated_with_historical_failures.py_docs.md`](./correlated_with_historical_failures.py_docs.md)
- [`public_bindings.py_docs.md`](./public_bindings.py_docs.md)
- [`interface.py_docs.md`](./interface.py_docs.md)
- [`historical_edited_files.py_docs.md`](./historical_edited_files.py_docs.md)


## Cross-References

- **File Documentation**: `mentioned_in_pr.py_docs.md`
- **Keyword Index**: `mentioned_in_pr.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

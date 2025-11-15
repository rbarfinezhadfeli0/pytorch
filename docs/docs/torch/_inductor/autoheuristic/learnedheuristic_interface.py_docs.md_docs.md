# Documentation: `docs/torch/_inductor/autoheuristic/learnedheuristic_interface.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/autoheuristic/learnedheuristic_interface.py_docs.md`
- **Size**: 5,414 bytes (5.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/autoheuristic/learnedheuristic_interface.py`

## File Metadata

- **Path**: `torch/_inductor/autoheuristic/learnedheuristic_interface.py`
- **Size**: 2,852 bytes (2.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import operator
from typing import Optional

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)


class LearnedHeuristic:
    """
    LearnedHeuristic is a base class for all learned heuristics.
    """

    def __init__(self) -> None:
        pass

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
        return True

    def get_decision(
        self, context: AHContext, choices: list[Choice]
    ) -> Optional[Choice]:
        return None

    def get_confidence_threshold(self) -> float:
        return 1.0

    def get_name(self) -> str:
        return ""

    def get_decisions_ranked(self, context: AHContext) -> Optional[list[str]]:
        return None


class LearnedHeuristicRegression(LearnedHeuristic):
    def __init__(self) -> None:
        super().__init__()

    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        return 1.0

    def get_decision(
        self, context: AHContext, choices: list[Choice]
    ) -> Optional[Choice]:
        choice2feedback = {}
        for choice in choices:
            predicted_feedback = self.get_feedback(context, choice)
            choice2feedback[choice] = predicted_feedback
        sorted_choices_feedback = sorted(
            choice2feedback.items(), key=operator.itemgetter(1)
        )
        highest_feedback = sorted_choices_feedback[-1][1]
        second_highest_feedback = sorted_choices_feedback[-2][1]
        if highest_feedback / second_highest_feedback > self.get_confidence_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None


class LearnedHeuristicDecision(LearnedHeuristic):
    def __init__(self) -> None:
        super().__init__()

    def get_choice(self, idx: int) -> Optional[str]:
        return None

    def get_decision(
        self, context: AHContext, choices: list[Choice]
    ) -> Optional[Choice]:
        best_choices = self.get_best_choices(context)
        if not best_choices:
            return None
        (best_choice_proba, best_choice_idx) = best_choices[0]
        if best_choice_proba <= self.get_confidence_threshold():
            return None
        return self.get_choice(best_choice_idx)

    def get_decisions_ranked(self, context: AHContext) -> Optional[list[str]]:
        feedback_idx_list = self.get_best_choices(context)
        if feedback_idx_list is None:
            return None
        choices = [
            self.get_choice(feedback_idx[1]) for feedback_idx in feedback_idx_list
        ]
        choices = [choice for choice in choices if choice is not None]
        return choices

    def get_best_choices(self, context: AHContext) -> Optional[list[tuple[float, int]]]:
        return []

```



## High-Level Overview

"""    LearnedHeuristic is a base class for all learned heuristics.

This Python file contains 4 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LearnedHeuristic`, `LearnedHeuristicRegression`, `LearnedHeuristicDecision`

**Functions defined**: `__init__`, `check_precondition`, `get_decision`, `get_confidence_threshold`, `get_name`, `get_decisions_ranked`, `__init__`, `get_feedback`, `get_decision`, `__init__`, `get_choice`, `get_decision`, `get_decisions_ranked`, `get_best_choices`

**Key imports**: operator, Optional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/autoheuristic`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `typing`: Optional


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/autoheuristic`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`learned_heuristic_controller.py_docs.md`](./learned_heuristic_controller.py_docs.md)
- [`autoheuristic_utils.py_docs.md`](./autoheuristic_utils.py_docs.md)
- [`autoheuristic.py_docs.md`](./autoheuristic.py_docs.md)


## Cross-References

- **File Documentation**: `learnedheuristic_interface.py_docs.md`
- **Keyword Index**: `learnedheuristic_interface.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/autoheuristic`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/autoheuristic`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_inductor/autoheuristic`):

- [`learnedheuristic_interface.py_kw.md_docs.md`](./learnedheuristic_interface.py_kw.md_docs.md)
- [`autoheuristic_utils.py_docs.md_docs.md`](./autoheuristic_utils.py_docs.md_docs.md)
- [`autoheuristic_utils.py_kw.md_docs.md`](./autoheuristic_utils.py_kw.md_docs.md)
- [`learned_heuristic_controller.py_docs.md_docs.md`](./learned_heuristic_controller.py_docs.md_docs.md)
- [`learned_heuristic_controller.py_kw.md_docs.md`](./learned_heuristic_controller.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`autoheuristic.py_kw.md_docs.md`](./autoheuristic.py_kw.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`autoheuristic.py_docs.md_docs.md`](./autoheuristic.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `learnedheuristic_interface.py_docs.md_docs.md`
- **Keyword Index**: `learnedheuristic_interface.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

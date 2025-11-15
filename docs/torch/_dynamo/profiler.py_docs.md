# Documentation: `torch/_dynamo/profiler.py`

## File Metadata

- **Path**: `torch/_dynamo/profiler.py`
- **Size**: 5,894 bytes (5.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Dynamo profiling implementation.

This module provides profiling functionality for Dynamo, including:
- ProfileMetrics: Class for collecting and aggregating performance metrics like
  execution time, operator counts, and fusion statistics
- ProfileResult: Class for analyzing and reporting profiling results
- Utilities for tracking missed/uncaptured operations
- Functions for instrumenting FX graphs with profiling capabilities

The profiler helps measure and optimize the performance of Dynamo-compiled code
by tracking both captured and total operations, timing, and graph statistics.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any
from typing_extensions import Self

import torch

from .utils import print_once


@dataclasses.dataclass
class ProfileMetrics:
    microseconds: float = 0.0
    operators: int = 0
    fusions: int = 0
    graphs: int = 0

    def __iadd__(self, other: Self) -> Self:
        self.microseconds += other.microseconds
        self.operators += other.operators
        self.fusions += other.fusions
        return self

    def __add__(self, other: ProfileMetrics) -> ProfileMetrics:
        assert isinstance(other, ProfileMetrics)
        return ProfileMetrics(
            self.microseconds + other.microseconds,
            self.operators + other.operators,
            self.fusions + other.fusions,
        )

    def __truediv__(self, other: Any) -> ProfileMetrics:
        if isinstance(other, int):
            other = ProfileMetrics(other, other, other)
        return ProfileMetrics(
            # pyrefly: ignore [no-matching-overload]
            self.microseconds / max(1, other.microseconds),
            # pyrefly: ignore [bad-argument-type]
            self.operators / max(1, other.operators),
            # pyrefly: ignore [bad-argument-type]
            self.fusions / max(1, other.fusions),
        )

    def __str__(self) -> str:
        return f"{self.operators:4.0%} ops {self.microseconds:4.0%} time"

    def tocsv(self) -> list[float]:
        return [self.operators, self.microseconds]


class ProfileResult:
    def __init__(
        self, captured: ProfileMetrics, total: ProfileMetrics, unique_graphs: int
    ) -> None:
        self.captured: ProfileMetrics = captured or ProfileMetrics()
        self.total: ProfileMetrics = total or ProfileMetrics()
        self.unique_graphs: int = unique_graphs

    def __iadd__(self, other: Self) -> Self:
        self.captured += other.captured
        self.total += other.total
        self.unique_graphs += other.unique_graphs
        return self

    def percent(self) -> ProfileMetrics:
        return self.captured / self.total

    def __str__(self) -> str:
        return (
            f"{self.unique_graphs:2} graphs {self.captured.graphs:2} graph calls "
            f"{self.captured.operators:4}/{self.total.operators:4} = "
            + str(self.percent())
        )

    def tocsv(self) -> list[Any]:
        return [
            self.unique_graphs,
            self.captured.graphs,
            self.captured.operators,
            self.total.operators,
        ] + self.percent().tocsv()


def should_print_missing() -> bool:
    return os.environ.get("TORCHDYNAMO_PRINT_MISSING") == "1"


def print_missing(stack: list[str]) -> None:
    if any("/torch/autograd/profiler.py" in x for x in stack):
        return
    stack = [
        x for x in stack if ("<built-in" not in x and "site-packages/torch/" not in x)
    ]
    print_once("MISSING", " >> ".join(stack[-3:]))


class Profiler:
    unique_graphs: int = 0

    def __init__(self) -> None:
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=should_print_missing(),
        )

    def results(self) -> ProfileResult:
        captured_regions = 0
        captured_ops = 0
        captured_microseconds = 0
        total_ops = 0
        total_microseconds = 0

        last_op_end_time = -1
        captured_region_end_time = -1
        events = sorted(self.prof.events(), key=lambda x: x.time_range.start)
        for e in events:
            if e.name == "TORCHDYNAMO":
                captured_region_end_time = e.time_range.end
                captured_regions += 1
                # ignore `handle = torch.zeros(1)` in record_function.__init__()
                total_ops -= 1
            elif e.time_range.start >= last_op_end_time:
                last_op_end_time = e.time_range.end
                if e.time_range.end <= captured_region_end_time:
                    captured_ops += 1
                    captured_microseconds += e.time_range.elapsed_us()
                elif should_print_missing():
                    print_missing(e.stack)
                total_ops += 1
                total_microseconds += e.time_range.elapsed_us()
            else:
                pass  # ops recursively called from other ops (ignored)

        unique_graphs = Profiler.unique_graphs
        Profiler.unique_graphs = 0
        # we counted one extra op that is part of the profiler setup code
        total_ops -= 1

        return ProfileResult(
            captured=ProfileMetrics(
                microseconds=captured_microseconds,
                operators=captured_ops,
                fusions=captured_ops - captured_regions,
                graphs=captured_regions,
            ),
            total=ProfileMetrics(
                microseconds=total_microseconds,
                operators=total_ops,
                fusions=total_ops - 1,
            ),
            unique_graphs=unique_graphs,
        )


def fx_insert_profiling(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Any:
    def _wrapped(*args: Any) -> Any:
        with torch.profiler.record_function("TORCHDYNAMO"):
            return gm.forward(*args)

    Profiler.unique_graphs += 1
    return _wrapped

```



## High-Level Overview

"""Dynamo profiling implementation.This module provides profiling functionality for Dynamo, including:- ProfileMetrics: Class for collecting and aggregating performance metrics like  execution time, operator counts, and fusion statistics- ProfileResult: Class for analyzing and reporting profiling results- Utilities for tracking missed/uncaptured operations- Functions for instrumenting FX graphs with profiling capabilitiesThe profiler helps measure and optimize the performance of Dynamo-compiled codeby tracking both captured and total operations, timing, and graph statistics.

This Python file contains 3 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ProfileMetrics`, `ProfileResult`, `Profiler`

**Functions defined**: `__iadd__`, `__add__`, `__truediv__`, `__str__`, `tocsv`, `__init__`, `__iadd__`, `percent`, `__str__`, `tocsv`, `should_print_missing`, `print_missing`, `__init__`, `results`, `fx_insert_profiling`, `_wrapped`

**Key imports**: annotations, dataclasses, os, Any, Self, torch, print_once


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `dataclasses`
- `os`
- `typing`: Any
- `typing_extensions`: Self
- `torch`
- `.utils`: print_once


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `profiler.py_docs.md`
- **Keyword Index**: `profiler.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

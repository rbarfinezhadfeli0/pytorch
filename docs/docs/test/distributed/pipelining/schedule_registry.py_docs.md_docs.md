# Documentation: `docs/test/distributed/pipelining/schedule_registry.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/pipelining/schedule_registry.py_docs.md`
- **Size**: 8,891 bytes (8.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/distributed/pipelining/schedule_registry.py`

## File Metadata

- **Path**: `test/distributed/pipelining/schedule_registry.py`
- **Size**: 6,239 bytes (6.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a Schedule zoo for testing torch.distributed.pipelining.
# It includes schedules designed purely for testing purposes
from collections.abc import Callable
from typing import Optional

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineScheduleRuntime,
    PipelineScheduleMulti,
    RECV_B,
    RECV_F,
    SEND_B,
    SEND_F,
)
from torch.distributed.pipelining.stage import _PipelineStageBase


F = _ComputationType.FORWARD
B = _ComputationType.FULL_BACKWARD
W = _ComputationType.BACKWARD_WEIGHT
I = _ComputationType.BACKWARD_INPUT


class ScheduleVShaped(PipelineScheduleMulti):
    n_stages = 4
    rank_stages = {
        0: [0, 3],
        1: [1, 2],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        # Go through one microbatch
        # Note(whc) - it might be easier to work with this schedules by writing them as a list of
        # ["0F0", ...] and then parsing them in the test infra to turn them into actions.
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                None,
                None,
                _Action(3, F, 0),
                _Action(3, B, 0),
                None,
                None,
                _Action(0, B, 0),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(2, F, 0),
                None,
                None,
                _Action(2, B, 0),
                _Action(1, B, 0),
                None,
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleUnbalanced(PipelineScheduleMulti):
    n_stages = 5
    rank_stages = {
        0: [0, 1, 4],
        1: [2, 3],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(1, F, 0),
                None,
                None,
                _Action(4, F, 0),
                _Action(4, B, 0),
                None,
                None,
                _Action(1, B, 0),
                _Action(0, B, 0),
            ],
            1: [
                None,
                None,
                _Action(2, F, 0),
                _Action(3, F, 0),
                None,
                None,
                _Action(3, B, 0),
                _Action(2, B, 0),
                None,
                None,
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleWithW(PipelineScheduleMulti):
    n_stages = 4
    num_microbatches = 2
    rank_stages = {
        0: [0, 2],
        1: [1, 3],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        enable_zero_bubble: bool = True,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        # Needs to be updated as part of all schedules using "W"
        self.use_full_backward = False

        # Go through two microbatches
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(2, F, 0),
                _Action(2, F, 1),
                None,
                _Action(2, I, 0),
                _Action(2, W, 0),
                _Action(0, I, 0),
                _Action(2, I, 1),
                _Action(0, W, 0),
                _Action(0, I, 1),
                _Action(2, W, 1),
                _Action(0, W, 1),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(3, F, 0),
                _Action(3, I, 0),
                _Action(3, F, 1),
                _Action(1, I, 0),
                _Action(3, I, 1),
                _Action(3, W, 0),
                _Action(1, I, 1),
                _Action(1, W, 0),
                _Action(3, W, 1),
                _Action(1, W, 1),
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleWithReorderedB(_PipelineScheduleRuntime):
    n_stages = 2
    num_microbatches = 2
    rank_stages = {
        0: [0],
        1: [1],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )
        # Go through two microbatches
        self.pipeline_order_with_comms = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(0, SEND_F, 0),
                _Action(0, SEND_F, 1),
                _Action(0, RECV_B, 0),
                _Action(0, RECV_B, 1),
                _Action(0, B, 0),
                _Action(0, B, 1),
            ],
            1: [
                _Action(1, RECV_F, 0),
                _Action(1, RECV_F, 1),
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(1, B, 0),
                _Action(1, B, 1),
                _Action(1, SEND_B, 0),
                _Action(1, SEND_B, 1),
            ],
        }

```



## High-Level Overview


This Python file contains 4 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ScheduleVShaped`, `ScheduleUnbalanced`, `ScheduleWithW`, `ScheduleWithReorderedB`

**Functions defined**: `__init__`, `__init__`, `__init__`, `__init__`

**Key imports**: Callable, Optional, _PipelineStageBase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Optional
- `torch.distributed.pipelining.stage`: _PipelineStageBase


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
python test/distributed/pipelining/schedule_registry.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/pipelining`):

- [`test_schedule_multiproc.py_docs.md`](./test_schedule_multiproc.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_microbatch.py_docs.md`](./test_microbatch.py_docs.md)
- [`test_schedule.py_docs.md`](./test_schedule.py_docs.md)
- [`test_pipe.py_docs.md`](./test_pipe.py_docs.md)
- [`model_registry.py_docs.md`](./model_registry.py_docs.md)
- [`test_transformer.py_docs.md`](./test_transformer.py_docs.md)
- [`test_stage.py_docs.md`](./test_stage.py_docs.md)
- [`test_unflatten.py_docs.md`](./test_unflatten.py_docs.md)


## Cross-References

- **File Documentation**: `schedule_registry.py_docs.md`
- **Keyword Index**: `schedule_registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/pipelining`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/pipelining`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/pipelining/schedule_registry.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/pipelining`):

- [`test_transformer.py_kw.md_docs.md`](./test_transformer.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_docs.md_docs.md`](./test_schedule_multiproc.py_docs.md_docs.md)
- [`model_registry.py_kw.md_docs.md`](./model_registry.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`test_stage.py_docs.md_docs.md`](./test_stage.py_docs.md_docs.md)
- [`schedule_registry.py_kw.md_docs.md`](./schedule_registry.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_kw.md_docs.md`](./test_schedule_multiproc.py_kw.md_docs.md)
- [`test_unflatten.py_kw.md_docs.md`](./test_unflatten.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `schedule_registry.py_docs.md_docs.md`
- **Keyword Index**: `schedule_registry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

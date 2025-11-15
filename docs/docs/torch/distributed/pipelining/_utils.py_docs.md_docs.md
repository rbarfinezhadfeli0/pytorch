# Documentation: `docs/torch/distributed/pipelining/_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/pipelining/_utils.py_docs.md`
- **Size**: 7,326 bytes (7.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/pipelining/_utils.py`

## File Metadata

- **Path**: `torch/distributed/pipelining/_utils.py`
- **Size**: 4,704 bytes (4.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import logging
from dataclasses import dataclass
from typing import Union

import torch
from torch import fx


logger = logging.getLogger(__name__)


def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.
    """
    flat_detached_args = []

    def extract_tensor_args(a):
        nonlocal flat_detached_args
        if isinstance(a, torch.Tensor):
            val = a.detach().requires_grad_(a.requires_grad)
            flat_detached_args.append(val)
            return val
        else:
            flat_detached_args.append(a)
            return a

    new_args = fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return new_args, flat_detached_args


def flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return flat_args


class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""


def validate_tensor_metadata(desc, expected, given):
    if not expected.shape == given.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given.shape}"
        )
    if not expected.dtype == given.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given.dtype}"
        )
    if not expected.stride() == given.stride():
        raise PipeliningShapeError(
            f"{desc} has a stride mismatch: expected {expected.stride()} actual {given.stride()}"
        )


def validate_tensors_metadata(
    desc,
    expected_tensors: Union[list[torch.Tensor], tuple[torch.Tensor, ...]],
    actual_tensors: Union[list[torch.Tensor], tuple[torch.Tensor, ...]],
):
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # dont change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, list[int]]:
    """
    Compute the rank to stage id mapping for either a looped or V-style schedule.

    This function inverts the stage_to_rank_mapping to get which stages are assigned to each rank.

    Returns a dictionary mapping rank -> list of stage indices assigned to that rank.
    """
    stage_to_rank = generate_stage_to_rank_mapping(pp_size, num_stages, style)

    # Invert the mapping: rank -> list of stages
    rank_to_stages: dict[int, list[int]] = {}
    for stage_id, rank in stage_to_rank.items():
        if rank not in rank_to_stages:
            rank_to_stages[rank] = []
        rank_to_stages[rank].append(stage_id)

    # Sort the stage lists for each rank to ensure consistent ordering
    for stages in rank_to_stages.values():
        stages.sort()

    return rank_to_stages


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool

```



## High-Level Overview

"""    Flatten the args into a list form and detach the tensors from computational graph.

This Python file contains 3 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PipeliningShapeError`, `PipeInfo`

**Functions defined**: `flatten_args_detach`, `extract_tensor_args`, `flatten_args`, `extract_tensor_args`, `validate_tensor_metadata`, `validate_tensors_metadata`, `generate_stage_to_rank_mapping`, `generate_rank_to_stage_mapping`

**Key imports**: logging, dataclass, Union, torch, fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/pipelining`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `dataclasses`: dataclass
- `typing`: Union
- `torch`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/pipelining`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_backward.py_docs.md`](./_backward.py_docs.md)
- [`_unflatten.py_docs.md`](./_unflatten.py_docs.md)
- [`schedules.py_docs.md`](./schedules.py_docs.md)
- [`microbatch.py_docs.md`](./microbatch.py_docs.md)
- [`stage.py_docs.md`](./stage.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_IR.py_docs.md`](./_IR.py_docs.md)
- [`_debug.py_docs.md`](./_debug.py_docs.md)


## Cross-References

- **File Documentation**: `_utils.py_docs.md`
- **Keyword Index**: `_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/pipelining`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/pipelining`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/pipelining`):

- [`schedules.py_docs.md_docs.md`](./schedules.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`_IR.py_kw.md_docs.md`](./_IR.py_kw.md_docs.md)
- [`_backward.py_docs.md_docs.md`](./_backward.py_docs.md_docs.md)
- [`stage.py_docs.md_docs.md`](./stage.py_docs.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_schedule_visualizer.py_kw.md_docs.md`](./_schedule_visualizer.py_kw.md_docs.md)
- [`microbatch.py_kw.md_docs.md`](./microbatch.py_kw.md_docs.md)
- [`_unflatten.py_docs.md_docs.md`](./_unflatten.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_utils.py_docs.md_docs.md`
- **Keyword Index**: `_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

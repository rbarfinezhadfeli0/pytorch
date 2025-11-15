# Documentation: `docs/torch/_inductor/comms_debug.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/comms_debug.py_docs.md`
- **Size**: 6,944 bytes (6.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/comms_debug.py`

## File Metadata

- **Path**: `torch/_inductor/comms_debug.py`
- **Size**: 4,227 bytes (4.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from torch._logging import trace_structured

from .memory import estimate_peak_memory_allocfree


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet

    from .memory import FreeableInputBuffer, SNodeMemory
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


def _debug_iterative_memory_recompute(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_names: str,
    snodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
    peak_memory: int,
    iter_curr_memory: dict[BaseSchedulerNode, tuple[int, int]],
    snodes_allocfree: dict[BaseSchedulerNode, SNodeMemory],
    tlparse_name: str,
    gn_to_bufs_last_use: dict[
        BaseSchedulerNode, list[Union[FreeableInputBuffer, SchedulerBuffer]]
    ],
) -> bool:
    iterative_recompute_error = False
    candidate_allocfree = snodes_allocfree[candidate]
    est_peak_memory, snodes_curr_memory, snodes_allocfree, _ = (
        estimate_peak_memory_allocfree(
            snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    est_curr_memory = dict(zip(snodes, snodes_curr_memory))
    iter_cm = iter_curr_memory[candidate]
    new_cm = est_curr_memory[candidate]
    log = ""
    if est_peak_memory > peak_memory:
        log = "ITERATIVE PEAK DOES NOT MATCH"
        iterative_recompute_error = True
    if iter_cm != new_cm:
        log = "ITERATIVE CURR MEMORY CANDIDATE DOES NOT MATCH"
        iterative_recompute_error = True
    for gn in gns:
        iter_gnm = iter_curr_memory[gn]
        new_gnm = est_curr_memory[gn]
        if iter_gnm != new_gnm:
            log = f"ITERATIVE GN CURR MEMORY DOES NOT MATCH:{gn.get_name()}"
            iterative_recompute_error = True
    if iterative_recompute_error:
        log += (
            f"\nCANDIDATE:{candidate.get_name()}"
            f"\nGROUP:{group_names}"
            f"\nPEAK_MEMORY_BEFORE:{peak_memory}"
            f"\nPEAK_MEMORY_AFTER_SWAP:{est_peak_memory}"
            f"\nCANDIDATE:{candidate.debug_str()}"
            f"\nCANDIDATE_ITER_CURR_MEMORY:{iter_cm}"
            f"\nCANDIDATE_NEW__CURR_MEMORY:{new_cm}"
            f"\nCANDIDATE_ITER_ALLOCFREE:{candidate_allocfree}"
            f"\nCANDIDATE_NEW_ALLOCFREE:{snodes_allocfree[candidate]}"
        )
        peak_log = ""
        for i, (pre, _post) in enumerate(snodes_curr_memory):
            if est_peak_memory == pre:
                n = snodes[i]
                peak_log = (
                    f"\nNEW_PEAK:{est_peak_memory}(BASE:{peak_memory})"
                    f" @ SNODE[{i}/{len(snodes)}]:{n.get_name()} {n.debug_str()}"
                )
                break
        group_log = ""
        for i, gn in enumerate(gns):
            iter_gnm = iter_curr_memory[gn]
            new_gnm = est_curr_memory[gn]
            group_log += (
                f"\nGROUP_NODE[{i}]:{gn.debug_str()}"
                f"\nGROUP_NODE[{i}] ITER_GNM[{gn.get_name()}]:{iter_gnm}"
                f"\nGROUP_NODE[{i}] ESTM_GNM[{gn.get_name()}]:{new_gnm}"
                f"\nGROUP_NODE[{i}] ITER_allocfree:{snodes_allocfree[gn]}"
                f"\nGROUP_NODE[{i}] ESTM_allocfree:{snodes_allocfree[gn]}"
            )
        log += peak_log
        log += group_log
        log += f"\nGN_TO_BUFS_LAST_USE:{gn_to_bufs_last_use}"
        log += "\n\n".join(
            [
                (
                    f"\nSNODE[{i}]\n{n.debug_str()}"
                    f"\nITER_cur_mem:{iter_curr_memory[n]}"
                    f"\nESTM_cur_mem:{est_curr_memory[n]}"
                    f"\nITER_allocfree:{snodes_allocfree[n]}"
                    f"\nESTM_allocfree:{snodes_allocfree[n]}"
                )
                for i, n in enumerate(snodes)
            ]
        )
        tname = f"{tlparse_name}_ITERATIVE_RECOMPUTE_ERROR"
        print(f"{tname}:\n{log}")
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": tname,
                "encoding": "string",
            },
            payload_fn=lambda: log,
        )
    return iterative_recompute_error

```



## High-Level Overview


This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_debug_iterative_memory_recompute`

**Key imports**: annotations, TYPE_CHECKING, Union, trace_structured, estimate_peak_memory_allocfree, OrderedSet, FreeableInputBuffer, SNodeMemory, BaseSchedulerNode, SchedulerBuffer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING, Union
- `torch._logging`: trace_structured
- `.memory`: estimate_peak_memory_allocfree
- `torch.utils._ordered_set`: OrderedSet
- `.scheduler`: BaseSchedulerNode, SchedulerBuffer


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `comms_debug.py_docs.md`
- **Keyword Index**: `comms_debug.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `comms_debug.py_docs.md_docs.md`
- **Keyword Index**: `comms_debug.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

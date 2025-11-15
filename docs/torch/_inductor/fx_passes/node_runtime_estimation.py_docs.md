# Documentation: `torch/_inductor/fx_passes/node_runtime_estimation.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/node_runtime_estimation.py`
- **Size**: 5,371 bytes (5.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Collective runtime estimation using CUDA events and power-of-2 rounding.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import torch
from torch._inductor.utils import clear_on_fresh_cache
from torch._logging import getArtifactLogger
from torch.fx.operator_schemas import normalize_function


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")


# TODO: Consider using a distributed-aware cache or rank-local disk cache
# not using local cache because different ranks might write to it concurrently.
# solvable in future, potentially with workflow to seed cache
@clear_on_fresh_cache
@lru_cache
def _get_collective_cache() -> dict[str, float]:
    """Get process-local cache for collective benchmarks."""
    return {}


def get_cached_runtime(key: str) -> Optional[float]:
    """Get cached runtime from process-local cache."""
    return _get_collective_cache().get(key)


def set_cached_runtime(key: str, value: float) -> None:
    """Set cached runtime in process-local cache."""
    _get_collective_cache()[key] = value


def get_hint(x: int | torch.SymInt) -> Optional[int]:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    return x.node.hint if x.node.has_hint() else None


def can_benchmark_collective() -> bool:
    """Check if we can benchmark collectives (not fake process group)."""
    import torch.distributed as c10d

    if not c10d.is_initialized():
        return False

    pg = c10d.distributed_c10d._get_default_group()
    if torch.distributed.distributed_c10d.get_backend(pg) == "fake":
        return False

    return True


def _benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nruns: int,
) -> float | None:
    """
    Core benchmarking logic using CUDA events and barriers.
    Returns runtime in ms or None on failure.
    """
    import torch.distributed as c10d

    # Warmup: call collective once and wait
    torch.cuda.synchronize()
    result = n.target(*args, **kwargs)  # type: ignore[operator]
    torch.ops._c10d_functional.wait_tensor(result)

    # Benchmark with CUDA events
    comm_time = 0.0
    for _ in range(nruns):
        c10d.barrier()
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        result = n.target(*args, **kwargs)  # type: ignore[operator]
        torch.ops._c10d_functional.wait_tensor(result)
        end_evt.record()
        end_evt.synchronize()

        comm_time += start_evt.elapsed_time(end_evt)

    return comm_time / nruns


def benchmark_collective_with_cuda_events(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    # context manager not allowed with profiler.
    with torch.utils._python_dispatch._disable_current_modes():
        return benchmark_collective_with_cuda_events_impl(n, nruns)


def benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    from torch._inductor import fx_utils
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    # Early check: can we actually run collectives?
    if not can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(n)

    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    assert opt_args_kwargs is not None
    group_name = opt_args_kwargs[1]["group_name"]
    group_size = _get_group_size_by_name(group_name)

    if not success:
        return None, ""

    # Extract actual input size in BYTES (first tensor argument)
    actual_bytes: Optional[int] = None

    def extract_tensor_info(t: torch.Tensor) -> torch.Tensor:
        nonlocal actual_bytes
        if actual_bytes is None:
            shape = [get_hint(dim) for dim in t.shape]
            if any(s is None for s in shape):
                return t

            total_elems = 1
            for dim in shape:
                assert dim is not None
                total_elems *= dim

            actual_bytes = total_elems * t.dtype.itemsize
        else:
            raise RuntimeError(f"should only be one input tensor to collective {n}")
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, extract_tensor_info, (args, kwargs))

    if actual_bytes is None:
        return None, ""

    # Cache key by BYTES (dtype-agnostic)
    key = f"{n.target}: ({group_size} group size, {actual_bytes} bytes)"

    # Check cache
    if (cached := get_cached_runtime(key)) is not None:
        return cached, key

    # Benchmark using CUDA events with actual args/kwargs
    runtime = _benchmark_collective_with_cuda_events_impl(n, args, kwargs, nruns)

    if runtime is None:
        return None, key

    # Cache the result
    set_cached_runtime(key, runtime)
    return runtime, key

```



## High-Level Overview

"""Collective runtime estimation using CUDA events and power-of-2 rounding.

This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_collective_cache`, `get_cached_runtime`, `set_cached_runtime`, `get_hint`, `can_benchmark_collective`, `_benchmark_collective_with_cuda_events_impl`, `benchmark_collective_with_cuda_events`, `benchmark_collective_with_cuda_events_impl`, `extract_tensor_info`

**Key imports**: annotations, lru_cache, Any, Optional, torch, clear_on_fresh_cache, getArtifactLogger, normalize_function, torch.distributed as c10d, torch.distributed as c10d, fx_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`: lru_cache
- `typing`: Any, Optional
- `torch`
- `torch._inductor.utils`: clear_on_fresh_cache
- `torch._logging`: getArtifactLogger
- `torch.fx.operator_schemas`: normalize_function
- `torch.distributed as c10d`
- `torch._inductor`: fx_utils
- `torch.distributed.distributed_c10d`: _get_group_size_by_name


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`numeric_utils.py_docs.md`](./numeric_utils.py_docs.md)
- [`dedupe_symint_uses.py_docs.md`](./dedupe_symint_uses.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `node_runtime_estimation.py_docs.md`
- **Keyword Index**: `node_runtime_estimation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/_inductor/runtime/debug_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/runtime/debug_utils.py_docs.md`
- **Size**: 7,313 bytes (7.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/runtime/debug_utils.py`

## File Metadata

- **Path**: `torch/_inductor/runtime/debug_utils.py`
- **Size**: 4,275 bytes (4.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import functools
import logging
import threading
import weakref

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

local = threading.local()
local.memory_tracker = None


class BufferMemoryTracker:
    """
    Tracks inductor runtime allocations and deallocations to compare against
    expected behavior.
    """

    def __init__(self) -> None:
        self.tensor_tracker: dict[str, torch.storage.UntypedStorage] = (
            weakref.WeakValueDictionary()  # type: ignore[assignment]
        )
        self.died_since_last_step: OrderedSet[str] = OrderedSet()
        self.added_since_last_step: OrderedSet[str] = OrderedSet()
        self.error = (
            torch._inductor.config.test_configs.track_memory_lifecycle == "assert"
        )

    def set_tensor(self, name: str, tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage()

        self.added_since_last_step.add(name)
        self.tensor_tracker[name] = storage

        def on_tensor_death() -> None:
            self.died_since_last_step.add(name)

        weakref.finalize(storage, on_tensor_death)

    def advance_step(self) -> None:
        self.died_since_last_step.clear()
        self.added_since_last_step.clear()

    def log_or_raise(self, msg: str) -> None:
        if self.error:
            raise RuntimeError(msg)
        else:
            log.info(msg)

    def check_step_delta(
        self,
        expected_allocated: list[str],
        expected_freed: list[str],
        is_final_step: bool,
    ) -> None:
        """Check only the delta changes since last step"""

        # Check expected deaths - we dont currently distinguish between nodes which die in last step
        # and are returned as outputs, so skip if final_step.
        if not is_final_step:
            missing_deaths = OrderedSet(expected_freed) - self.died_since_last_step
            if missing_deaths:
                self.log_or_raise(
                    f"Expected tensors to die but still alive: {missing_deaths}"
                )

        # Check for unexpected deaths
        unexpected_deaths = self.died_since_last_step - OrderedSet(expected_freed)
        if unexpected_deaths:
            self.log_or_raise(f"Unexpected tensor deaths: {unexpected_deaths}")

        # Check newly alive tensors - separate messages like deaths
        actual_allocated = self.added_since_last_step
        expected_allocated_set = OrderedSet(expected_allocated)

        extra_alive = actual_allocated - expected_allocated_set
        if extra_alive:
            self.log_or_raise(f"Unexpected allocated tensors: {extra_alive}")

        missing_alive = expected_allocated_set - actual_allocated
        if missing_alive:
            self.log_or_raise(
                f"Expected allocated tensors but missing: {missing_alive}"
            )

        # Reset for next step
        self.advance_step()

        if is_final_step:
            local.memory_tracker = None


def get_mem_tracker() -> BufferMemoryTracker:
    if local.memory_tracker is None:
        local.memory_tracker = BufferMemoryTracker()
    return local.memory_tracker


def track_tensor(tensor: torch.Tensor, name: str) -> None:
    get_mem_tracker().set_tensor(name, tensor)


def tracked_empty_strided(
    size: list[int],
    stride: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    o = torch.empty_strided(size, stride, dtype=dtype, device=device)
    track_tensor(o, name)
    return o


def check_memory_step(
    allocated: list[str], freed: list[str], is_final_step: bool = False
) -> None:
    tracker = get_mem_tracker()
    tracker.check_step_delta(allocated, freed, is_final_step)


@functools.lru_cache(None)
def register_check_mem_op() -> None:
    lib = torch.library.Library("_inductor_debug", "FRAGMENT")  # noqa: TOR901
    lib.define(
        "check_memory_step(str[] allocated, str[] freed, bool is_final_step) -> ()"
    )
    lib.impl("check_memory_step", check_memory_step, "BackendSelect")
    from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

    _register_effectful_op(
        torch.ops._inductor_debug.check_memory_step.default,
        _EffectType.ORDERED,
    )

```



## High-Level Overview

"""    Tracks inductor runtime allocations and deallocations to compare against    expected behavior.

This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BufferMemoryTracker`

**Functions defined**: `__init__`, `set_tensor`, `on_tensor_death`, `advance_step`, `log_or_raise`, `check_step_delta`, `get_mem_tracker`, `track_tensor`, `tracked_empty_strided`, `check_memory_step`, `register_check_mem_op`

**Key imports**: functools, logging, threading, weakref, torch, OrderedSet, _EffectType, _register_effectful_op


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `logging`
- `threading`
- `weakref`
- `torch`
- `torch.utils._ordered_set`: OrderedSet
- `torch._higher_order_ops.effects`: _EffectType, _register_effectful_op


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_inductor/runtime`):

- [`static_cuda_launcher.py_docs.md`](./static_cuda_launcher.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`hints.py_docs.md`](./hints.py_docs.md)
- [`coordinate_descent_tuner.py_docs.md`](./coordinate_descent_tuner.py_docs.md)
- [`autotune_cache.py_docs.md`](./autotune_cache.py_docs.md)
- [`triton_heuristics.py_docs.md`](./triton_heuristics.py_docs.md)
- [`compile_tasks.py_docs.md`](./compile_tasks.py_docs.md)
- [`triton_compat.py_docs.md`](./triton_compat.py_docs.md)
- [`cache_dir_utils.py_docs.md`](./cache_dir_utils.py_docs.md)


## Cross-References

- **File Documentation**: `debug_utils.py_docs.md`
- **Keyword Index**: `debug_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_inductor/runtime`):

- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`hints.py_kw.md_docs.md`](./hints.py_kw.md_docs.md)
- [`cache_dir_utils.py_kw.md_docs.md`](./cache_dir_utils.py_kw.md_docs.md)
- [`cache_dir_utils.py_docs.md_docs.md`](./cache_dir_utils.py_docs.md_docs.md)
- [`halide_helpers.py_docs.md_docs.md`](./halide_helpers.py_docs.md_docs.md)
- [`runtime_utils.py_kw.md_docs.md`](./runtime_utils.py_kw.md_docs.md)
- [`static_cuda_launcher.py_docs.md_docs.md`](./static_cuda_launcher.py_docs.md_docs.md)
- [`static_cuda_launcher.py_kw.md_docs.md`](./static_cuda_launcher.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `debug_utils.py_docs.md_docs.md`
- **Keyword Index**: `debug_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

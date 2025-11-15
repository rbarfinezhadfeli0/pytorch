# Documentation: `docs/torch/distributed/_tools/memory_tracker.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/_tools/memory_tracker.py_docs.md`
- **Size**: 15,947 bytes (15.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/distributed/_tools/memory_tracker.py`

## File Metadata

- **Path**: `torch/distributed/_tools/memory_tracker.py`
- **Size**: 11,862 bytes (11.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
import operator
import pickle
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, no_type_check, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


BYTES_PER_MB = 1024 * 1024.0


class MemoryProfileDispatchMode(TorchDispatchMode):
    """Run in ``TorchDispatchMode`` to get memory stats at operator level."""

    def __init__(self, memory_tracker) -> None:
        self.memory_tracker = memory_tracker

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        if func is torch.ops.aten.detach.default:
            return rs
        func_name: str = (
            self.memory_tracker._cur_module_name
            + "."
            + func.__name__
            + "_"
            + str(self.memory_tracker._operator_names[func.__name__])
        )
        self.memory_tracker._operator_names[func.__name__] = (
            self.memory_tracker._operator_names[func.__name__] + 1
        )
        self.memory_tracker._record_memory_stats(func_name)

        return rs


class MemoryTracker:
    """
    Collect and plot the memory stats at operator level.

    Includes ``memories_allocated``, ``memories_active`` and ``memories_reserved``.
    It also prints a summary for the top 20 operators that generate the most memories.

    Example usage:

        >>> # xdoctest: +SKIP(failing)
        >>> net.cuda()
        >>> input = input.cuda()

        >>> mem_tracker = MemoryTracker()
        >>> mem_tracker.start_monitor(net)

        >>> net.zero_grad(True)
        >>> loss = net(input)
        >>> if isinstance(loss, dict):
        >>>    loss = loss['out']
        >>> loss.sum().backward()
        >>> net.zero_grad(set_to_none=True)

        >>> mem_tracker.stop()
        >>> mem_tracker.summary()
        >>> mem_tracker.show_traces()
    """

    def __init__(self) -> None:
        torch._C._log_api_usage_once("torch.distributed.memory_tracker")
        self._hooks: list[RemovableHandle] = []
        self._operator_names: dict[str, int] = defaultdict(int)
        self.memories_allocated: dict[int, dict[str, float]] = defaultdict()
        self.memories_active: dict[int, dict[str, float]] = defaultdict()
        self.memories_reserved: dict[int, dict[str, float]] = defaultdict()
        self._markers: dict[str, int] = defaultdict(int)
        self._cur_module_name: str = ""
        self._op_index: int = 0
        self._num_alloc_retries: int = 0
        self._device_module = torch.get_device_module()

    @no_type_check
    def start_monitor(self, root_module: nn.Module) -> None:
        """
        Register module hooks and entering ``MemoryProfileDispatchMode``.

        This enables operator level memory stats can be tracked during module runtime.
        """
        self._clear_state()
        root_module.__setattr__("_memory_tracker_is_root", True)
        for name, m in root_module.named_modules():
            if m is not root_module:
                m.__setattr__("_memory_tracker_is_root", False)
            # fused_proxy_group does not support hooks
            if ".fused_proxy_grouped_embedding_bag" in name:
                continue
            # hook ordering with other hooks added by users is not managed, so
            # the memory stats tracked here may not completely accurate.
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            # it does not work well with jagged tensor somehow, the root cause is not
            # clear and remove it for now as it does not really capture important info.
            # h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2])
        self._device_module.empty_cache()
        assert getattr(self, "profile_mode", None) is None
        self.profile_mode = MemoryProfileDispatchMode(self)
        self.profile_mode.__enter__()

    @no_type_check
    def stop(self) -> None:
        """
        Remove module hooks and exit ``MemoryProfileDispatchMode`` to stop tracking memory stats at operator level.

        Get some aggregated stats when the memory_tracker() is enabled, like ``num_alloc_retries``.
        """
        self._num_alloc_retries = self._device_module.memory_stats().get(
            "num_alloc_retries", 0
        )

        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        assert getattr(self, "profile_mode", None) is not None
        self.profile_mode.__exit__(None, None, None)
        self.profile_mode = None

    @no_type_check
    def summary(self, top: int = 20) -> None:
        """
        Print out the top operators that generate the most memories.

        The number of the top operators can be configured.
        """
        op_diff: dict[str, float] = defaultdict(float)
        op_name, previous_allocated_memory = self.memories_allocated[0]
        for i in range(1, self._op_index):
            op_name, current_allocated_memory = self.memories_allocated[i]
            op_diff[op_name] = current_allocated_memory - previous_allocated_memory
            previous_allocated_memory = current_allocated_memory

        print("------------------------------------------------")
        print(f"The number of alloc retries are: {self._num_alloc_retries}")
        print(f"Top {top} ops that generates memory are:")
        for k, v in sorted(op_diff.items(), key=operator.itemgetter(1), reverse=True)[
            :top
        ]:
            print(f"{k}: {v}MB")
        print("------------------------------------------------")

    @no_type_check
    def show_traces(self, path: str = "") -> None:
        import matplotlib.pyplot as plt

        def _plot_figure(x, y_values, labels):
            min_val = min(chain.from_iterable(y_values)) * 0.999
            max_val = max(chain.from_iterable(y_values)) * 1.001
            plt.figure()
            for y, label in zip(y_values, labels):
                plt.plot(x, y, label=label)
            plt.xlabel("# Operator Calls")
            plt.ylabel("Memory (MB)")
            plt.legend()
            for marker_name, marker in self._markers.items():
                if marker_name == "fw_bw_boundary":
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "r",
                        lw=2,
                        label=marker_name,
                    )
                else:
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "k-",
                        lw=2,
                        label=marker_name,
                    )

        if path != "":
            self.load(path)

        y_1 = [gb for (name, gb) in self.memories_allocated.values()]
        y_2 = [gb for (name, gb) in self.memories_active.values()]
        y_3 = [gb for (name, gb) in self.memories_reserved.values()]
        x = list(range(len(y_1)))
        # Split figures when there is big difference between
        # "reserved_memory" and "allocated_memory" or "active_memory".
        _plot_figure(
            x,
            [list(y_1), list(y_2), list(y_3)],
            ["allocated_memory", "active_memory", "reserved_memory"],
        )
        _plot_figure(x, [list(y_1)], ["allocated_memory"])
        _plot_figure(x, [list(y_2)], ["active_memory"])
        _plot_figure(x, [list(y_3)], ["reserved_memory"])

    def save_stats(self, path: str) -> None:
        """Save the stats using pickle during runtime if users want to plot the traces in other places like notebook."""
        stats = {
            "memories_allocated": self.memories_allocated,
            "memories_active": self.memories_active,
            "memories_reserved": self.memories_reserved,
            "markers": self._markers,
            "num_alloc_retries": self._num_alloc_retries,
        }

        with open(path, "wb") as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load the pickled memory stats to plot the traces or print the summary."""
        with open(path, "rb") as f:
            stats = pickle.load(f)

        self.memories_allocated = stats["memories_allocated"]
        self.memories_active = stats["memories_active"]
        self.memories_reserved = stats["memories_reserved"]
        self._markers = stats["markers"]
        self._num_alloc_retries = stats["num_alloc_retries"]

    def _create_pre_forward_hook(self, name: str) -> Callable:
        """Prefix operator name with current module and 'forward', and insert 'fw_start' marker at forward pass start."""

        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            self._cur_module_name = f"{name}.forward"
            if (
                # pyrefly: ignore [invalid-argument]
                hasattr(module, "_memory_tracker_is_root")
                # pyrefly: ignore [not-callable]
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_start")

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str) -> Callable:
        """Insert the marker 'fw_bw_boundary' at the boundary of forward and backward pass."""

        def _post_forward_hook(
            module: nn.Module,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
        ) -> None:
            if (
                # pyrefly: ignore [invalid-argument]
                hasattr(module, "_memory_tracker_is_root")
                # pyrefly: ignore [not-callable]
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_bw_boundary")

        return _post_forward_hook

    def _create_backward_hook(self, name: str) -> Callable:
        """Insert the current module name with backward prefix for the operator name."""

        def _backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ) -> None:
            self._cur_module_name = f"{name}.backward"

        return _backward_hook

    @no_type_check
    def _record_memory_stats(self, fn_name: str) -> None:
        """
        Record current memory allocated, current memory active and current memory reserved.

        The memory stats dict is indexed with ``self._op_index``.
        """
        memory_allocated: float = self._device_module.memory_allocated() / BYTES_PER_MB
        memory_reserved: float = self._device_module.memory_reserved() / BYTES_PER_MB
        memory_active: float = (
            self._device_module.memory_stats().get("active_bytes.all.current", 0)
            / BYTES_PER_MB
        )
        self.memories_allocated[self._op_index] = (fn_name, memory_allocated)
        self.memories_reserved[self._op_index] = (fn_name, memory_reserved)
        self.memories_active[self._op_index] = (fn_name, memory_active)
        self._op_index += 1

    def _add_marker(self, marker_name: str) -> None:
        """Set the marker's x-axis value."""
        marker_val = len(self.memories_allocated.values())
        self._markers[marker_name] = marker_val

    def _clear_state(self) -> None:
        """Clear states when start_monitor() is called."""
        self._operator_names.clear()
        self.memories_allocated.clear()
        self.memories_active.clear()
        self.memories_reserved.clear()
        self._markers.clear()
        self._cur_module_name = ""
        self._op_index = 0
        self._num_alloc_retries = 0

```



## High-Level Overview

"""Run in ``TorchDispatchMode`` to get memory stats at operator level."""    def __init__(self, memory_tracker) -> None:        self.memory_tracker = memory_tracker    def __torch_dispatch__(self, func, types, args=..., kwargs=None):        rs = func(*args, **kwargs)        if func is torch.ops.aten.detach.default:            return rs        func_name: str = (            self.memory_tracker._cur_module_name            + "."            + func.__name__            + "_"            + str(self.memory_tracker._operator_names[func.__name__])        )        self.memory_tracker._operator_names[func.__name__] = (            self.memory_tracker._operator_names[func.__name__] + 1        )        self.memory_tracker._record_memory_stats(func_name)        return rsclass MemoryTracker:

This Python file contains 2 class(es) and 19 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MemoryProfileDispatchMode`, `MemoryTracker`

**Functions defined**: `__init__`, `__torch_dispatch__`, `__init__`, `start_monitor`, `stop`, `summary`, `show_traces`, `_plot_figure`, `save_stats`, `load`, `_create_pre_forward_hook`, `_pre_forward_hook`, `_create_post_forward_hook`, `_post_forward_hook`, `_create_backward_hook`, `_backward_hook`, `_record_memory_stats`, `_add_marker`, `_clear_state`

**Key imports**: operator, pickle, defaultdict, Callable, Sequence, chain, Any, no_type_check, TYPE_CHECKING, torch, torch.nn as nn, TorchDispatchMode, RemovableHandle


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/_tools`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `pickle`
- `collections`: defaultdict
- `collections.abc`: Callable, Sequence
- `itertools`: chain
- `typing`: Any, no_type_check, TYPE_CHECKING
- `torch`
- `torch.nn as nn`
- `torch.utils._python_dispatch`: TorchDispatchMode
- `torch.utils.hooks`: RemovableHandle
- `matplotlib.pyplot as plt`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/_tools`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ilp_utils.py_docs.md`](./ilp_utils.py_docs.md)
- [`mod_tracker.py_docs.md`](./mod_tracker.py_docs.md)
- [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- [`sac_estimator.py_docs.md`](./sac_estimator.py_docs.md)
- [`sac_ilp.py_docs.md`](./sac_ilp.py_docs.md)
- [`mem_tracker.py_docs.md`](./mem_tracker.py_docs.md)
- [`runtime_estimator.py_docs.md`](./runtime_estimator.py_docs.md)
- [`fsdp2_mem_tracker.py_docs.md`](./fsdp2_mem_tracker.py_docs.md)


## Cross-References

- **File Documentation**: `memory_tracker.py_docs.md`
- **Keyword Index**: `memory_tracker.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/_tools`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_tools`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/_tools`):

- [`fsdp2_mem_tracker.py_docs.md_docs.md`](./fsdp2_mem_tracker.py_docs.md_docs.md)
- [`common_utils.py_kw.md_docs.md`](./common_utils.py_kw.md_docs.md)
- [`runtime_estimator.py_kw.md_docs.md`](./runtime_estimator.py_kw.md_docs.md)
- [`mod_tracker.py_kw.md_docs.md`](./mod_tracker.py_kw.md_docs.md)
- [`sac_estimator.py_docs.md_docs.md`](./sac_estimator.py_docs.md_docs.md)
- [`ilp_utils.py_kw.md_docs.md`](./ilp_utils.py_kw.md_docs.md)
- [`sac_estimator.py_kw.md_docs.md`](./sac_estimator.py_kw.md_docs.md)
- [`fake_collectives.py_docs.md_docs.md`](./fake_collectives.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`runtime_estimator.py_docs.md_docs.md`](./runtime_estimator.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `memory_tracker.py_docs.md_docs.md`
- **Keyword Index**: `memory_tracker.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/autograd/profiler_legacy.py_docs.md`

## File Metadata

- **Path**: `docs/torch/autograd/profiler_legacy.py_docs.md`
- **Size**: 15,458 bytes (15.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/autograd/profiler_legacy.py`

## File Metadata

- **Path**: `torch/autograd/profiler_legacy.py`
- **Size**: 12,210 bytes (11.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import itertools
import warnings
from typing_extensions import deprecated

import torch
import torch.cuda
from torch.autograd import (
    _disable_profiler_legacy,
    _enable_profiler_legacy,
    DeviceType,
    ProfilerConfig,
    ProfilerState,
)
from torch.autograd.profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
)


__all__ = ["profile"]


@deprecated(
    "`torch.autograd.profiler_legacy.profile` is deprecated and will be removed in a future release. "
    "Please use `torch.profiler` instead.",
    category=None,  # TODO: change to `FutureWarning`
)
class profile:
    """DEPRECATED: use torch.profiler instead."""

    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        self.function_events = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules

        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available, disabling CUDA profiling",
                stacklevel=2,
            )
            self.use_cuda = False

        if self.use_cuda:
            self.profiler_kind = ProfilerState.CUDA
        else:
            self.profiler_kind = ProfilerState.CPU

    def config(self):
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            # avoid exposing _ExperimentalConfig this in legacy public API
            torch._C._profiler._ExperimentalConfig(),
        )

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self.entered = True
        self._start_trace()
        return self

    def _start_trace(self):
        _enable_profiler_legacy(self.config())

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize()

        records = _disable_profiler_legacy()
        parsed_results = _parse_legacy_records(records)
        # pyrefly: ignore [bad-assignment]
        self.function_events = EventList(
            parsed_results,
            use_device="cuda" if self.use_cuda else None,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        # pyrefly: ignore [missing-attribute]
        self.function_events._build_tree()
        return False

    def __repr__(self):
        if self.function_events is None:
            return "<unfinished profiler_legacy.profile>"
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return "<unfinished profile.profiler_legacy.profile>"
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.export_chrome_trace(path)

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        if not self.with_stack:
            raise AssertionError("export_stacks() requires with_stack=True")
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Return CPU time as the sum of self times across all events."""
        self._check_finish()
        if self.function_events is None:
            raise AssertionError("Expected profiling results")
        return self.function_events.self_cpu_time_total


def _parse_legacy_records(thread_records):
    def _get_record_key(record):
        """Return a tuple for correlating start and end records in `_parse_legacy_records`."""
        return (record.handle(), record.node_id())

    start_record = None
    functions = []

    # '__start_profile' is not guaranteed to be first, so we must find it here
    for record in itertools.chain.from_iterable(thread_records):
        name = record.name()
        if start_record is None and name == "__start_profile":
            start_record = record

    if start_record is None or start_record.is_remote():
        raise AssertionError("Expected a valid local start_record")

    for thread_record_list in thread_records:
        # accumulated memory allocations per handle
        cpu_memory_allocs = {}
        cuda_memory_allocs = {}
        # ranges per handle
        range_starts = {}

        filtered_handles = set()
        prev_record = None
        for record in thread_record_list:
            record_key = _get_record_key(record)
            if _filter_name(record.name()) or record_key in filtered_handles:
                filtered_handles.add(record_key)
                continue

            if record.kind() == "push":
                # workaround to reduce double logging from operator
                # wrappers and redispatch
                if prev_record is not None:
                    duplicate = (
                        prev_record.name() == record.name()
                        and prev_record.kind() == record.kind()
                        and prev_record.node_id() == record.node_id()
                    )
                    if duplicate:
                        filtered_handles.add(record_key)
                        continue

                range_starts[record_key] = record
                cpu_memory_allocs[record_key] = 0
                cuda_memory_allocs[record_key] = 0
            elif record.kind() == "pop":
                if record_key not in range_starts:
                    raise AssertionError(
                        f"Expected record with key {record_key} to exist in range_starts. "
                        "This means that the pop event did not have a corresponding push."
                    )

                start = range_starts[record_key]

                cpu_memory_usage = cpu_memory_allocs[record_key]
                cuda_memory_usage = cuda_memory_allocs[record_key]
                is_async = start.is_async() or (start.thread_id() != record.thread_id())
                is_remote_event = record.is_remote()
                start_flops = start.flops()

                fe = FunctionEvent(
                    id=record.handle(),
                    node_id=record.node_id(),
                    name=_rewrite_name(name=start.name(), with_wildcard=True),
                    trace_name=_rewrite_name(name=start.name(), with_wildcard=False),
                    thread=start.thread_id(),
                    start_us=start_record.cpu_elapsed_us(start),
                    end_us=start_record.cpu_elapsed_us(record),
                    fwd_thread=start.fwd_thread_id(),
                    input_shapes=start.shapes(),
                    stack=[
                        entry for entry in start.stack() if _filter_stack_entry(entry)
                    ],
                    scope=start.scope(),
                    use_device="cuda" if start.has_cuda() else None,
                    cpu_memory_usage=cpu_memory_usage,
                    device_memory_usage=cuda_memory_usage,
                    is_async=is_async,
                    is_remote=is_remote_event,
                    sequence_nr=start.sequence_nr(),
                    device_type=DeviceType.CPU,
                    is_legacy=True,
                    flops=start_flops,
                )
                # note: async events have only cpu total time
                if not is_async and start.has_cuda():
                    duration = start.cuda_elapsed_us(record)
                    if duration > 0:
                        fe.append_kernel(start.name(), start.device(), duration)
                functions.append(fe)
                del range_starts[record_key]
                del cpu_memory_allocs[record_key]
                del cuda_memory_allocs[record_key]
            elif record.kind() == "memory_alloc":
                num_open_handles_cpu = len(cpu_memory_allocs)
                num_open_handles_cuda = len(cuda_memory_allocs)
                if num_open_handles_cpu != num_open_handles_cuda:
                    raise AssertionError(
                        f"Expected CPU and CUDA memory allocation handles to match, "
                        f"but got {num_open_handles_cpu} CPU and {num_open_handles_cuda} CUDA"
                    )
                for handle in cpu_memory_allocs:
                    cpu_memory_allocs[handle] += record.cpu_memory_usage()
                for handle in cuda_memory_allocs:
                    cuda_memory_allocs[handle] += record.cuda_memory_usage()
                if num_open_handles_cpu == 0:
                    # output event as a top-level memory event
                    fe = FunctionEvent(
                        id=0,
                        name=MEMORY_EVENT_NAME,
                        trace_name=None,
                        thread=0,
                        start_us=0,
                        end_us=0,
                        stack=[],
                        cpu_memory_usage=record.cpu_memory_usage(),
                        device_memory_usage=record.cuda_memory_usage(),
                        is_legacy=True,
                    )
                    functions.append(fe)
            prev_record = record

    # Sort functions by start time then by end time ascending.
    # This ensures that--in the case of nested events which
    # have the same start time (which may happen due to the
    # granularity of the given clock tick)--we always show
    # the outermost nested call first. This adds stability
    # in how FunctionEvents appear
    functions.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])
    return functions

```



## High-Level Overview

"""DEPRECATED: use torch.profiler instead."""    def __init__(        self,        enabled=True,        *,        use_cuda=False,        record_shapes=False,        with_flops=False,        profile_memory=False,        with_stack=False,        with_modules=False,    ):        self.enabled: bool = enabled        if not self.enabled:            return        self.use_cuda = use_cuda

This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `profile`

**Functions defined**: `__init__`, `config`, `__enter__`, `_start_trace`, `__exit__`, `__repr__`, `__str__`, `_check_finish`, `table`, `export_chrome_trace`, `export_stacks`, `key_averages`, `total_average`, `self_cpu_time_total`, `_parse_legacy_records`, `_get_record_key`

**Key imports**: itertools, warnings, deprecated, torch, torch.cuda


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `warnings`
- `typing_extensions`: deprecated
- `torch`
- `torch.cuda`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/autograd`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`function.py_docs.md`](./function.py_docs.md)
- [`variable.py_docs.md`](./variable.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`forward_ad.py_docs.md`](./forward_ad.py_docs.md)
- [`gradcheck.py_docs.md`](./gradcheck.py_docs.md)
- [`functional.py_docs.md`](./functional.py_docs.md)
- [`profiler.py_docs.md`](./profiler.py_docs.md)
- [`anomaly_mode.py_docs.md`](./anomaly_mode.py_docs.md)
- [`profiler_util.py_docs.md`](./profiler_util.py_docs.md)


## Cross-References

- **File Documentation**: `profiler_legacy.py_docs.md`
- **Keyword Index**: `profiler_legacy.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/autograd`):

- [`profiler.py_docs.md_docs.md`](./profiler.py_docs.md_docs.md)
- [`profiler_util.py_kw.md_docs.md`](./profiler_util.py_kw.md_docs.md)
- [`profiler_util.py_docs.md_docs.md`](./profiler_util.py_docs.md_docs.md)
- [`variable.py_docs.md_docs.md`](./variable.py_docs.md_docs.md)
- [`forward_ad.py_kw.md_docs.md`](./forward_ad.py_kw.md_docs.md)
- [`graph.py_kw.md_docs.md`](./graph.py_kw.md_docs.md)
- [`forward_ad.py_docs.md_docs.md`](./forward_ad.py_docs.md_docs.md)
- [`functional.py_kw.md_docs.md`](./functional.py_kw.md_docs.md)
- [`profiler.py_kw.md_docs.md`](./profiler.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `profiler_legacy.py_docs.md_docs.md`
- **Keyword Index**: `profiler_legacy.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

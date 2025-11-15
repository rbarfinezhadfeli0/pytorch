# Documentation: `torch/_functorch/benchmark_utils.py`

## File Metadata

- **Path**: `torch/_functorch/benchmark_utils.py`
- **Size**: 6,305 bytes (6.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

import contextlib
import json
import operator
import os
import time

import torch
from torch.profiler import profile, ProfilerActivity


def synchronize():
    pass


def dump_chrome_trace(
    f,
    input,
    trace_filename,
    optimize_ctx,
    activities,
    num_runs=1,
    devices=None,
    kwargs_for_f=None,
    kwargs_for_profiler=None,
):
    """
    Output the chrome trace of running f(input, **kwargs_for_f) with [optimize_ctx]
    [num_runs] times to [trace_filename].

    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.
    Return total runtime without the profiler

    Outputs to trace_filename
    """

    if devices is None:
        devices = ["cuda"]

    global synchronize
    if devices != ["cpu"] and torch.cuda.is_available():
        synchronize = torch.cuda.synchronize

    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}

    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5):  # warmup runs
            f(input, **kwargs_for_f)
            synchronize()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0

    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input, **kwargs_for_f)
                synchronize()
    prof.export_chrome_trace(trace_filename)

    return timing


def get_chrome_trace_events(filename):
    with open(filename) as f:
        data = json.load(f)
    events = data["traceEvents"]
    return events


def is_gpu_compute_event(event):
    global gpu_pids
    return (
        "pid" in event
        and event["pid"] in gpu_pids
        and "ph" in event
        and event["ph"] == "X"
    )


def get_sorted_gpu_events(events):
    sorted_gpu_events = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=operator.itemgetter("ts"))


def get_duration(sorted_gpu_events):
    if len(sorted_gpu_events) == 0:
        return 0
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration


def get_sorted_gpu_mm_conv_events(events):
    def is_mm_conv_event(event):
        return "name" in event and (
            "gemm" in event["name"]
            or "conv" in event["name"]
            or "cutlass" in event["name"]
            or "wgrad" in event["name"]
        )

    gpu_events = get_sorted_gpu_events(events)
    sorted_events = []
    for event in gpu_events:
        if not is_mm_conv_event(event):
            continue
        sorted_events.append(event)
    return sorted_events


gpu_pids = []


def compute_utilization(filename: str, total_length: float):
    """
    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization
    and percent of times spent on matmul and convolution

    Args:
        filename(str): Name of chrome traces file produced by pytorch profiler

        total_length(float): total length of the process without profiler in second

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)
    """
    events = get_chrome_trace_events(filename)

    # get pids of GPU events
    global gpu_pids
    gpu_pids = []
    for event in events:
        if "name" not in event:
            continue
        if event["name"] == "process_labels" and "GPU" in event["args"]["labels"]:
            gpu_pids.append(event["pid"])

    total_length = total_length * 1e6
    sorted_gpu_events = get_sorted_gpu_events(events)
    utilization = get_duration(sorted_gpu_events) / total_length

    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

    return utilization, mm_conv_utilization


def benchmark_utilization(
    f,
    input,
    trace_folder,
    optimize_ctx=None,
    trace_file_name="tmp_chrome_trace",
    num_runs=1,
):
    """
    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of
    running f(input, **kwargs_for_f) with [optimize_ctx] [num_runs] times.
    It will produce a chrome trace file in trace_folder/trace_file_name.json

    Example:

    ```
    def f(a):
        return a.sum()


    a = torch.rand(2**20, device="cuda")
    utilization, mm_conv_utilization = benchmark_utilization(
        f, a, "tmp", trace_file_name="tmp_chrome_trace"
    )
    ```

    Args:
        f: function to benchmark

        input: input to :attr:`f`

        trace_folder: name of the folder to store the chrome trace

        optimize_ctx: the context in which f will run

        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"

        num_runs: number of times to run f, excluding the warm-up runs, default to 1.

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)

    """
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print("create folder " + trace_folder)

    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()

    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + ".json")
    total_length = dump_chrome_trace(
        f,
        input,
        chrome_trace_file_name,
        optimize_ctx,
        [ProfilerActivity.CUDA],
        num_runs=num_runs,
        devices=["cuda"],
    )
    utilization, mm_conv_utilization = compute_utilization(
        chrome_trace_file_name, total_length
    )

    return utilization, mm_conv_utilization

```



## High-Level Overview

"""    Output the chrome trace of running f(input, **kwargs_for_f) with [optimize_ctx]    [num_runs] times to [trace_filename].    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.    Return total runtime without the profiler    Outputs to trace_filename

This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `synchronize`, `dump_chrome_trace`, `get_chrome_trace_events`, `is_gpu_compute_event`, `get_sorted_gpu_events`, `get_duration`, `get_sorted_gpu_mm_conv_events`, `is_mm_conv_event`, `compute_utilization`, `benchmark_utilization`, `f`

**Key imports**: contextlib, json, operator, os, time, torch, profile, ProfilerActivity


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `json`
- `operator`
- `os`
- `time`
- `torch`
- `torch.profiler`: profile, ProfilerActivity


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/_functorch`):

- [`predispatch.py_docs.md`](./predispatch.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`batch_norm_replacement.py_docs.md`](./batch_norm_replacement.py_docs.md)
- [`pytree_hacks.py_docs.md`](./pytree_hacks.py_docs.md)
- [`python_key.py_docs.md`](./python_key.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`partitioners.py_docs.md`](./partitioners.py_docs.md)
- [`vmap.py_docs.md`](./vmap.py_docs.md)
- [`eager_transforms.py_docs.md`](./eager_transforms.py_docs.md)
- [`pyfunctorch.py_docs.md`](./pyfunctorch.py_docs.md)


## Cross-References

- **File Documentation**: `benchmark_utils.py_docs.md`
- **Keyword Index**: `benchmark_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `functorch/benchmarks/pointwise_scorecard.py`

## File Metadata

- **Path**: `functorch/benchmarks/pointwise_scorecard.py`
- **Size**: 6,576 bytes (6.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import inspect
import itertools
import sys
import time

import torch
from functorch import pointwise_operator


torch.set_num_threads(1)
torch._C._debug_set_fusion_group_inlining(False)


def rand(*shape):
    return torch.rand(*shape).mul(16).add(1)


# ------------------------------------------------------------------------------
# Shape test cases
# ------------------------------------------------------------------------------
def scalar():
    return (rand(1), rand(1))


def small():
    return (rand(32), rand(32))


def small_2d():
    return (rand(1, 32), rand(1, 32))


def small_broadcast():
    return (rand(4, 32), rand(32))


def medium():
    return (rand(32, 12, 64, 64), rand(32, 12, 64, 64))


def medium_sliced():
    return (rand(32, 12, 64, 64)[..., ::2], rand(32, 12, 64, 64)[..., ::2])


def medium_transpose():
    return (
        rand(32, 12, 64, 64).transpose(-1, -2),
        rand(32, 12, 64, 64).transpose(-1, -2),
    )


def medium2():
    return (rand(32, 3, 224, 224), rand(32, 3, 224, 224))


def medium3d():
    return (rand(16, 32, 64), rand(16, 32, 64))


def medium_channels_last():
    return (
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
        rand(32, 3, 224, 224).to(memory_format=torch.channels_last),
    )


def medium_broadcast():
    return (rand(32, 12, 64, 64), rand(64))


def medium_broadcast_channels_last():
    return (rand(32, 3, 223, 223).to(memory_format=torch.channels_last), rand(3, 1, 1))


def large():
    return (rand(8192, 8192), rand(8192, 8192))


def large_transpose():
    return (rand(8192, 8192).transpose(0, 1), rand(8192, 8192).transpose(0, 1))


def large_channels_last():
    return (
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
        rand(32, 32, 256, 256).to(memory_format=torch.channels_last),
    )


def pathological_broadcast():
    return (rand(1, 32, 32, 2), rand(1024, 1, 1, 2))


# ------------------------------------------------------------------------------
# Operator test cases
# ------------------------------------------------------------------------------
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def relu(a):
    return a.relu()


def sigmoid(a):
    return a.sigmoid()


def tanh(a):
    return a.tanh()


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def square(a):
    return a**2


def fma(a, b):
    return a * b + b


def hardswish(a):
    return a * (a + 3.0).clamp(0.0, 6.0) / 6.0


def native_hardswish(a):
    return torch._C._nn.hardswish(a)


def softplus(a):
    return (a * 1.0).exp().log1p() / 1.0


def mish(a):
    return a * ((a * 1.0).exp().log1p() / 1.0).tanh()


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def time_cpu(fn, args, iters):
    s = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    e = time.perf_counter()
    return e - s


def time_cuda(fn, args, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3


def benchmark_with_timer(fn, args, timer):
    timer(fn, args, 3)
    calibration = timer(fn, args, 1)
    iters = int(1.0 / calibration)
    return timer(fn, args, iters) / iters


def benchmark(fn, args):
    timer = time_cpu if args[0].device.type == "cpu" else time_cuda
    return benchmark_with_timer(fn, args, timer)


def micros(s):
    return f"{s * 1e6:.1f}"


shapes = [
    scalar,
    small,
    small_2d,
    small_broadcast,
    medium,
    medium2,
    medium3d,
    medium_sliced,
    medium_transpose,
    medium_channels_last,
    medium_broadcast,
    medium_broadcast_channels_last,
    large,
    large_transpose,
    large_channels_last,
    pathological_broadcast,
]

operators = [
    add,
    sub,
    mul,
    div,
    relu,
    sigmoid,
    tanh,
    log,
    exp,
    square,
    fma,
    hardswish,
    native_hardswish,
]

nope = set()
for shape, operator in itertools.product(shapes, operators):
    nargs = len(inspect.signature(operator).parameters)
    args = shape()[:nargs]

    try:
        if shape == medium_transpose:
            raise RuntimeError("pointwise_operator hangs on medium_transpose")
        pw_op = pointwise_operator(operator)
        torch.testing.assert_close(operator(*args), pw_op(*args))
    except Exception:
        print(f"pointwise_operator failed on {operator.__name__}, {shape.__name__}")
        nope.add((operator, shape))

    ts_op = torch.jit.script(operator)
    torch.testing.assert_close(operator(*args), ts_op(*args))


print("fuser,device,operator,shape,time")
results = []
for shape, operator in itertools.product(shapes, operators):
    nargs = len(inspect.signature(operator).parameters)
    args = shape()[:nargs]

    result = benchmark(operator, args)
    print(
        ",".join(
            [
                "eager",
                args[0].device.type,
                operator.__name__,
                shape.__name__,
                micros(result),
            ]
        )
    )
    try:
        if shape == medium_transpose:
            raise RuntimeError("pointwise_operator hangs on medium_transpose")
        if (operator, shape) in nope:
            raise RuntimeError("pointwise_operator fails on medium_transpose")
        pw_op = pointwise_operator(operator)
        result = benchmark(pw_op, args)
        print(
            ",".join(
                [
                    "pointwise",
                    args[0].device.type,
                    operator.__name__,
                    shape.__name__,
                    micros(result),
                ]
            )
        )
    except Exception:
        print(
            ",".join(
                [
                    "pointwise",
                    args[0].device.type,
                    operator.__name__,
                    shape.__name__,
                    micros(float("nan")),
                ]
            )
        )

    ts_op = torch.jit.script(operator)
    result = benchmark(ts_op, args)
    print(
        ",".join(
            [
                "fuser",
                args[0].device.type,
                operator.__name__,
                shape.__name__,
                micros(result),
            ]
        )
    )
    sys.stdout.flush()

```



## High-Level Overview


This Python file contains 0 class(es) and 37 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `rand`, `scalar`, `small`, `small_2d`, `small_broadcast`, `medium`, `medium_sliced`, `medium_transpose`, `medium2`, `medium3d`, `medium_channels_last`, `medium_broadcast`, `medium_broadcast_channels_last`, `large`, `large_transpose`, `large_channels_last`, `pathological_broadcast`, `add`, `sub`, `mul`

**Key imports**: inspect, itertools, sys, time, torch, pointwise_operator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/benchmarks`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `itertools`
- `sys`
- `time`
- `torch`
- `functorch`: pointwise_operator


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`functorch/benchmarks`):

- [`operator_authoring.py_docs.md`](./operator_authoring.py_docs.md)
- [`per_sample_grads.py_docs.md`](./per_sample_grads.py_docs.md)
- [`chrome_trace_parser.py_docs.md`](./chrome_trace_parser.py_docs.md)
- [`process_scorecard.py_docs.md`](./process_scorecard.py_docs.md)
- [`cse.py_docs.md`](./cse.py_docs.md)


## Cross-References

- **File Documentation**: `pointwise_scorecard.py_docs.md`
- **Keyword Index**: `pointwise_scorecard.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

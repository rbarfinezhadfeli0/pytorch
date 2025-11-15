# Documentation: `benchmarks/dynamo/microbenchmarks/microbench.py`

## File Metadata

- **Path**: `benchmarks/dynamo/microbenchmarks/microbench.py`
- **Size**: 5,574 bytes (5.44 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

import argparse
import inspect
import sys

import numpy as np
import tabulate

import torch
import torch._inductor
from torch._dynamo.backends.cudagraphs import cudagraphs_inner
from torch._dynamo.testing import same
from torch._inductor.compile_fx import compile_fx
from torch._inductor.utils import timed


aten = torch.ops.aten

try:
    import test.test_torchinductor as tti
except ImportError:
    tti = None


def compute_speedups(args, models, example_inputs):
    expected = models[0](*example_inputs)
    for model in models[1:]:
        actual = model(*example_inputs)
        assert same(actual, expected), expected[0] - actual[0]

    timings = np.zeros((args.repeat, len(models)), np.float64)
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        for m, model in enumerate(models):
            timings[rep, m] = timed(model, example_inputs)
    median = np.median(timings, axis=0)
    return (median[0] / median[1:]).tolist()


def microbenchmark(args, model, example_inputs):
    compiled_fn = compile_fx(torch.fx.symbolic_trace(model), example_inputs)
    cudagraphs_eager = cudagraphs_inner(model, example_inputs, copy_outputs=False)
    cudagraphs_jit = cudagraphs_inner(
        torch.jit.trace(model, example_inputs), example_inputs, copy_outputs=False
    )
    return compute_speedups(
        args,
        [cudagraphs_eager, cudagraphs_jit, compiled_fn],
        example_inputs,
    )


class MyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
        )

    def forward(self, input):
        # return (self.model(input) + 1,)
        return (self.model(input),)


class MyModel2(torch.nn.Module):
    def forward(self, x, y):
        # return x / (torch.abs(x) + 1.0),
        return (x + y,)


class MicroBenchmarks:
    @staticmethod
    def add(a, b):
        return (a + b,)

    @staticmethod
    def scale(x, m, d):
        return ((x - m) / torch.clip(d, 1e-4),)

    @staticmethod
    def abs_norm(x):
        return (x / (torch.abs(x) + 1),)

    @staticmethod
    def add_relu_softmax(x, a):
        return (torch.softmax(torch.relu(x + a), -1),)

    @staticmethod
    def sum(a, b):
        return ((a + b).sum(),)

    @staticmethod
    def view(x):
        return (aten.alias(x),)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument("--size", "-s", action="append", help="cpu or cuda")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    parser.add_argument(
        "--threads", "-t", type=int, help="number of threads to use for eager"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    parser.add_argument(
        "--nvfuser", action="store_true", help="enable nvfuser globally"
    )
    parser.add_argument("--transpose", action="store_true", help="transpose one input")
    parser.add_argument("--broadcast", action="store_true", help="broadcast one input")
    args = parser.parse_args()

    # defaults
    args.devices = args.devices or ["cpu", "cuda"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]
    args.size = args.size or [64, 256, 1024, 4096, 8192]

    if args.nvfuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        torch._C._jit_override_can_fuse_on_cpu(torch._C._llvm_enabled())
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        if torch.cuda.is_available():
            torch._C._jit_set_nvfuser_enabled(False)

    if args.threads:
        torch.set_num_threads(args.threads)
        torch._inductor.config.cpp.threads = args.threads

    if args.verbose:
        torch._inductor.config.debug = True

    torch._inductor.config.triton.autotune_pointwise = True

    rows = []
    for model in (MicroBenchmarks.sum, MicroBenchmarks.view):
        nargs = len(inspect.signature(model).parameters)
        for device in args.devices:
            for n in args.size:
                n = int(n)
                sys.stdout.write(f"{model.__name__:10} {device:4} {n:5} ")
                sys.stdout.flush()
                inputs = [torch.rand((n, n), device=device) for _ in range(nargs)]
                if args.broadcast:
                    inputs[-1] = torch.rand((1, n), device=device)
                if args.transpose:
                    inputs[-1] = inputs[-1].transpose(0, 1)
                result = microbenchmark(args, model, inputs)
                rows.append([model.__name__, device, str(n)] + result)
                print(" ".join(f"{v:.2f}x" for v in result))

    print(
        tabulate.tabulate(
            rows,
            headers=[
                "model",
                "dev",
                "n",
                "ts",
                "inductor",
            ],
        )
    )


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 3 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModel1`, `MyModel2`, `MicroBenchmarks`

**Functions defined**: `compute_speedups`, `microbenchmark`, `__init__`, `forward`, `forward`, `add`, `scale`, `abs_norm`, `add_relu_softmax`, `sum`, `view`, `main`

**Key imports**: argparse, inspect, sys, numpy as np, tabulate, torch, torch._inductor, cudagraphs_inner, same, compile_fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/microbenchmarks`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `inspect`
- `sys`
- `numpy as np`
- `tabulate`
- `torch`
- `torch._inductor`
- `torch._dynamo.backends.cudagraphs`: cudagraphs_inner
- `torch._dynamo.testing`: same
- `torch._inductor.compile_fx`: compile_fx
- `torch._inductor.utils`: timed
- `test.test_torchinductor as tti`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`benchmarks/dynamo/microbenchmarks`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`bench_mm_fusion.py_docs.md`](./bench_mm_fusion.py_docs.md)
- [`dynamo_microbenchmarks.py_docs.md`](./dynamo_microbenchmarks.py_docs.md)
- [`benchmark_helper.py_docs.md`](./benchmark_helper.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`dynamo_guard_eval.py_docs.md`](./dynamo_guard_eval.py_docs.md)
- [`overheads.py_docs.md`](./overheads.py_docs.md)
- [`operatorbench.py_docs.md`](./operatorbench.py_docs.md)
- [`tensor_layout_mini_benchmark.py_docs.md`](./tensor_layout_mini_benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `microbench.py_docs.md`
- **Keyword Index**: `microbench.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

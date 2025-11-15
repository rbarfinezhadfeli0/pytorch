# Documentation: `benchmarks/dynamo/pr_time_benchmarks/benchmarks/basic_modules_benchmarks.py`

## File Metadata

- **Path**: `benchmarks/dynamo/pr_time_benchmarks/benchmarks/basic_modules_benchmarks.py`
- **Size**: 2,349 bytes (2.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_cache


class ListOfLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Benchmark(BenchmarkBase):
    def __init__(
        self, ModuleClass, backend, is_gpu=False, dynamic=False, force_shape_pad=False
    ):
        self.ModuleClass = ModuleClass
        self._name = ModuleClass.__name__
        self._is_gpu = is_gpu
        self._force_shape_pad = force_shape_pad

        super().__init__(
            category="basic_modules",
            backend=backend,
            device="cuda" if self._is_gpu else "cpu",
            dynamic=dynamic,
        )

    def name(self):
        prefix = f"{self.category()}_{self._name}_{self.backend()}"
        if self.is_dynamic():
            prefix += "_dynamic"
        if self._is_gpu:
            prefix += "_gpu"
        if self._force_shape_pad:
            prefix += "_force_shape_pad"
        return prefix

    def _prepare_once(self):
        self.m = self.ModuleClass()
        torch.set_float32_matmul_precision("high")
        self.input = torch.ones(10, device=self.device())

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        with (
            fresh_cache(),
            torch._inductor.config.patch(force_shape_pad=self._force_shape_pad),
        ):
            opt_m = torch.compile(backend=self.backend(), dynamic=self.is_dynamic())(
                self.m.cuda() if self._is_gpu else self.m
            )
            opt_m(self.input)


def main():
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(ListOfLinears, "eager"),
        Benchmark(ListOfLinears, "inductor"),
        Benchmark(ListOfLinears, "inductor", is_gpu=True),
        Benchmark(ListOfLinears, "inductor", is_gpu=True, force_shape_pad=True),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ListOfLinears`, `Benchmark`

**Functions defined**: `__init__`, `forward`, `__init__`, `name`, `_prepare_once`, `_prepare`, `_work`, `main`

**Key imports**: sys, BenchmarkBase, torch, torch.nn as nn, fresh_cache


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo/pr_time_benchmarks/benchmarks`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `benchmark_base`: BenchmarkBase
- `torch`
- `torch.nn as nn`
- `torch._inductor.utils`: fresh_cache


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`benchmarks/dynamo/pr_time_benchmarks/benchmarks`):

- [`add_loop.py_docs.md`](./add_loop.py_docs.md)
- [`benchmark_base.py_docs.md`](./benchmark_base.py_docs.md)
- [`dtensor.py_docs.md`](./dtensor.py_docs.md)
- [`float_args.py_docs.md`](./float_args.py_docs.md)
- [`aotdispatcher.py_docs.md`](./aotdispatcher.py_docs.md)
- [`runtime_overhead.py_docs.md`](./runtime_overhead.py_docs.md)
- [`mm_loop.py_docs.md`](./mm_loop.py_docs.md)
- [`aotdispatcher_partitioner.py_docs.md`](./aotdispatcher_partitioner.py_docs.md)
- [`sum_floordiv.py_docs.md`](./sum_floordiv.py_docs.md)


## Cross-References

- **File Documentation**: `basic_modules_benchmarks.py_docs.md`
- **Keyword Index**: `basic_modules_benchmarks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

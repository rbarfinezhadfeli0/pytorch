# Documentation: `torch/utils/throughput_benchmark.py`

## File Metadata

- **Path**: `torch/utils/throughput_benchmark.py`
- **Size**: 6,625 bytes (6.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import torch._C


def format_time(time_us=None, time_ms=None, time_s=None) -> str:
    """Define time formatting."""
    if sum([time_us is not None, time_ms is not None, time_s is not None]) != 1:
        raise AssertionError("Expected only one of time_us, time_ms, time_s is given.")

    US_IN_SECOND = 1e6
    US_IN_MS = 1e3

    if time_us is None:
        if time_ms is not None:
            time_us = time_ms * US_IN_MS
        elif time_s is not None:
            time_us = time_s * US_IN_SECOND
        else:
            raise AssertionError("Shouldn't reach here :)")

    if time_us >= US_IN_SECOND:
        return f'{time_us / US_IN_SECOND:.3f}s'
    if time_us >= US_IN_MS:
        return f'{time_us / US_IN_MS:.3f}ms'
    return f'{time_us:.3f}us'


class ExecutionStats:
    def __init__(self, c_stats, benchmark_config) -> None:
        self._c_stats = c_stats
        self.benchmark_config = benchmark_config

    @property
    def latency_avg_ms(self):
        return self._c_stats.latency_avg_ms

    @property
    def num_iters(self):
        return self._c_stats.num_iters

    @property
    def iters_per_second(self):
        """Return total number of iterations per second across all calling threads."""
        return self.num_iters / self.total_time_seconds

    @property
    def total_time_seconds(self):
        return self.num_iters * (
            self.latency_avg_ms / 1000.0) / self.benchmark_config.num_calling_threads

    def __str__(self) -> str:
        return '\n'.join([
            "Average latency per example: " + format_time(time_ms=self.latency_avg_ms),
            f"Total number of iterations: {self.num_iters}",
            f"Total number of iterations per second (across all threads): {self.iters_per_second:.2f}",
            "Total time: " + format_time(time_s=self.total_time_seconds)
        ])


class ThroughputBenchmark:
    """
    This class is a wrapper around a c++ component throughput_benchmark::ThroughputBenchmark.

    This wrapper on the throughput_benchmark::ThroughputBenchmark component is responsible
    for executing a PyTorch module (nn.Module or ScriptModule) under an inference
    server like load. It can emulate multiple calling threads to a single module
    provided. In the future we plan to enhance this component to support inter and
    intra-op parallelism as well as multiple models running in a single process.

    Please note that even though nn.Module is supported, it might incur an overhead
    from the need to hold GIL every time we execute Python code or pass around
    inputs as Python objects. As soon as you have a ScriptModule version of your
    model for inference deployment it is better to switch to using it in this
    benchmark.

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> from torch.utils import ThroughputBenchmark
        >>> bench = ThroughputBenchmark(my_module)
        >>> # Pre-populate benchmark's data set with the inputs
        >>> for input in inputs:
        ...     # Both args and kwargs work, same as any PyTorch Module / ScriptModule
        ...     bench.add_input(input[0], x2=input[1])
        >>> # Inputs supplied above are randomly used during the execution
        >>> stats = bench.benchmark(
        ...     num_calling_threads=4,
        ...     num_warmup_iters = 100,
        ...     num_iters = 1000,
        ... )
        >>> print("Avg latency (ms): {}".format(stats.latency_avg_ms))
        >>> print("Number of iterations: {}".format(stats.num_iters))
    """

    def __init__(self, module) -> None:
        if isinstance(module, torch.jit.ScriptModule):
            self._benchmark = torch._C.ThroughputBenchmark(module._c)
        else:
            self._benchmark = torch._C.ThroughputBenchmark(module)

    def run_once(self, *args, **kwargs):
        """
        Given input id (input_idx) run benchmark once and return prediction.

        This is useful for testing that benchmark actually runs the module you
        want it to run. input_idx here is an index into inputs array populated
        by calling add_input() method.
        """
        return self._benchmark.run_once(*args, **kwargs)

    def add_input(self, *args, **kwargs) -> None:
        """
        Store a single input to a module into the benchmark memory and keep it there.

        During the benchmark execution every thread is going to pick up a
        random input from the all the inputs ever supplied to the benchmark via
        this function.
        """
        self._benchmark.add_input(*args, **kwargs)

    def benchmark(
            self,
            num_calling_threads=1,
            num_warmup_iters=10,
            num_iters=100,
            profiler_output_path=""):
        """
        Run a benchmark on the module.

        Args:
            num_warmup_iters (int): Warmup iters are used to make sure we run a module
                a few times before actually measuring things. This way we avoid cold
                caches and any other similar problems. This is the number of warmup
                iterations for each of the thread in separate

            num_iters (int): Number of iterations the benchmark should run with.
                This number is separate from the warmup iterations. Also the number is
                shared across all the threads. Once the num_iters iterations across all
                the threads is reached, we will stop execution. Though total number of
                iterations might be slightly larger. Which is reported as
                stats.num_iters where stats is the result of this function

            profiler_output_path (str): Location to save Autograd Profiler trace.
                If not empty, Autograd Profiler will be enabled for the main benchmark
                execution (but not the warmup phase). The full trace will be saved
                into the file path provided by this argument


        This function returns BenchmarkExecutionStats object which is defined via pybind11.
        It currently has two fields:
            - num_iters - number of actual iterations the benchmark have made
            - avg_latency_ms - average time it took to infer on one input example in milliseconds
        """
        config = torch._C.BenchmarkConfig()
        config.num_calling_threads = num_calling_threads
        config.num_warmup_iters = num_warmup_iters
        config.num_iters = num_iters
        config.profiler_output_path = profiler_output_path
        c_stats = self._benchmark.benchmark(config)
        return ExecutionStats(c_stats, config)

```



## High-Level Overview

"""Define time formatting."""    if sum([time_us is not None, time_ms is not None, time_s is not None]) != 1:        raise AssertionError("Expected only one of time_us, time_ms, time_s is given.")    US_IN_SECOND = 1e6    US_IN_MS = 1e3    if time_us is None:        if time_ms is not None:            time_us = time_ms * US_IN_MS        elif time_s is not None:            time_us = time_s * US_IN_SECOND        else:            raise AssertionError("Shouldn't reach here :)")    if time_us >= US_IN_SECOND:        return f'{time_us / US_IN_SECOND:.3f}s'    if time_us >= US_IN_MS:        return f'{time_us / US_IN_MS:.3f}ms'    return f'{time_us:.3f}us'class ExecutionStats:    def __init__(self, c_stats, benchmark_config) -> None:        self._c_stats = c_stats        self.benchmark_config = benchmark_config    @property    def latency_avg_ms(self):        return self._c_stats.latency_avg_ms    @property    def num_iters(self):        return self._c_stats.num_iters    @property    def iters_per_second(self):

This Python file contains 3 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExecutionStats`, `ThroughputBenchmark`

**Functions defined**: `format_time`, `__init__`, `latency_avg_ms`, `num_iters`, `iters_per_second`, `total_time_seconds`, `__str__`, `__init__`, `run_once`, `add_input`, `benchmark`

**Key imports**: torch._C, ThroughputBenchmark


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch._C`
- `torch.utils`: ThroughputBenchmark


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

Files in the same folder (`torch/utils`):

- [`_zip.py_docs.md`](./_zip.py_docs.md)
- [`weak.py_docs.md`](./weak.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_cpp_embed_headers.py_docs.md`](./_cpp_embed_headers.py_docs.md)
- [`_cpp_extension_versioner.py_docs.md`](./_cpp_extension_versioner.py_docs.md)
- [`module_tracker.py_docs.md`](./module_tracker.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`_content_store.py_docs.md`](./_content_store.py_docs.md)
- [`_triton.py_docs.md`](./_triton.py_docs.md)
- [`file_baton.py_docs.md`](./file_baton.py_docs.md)


## Cross-References

- **File Documentation**: `throughput_benchmark.py_docs.md`
- **Keyword Index**: `throughput_benchmark.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

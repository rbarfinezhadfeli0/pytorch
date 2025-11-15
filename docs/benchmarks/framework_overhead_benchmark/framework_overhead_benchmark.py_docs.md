# Documentation: `benchmarks/framework_overhead_benchmark/framework_overhead_benchmark.py`

## File Metadata

- **Path**: `benchmarks/framework_overhead_benchmark/framework_overhead_benchmark.py`
- **Size**: 3,983 bytes (3.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse

from pt_wrapper_module import WrapperModule
from SimpleAddModule import add_tensors_loop, SimpleAddModule

from utils import benchmark_module, BenchmarkConfig, ModuleConfig, ms_to_us


""" Framework overhead benchmark script.
Benchmark framework overhead.
Currently supported ops: add.
As of now runs only forward pass.
Supports both graph mode and eager mode. In graph mode the module is traced via JIT tracing.
Debug option prints the traced graph is graph_mode is enabled.
Graph can be saved via save option. Saved in the directory where benchmark is run.
Example build/run:
To run PT benchmark:
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add-op --graph-mode --eager-mode (Runs both graph mode and eager mode)
buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark --
 --add-op --graph-mode (Runs only graph mode)
"""

SUPPORTED_OPS = {"add_op"}


def parse_op_args(op):
    op_list = op.split(",")  # noqa: F841


def print_results(result):
    print("===================================")
    for key, value in result.items():
        print(f"{key}, latency per iter (us):{ms_to_us(value)}")
    print("===================================")


def benchmark_simple_fn(args, config, module_config, module_type, result):
    """Benchmarks a PyTorch traceable function specified in the config.
    Instantiates a wrapper object that wraps the object of module_type and runs the forward
    method using benchmark_module.
    Args:
        config:         contains number of warmup and benchmark iterations.
        module_config:  module_config which contains op, number of parameters that op takes
                    and whether graph mode is enabled or not.
        module_type:    Type of the module to be wrapped. e.g. SimpleAddModule for add op.
        result:         dictionary instance to be populated with the benchmark result (latency per iter).
    """
    print(f"Benchmarking {module_type.__name__}")
    f_name = (
        module_config.pt_fn.__name__ + ":Num Operands=" + str(module_config.num_params)
    )
    graph_mode_str = "Graph mode" + ":" + str(module_config.graph_mode)
    result_key = ",".join((f_name, graph_mode_str))
    module = WrapperModule(module_type, module_config, args.debug, args.save)
    latency_per_iter_ms = benchmark_module(
        config, module, args.use_throughput_benchmark
    )
    result[result_key] = latency_per_iter_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="add_op", dest="op", type=str)
    parser.add_argument(
        "--use-throughput-benchmark",
        "--use_throughput_benchmark",
        default=False,
        dest="use_throughput_benchmark",
        action="store_true",
    )
    parser.add_argument("--debug", default=False, dest="debug", action="store_true")
    parser.add_argument("--save", default=False, dest="save", action="store_true")
    parser.add_argument(
        "--eager-mode",
        "--eager_mode",
        default=False,
        dest="eager_mode",
        action="store_true",
    )
    parser.add_argument(
        "--num-warmup-iters", "--num_warmup_iters", type=int, default=100
    )
    parser.add_argument("--num-iters", "--num_iters", type=int, default=1000)
    args = parser.parse_args()

    if args.op not in SUPPORTED_OPS:
        print(f"Op {args.op} is not supported: Supported ops are:{SUPPORTED_OPS}")
        return
    num_warmup_iters = args.num_warmup_iters
    num_iters = args.num_iters
    config = BenchmarkConfig(num_warmup_iters, num_iters)
    graph_mode = True
    if args.eager_mode:
        graph_mode = False
    result = {}
    if args.op == "add_op":
        num_params = 2
        module_config = ModuleConfig(add_tensors_loop, None, num_params, graph_mode)
        benchmark_simple_fn(args, config, module_config, SimpleAddModule, result)
    print_results(result)


if __name__ == "__main__":
    main()

```



## High-Level Overview

""" Framework overhead benchmark script.Benchmark framework overhead.Currently supported ops: add.As of now runs only forward pass.Supports both graph mode and eager mode. In graph mode the module is traced via JIT tracing.Debug option prints the traced graph is graph_mode is enabled.Graph can be saved via save option. Saved in the directory where benchmark is run.Example build/run:To run PT benchmark:buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark -- --add-op --graph-mode --eager-mode (Runs both graph mode and eager mode)buck run @mode/opt <path-to-framework_overhead_benchmark>:framework_overhead_benchmark -- --add-op --graph-mode (Runs only graph mode)

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parse_op_args`, `print_results`, `benchmark_simple_fn`, `main`

**Key imports**: argparse, WrapperModule, add_tensors_loop, SimpleAddModule, benchmark_module, BenchmarkConfig, ModuleConfig, ms_to_us


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/framework_overhead_benchmark`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `pt_wrapper_module`: WrapperModule
- `SimpleAddModule`: add_tensors_loop, SimpleAddModule
- `utils`: benchmark_module, BenchmarkConfig, ModuleConfig, ms_to_us


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

Files in the same folder (`benchmarks/framework_overhead_benchmark`):

- [`utils.py_docs.md`](./utils.py_docs.md)
- [`pt_wrapper_module.py_docs.md`](./pt_wrapper_module.py_docs.md)
- [`SimpleAddModule.py_docs.md`](./SimpleAddModule.py_docs.md)


## Cross-References

- **File Documentation**: `framework_overhead_benchmark.py_docs.md`
- **Keyword Index**: `framework_overhead_benchmark.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

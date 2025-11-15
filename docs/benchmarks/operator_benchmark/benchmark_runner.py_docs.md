# Documentation: `benchmarks/operator_benchmark/benchmark_runner.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/benchmark_runner.py`
- **Size**: 5,564 bytes (5.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse

import benchmark_core

import benchmark_utils

import torch


"""Performance microbenchmarks's main binary.

This is the main function for running performance microbenchmark tests.
It also registers existing benchmark tests via Python module imports.
"""
parser = argparse.ArgumentParser(
    description="Run microbenchmarks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler="resolve",
)


def parse_args():
    parser.add_argument(
        "--tag-filter",
        "--tag_filter",
        help="tag_filter can be used to run the shapes which matches the tag. (all is used to run all the shapes)",
        default="short",
    )

    # This option is used to filter test cases to run.
    parser.add_argument(
        "--operators",
        help="Filter tests based on comma-delimited list of operators to test",
        default=None,
    )

    parser.add_argument(
        "--operator-range",
        "--operator_range",
        help="Filter tests based on operator_range(e.g. a-c or b,c-d)",
        default=None,
    )

    parser.add_argument(
        "--test-name",
        "--test_name",
        help="Run tests that have the provided test_name",
        default=None,
    )

    parser.add_argument(
        "--list-ops",
        "--list_ops",
        help="List operators without running them",
        action="store_true",
    )

    parser.add_argument(
        "--output-json",
        "--output_json",
        help="JSON file path to write the results to",
        default=None,
    )

    parser.add_argument(
        "--benchmark-name",
        "--benchmark_name",
        help="Name of the benchmark to store results to",
        default="PyTorch operator benchmark",
    )

    parser.add_argument(
        "--list-tests",
        "--list_tests",
        help="List all test cases without running them",
        action="store_true",
    )

    parser.add_argument(
        "--iterations",
        help="Repeat each operator for the number of iterations",
        type=int,
    )

    parser.add_argument(
        "--num-runs",
        "--num_runs",
        help="Run each test for num_runs. Each run executes an operator for number of <--iterations>",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-time-per-test",
        "--min_time_per_test",
        help="Set the minimum time (unit: seconds) to run each test",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--warmup-iterations",
        "--warmup_iterations",
        help="Number of iterations to ignore before measuring performance",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--omp-num-threads",
        "--omp_num_threads",
        help="Number of OpenMP threads used in PyTorch runtime",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--mkl-num-threads",
        "--mkl_num_threads",
        help="Number of MKL threads used in PyTorch runtime",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--report-aibench",
        "--report_aibench",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Print result when running on AIBench",
    )

    parser.add_argument(
        "--use-jit",
        "--use_jit",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run operators with PyTorch JIT mode",
    )

    parser.add_argument(
        "--use-compile",
        "--use_compile",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run operators with PyTorch Compile mode",
    )

    parser.add_argument(
        "--forward-only",
        "--forward_only",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Only run the forward path of operators",
    )

    parser.add_argument(
        "--device",
        help="Run tests on the provided architecture (cpu, cuda)",
        default="None",
    )

    parser.add_argument(
        "--output-csv",
        "--output_csv",
        help="CSV file path to store the results",
        default="benchmark_logs",
    )

    parser.add_argument(
        "--output-json-for-dashboard",
        "--output_json_for_dashboard",
        help="Save results in JSON format for display on the OSS dashboard",
        default="benchmark-results.json",
    )

    args, _ = parser.parse_known_args()

    if args.omp_num_threads:
        # benchmark_utils.set_omp_threads sets the env variable OMP_NUM_THREADS
        # which doesn't have any impact as C2 init logic has already been called
        # before setting the env var.

        # In general, OMP_NUM_THREADS (and other OMP env variables) needs to be set
        # before the program is started.
        # From Chapter 4 in OMP standard: https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf
        # "Modifications to the environment variables after the program has started,
        # even if modified by the program itself, are ignored by the OpenMP implementation"
        benchmark_utils.set_omp_threads(args.omp_num_threads)
        torch.set_num_threads(args.omp_num_threads)
    if args.mkl_num_threads:
        benchmark_utils.set_mkl_threads(args.mkl_num_threads)

    return args


def main():
    args = parse_args()
    benchmark_core.BenchmarkRunner(args).run()


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Performance microbenchmarks's main binary.This is the main function for running performance microbenchmark tests.It also registers existing benchmark tests via Python module imports.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parse_args`, `main`

**Key imports**: argparse, benchmark_core, benchmark_utils, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `benchmark_core`
- `benchmark_utils`
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`benchmarks/operator_benchmark`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`benchmark_test_generator.py_docs.md`](./benchmark_test_generator.py_docs.md)
- [`x86_64_expected_ci_operator_benchmark_eager_float32_cpu.csv_docs.md`](./x86_64_expected_ci_operator_benchmark_eager_float32_cpu.csv_docs.md)
- [`benchmark_pytorch.py_docs.md`](./benchmark_pytorch.py_docs.md)
- [`benchmark_all_other_test.py_docs.md`](./benchmark_all_other_test.py_docs.md)
- [`check_perf_csv.py_docs.md`](./check_perf_csv.py_docs.md)
- [`operator_benchmark.py_docs.md`](./operator_benchmark.py_docs.md)
- [`benchmark_core.py_docs.md`](./benchmark_core.py_docs.md)
- [`benchmark_all_test.py_docs.md`](./benchmark_all_test.py_docs.md)
- [`aarch64_expected_ci_operator_benchmark_eager_float32_cpu.csv_docs.md`](./aarch64_expected_ci_operator_benchmark_eager_float32_cpu.csv_docs.md)


## Cross-References

- **File Documentation**: `benchmark_runner.py_docs.md`
- **Keyword Index**: `benchmark_runner.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

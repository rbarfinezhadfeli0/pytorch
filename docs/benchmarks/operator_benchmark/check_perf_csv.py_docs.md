# Documentation: `benchmarks/operator_benchmark/check_perf_csv.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/check_perf_csv.py`
- **Size**: 3,502 bytes (3.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import sys
import textwrap

import pandas as pd


SKIP_TEST_LISTS = [
    # https://github.com/pytorch/pytorch/issues/143852
    "channel_shuffle_batch_size4_channels_per_group64_height64_width64_groups4_channel_lastTrue",
    "batchnorm_N3136_C256_cpu_trainingTrue_cudnnFalse",
    "index_add__M256_N512_K1_dim1_cpu_dtypetorch.float32",
    "interpolate_input_size(1,3,600,400)_output_size(240,240)_channels_lastTrue_modelinear",
    "original_kernel_tensor_N1_C3_H512_W512_zero_point_dtypetorch.int32_nbits4_cpu",
    "original_kernel_tensor_N1_C3_H512_W512_zero_point_dtypetorch.int32_nbits8_cpu",
]


def get_field(csv, case: str, field: str):
    try:
        return csv.loc[csv["Case Name"] == case][field].item()
    except Exception:
        return None


def check_perf(actual_csv, expected_csv, expected_filename, threshold):
    failed = []
    improved = []
    baseline_not_found = []

    actual_csv = actual_csv[~actual_csv["Case Name"].isin(set(SKIP_TEST_LISTS))]

    for case in actual_csv["Case Name"]:
        perf = get_field(actual_csv, case, "Execution Time")
        expected_perf = get_field(expected_csv, case, "Execution Time")

        if expected_perf is None:
            status = "Baseline Not Found"
            print(f"{case:34}  {status}")
            baseline_not_found.append(case)
            continue

        speed_up = expected_perf / perf

        if (1 - threshold) <= speed_up < (1 + threshold):
            status = "PASS"
            print(f"{case:34}  {status}")
            continue
        elif speed_up >= 1 + threshold:
            status = "IMPROVED:"
            improved.append(case)
        else:
            status = "FAILED:"
            failed.append(case)
        print(f"{case:34}  {status:9} perf={perf}, expected={expected_perf}")

    msg = ""
    if failed or improved or baseline_not_found:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have performance status regressed:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have performance status improved:
                {" ".join(improved)}

            """
            )

        if baseline_not_found:
            msg += textwrap.dedent(
                f"""
            Baseline Not Found: {len(baseline_not_found)} models don't have the baseline data:
                {" ".join(baseline_not_found)}

            """
            )

        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        """
        )
    return failed or improved or baseline_not_found, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold to define regression/improvement",
    )
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    actual.drop_duplicates(subset=["Case Name"], keep="first", inplace=True)
    expected = pd.read_csv(args.expected)

    failed, msg = check_perf(actual, expected, args.expected, args.threshold)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_field`, `check_perf`, `main`

**Key imports**: argparse, sys, textwrap, pandas as pd


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `sys`
- `textwrap`
- `pandas as pd`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


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
- [`operator_benchmark.py_docs.md`](./operator_benchmark.py_docs.md)
- [`benchmark_core.py_docs.md`](./benchmark_core.py_docs.md)
- [`benchmark_all_test.py_docs.md`](./benchmark_all_test.py_docs.md)
- [`aarch64_expected_ci_operator_benchmark_eager_float32_cpu.csv_docs.md`](./aarch64_expected_ci_operator_benchmark_eager_float32_cpu.csv_docs.md)


## Cross-References

- **File Documentation**: `check_perf_csv.py_docs.md`
- **Keyword Index**: `check_perf_csv.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/utils/benchmark/examples/sparse/op_benchmark.py`

## File Metadata

- **Path**: `torch/utils/benchmark/examples/sparse/op_benchmark.py`
- **Size**: 4,456 bytes (4.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""Example use of Timer and sparse op fuzzers to measure kernel performance.

$ python -m examples.sparse.op_benchmark
"""

import numpy as np
import torch

from torch.utils.benchmark import Timer
from torch.utils.benchmark.op_fuzzers.sparse_unary import UnaryOpSparseFuzzer
from torch.utils.benchmark.op_fuzzers.sparse_binary import BinaryOpSparseFuzzer
import operator

_MEASURE_TIME = 1.0

def assert_dicts_equal(dict_0, dict_1) -> None:
    """Builtin dict comparison will not compare numpy arrays.
    e.g.
        x = {"a": np.ones((2, 1))}
        x == x  # Raises ValueError
    """
    if set(dict_0.keys()) != set(dict_0.keys()):
        raise AssertionError("dicts must have the same keys")
    if all(np.all(v != dict_1[k]) for k, v in dict_0.items() if k != "dtype"):
        raise AssertionError("dict values differ for keys other than 'dtype'")

def run(n, stmt, fuzzer_cls) -> None:
    float_iter = fuzzer_cls(seed=0, dtype=torch.float32).take(n)
    double_iter = fuzzer_cls(seed=0, dtype=torch.float64).take(n)
    raw_results = []
    for i, (float_values, int_values) in enumerate(zip(float_iter, double_iter, strict=True)):
        float_tensors, float_tensor_params, float_params = float_values
        int_tensors, int_tensor_params, int_params = int_values

        assert_dicts_equal(float_params, int_params)
        assert_dicts_equal(float_tensor_params["x"], int_tensor_params["x"])

        float_measurement, int_measurement = (
            Timer(
                stmt,
                globals=tensors,
            ).blocked_autorange(min_run_time=_MEASURE_TIME)
            for tensors in (float_tensors, int_tensors)
        )

        descriptions = []
        for name in float_tensors:
            shape_str = "(" + ", ".join([
                f"2 ** {int(np.log2(i))}"
                if 2 ** int(np.log2(i)) == i and i > 1
                else str(i)
                for i in float_tensors[name].shape
            ]) + ")"
            sparse_dim = float_tensor_params[name]["sparse_dim"]
            sparse_dim_str = str(sparse_dim)
            is_coalesced = float_tensor_params[name]["is_coalesced"]
            is_coalesced_str = "True" if is_coalesced else "False"
            descriptions.append((name, shape_str, sparse_dim_str, is_coalesced_str))
        raw_results.append((float_measurement, int_measurement, descriptions))

        print(f"\r{i + 1} / {n}", end="")
    print()

    parsed_results, name_len, shape_len, sparse_dim_len, is_coalesced_len = [], 0, 0, 0, 0
    for float_measurement, int_measurement, descriptions in raw_results:
        t_float = float_measurement.median * 1e6
        t_int = int_measurement.median * 1e6
        rel_diff = abs(t_float - t_int) / (t_float + t_int) * 2
        parsed_results.append((t_float, t_int, rel_diff, descriptions))
        for name, shape, sparse_dim, is_coalesced in descriptions:
            name_len = max(name_len, len(name))
            shape_len = max(shape_len, len(shape))
            sparse_dim_len = max(sparse_dim_len, len(sparse_dim))
            is_coalesced_len = max(is_coalesced_len, len(is_coalesced))

    parsed_results.sort(key=operator.itemgetter(2))

    print(f"stmt: {stmt}")
    print(f" diff    faster{'':>17}{' ' * name_len} ", end="")
    print(f"{'shape'.ljust(shape_len)}{'':>12}{'sparse_dim'.ljust(sparse_dim_len)}", end="")
    print(f"          is_coalesced\n{'-' * 100}")
    for results, spacer in [(parsed_results[:10], "..."), (parsed_results[-10:], "")]:
        for t_float, t_int, rel_diff, descriptions in results:
            time_str = [f"{rel_diff * 100:>4.1f}%    {'int' if t_int < t_float else 'float':<20}"]
            time_str.extend(["".ljust(len(time_str[0])) for _ in descriptions[:-1]])
            for t_str, (name, shape, sparse_dim, is_coalesced) in zip(time_str, descriptions, strict=True):
                name = f"{name}:".ljust(name_len + 1)
                shape = shape.ljust(shape_len + 10)
                sparse_dim = sparse_dim.ljust(sparse_dim_len)
                print(f"{t_str} {name}  {shape}|     {sparse_dim}      |   {is_coalesced}")
        print(spacer)


def main() -> None:
    run(n=100, stmt="torch.sparse.sum(x, dim=0)", fuzzer_cls=UnaryOpSparseFuzzer)
    run(n=100, stmt="torch.sparse.softmax(x, dim=0)", fuzzer_cls=UnaryOpSparseFuzzer)
    run(n=100, stmt="x + y", fuzzer_cls=BinaryOpSparseFuzzer)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Example use of Timer and sparse op fuzzers to measure kernel performance.$ python -m examples.sparse.op_benchmark

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `assert_dicts_equal`, `run`, `main`

**Key imports**: numpy as np, torch, Timer, UnaryOpSparseFuzzer, BinaryOpSparseFuzzer, operator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/benchmark/examples/sparse`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `torch`
- `torch.utils.benchmark`: Timer
- `torch.utils.benchmark.op_fuzzers.sparse_unary`: UnaryOpSparseFuzzer
- `torch.utils.benchmark.op_fuzzers.sparse_binary`: BinaryOpSparseFuzzer
- `operator`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/utils/benchmark/examples/sparse`):

- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`compare.py_docs.md`](./compare.py_docs.md)


## Cross-References

- **File Documentation**: `op_benchmark.py_docs.md`
- **Keyword Index**: `op_benchmark.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `benchmarks/operator_benchmark/pt/nan_to_num_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/nan_to_num_test.py`
- **Size**: 1,744 bytes (1.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import math

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.nan_to_num / nan_to_num_ operators"""

# Configs for PT torch.nan_to_num / nan_to_num_ operators

nan_to_num_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["nan_to_num", torch.nan_to_num],
        ["nan_to_num_", torch.nan_to_num_],
    ],
)

nan_to_num_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128],
    N=range(32, 128, 32),
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["long"],
)


nan_to_num_short_configs = op_bench.cross_product_configs(
    M=[16, 64],
    N=[64, 64],
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["short"],
)


class ReplaceNaNBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dtype, replace_inf, op_func):
        input = torch.randn(M, N, dtype=dtype)
        input[0][0] = float("nan")
        self.inputs = {"input": input, "replace_inf": replace_inf}
        self.op_func = op_func
        self.set_module_name("nan_to_num")

        #  To make casename unique as nan_to_num and nan_to_num_ are two different functions.
        if op_func is torch.nan_to_num_:
            self.set_module_name("nan_to_num_")

    def forward(self, input, replace_inf: bool):
        # compare inplace
        if replace_inf:
            return self.op_func(input, nan=1.0)
        else:
            return self.op_func(input, nan=1.0, posinf=math.inf, neginf=-math.inf)


op_bench.generate_pt_tests_from_op_list(
    nan_to_num_ops_list,
    nan_to_num_long_configs + nan_to_num_short_configs,
    ReplaceNaNBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for torch.nan_to_num / nan_to_num_ operators"""# Configs for PT torch.nan_to_num / nan_to_num_ operatorsnan_to_num_ops_list = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[        ["nan_to_num", torch.nan_to_num],        ["nan_to_num_", torch.nan_to_num_],    ],)nan_to_num_long_configs = op_bench.cross_product_configs(    M=[32, 64, 128],    N=range(32, 128, 32),    dtype=[torch.float, torch.double],    replace_inf=[True, False],    tags=["long"],)nan_to_num_short_configs = op_bench.cross_product_configs(    M=[16, 64],    N=[64, 64],    dtype=[torch.float, torch.double],    replace_inf=[True, False],    tags=["short"],)class ReplaceNaNBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, dtype, replace_inf, op_func):        input = torch.randn(M, N, dtype=dtype)        input[0][0] = float("nan")        self.inputs = {"input": input, "replace_inf": replace_inf}        self.op_func = op_func        self.set_module_name("nan_to_num")        #  To make casename unique as nan_to_num and nan_to_num_ are two different functions.        if op_func is torch.nan_to_num_:            self.set_module_name("nan_to_num_")    def forward(self, input, replace_inf: bool):

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ReplaceNaNBenchmark`

**Functions defined**: `init`, `forward`

**Key imports**: math, operator_benchmark as op_bench, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `operator_benchmark as op_bench`
- `torch`


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

This is a test file. Run it with:

```bash
python benchmarks/operator_benchmark/pt/nan_to_num_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/operator_benchmark/pt`):

- [`qarithmetic_test.py_docs.md`](./qarithmetic_test.py_docs.md)
- [`bmm_test.py_docs.md`](./bmm_test.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gather_test.py_docs.md`](./gather_test.py_docs.md)
- [`clip_ranges_test.py_docs.md`](./clip_ranges_test.py_docs.md)
- [`split_test.py_docs.md`](./split_test.py_docs.md)
- [`groupnorm_test.py_docs.md`](./groupnorm_test.py_docs.md)
- [`sum_test.py_docs.md`](./sum_test.py_docs.md)
- [`matrix_mult_test.py_docs.md`](./matrix_mult_test.py_docs.md)
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `nan_to_num_test.py_docs.md`
- **Keyword Index**: `nan_to_num_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

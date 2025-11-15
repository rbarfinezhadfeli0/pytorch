# Documentation: `benchmarks/operator_benchmark/pt/binary_inplace_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/binary_inplace_test.py`
- **Size**: 3,303 bytes (3.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for inplace binary operators."""


def add_(in1, in2):
    return in1.add_(in2)


def sub_(in1, in2):
    return in1.sub_(in2)


def div_(in1, in2):
    return in1.div_(in2)


def mul_(in1, in2):
    return in1.mul_(in2)


def copy_(in1, in2):
    return in1.copy_(in2)


######
# Benchmark ops performance for inplace add + sub + mul + copy
######
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add_", add_],
        ["sub_", sub_],
        # ["div_",  div_ ], # done separately below because of data type
        ["mul_", mul_],
        ["copy_", copy_],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [torch.int32],
        "dtype_two": [torch.int32],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.int8, torch.int32],
    dtype_two=[torch.int8, torch.int32],
    tags=["long"],
)


class InpBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, InpBinaryOpBenchmark
)


######
# Benchmark ops performance for inplace div
######
# Performing division inplace benchmarks separately, as data needs to be float
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["div_", div_],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [torch.float],
        "dtype_two": [torch.float],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.float, torch.float],
    dtype_two=[torch.float, torch.float],
    tags=["long"],
)


class InpBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, InpBinaryOpBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for inplace binary operators."""def add_(in1, in2):    return in1.add_(in2)def sub_(in1, in2):    return in1.sub_(in2)def div_(in1, in2):    return in1.div_(in2)def mul_(in1, in2):    return in1.mul_(in2)def copy_(in1, in2):    return in1.copy_(in2)####### Benchmark ops performance for inplace add + sub + mul + copy######binary_ops_list = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[        ["add_", add_],        ["sub_", sub_],        # ["div_",  div_ ], # done separately below because of data type        ["mul_", mul_],        ["copy_", copy_],    ],)binary_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={

This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `InpBinaryOpBenchmark`, `InpBinaryOpBenchmark`

**Functions defined**: `add_`, `sub_`, `div_`, `mul_`, `copy_`, `init`, `forward`, `init`, `forward`

**Key imports**: operator_benchmark as op_bench, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`
- `torch`


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

This is a test file. Run it with:

```bash
python benchmarks/operator_benchmark/pt/binary_inplace_test.py
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

- **File Documentation**: `binary_inplace_test.py_docs.md`
- **Keyword Index**: `binary_inplace_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

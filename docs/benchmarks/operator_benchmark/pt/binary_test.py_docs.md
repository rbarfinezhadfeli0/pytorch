# Documentation: `benchmarks/operator_benchmark/pt/binary_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/binary_test.py`
- **Size**: 5,417 bytes (5.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
    ],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16, torch.float64],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
        ["asr", torch.bitwise_right_shift],
        ["lsl", torch.bitwise_left_shift],
        ["xor", torch.bitwise_xor],
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
        "dtype_one": [torch.int32, torch.uint8],
        "dtype_two": [torch.int32, torch.uint8],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.int8, torch.int32, torch.uint8],
    dtype_two=[torch.int8, torch.int32, torch.uint8],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


######
# Benchmark ops performance for boolean dtype
######


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.bool],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, in_one, in_two, dtype, device, op_func):
        self.inputs = {
            "in_one": torch.bernoulli(0.5 * torch.ones(in_one, device=device)).to(
                dtype=dtype
            ),
            "in_two": torch.bernoulli(0.5 * torch.ones(in_two, device=device)).to(
                dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
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
        "dtype_one": [torch.bool],
        "dtype_two": [torch.bool],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.bool, torch.bool],
    dtype_two=[torch.bool, torch.bool],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        self.inputs = {
            "input_one": torch.bernoulli(0.5 * torch.ones(M, N, K, device=device)).to(
                dtype=dtype_one
            ),
            "input_two": torch.bernoulli(0.5 * torch.ones(M, N, K, device=device)).to(
                dtype=dtype_two
            ),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for binary operators."""# Benchmark ops performance with broadcastbinary_ops_bcast_list = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[        ["add", torch.add],        ["sub", torch.sub],        ["div", torch.div],        ["mul", torch.mul],    ],)# Configs with broadcastbinary_configs_broadcast = op_bench.config_list(    attr_names=["in_one", "in_two"],    attrs=[        [[64, 1, 64], [1, 64, 1]],    ],    cross_product_configs={        "device": ["cpu"],        "dtype": [torch.float, torch.bfloat16, torch.float64],    },    tags=["short"],)class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):    def init(self, in_one, in_two, dtype, device, op_func):        self.inputs = {            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),        }        self.op_func = op_func    def forward(self, in_one, in_two):        return self.op_func(in_one, in_two)op_bench.generate_pt_tests_from_op_list(    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark)

This Python file contains 4 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BinaryOpBcastBenchmark`, `BinaryOpBenchmark`, `BinaryOpBcastBenchmark`, `BinaryOpBenchmark`

**Functions defined**: `init`, `forward`, `init`, `forward`, `init`, `forward`, `init`, `forward`

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
python benchmarks/operator_benchmark/pt/binary_test.py
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

- **File Documentation**: `binary_test.py_docs.md`
- **Keyword Index**: `binary_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `benchmarks/operator_benchmark/pt/remainder_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/remainder_test.py`
- **Size**: 1,654 bytes (1.62 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for remainder operators."""


# Benchmark ops performance with broadcast
remainder_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["fmod", torch.fmod],
        ["remainder", torch.remainder],
    ],
)

remainder_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.int32, torch.float, torch.double],
    },
    tags=["short"],
)

remainder_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype=[torch.int32, torch.float, torch.double],
    tags=["long"],
)


class RemainderOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype, op_func):
        self.dividend = torch.rand(M, N, K, device=device)
        self.dividend = (self.dividend * 1000 - 500).to(dtype=dtype)

        self.divisor = torch.rand(M, N, K, device=device)
        # +1 so we don't divide by zero
        self.divisor = (self.divisor * 40 + 1).to(dtype=dtype)

        self.inputs = {"dividend": self.dividend, "divisor": self.divisor}

        self.op_func = op_func

    def forward(self, dividend, divisor):
        return self.op_func(dividend, divisor)


op_bench.generate_pt_tests_from_op_list(
    remainder_ops_list,
    remainder_short_configs + remainder_long_configs,
    RemainderOpBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for remainder operators."""# Benchmark ops performance with broadcastremainder_ops_list = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[        ["fmod", torch.fmod],        ["remainder", torch.remainder],    ],)remainder_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={        "device": ["cpu", "cuda"],        "dtype": [torch.int32, torch.float, torch.double],    },    tags=["short"],)remainder_long_configs = op_bench.cross_product_configs(    M=[8, 128],    N=[32, 64],    K=[256, 512],    device=["cpu", "cuda"],    dtype=[torch.int32, torch.float, torch.double],    tags=["long"],)class RemainderOpBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, device, dtype, op_func):        self.dividend = torch.rand(M, N, K, device=device)        self.dividend = (self.dividend * 1000 - 500).to(dtype=dtype)        self.divisor = torch.rand(M, N, K, device=device)        # +1 so we don't divide by zero        self.divisor = (self.divisor * 40 + 1).to(dtype=dtype)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RemainderOpBenchmark`

**Functions defined**: `init`, `forward`

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
python benchmarks/operator_benchmark/pt/remainder_test.py
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

- **File Documentation**: `remainder_test.py_docs.md`
- **Keyword Index**: `remainder_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

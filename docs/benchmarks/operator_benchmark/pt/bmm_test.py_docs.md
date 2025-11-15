# Documentation: `benchmarks/operator_benchmark/pt/bmm_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/bmm_test.py`
- **Size**: 2,892 bytes (2.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for batched operators."""

# binary ops (two inputs in shape of batches)
batched_binary_ops = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["bmm", torch.bmm],
    ],
)

batched_binary_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [2, 1, 8, 2],
        [128, 64, 32, 64],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16],
    },
    tags=["short"],
)

batched_binary_configs_long = op_bench.cross_product_configs(
    B=[8, 32],
    M=[256, 1024],
    N=[256, 1024],
    K=[64, 128],
    device=["cuda"],
    dtype=[torch.float32, torch.bfloat16, torch.float16],
    tags=["long"],
)


class BatchedBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype, op_func):
        self.inputs = {
            "batch1": torch.rand(
                (B, M, N), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (B, N, K), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
        }
        self.op_func = op_func

    def forward(self, batch1, batch2):
        return self.op_func(batch1, batch2)


op_bench.generate_pt_tests_from_op_list(
    batched_binary_ops,
    batched_binary_configs_short + batched_binary_configs_long,
    BatchedBinaryOpBenchmark,
)
op_bench.generate_pt_gradient_tests_from_op_list(
    batched_binary_ops,
    batched_binary_configs_long,
    BatchedBinaryOpBenchmark,
)


# batched ternary ops
batched_ternary_ops = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["baddbmm", torch.baddbmm]],
)


class BatchedTernaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype, op_func):
        self.inputs = {
            "input_": torch.rand(
                (B, M, K), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
            "batch1": torch.rand(
                (B, M, N), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (B, N, K), device=device, dtype=dtype, requires_grad=self.auto_set()
            ),
        }
        self.op_func = op_func

    def forward(self, input_, batch1, batch2):
        return self.op_func(input_, batch1, batch2)


op_bench.generate_pt_tests_from_op_list(
    batched_ternary_ops,
    batched_binary_configs_short + batched_binary_configs_long,
    BatchedTernaryOpBenchmark,
)
op_bench.generate_pt_gradient_tests_from_op_list(
    batched_ternary_ops,
    batched_binary_configs_long,
    BatchedTernaryOpBenchmark,
)


# TODO: does it automatically register new scripts?

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for batched operators."""# binary ops (two inputs in shape of batches)batched_binary_ops = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[        ["bmm", torch.bmm],    ],)batched_binary_configs_short = op_bench.config_list(    attr_names=["B", "M", "N", "K"],    attrs=[        [2, 1, 8, 2],        [128, 64, 32, 64],    ],    cross_product_configs={        "device": ["cpu"],        "dtype": [torch.float, torch.bfloat16],    },    tags=["short"],)batched_binary_configs_long = op_bench.cross_product_configs(    B=[8, 32],    M=[256, 1024],    N=[256, 1024],    K=[64, 128],    device=["cuda"],    dtype=[torch.float32, torch.bfloat16, torch.float16],    tags=["long"],)class BatchedBinaryOpBenchmark(op_bench.TorchBenchmarkBase):    def init(self, B, M, N, K, device, dtype, op_func):        self.inputs = {            "batch1": torch.rand(                (B, M, N), device=device, dtype=dtype, requires_grad=self.auto_set()            ),            "batch2": torch.rand(                (B, N, K), device=device, dtype=dtype, requires_grad=self.auto_set()            ),        }        self.op_func = op_func

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BatchedBinaryOpBenchmark`, `BatchedTernaryOpBenchmark`

**Functions defined**: `init`, `forward`, `init`, `forward`

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
python benchmarks/operator_benchmark/pt/bmm_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/operator_benchmark/pt`):

- [`qarithmetic_test.py_docs.md`](./qarithmetic_test.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gather_test.py_docs.md`](./gather_test.py_docs.md)
- [`clip_ranges_test.py_docs.md`](./clip_ranges_test.py_docs.md)
- [`split_test.py_docs.md`](./split_test.py_docs.md)
- [`groupnorm_test.py_docs.md`](./groupnorm_test.py_docs.md)
- [`sum_test.py_docs.md`](./sum_test.py_docs.md)
- [`matrix_mult_test.py_docs.md`](./matrix_mult_test.py_docs.md)
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `bmm_test.py_docs.md`
- **Keyword Index**: `bmm_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

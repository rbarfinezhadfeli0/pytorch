# Documentation: `benchmarks/operator_benchmark/pt/matrix_mult_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/matrix_mult_test.py`
- **Size**: 2,980 bytes (2.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""
Microbenchmarks for batch matrix mult with einsum and torch.bmm.
"""

batch_mm_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [4, 5, 3, 2],
        [32, 25, 20, 30],
        [128, 100, 120, 110],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

batch_mm_configs_long = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [128, 256, 128, 256],
        [512, 1024, 1024, 512],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)

batch_mm_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_bmm", torch.einsum],
        # ["bmm", torch.bmm],
    ],
)


class BatchMatrixMultBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, op_func):
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, N, K, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bjk->bik", input_one, input_two)
        else:
            return torch.bmm(input_one, input_two)


"""
Microbenchmarks for element-wise matrix mult with einsum and torch.mul.
"""

batch_elementwise_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N"],
    attrs=[
        [4, 5, 3],
        [32, 25, 20],
        [100, 90, 110],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


batch_elementwise_configs_long = op_bench.cross_product_configs(
    B=[128, 512, 1024],
    M=[128, 512, 1024],
    N=[128, 512, 1024],
    device=["cpu", "cuda"],
    tags=["long"],
)

batch_elementwise_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_elementwise", torch.einsum],
        ["mul", torch.mul],
    ],
)


class BatchElementWiseBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, device, op_func):
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, M, N, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bij->bij", input_one, input_two)
        else:
            return torch.mul(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    batch_mm_op_list,
    batch_mm_configs_short + batch_mm_configs_long,
    BatchMatrixMultBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    batch_elementwise_op_list,
    batch_elementwise_configs_short + batch_elementwise_configs_long,
    BatchElementWiseBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for batch matrix mult with einsum and torch.bmm.

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BatchMatrixMultBenchmark`, `BatchElementWiseBenchmark`

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
python benchmarks/operator_benchmark/pt/matrix_mult_test.py
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
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `matrix_mult_test.py_docs.md`
- **Keyword Index**: `matrix_mult_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `benchmarks/operator_benchmark/pt/matmul_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/matmul_test.py`
- **Size**: 2,005 bytes (1.96 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for MatMul operator"""

# Configs for PT Matmul operator
mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "trans_a", "trans_b"],
    attrs=[
        [1, 1, 1, True, False],
        [128, 128, 128, True, False],
        [256, 256, 256, False, True],
    ],
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["short"],
)


mm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, trans_a, trans_b, device, dtype=torch.float):
        # Create tensors without requires_grad first, then set it separately
        # This avoids creating graph leaves that cannot be deep copied
        if trans_a:
            input_one = torch.rand(M, N, device=device, dtype=dtype)
        else:
            input_one = torch.rand(N, M, device=device, dtype=dtype).t()

        if trans_b:
            input_two = torch.rand(N, K, device=device, dtype=dtype)
        else:
            input_two = torch.rand(K, N, device=device, dtype=dtype).t()

        # Set requires_grad after tensor creation to avoid graph leaf issues
        if self.auto_set():
            input_one.requires_grad_(True)
        if self.auto_set():
            input_two.requires_grad_(True)

        self.inputs = {
            "input_one": input_one,
            "input_two": input_two,
        }
        self.set_module_name("matmul")

    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)


op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)
op_bench.generate_pt_gradient_test(mm_long_configs, MatMulBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for MatMul operator"""# Configs for PT Matmul operatormm_short_configs = op_bench.config_list(    attr_names=["M", "N", "K", "trans_a", "trans_b"],    attrs=[        [1, 1, 1, True, False],        [128, 128, 128, True, False],        [256, 256, 256, False, True],    ],    cross_product_configs={"device": ["cpu", "cuda"]},    tags=["short"],)mm_long_configs = op_bench.cross_product_configs(    M=[256, 1024, 3000],    N=[512, 4096],    K=[512, 4096],    trans_a=[False, True],    trans_b=[True, False],    device=["cuda"],    dtype=[torch.float16, torch.bfloat16, torch.float32],    tags=["long"],)class MatMulBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, trans_a, trans_b, device, dtype=torch.float):        # Create tensors without requires_grad first, then set it separately        # This avoids creating graph leaves that cannot be deep copied        if trans_a:            input_one = torch.rand(M, N, device=device, dtype=dtype)        else:            input_one = torch.rand(N, M, device=device, dtype=dtype).t()        if trans_b:            input_two = torch.rand(N, K, device=device, dtype=dtype)        else:            input_two = torch.rand(K, N, device=device, dtype=dtype).t()        # Set requires_grad after tensor creation to avoid graph leaf issues        if self.auto_set():            input_one.requires_grad_(True)        if self.auto_set():

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MatMulBenchmark`

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
python benchmarks/operator_benchmark/pt/matmul_test.py
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

- **File Documentation**: `matmul_test.py_docs.md`
- **Keyword Index**: `matmul_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

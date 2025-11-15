# Documentation: `benchmarks/operator_benchmark/pt/addmm_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/addmm_test.py`
- **Size**: 3,049 bytes (2.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for add_(matmul) operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator
addmm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    device=["cuda"],
    tags=["long"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
)


addmm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float],
    },
    tags=["short"],
)


"""Mircobenchmark for addmm operator."""


class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                M, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "mat1": torch.rand(
                M, N, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "mat2": torch.rand(
                N, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("addmm")

    def forward(self, input_one, mat1, mat2):
        return torch.addmm(input_one, mat1, mat2)


op_bench.generate_pt_test(addmm_short_configs + addmm_long_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(addmm_long_configs, AddmmBenchmark)

"""Mircobenchmark for addbmm operator."""


class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
                dtype=dtype,
            ),
        }
        self.set_module_name("addbmm")

    def forward(self, input_one, batch1, batch2):
        return torch.addbmm(input_one, batch1, batch2)


addbmm_long_configs = op_bench.cross_product_configs(
    B=[8, 32],
    M=[256, 1024],
    N=[256, 1024],
    K=[64, 128],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)
addbmm_short_configs = op_bench.cross_product_configs(
    B=[1, 8],
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["short"],
)

op_bench.generate_pt_test(addbmm_long_configs + addbmm_short_configs, AddbmmBenchmark)
op_bench.generate_pt_gradient_test(addbmm_long_configs, AddbmmBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for add_(matmul) operator. Supports both Caffe2/PyTorch."""# Configs for PT add operatoraddmm_long_configs = op_bench.cross_product_configs(    M=[256, 1024, 3000],    N=[512, 4096],    K=[512, 4096],    device=["cuda"],    tags=["long"],    dtype=[torch.float16, torch.bfloat16, torch.float32],)addmm_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={        "device": ["cpu", "cuda"],        "dtype": [torch.float],    },    tags=["short"],)

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AddmmBenchmark`, `AddbmmBenchmark`

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
python benchmarks/operator_benchmark/pt/addmm_test.py
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

- **File Documentation**: `addmm_test.py_docs.md`
- **Keyword Index**: `addmm_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

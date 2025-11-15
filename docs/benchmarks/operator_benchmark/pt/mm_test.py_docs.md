# Documentation: `benchmarks/operator_benchmark/pt/mm_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/mm_test.py`
- **Size**: 1,505 bytes (1.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.mm."""

# Benchmark ops performance without broadcast
ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["mm", torch.mm]],
)

mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)

mm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)


class MmOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype, op_func):
        self.inputs = {
            "input_one": torch.randn(
                M, N, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "input_two": torch.randn(
                N, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    ops_list, mm_short_configs + mm_long_configs, MmOpBenchmark
)
op_bench.generate_pt_gradient_tests_from_op_list(
    ops_list, mm_long_configs, MmOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for torch.mm."""# Benchmark ops performance without broadcastops_list = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[["mm", torch.mm]],)mm_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},    tags=["short"],)mm_long_configs = op_bench.cross_product_configs(    M=[256, 1024, 3000],    N=[512, 4096],    K=[512, 4096],    device=["cuda"],    dtype=[torch.float16, torch.bfloat16, torch.float32],    tags=["long"],)class MmOpBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, device, dtype, op_func):        self.inputs = {            "input_one": torch.randn(                M, N, device=device, requires_grad=self.auto_set(), dtype=dtype            ),            "input_two": torch.randn(                N, K, device=device, requires_grad=self.auto_set(), dtype=dtype            ),        }        self.op_func = op_func    def forward(self, input_one, input_two):        return self.op_func(input_one, input_two)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MmOpBenchmark`

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
python benchmarks/operator_benchmark/pt/mm_test.py
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

- **File Documentation**: `mm_test.py_docs.md`
- **Keyword Index**: `mm_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

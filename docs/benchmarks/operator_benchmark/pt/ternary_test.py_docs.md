# Documentation: `benchmarks/operator_benchmark/pt/ternary_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/ternary_test.py`
- **Size**: 1,391 bytes (1.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for ternary operators."""


ternary_ops = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["addcmul", torch.addcmul],
        ["addcdiv", torch.addcdiv],
    ],
)

ternary_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [1, 2],
        [32, 64],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16],
    },
    tags=["short"],
)

ternary_configs_long = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    device=["cpu", "cuda"],
    dtype=[torch.float, torch.bfloat16],
    tags=["long"],
)


class TernaryOpBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype, op_func):
        self.inputs = {
            "input_": torch.rand((M, N), device=device).to(dtype=dtype),
            "tensor1": torch.rand((M, N), device=device).to(dtype=dtype),
            "tensor2": torch.rand((M, N), device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, input_, tensor1, tensor2):
        return self.op_func(input_, tensor1, tensor2)


op_bench.generate_pt_tests_from_op_list(
    ternary_ops, ternary_configs_short + ternary_configs_long, TernaryOpBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for ternary operators."""ternary_ops = op_bench.op_list(    attr_names=["op_name", "op_func"],    attrs=[        ["addcmul", torch.addcmul],        ["addcdiv", torch.addcdiv],    ],)ternary_configs_short = op_bench.config_list(    attr_names=["M", "N"],    attrs=[        [1, 2],        [32, 64],    ],    cross_product_configs={        "device": ["cpu"],        "dtype": [torch.float, torch.bfloat16],    },    tags=["short"],)ternary_configs_long = op_bench.cross_product_configs(    M=[8, 128],    N=[32, 64],    device=["cpu", "cuda"],    dtype=[torch.float, torch.bfloat16],    tags=["long"],)class TernaryOpBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, device, dtype, op_func):        self.inputs = {            "input_": torch.rand((M, N), device=device).to(dtype=dtype),            "tensor1": torch.rand((M, N), device=device).to(dtype=dtype),            "tensor2": torch.rand((M, N), device=device).to(dtype=dtype),        }        self.op_func = op_func    def forward(self, input_, tensor1, tensor2):        return self.op_func(input_, tensor1, tensor2)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TernaryOpBenchmark`

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
python benchmarks/operator_benchmark/pt/ternary_test.py
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

- **File Documentation**: `ternary_test.py_docs.md`
- **Keyword Index**: `ternary_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

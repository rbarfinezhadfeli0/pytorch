# Documentation: `benchmarks/operator_benchmark/pt/where_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/where_test.py`
- **Size**: 1,403 bytes (1.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for where operator."""


configs_short = op_bench.config_list(
    attr_names=["cond_shape", "input_shape", "other_shape"],
    attrs=[
        [(8, 16, 1), (1,), (1,)],
        [(8, 16, 1), (16, 1), (8, 16, 1)],
        [(8, 16, 1), (8, 1, 1), (1,)],
    ],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)


configs_long = op_bench.cross_product_configs(
    cond_shape=[(64, 16, 1), (64, 16, 8), (1024, 64, 16, 128)],
    input_shape=[(1,), (16, 1), (64, 16, 1)],
    other_shape=[(1,), (16, 1), (64, 16, 1)],
    device=["cpu", "cuda"],
    dtype=[torch.float],
    tags=["long"],
)


class WhereBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, cond_shape, input_shape, other_shape, dtype, device):
        def _create_tensor(shape):
            return torch.randn(*shape, dtype=dtype, device=device)

        self.inputs = {
            "condition": _create_tensor(cond_shape) > 0,
            "input": _create_tensor(input_shape),
            "other": _create_tensor(other_shape),
        }
        self.set_module_name("where")

    def forward(self, condition, input, other):
        return torch.where(condition, input, other)


op_bench.generate_pt_test(configs_short + configs_long, WhereBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for where operator."""configs_short = op_bench.config_list(    attr_names=["cond_shape", "input_shape", "other_shape"],    attrs=[        [(8, 16, 1), (1,), (1,)],        [(8, 16, 1), (16, 1), (8, 16, 1)],        [(8, 16, 1), (8, 1, 1), (1,)],    ],    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},    tags=["short"],)configs_long = op_bench.cross_product_configs(    cond_shape=[(64, 16, 1), (64, 16, 8), (1024, 64, 16, 128)],    input_shape=[(1,), (16, 1), (64, 16, 1)],    other_shape=[(1,), (16, 1), (64, 16, 1)],    device=["cpu", "cuda"],    dtype=[torch.float],    tags=["long"],)class WhereBenchmark(op_bench.TorchBenchmarkBase):    def init(self, cond_shape, input_shape, other_shape, dtype, device):        def _create_tensor(shape):            return torch.randn(*shape, dtype=dtype, device=device)        self.inputs = {            "condition": _create_tensor(cond_shape) > 0,            "input": _create_tensor(input_shape),            "other": _create_tensor(other_shape),        }        self.set_module_name("where")    def forward(self, condition, input, other):        return torch.where(condition, input, other)op_bench.generate_pt_test(configs_short + configs_long, WhereBenchmark)if __name__ == "__main__":

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WhereBenchmark`

**Functions defined**: `init`, `_create_tensor`, `forward`

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
python benchmarks/operator_benchmark/pt/where_test.py
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

- **File Documentation**: `where_test.py_docs.md`
- **Keyword Index**: `where_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

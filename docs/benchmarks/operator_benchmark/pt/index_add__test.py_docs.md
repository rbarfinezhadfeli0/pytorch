# Documentation: `benchmarks/operator_benchmark/pt/index_add__test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/index_add__test.py`
- **Size**: 1,589 bytes (1.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for index_add_ operator."""


configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "dim"],
    attrs=[[8, 32, 1, 0], [256, 512, 1, 1], [512, 512, 1, 2]],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)


configs_long = op_bench.cross_product_configs(
    M=[1, 128, 1024],
    N=[2, 256, 512],
    K=[1, 2, 8],
    dim=[0, 1, 2],
    device=["cpu", "cuda"],
    dtype=[torch.float],
    tags=["long"],
)


class IndexAddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dim, dtype, device):
        # creating the original tensor
        tensor = torch.rand(M, N, K, dtype=dtype, device=device)

        # creating index
        index_max_len = tensor.shape[dim]
        index_len = numpy.random.randint(1, index_max_len + 1)
        index = torch.tensor(
            numpy.random.choice(index_max_len, index_len, replace=False), device=device
        )

        src_dims = [M, N, K]
        src_dims[dim] = index_len
        source = torch.rand(*src_dims, dtype=dtype, device=device)

        self.inputs = {
            "tensor": tensor,
            "dim": dim,
            "index": index,
            "source": source,
        }
        self.set_module_name("index_add_")

    def forward(self, tensor, dim, index, source):
        return tensor.index_add_(dim, index, source)


op_bench.generate_pt_test(configs_short + configs_long, IndexAddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for index_add_ operator."""configs_short = op_bench.config_list(    attr_names=["M", "N", "K", "dim"],    attrs=[[8, 32, 1, 0], [256, 512, 1, 1], [512, 512, 1, 2]],    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},    tags=["short"],)configs_long = op_bench.cross_product_configs(    M=[1, 128, 1024],    N=[2, 256, 512],    K=[1, 2, 8],    dim=[0, 1, 2],    device=["cpu", "cuda"],    dtype=[torch.float],    tags=["long"],)class IndexAddBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, dim, dtype, device):        # creating the original tensor        tensor = torch.rand(M, N, K, dtype=dtype, device=device)        # creating index        index_max_len = tensor.shape[dim]        index_len = numpy.random.randint(1, index_max_len + 1)        index = torch.tensor(            numpy.random.choice(index_max_len, index_len, replace=False), device=device        )        src_dims = [M, N, K]        src_dims[dim] = index_len        source = torch.rand(*src_dims, dtype=dtype, device=device)        self.inputs = {            "tensor": tensor,            "dim": dim,            "index": index,            "source": source,

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IndexAddBenchmark`

**Functions defined**: `init`, `forward`

**Key imports**: numpy, operator_benchmark as op_bench, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `numpy`
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
python benchmarks/operator_benchmark/pt/index_add__test.py
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

- **File Documentation**: `index_add__test.py_docs.md`
- **Keyword Index**: `index_add__test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

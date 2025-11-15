# Documentation: `benchmarks/operator_benchmark/pt/gather_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/gather_test.py`
- **Size**: 1,261 bytes (1.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for gather operator."""

# An example input from this configuration is M=4, N=4, dim=0.
gather_configs_short = op_bench.config_list(
    attr_names=["M", "N", "dim"],
    attrs=[
        [256, 512, 0],
        [512, 512, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


gather_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], dim=[0, 1], device=["cpu", "cuda"], tags=["long"]
)


class GatherBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, dim, device):
        min_val = M if dim == 0 else N
        numpy.random.seed((1 << 32) - 1)
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, min_val, (M, N)), device=device
            ),
        }
        self.set_module_name("gather")

    def forward(self, input_one, dim: int, index):
        return torch.gather(input_one, dim, index)


op_bench.generate_pt_test(gather_configs_short + gather_configs_long, GatherBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for gather operator."""# An example input from this configuration is M=4, N=4, dim=0.gather_configs_short = op_bench.config_list(    attr_names=["M", "N", "dim"],    attrs=[        [256, 512, 0],        [512, 512, 1],    ],    cross_product_configs={        "device": ["cpu", "cuda"],    },    tags=["short"],)gather_configs_long = op_bench.cross_product_configs(    M=[128, 1024], N=[128, 1024], dim=[0, 1], device=["cpu", "cuda"], tags=["long"])class GatherBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, dim, device):        min_val = M if dim == 0 else N        numpy.random.seed((1 << 32) - 1)        self.inputs = {            "input_one": torch.rand(M, N, device=device),            "dim": dim,            "index": torch.tensor(                numpy.random.randint(0, min_val, (M, N)), device=device            ),        }        self.set_module_name("gather")    def forward(self, input_one, dim: int, index):        return torch.gather(input_one, dim, index)op_bench.generate_pt_test(gather_configs_short + gather_configs_long, GatherBenchmark)if __name__ == "__main__":    op_bench.benchmark_runner.main()

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GatherBenchmark`

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
python benchmarks/operator_benchmark/pt/gather_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/operator_benchmark/pt`):

- [`qarithmetic_test.py_docs.md`](./qarithmetic_test.py_docs.md)
- [`bmm_test.py_docs.md`](./bmm_test.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`clip_ranges_test.py_docs.md`](./clip_ranges_test.py_docs.md)
- [`split_test.py_docs.md`](./split_test.py_docs.md)
- [`groupnorm_test.py_docs.md`](./groupnorm_test.py_docs.md)
- [`sum_test.py_docs.md`](./sum_test.py_docs.md)
- [`matrix_mult_test.py_docs.md`](./matrix_mult_test.py_docs.md)
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `gather_test.py_docs.md`
- **Keyword Index**: `gather_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

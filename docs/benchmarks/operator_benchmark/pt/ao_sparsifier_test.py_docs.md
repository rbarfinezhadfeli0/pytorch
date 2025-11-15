# Documentation: `benchmarks/operator_benchmark/pt/ao_sparsifier_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/ao_sparsifier_test.py`
- **Size**: 1,530 bytes (1.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch
from torch import nn
from torch.ao import pruning


"""Microbenchmarks for sparsifier."""

sparse_configs_short = op_bench.config_list(
    attr_names=["M", "SL", "SBS", "ZPB"],
    attrs=[
        [(32, 16), 0.3, (4, 1), 2],
        [(32, 16), 0.6, (1, 4), 4],
        [(17, 23), 0.9, (1, 1), 1],
    ],
    tags=("short",),
)

sparse_configs_long = op_bench.cross_product_configs(
    M=((128, 128), (255, 324)),  # Mask shape
    SL=(0.0, 1.0, 0.3, 0.6, 0.9, 0.99),  # Sparsity level
    SBS=((1, 4), (1, 8), (4, 1), (8, 1)),  # Sparse block shape
    ZPB=(0, 1, 2, 3, 4, None),  # Zeros per block
    tags=("long",),
)


class WeightNormSparsifierBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, SL, SBS, ZPB):
        weight = torch.ones(M)
        model = nn.Module()
        model.register_buffer("weight", weight)

        sparse_config = [{"tensor_fqn": "weight"}]
        self.sparsifier = pruning.WeightNormSparsifier(
            sparsity_level=SL,
            sparse_block_shape=SBS,
            zeros_per_block=ZPB,
        )
        self.sparsifier.prepare(model, config=sparse_config)
        self.inputs = {}  # All benchmarks need inputs :)
        self.set_module_name("weight_norm_sparsifier_step")

    def forward(self):
        self.sparsifier.step()


all_tests = sparse_configs_short + sparse_configs_long
op_bench.generate_pt_test(all_tests, WeightNormSparsifierBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for sparsifier."""sparse_configs_short = op_bench.config_list(    attr_names=["M", "SL", "SBS", "ZPB"],    attrs=[        [(32, 16), 0.3, (4, 1), 2],        [(32, 16), 0.6, (1, 4), 4],        [(17, 23), 0.9, (1, 1), 1],    ],    tags=("short",),)sparse_configs_long = op_bench.cross_product_configs(    M=((128, 128), (255, 324)),  # Mask shape    SL=(0.0, 1.0, 0.3, 0.6, 0.9, 0.99),  # Sparsity level    SBS=((1, 4), (1, 8), (4, 1), (8, 1)),  # Sparse block shape    ZPB=(0, 1, 2, 3, 4, None),  # Zeros per block    tags=("long",),)class WeightNormSparsifierBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, SL, SBS, ZPB):        weight = torch.ones(M)        model = nn.Module()        model.register_buffer("weight", weight)        sparse_config = [{"tensor_fqn": "weight"}]        self.sparsifier = pruning.WeightNormSparsifier(            sparsity_level=SL,            sparse_block_shape=SBS,            zeros_per_block=ZPB,        )        self.sparsifier.prepare(model, config=sparse_config)        self.inputs = {}  # All benchmarks need inputs :)        self.set_module_name("weight_norm_sparsifier_step")    def forward(self):        self.sparsifier.step()all_tests = sparse_configs_short + sparse_configs_longop_bench.generate_pt_test(all_tests, WeightNormSparsifierBenchmark)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `WeightNormSparsifierBenchmark`

**Functions defined**: `init`, `forward`

**Key imports**: operator_benchmark as op_bench, torch, nn, pruning


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`
- `torch`
- `torch.ao`: pruning


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python benchmarks/operator_benchmark/pt/ao_sparsifier_test.py
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

- **File Documentation**: `ao_sparsifier_test.py_docs.md`
- **Keyword Index**: `ao_sparsifier_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

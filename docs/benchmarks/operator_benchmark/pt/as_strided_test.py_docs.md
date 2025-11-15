# Documentation: `benchmarks/operator_benchmark/pt/as_strided_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/as_strided_test.py`
- **Size**: 1,406 bytes (1.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for as_strided operator"""


# Configs for PT as_strided operator
as_strided_configs_short = op_bench.config_list(
    attr_names=["M", "N", "size", "stride", "storage_offset"],
    attrs=[
        [8, 8, (2, 2), (1, 1), 0],
        [256, 256, (32, 32), (1, 1), 0],
        [512, 512, (64, 64), (2, 2), 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

as_strided_configs_long = op_bench.cross_product_configs(
    M=[512],
    N=[1024],
    size=[(16, 16), (128, 128)],
    stride=[(1, 1)],
    storage_offset=[0, 1],
    device=["cpu", "cuda"],
    tags=["long"],
)


class As_stridedBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, size, stride, storage_offset, device):
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "size": size,
            "stride": stride,
            "storage_offset": storage_offset,
        }
        self.set_module_name("as_strided")

    def forward(
        self, input_one, size: list[int], stride: list[int], storage_offset: int
    ):
        return torch.as_strided(input_one, size, stride, storage_offset)


op_bench.generate_pt_test(
    as_strided_configs_short + as_strided_configs_long, As_stridedBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for as_strided operator"""# Configs for PT as_strided operatoras_strided_configs_short = op_bench.config_list(    attr_names=["M", "N", "size", "stride", "storage_offset"],    attrs=[        [8, 8, (2, 2), (1, 1), 0],        [256, 256, (32, 32), (1, 1), 0],        [512, 512, (64, 64), (2, 2), 1],    ],    cross_product_configs={        "device": ["cpu", "cuda"],    },    tags=["short"],)as_strided_configs_long = op_bench.cross_product_configs(    M=[512],    N=[1024],    size=[(16, 16), (128, 128)],    stride=[(1, 1)],    storage_offset=[0, 1],    device=["cpu", "cuda"],    tags=["long"],)class As_stridedBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, size, stride, storage_offset, device):        self.inputs = {            "input_one": torch.rand(M, N, device=device),            "size": size,            "stride": stride,            "storage_offset": storage_offset,        }        self.set_module_name("as_strided")    def forward(        self, input_one, size: list[int], stride: list[int], storage_offset: int    ):        return torch.as_strided(input_one, size, stride, storage_offset)op_bench.generate_pt_test(

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `As_stridedBenchmark`

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
python benchmarks/operator_benchmark/pt/as_strided_test.py
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

- **File Documentation**: `as_strided_test.py_docs.md`
- **Keyword Index**: `as_strided_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

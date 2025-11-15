# Documentation: `benchmarks/operator_benchmark/pt/clip_ranges_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/clip_ranges_test.py`
- **Size**: 1,412 bytes (1.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for ClipRanges operator."""
torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")

# Configs for C2 ClipRanges operator
clip_ranges_long_configs = op_bench.cross_product_configs(
    LENGTH=range(1, 100),
    M=[1],
    N=[2],
    MAX_LENGTH=range(1, 100),
    device=["cpu", "cuda"],
    dtype=[torch.int32],
    tags=["long"],
)


clip_ranges_short_configs = op_bench.config_list(
    attrs=[
        [6, 1, 2, 1, torch.int32],
        [7, 1, 2, 2, torch.int32],
        [8, 1, 2, 3, torch.int32],
        [9, 1, 2, 4, torch.int32],
        [10, 1, 2, 5, torch.int32],
    ],
    attr_names=["LENGTH", "M", "N", "MAX_LENGTH", "dtype"],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class ClipRangesBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, LENGTH, M, N, MAX_LENGTH, device, dtype):
        self.inputs = {
            "input": torch.rand(LENGTH, M, N, device=device).type(dtype),
            "max_length": MAX_LENGTH,
        }
        self.set_module_name("clip_ranges")

    def forward(self, input, max_length: int):
        return torch.ops.fb.clip_ranges(input, max_length)


op_bench.generate_pt_test(
    clip_ranges_long_configs + clip_ranges_short_configs, ClipRangesBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for ClipRanges operator."""torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")# Configs for C2 ClipRanges operatorclip_ranges_long_configs = op_bench.cross_product_configs(    LENGTH=range(1, 100),    M=[1],    N=[2],    MAX_LENGTH=range(1, 100),    device=["cpu", "cuda"],    dtype=[torch.int32],    tags=["long"],)clip_ranges_short_configs = op_bench.config_list(    attrs=[        [6, 1, 2, 1, torch.int32],        [7, 1, 2, 2, torch.int32],        [8, 1, 2, 3, torch.int32],        [9, 1, 2, 4, torch.int32],        [10, 1, 2, 5, torch.int32],    ],    attr_names=["LENGTH", "M", "N", "MAX_LENGTH", "dtype"],    cross_product_configs={        "device": ["cpu", "cuda"],    },    tags=["short"],)class ClipRangesBenchmark(op_bench.TorchBenchmarkBase):    def init(self, LENGTH, M, N, MAX_LENGTH, device, dtype):        self.inputs = {            "input": torch.rand(LENGTH, M, N, device=device).type(dtype),            "max_length": MAX_LENGTH,        }        self.set_module_name("clip_ranges")    def forward(self, input, max_length: int):        return torch.ops.fb.clip_ranges(input, max_length)op_bench.generate_pt_test(    clip_ranges_long_configs + clip_ranges_short_configs, ClipRangesBenchmark

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ClipRangesBenchmark`

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
python benchmarks/operator_benchmark/pt/clip_ranges_test.py
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
- [`split_test.py_docs.md`](./split_test.py_docs.md)
- [`groupnorm_test.py_docs.md`](./groupnorm_test.py_docs.md)
- [`sum_test.py_docs.md`](./sum_test.py_docs.md)
- [`matrix_mult_test.py_docs.md`](./matrix_mult_test.py_docs.md)
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `clip_ranges_test.py_docs.md`
- **Keyword Index**: `clip_ranges_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

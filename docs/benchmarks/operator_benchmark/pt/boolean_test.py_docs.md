# Documentation: `benchmarks/operator_benchmark/pt/boolean_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/boolean_test.py`
- **Size**: 1,805 bytes (1.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for boolean operators. Supports both Caffe2/PyTorch."""

# Configs for PT all operator
all_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
)


all_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class AllBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.randint(0, 2, (M, N, K), device=device, dtype=torch.bool)
        }
        self.set_module_name("all")

    def forward(self, input_one):
        return torch.all(input_one)


# The generated test names based on all_short_configs will be in the following pattern:
# all_M8_N16_K32_devicecpu
# all_M8_N16_K32_devicecpu_bwdall
# all_M8_N16_K32_devicecpu_bwd1
# all_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(all_long_configs + all_short_configs, AllBenchmark)

"""Mircobenchmark for any operator."""


class AnyBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device):
        self.inputs = {
            "input_one": torch.randint(0, 2, (M, N), device=device, dtype=torch.bool)
        }
        self.set_module_name("any")

    def forward(self, input_one):
        return torch.any(input_one)


any_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    tags=["any"],
)

op_bench.generate_pt_test(any_configs, AnyBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for boolean operators. Supports both Caffe2/PyTorch."""# Configs for PT all operatorall_long_configs = op_bench.cross_product_configs(    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"])all_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={        "device": ["cpu", "cuda"],    },    tags=["short"],)class AllBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, device):        self.inputs = {            "input_one": torch.randint(0, 2, (M, N, K), device=device, dtype=torch.bool)        }        self.set_module_name("all")    def forward(self, input_one):        return torch.all(input_one)# The generated test names based on all_short_configs will be in the following pattern:# all_M8_N16_K32_devicecpu# all_M8_N16_K32_devicecpu_bwdall# all_M8_N16_K32_devicecpu_bwd1# all_M8_N16_K32_devicecpu_bwd2# ...# Those names can be used to filter tests.op_bench.generate_pt_test(all_long_configs + all_short_configs, AllBenchmark)

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AllBenchmark`, `AnyBenchmark`

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
python benchmarks/operator_benchmark/pt/boolean_test.py
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

- **File Documentation**: `boolean_test.py_docs.md`
- **Keyword Index**: `boolean_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

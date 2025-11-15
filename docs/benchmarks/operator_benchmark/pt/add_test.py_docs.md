# Documentation: `benchmarks/operator_benchmark/pt/add_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/add_test.py`
- **Size**: 2,508 bytes (2.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator
add_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
)


add_short_configs = op_bench.config_list(
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


class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "input_two": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
        }
        self.set_module_name("add")

    def forward(self, input_one, input_two):
        return torch.add(input_one, input_two)


# The generated test names based on add_short_configs will be in the following pattern:
# add_M8_N16_K32_devicecpu
# add_M8_N16_K32_devicecpu_bwdall
# add_M8_N16_K32_devicecpu_bwd1
# add_M8_N16_K32_devicecpu_bwd2
# ...
# Those names can be used to filter tests.

op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddBenchmark)

"""Mircobenchmark for addr operator."""


class AddrBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec1": torch.rand(
                (M,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec2": torch.rand(
                (N,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("addr")

    def forward(self, input_one, vec1, vec2):
        return torch.addr(input_one, vec1, vec2)


addr_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    dtype=[torch.double, torch.half],
    tags=["addr"],
)

op_bench.generate_pt_test(addr_configs, AddrBenchmark)
op_bench.generate_pt_gradient_test(addr_configs, AddrBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""# Configs for PT add operatoradd_long_configs = op_bench.cross_product_configs(    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"])add_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={        "device": ["cpu", "cuda"],    },    tags=["short"],)class AddBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, device):        self.inputs = {            "input_one": torch.rand(                M, N, K, device=device, requires_grad=self.auto_set()            ),            "input_two": torch.rand(                M, N, K, device=device, requires_grad=self.auto_set()            ),        }        self.set_module_name("add")    def forward(self, input_one, input_two):        return torch.add(input_one, input_two)# The generated test names based on add_short_configs will be in the following pattern:# add_M8_N16_K32_devicecpu# add_M8_N16_K32_devicecpu_bwdall# add_M8_N16_K32_devicecpu_bwd1# add_M8_N16_K32_devicecpu_bwd2# ...# Those names can be used to filter tests.

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AddBenchmark`, `AddrBenchmark`

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
python benchmarks/operator_benchmark/pt/add_test.py
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

- **File Documentation**: `add_test.py_docs.md`
- **Keyword Index**: `add_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

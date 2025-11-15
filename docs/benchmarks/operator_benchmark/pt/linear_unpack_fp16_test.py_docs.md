# Documentation: `benchmarks/operator_benchmark/pt/linear_unpack_fp16_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/linear_unpack_fp16_test.py`
- **Size**: 1,508 bytes (1.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for linear_unpack_fp16_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT linear_unpack_fp16 operator
linear_unpack_fp16_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu"], tags=["long"]
)

linear_unpack_fp16_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu"],
    },
    tags=["short"],
)


class LinearUnpackFP16Benchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        # input to unpack operator must be what the output is for prepack operator
        self.inputs = {
            "input_one": torch.ops.quantized.linear_prepack_fp16(
                torch.rand(
                    M, N, K, device=device, requires_grad=False, dtype=torch.float32
                )
            )
        }
        self.set_module_name("linear_unpack_fp16")

    def forward(self, input_one):
        return torch.ops.quantized.linear_unpack_fp16(input_one)


# The generated test names based on linear_unpack_fp16_short_configs will be in the following pattern:
# linear_unpack_fp16_M8_N16_K32_devicecpu

op_bench.generate_pt_test(
    linear_unpack_fp16_long_configs + linear_unpack_fp16_short_configs,
    LinearUnpackFP16Benchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for linear_unpack_fp16_ operator. Supports both Caffe2/PyTorch."""# Configs for PT linear_unpack_fp16 operatorlinear_unpack_fp16_long_configs = op_bench.cross_product_configs(    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu"], tags=["long"])linear_unpack_fp16_short_configs = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 1, 1],        [64, 64, 64],        [64, 64, 128],    ],    cross_product_configs={        "device": ["cpu"],    },    tags=["short"],)class LinearUnpackFP16Benchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, device):        # input to unpack operator must be what the output is for prepack operator        self.inputs = {            "input_one": torch.ops.quantized.linear_prepack_fp16(                torch.rand(                    M, N, K, device=device, requires_grad=False, dtype=torch.float32                )            )        }        self.set_module_name("linear_unpack_fp16")    def forward(self, input_one):        return torch.ops.quantized.linear_unpack_fp16(input_one)# The generated test names based on linear_unpack_fp16_short_configs will be in the following pattern:# linear_unpack_fp16_M8_N16_K32_devicecpuop_bench.generate_pt_test(    linear_unpack_fp16_long_configs + linear_unpack_fp16_short_configs,    LinearUnpackFP16Benchmark,)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearUnpackFP16Benchmark`

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
python benchmarks/operator_benchmark/pt/linear_unpack_fp16_test.py
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

- **File Documentation**: `linear_unpack_fp16_test.py_docs.md`
- **Keyword Index**: `linear_unpack_fp16_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `benchmarks/operator_benchmark/pt/qinstancenorm_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qinstancenorm_test.py`
- **Size**: 1,337 bytes (1.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    dtype=(torch.qint8,),
    tags=["short"],
)


class QInstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dims, dtype):
        X = (torch.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0

        self.inputs = {
            "qX": torch.quantize_per_tensor(
                X, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):
        return torch.ops.quantized.instance_norm(
            qX,
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )


op_bench.generate_pt_test(instancenorm_configs_short, QInstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for quantized instancenorm operator."""instancenorm_configs_short = op_bench.cross_product_configs(    dims=(        (32, 8, 16),        (32, 8, 56, 56),    ),    dtype=(torch.qint8,),    tags=["short"],)class QInstanceNormBenchmark(op_bench.TorchBenchmarkBase):    def init(self, dims, dtype):        X = (torch.rand(*dims) - 0.5) * 256        num_channels = dims[1]        scale = 1.0        zero_point = 0        self.inputs = {            "qX": torch.quantize_per_tensor(                X, scale=scale, zero_point=zero_point, dtype=dtype            ),            "weight": torch.rand(num_channels, dtype=torch.float),            "bias": torch.rand(num_channels, dtype=torch.float),            "eps": 1e-5,            "Y_scale": 0.1,            "Y_zero_point": 0,        }    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):        return torch.ops.quantized.instance_norm(            qX,            weight=weight,            bias=bias,            eps=eps,            output_scale=Y_scale,            output_zero_point=Y_zero_point,        )op_bench.generate_pt_test(instancenorm_configs_short, QInstanceNormBenchmark)if __name__ == "__main__":

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QInstanceNormBenchmark`

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
python benchmarks/operator_benchmark/pt/qinstancenorm_test.py
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

- **File Documentation**: `qinstancenorm_test.py_docs.md`
- **Keyword Index**: `qinstancenorm_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

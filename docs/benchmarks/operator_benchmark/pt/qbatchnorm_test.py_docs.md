# Documentation: `benchmarks/operator_benchmark/pt/qbatchnorm_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qbatchnorm_test.py`
- **Size**: 2,513 bytes (2.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized batchnorm operator."""

batchnorm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 256, 3136],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": (torch.qint8,),
    },
    tags=["short"],
)


class QBatchNormBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype):
        self._init(M, N, K, device)
        x_scale = 0.1
        x_zero_point = 0
        self.inputs = {
            "q_input_one": torch.quantize_per_tensor(
                self.input_one, scale=x_scale, zero_point=x_zero_point, dtype=dtype
            ),
            "mean": torch.rand(N),
            "var": torch.rand(N),
            "weight": torch.rand(N),
            "bias": torch.rand(N),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def _init(self, M, N, K, device):
        pass

    def forward(self):
        pass


class QBatchNorm1dBenchmark(QBatchNormBenchmark):
    def _init(self, M, N, K, device):
        self.set_module_name("QBatchNorm1d")
        self.input_one = torch.rand(
            M, N, K, device=device, requires_grad=self.auto_set()
        )

    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.batch_norm1d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )


class QBatchNorm2dBenchmark(QBatchNormBenchmark):
    def _init(self, M, N, K, device):
        self.set_module_name("QBatchNorm2d")
        # Note: quantized implementation requires rank 4, which is why we
        # add a 1 as the last dimension
        self.input_one = torch.rand(
            M, N, K, 1, device=device, requires_grad=self.auto_set()
        )

    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        return torch.ops.quantized.batch_norm2d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )


op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm1dBenchmark)
op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm2dBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for quantized batchnorm operator."""batchnorm_configs_short = op_bench.config_list(    attr_names=["M", "N", "K"],    attrs=[        [1, 256, 3136],    ],    cross_product_configs={        "device": ["cpu"],        "dtype": (torch.qint8,),    },    tags=["short"],)class QBatchNormBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, device, dtype):        self._init(M, N, K, device)        x_scale = 0.1        x_zero_point = 0        self.inputs = {            "q_input_one": torch.quantize_per_tensor(                self.input_one, scale=x_scale, zero_point=x_zero_point, dtype=dtype            ),            "mean": torch.rand(N),            "var": torch.rand(N),            "weight": torch.rand(N),            "bias": torch.rand(N),            "eps": 1e-5,            "Y_scale": 0.1,            "Y_zero_point": 0,        }    def _init(self, M, N, K, device):        pass    def forward(self):        passclass QBatchNorm1dBenchmark(QBatchNormBenchmark):    def _init(self, M, N, K, device):        self.set_module_name("QBatchNorm1d")        self.input_one = torch.rand(            M, N, K, device=device, requires_grad=self.auto_set()

This Python file contains 3 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QBatchNormBenchmark`, `QBatchNorm1dBenchmark`, `QBatchNorm2dBenchmark`

**Functions defined**: `init`, `_init`, `forward`, `_init`, `forward`, `_init`, `forward`

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
python benchmarks/operator_benchmark/pt/qbatchnorm_test.py
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

- **File Documentation**: `qbatchnorm_test.py_docs.md`
- **Keyword Index**: `qbatchnorm_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

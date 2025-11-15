# Documentation: `benchmarks/operator_benchmark/pt/qconv_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qconv_test.py`
- **Size**: 2,807 bytes (2.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq


"""
Microbenchmarks for qConv operators.
"""


class QConv1dBenchmark(op_bench.TorchBenchmarkBase):
    # def init(self, N, IC, OC, L, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, L, device):
        G = 1
        pad = 0
        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(N, IC, L, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, IC // G, kernel, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        self.inputs = {"input": qX}

        self.qconv1d = nnq.Conv1d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv1d.set_weight_bias(self.qW, None)
        self.qconv1d.scale = torch.tensor(self.scale, dtype=torch.double)
        self.qconv1d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        self.set_module_name("QConv1d")

    def forward(self, input):
        return self.qconv1d(input)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    # def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        # super().init(N, IC, OC, (H, W), G, (kernel, kernel), stride, pad)

        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        self.inputs = {"input": qX}

        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d.set_weight_bias(self.qW, None)
        self.qconv2d.scale = torch.tensor(self.scale, dtype=torch.double)
        self.qconv2d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        self.set_module_name("QConv2d")

    def forward(self, input):
        return self.qconv2d(input)


op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    QConv1dBenchmark,
)
op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    QConv2dBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for qConv operators.

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QConv1dBenchmark`, `QConv2dBenchmark`

**Functions defined**: `init`, `init`, `forward`, `init`, `init`, `forward`

**Key imports**: configs, operator_benchmark as op_bench, torch, torch.ao.nn.quantized as nnq


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `pt`: configs
- `operator_benchmark as op_bench`
- `torch`
- `torch.ao.nn.quantized as nnq`


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
python benchmarks/operator_benchmark/pt/qconv_test.py
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

- **File Documentation**: `qconv_test.py_docs.md`
- **Keyword Index**: `qconv_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

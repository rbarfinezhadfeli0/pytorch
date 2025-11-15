# Documentation: `benchmarks/operator_benchmark/pt/qarithmetic_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qarithmetic_test.py`
- **Size**: 2,569 bytes (2.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch
from torch._ops import ops


qarithmetic_binary_configs = op_bench.cross_product_configs(
    N=(2, 8, 64, 512),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    contig=(False, True),
    tags=("short",),
)


qarithmetic_binary_ops = op_bench.op_list(
    attrs=(
        ("add", ops.quantized.add),
        ("add_relu", ops.quantized.add_relu),
        ("mul", ops.quantized.mul),
    ),
    attr_names=("op_name", "op_func"),
)

qarithmetic_binary_scalar_ops = op_bench.op_list(
    attrs=(
        ("add_scalar", ops.quantized.add_scalar),
        ("mul_scalar", ops.quantized.mul_scalar),
    ),
    attr_names=("op_name", "op_func"),
)


class _QFunctionalBinaryArithmeticBenchmarkBase(op_bench.TorchBenchmarkBase):
    def setup(self, N, dtype, contig):
        self.qfunctional = torch.ao.nn.quantized.QFunctional()

        # TODO: Consider more diverse shapes
        f_input = (torch.rand(N, N) - 0.5) * 256
        self.scale = 1.0
        self.zero_point = 0
        self.q_input_a = torch.quantize_per_tensor(
            f_input, scale=self.scale, zero_point=self.zero_point, dtype=dtype
        )

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            self.q_input_a = self.q_input_a.permute(permute_dims)


class QFunctionalBenchmark(_QFunctionalBinaryArithmeticBenchmarkBase):
    def init(self, N, dtype, contig, op_func):
        super().setup(N, dtype, contig)
        self.inputs = {
            "q_input_a": self.q_input_a,
            "q_input_b": self.q_input_a,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }
        self.op_func = op_func

    def forward(self, q_input_a, q_input_b, scale: float, zero_point: int):
        return self.op_func(q_input_a, q_input_b, scale=scale, zero_point=zero_point)


op_bench.generate_pt_tests_from_op_list(
    qarithmetic_binary_ops, qarithmetic_binary_configs, QFunctionalBenchmark
)


class QFunctionalScalarBenchmark(_QFunctionalBinaryArithmeticBenchmarkBase):
    def init(self, N, dtype, contig, op_func):
        super().setup(N, dtype, contig)
        self.inputs = {"q_input": self.q_input_a, "scalar_input": 42}
        self.op_func = op_func

    def forward(self, q_input, scalar_input: int):
        return self.op_func(q_input, scalar_input)


op_bench.generate_pt_tests_from_op_list(
    qarithmetic_binary_scalar_ops,
    qarithmetic_binary_configs,
    QFunctionalScalarBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview


This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_QFunctionalBinaryArithmeticBenchmarkBase`, `QFunctionalBenchmark`, `QFunctionalScalarBenchmark`

**Functions defined**: `setup`, `init`, `forward`, `init`, `forward`

**Key imports**: operator_benchmark as op_bench, torch, ops


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`
- `torch`
- `torch._ops`: ops


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
python benchmarks/operator_benchmark/pt/qarithmetic_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/operator_benchmark/pt`):

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

- **File Documentation**: `qarithmetic_test.py_docs.md`
- **Keyword Index**: `qarithmetic_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

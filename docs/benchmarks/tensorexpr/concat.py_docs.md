# Documentation: `benchmarks/tensorexpr/concat.py`

## File Metadata

- **Path**: `benchmarks/tensorexpr/concat.py`
- **Size**: 4,203 bytes (4.10 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import numpy as np

import torch

from . import benchmark


class Concat2D2InputBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1
        self.I1_D2 = I1_D2
        self.I2_D1 = I2_D1
        self.I2_D2 = I2_D2
        self.concat_dim = concat_dim
        self.input1 = self.randn(
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]

    def forward(self, input1, input2):
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        y = self.cat((x1, x2), dim=self.concat_dim)
        return y

    def reference(self):
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    @staticmethod
    def module():
        return "concat2d2input"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [
            [1, 160, 1, 14, 1],
            [1, 580, 1, 174, 1],
            [20, 160, 20, 14, 1],
            [20, 580, 20, 174, 1],
            [8, 512, 8, 512, 1],
            [1 << 13, 1060, 1 << 13, 1040, 1],
            [1 << 13, 2000, 1 << 13, 1074, 1],
            [1 << 15, 1060, 1 << 15, 2670, 1],
            [1 << 15, 5120, 1 << 15, 2512, 1],
        ]


benchmark.register_benchmark_class(Concat2D2InputBench)


class ConcatGraphOptBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1
        self.I1_D2 = I1_D2
        self.I2_D1 = I2_D1
        self.I2_D2 = I2_D2
        self.concat_dim = concat_dim
        self.input1 = self.randn(
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_cat_wo_conditionals(True)

    def forward(self, input1, input2):
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        y = self.cat((x1, x2), dim=self.concat_dim)
        z = self.relu(y)
        return z

    def reference(self):
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    @staticmethod
    def module():
        return "concatGraphOpt"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [
            [1 << 13, 1060, 1 << 13, 1040, 1],
            [1 << 13, 2000, 1 << 13, 1074, 1],
            [1 << 15, 1060, 1 << 15, 2670, 1],
            [1 << 15, 5120, 1 << 15, 2512, 1],
        ]


benchmark.register_benchmark_class(ConcatGraphOptBench)

```



## High-Level Overview


This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Concat2D2InputBench`, `ConcatGraphOptBench`

**Functions defined**: `__init__`, `forward`, `reference`, `config`, `module`, `memory_workload`, `default_configs`, `__init__`, `forward`, `reference`, `config`, `module`, `memory_workload`, `default_configs`

**Key imports**: numpy as np, torch, benchmark


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/tensorexpr`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `torch`
- `.`: benchmark


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/tensorexpr`):

- [`microbenchmarks.py_docs.md`](./microbenchmarks.py_docs.md)
- [`HowToRun.md_docs.md`](./HowToRun.md_docs.md)
- [`rnn_eltwise.py_docs.md`](./rnn_eltwise.py_docs.md)
- [`matmul.py_docs.md`](./matmul.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`reduction.py_docs.md`](./reduction.py_docs.md)
- [`broadcast.py_docs.md`](./broadcast.py_docs.md)
- [`swish.py_docs.md`](./swish.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `concat.py_docs.md`
- **Keyword Index**: `concat.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

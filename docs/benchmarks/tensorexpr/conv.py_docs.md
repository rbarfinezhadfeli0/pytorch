# Documentation: `benchmarks/tensorexpr/conv.py`

## File Metadata

- **Path**: `benchmarks/tensorexpr/conv.py`
- **Size**: 2,945 bytes (2.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
from . import benchmark


class ConvImplBench(benchmark.Benchmark):
    def __init__(self, case, mode, device, dtype, kernel_size, N, iC, H, W, oC):
        super().__init__(mode, device, dtype)
        self.case = case
        self.kernel_size = kernel_size
        self.N = N
        self.iC = iC
        self.H = H
        self.W = W
        self.oC = oC
        self.data = self.rand(
            [N, iC, H, W], device=device, requires_grad=self.requires_grad
        )
        if case == "conv":
            self.groups = 1
        elif case == "depthwise_conv":
            self.groups = iC
        else:
            raise ValueError(f"invalid case: {case}")

        self.conv = self.conv2d_layer(iC, oC, kernel_size, groups=self.groups)
        if device != "cpu":
            self.to_device(self.conv, device)

    def forward(self):
        y = self.conv(self.data)
        return y

    def config(self):
        return [self.kernel_size, self.N, self.iC, self.H, self.W, self.oC]

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = {"i": 1, "o": 1, "k": 1}
            algorithmic_count = {"i": 1, "o": 1, "k": 1}
        else:
            sol_count = {"i": 1 + 1, "o": 1 + 1, "k": 1 + 1}
            algorithmic_count = {"i": 1 + (1 + 1), "o": 1 + (1 + 1), "k": 1 + (1 + 1)}

        buffer_size = {
            "i": self.N * self.iC * self.H * self.W,
            "o": self.N * self.oC * self.H * self.W,
            "k": self.oC
            * (self.iC / self.groups)
            * self.kernel_size
            * self.kernel_size,
        }
        sol_size = 0
        algorithmic_size = 0
        for key in sol_count:
            sol_size += buffer_size[key] * sol_count[key]
            algorithmic_size += buffer_size[key] * algorithmic_count[key]
        return {"sol": sol_size, "algorithmic": algorithmic_size}

    def compute_workload(self):
        if self.mode == "fwd":
            count = 1
        elif self.mode == "both":
            count = 1 + (1 + 1)
        else:
            raise ValueError(f"invalid mode: {self.mode}")

        op_count = (
            self.N
            * self.iC
            / self.groups
            * self.oC
            * self.kernel_size
            * self.kernel_size
            * self.H
            * self.W
        )
        op_count *= 2

        return op_count * count

    @staticmethod
    def default_configs():
        return [
            [3, 64, 32, 128, 128, 64],
        ]


class ConvBench(ConvImplBench):
    def __init__(self, *args):
        super().__init__("conv", *args)

    @staticmethod
    def module():
        return "conv"


class DepthwiseConvBench(ConvImplBench):
    def __init__(self, *args):
        super().__init__("depthwise_conv", *args)

    @staticmethod
    def module():
        return "depthwise_conv"


benchmark.register_benchmark_class(ConvBench)
benchmark.register_benchmark_class(DepthwiseConvBench)

```



## High-Level Overview


This Python file contains 3 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConvImplBench`, `ConvBench`, `DepthwiseConvBench`

**Functions defined**: `__init__`, `forward`, `config`, `memory_workload`, `compute_workload`, `default_configs`, `__init__`, `module`, `__init__`, `module`

**Key imports**: benchmark


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/tensorexpr`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `.`: benchmark


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/tensorexpr`):

- [`microbenchmarks.py_docs.md`](./microbenchmarks.py_docs.md)
- [`HowToRun.md_docs.md`](./HowToRun.md_docs.md)
- [`rnn_eltwise.py_docs.md`](./rnn_eltwise.py_docs.md)
- [`concat.py_docs.md`](./concat.py_docs.md)
- [`matmul.py_docs.md`](./matmul.py_docs.md)
- [`reduction.py_docs.md`](./reduction.py_docs.md)
- [`broadcast.py_docs.md`](./broadcast.py_docs.md)
- [`swish.py_docs.md`](./swish.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `conv.py_docs.md`
- **Keyword Index**: `conv.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

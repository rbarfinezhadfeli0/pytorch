# Documentation: `benchmarks/tensorexpr/rnn_eltwise.py`

## File Metadata

- **Path**: `benchmarks/tensorexpr/rnn_eltwise.py`
- **Size**: 3,411 bytes (3.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
import torch

from . import benchmark


class RNNEltwise(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, b, hs):
        super().__init__(mode, device, dtype)
        self.b = b
        self.hs = hs
        self.input = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.hx = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.cx = self.rand(
            [b, hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.b_ih = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.b_hh = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    def forward(self, input, hx, cx, b_ih, b_hh):
        gates = input + hx + b_ih + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def config(self):
        return [self.b, self.hs]

    @staticmethod
    def module():
        return "rnn_eltwise"

    def memory_workload(self):
        def memsize(t):
            return t.numel() * t.element_size()

        input_size = sum(memsize(t) for t in self.inputs)
        output_size = 2 * memsize(self.cx)
        io_size = input_size + output_size
        return {"sol": io_size, "algorithmic": io_size}

    @staticmethod
    def default_configs():
        return [[64, 512]]


benchmark.register_benchmark_class(RNNEltwise)


class DynamicLSTM(benchmark.DynamicShape, RNNEltwise):
    def __init__(self, mode, device, dtype, b, hs):
        benchmark.DynamicShape.__init__(self)
        RNNEltwise.__init__(self, mode, device, dtype, b, hs)

    def instantiate_input(self):
        b, hs = self.rand_shape([self.b, self.hs])

        self.input = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.hx = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.cx = self.rand(
            [b, hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.b_ih = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.b_hh = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    @staticmethod
    def module():
        return "dynamic_lstm"


benchmark.register_benchmark_class(DynamicLSTM)

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RNNEltwise`, `DynamicLSTM`

**Functions defined**: `__init__`, `forward`, `config`, `module`, `memory_workload`, `memsize`, `default_configs`, `__init__`, `instantiate_input`, `module`

**Key imports**: torch, benchmark


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/tensorexpr`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
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
- [`concat.py_docs.md`](./concat.py_docs.md)
- [`matmul.py_docs.md`](./matmul.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`reduction.py_docs.md`](./reduction.py_docs.md)
- [`broadcast.py_docs.md`](./broadcast.py_docs.md)
- [`swish.py_docs.md`](./swish.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `rnn_eltwise.py_docs.md`
- **Keyword Index**: `rnn_eltwise.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

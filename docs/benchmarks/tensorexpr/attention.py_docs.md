# Documentation: `benchmarks/tensorexpr/attention.py`

## File Metadata

- **Path**: `benchmarks/tensorexpr/attention.py`
- **Size**: 2,884 bytes (2.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
# This is a copy of rnn_attention from MLPerf, with some common sizes hardcoded
# for benchmarking and some control flow stripped out.
# https://github.com/mlcommons/training/blob/master/retired_benchmarks/gnmt/pytorch/seq2seq/models/attention.py

import torch

from . import benchmark


class BahdanauAttention(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, b, t_q, t_k, n):
        super().__init__(mode, device, dtype)
        self.b = b
        self.t_q = t_q
        self.t_k = t_k
        self.n = n
        self.att_query = self.rand(
            [b, t_q, n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.att_keys = self.rand(
            [b, t_k, n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.normalize_bias = self.rand(
            [n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.linear_att = self.rand(
            [n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.att_query,
            self.att_keys,
            self.normalize_bias,
            self.linear_att,
        ]

    def forward(self, att_query, att_keys, normalize_bias, linear_att):
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        return b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys + normalize_bias
        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def reference(self):
        return self.numpy(self.forward(*self.inputs))

    def config(self):
        return [self.b, self.t_q, self.t_k, self.n]

    @staticmethod
    def module():
        return "attention"

    def memory_workload(self):
        def memsize(t):
            return t.numel() * t.element_size()

        input_size = (
            memsize(self.att_query)
            + memsize(self.att_keys)
            + memsize(self.normalize_bias)
            + memsize(self.linear_att)
        )
        output_size = 4 * torch.Size([self.b, self.t_q, self.t_k]).numel()
        io_size = input_size + output_size

        # If matmul is not fused, must write and then read `sum_qk`.
        intermediate_size = (
            2 * 4 * torch.Size([self.b, self.t_q, self.t_k, self.n]).numel()
        )
        return {"sol": io_size, "algorithmic": io_size + intermediate_size}

    @staticmethod
    def default_configs():
        mlperf_inference = [1280, 1, 66, 1024]
        nvidia = [128, 10, 128, 1024]
        return [mlperf_inference, nvidia]


benchmark.register_benchmark_class(BahdanauAttention)

```



## High-Level Overview

"""        Calculate Bahdanau score        :param att_query: b x t_q x n        :param att_keys: b x t_k x n        return b x t_q x t_k scores

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BahdanauAttention`

**Functions defined**: `__init__`, `forward`, `reference`, `config`, `module`, `memory_workload`, `memsize`, `default_configs`

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
- [`rnn_eltwise.py_docs.md`](./rnn_eltwise.py_docs.md)
- [`concat.py_docs.md`](./concat.py_docs.md)
- [`matmul.py_docs.md`](./matmul.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`reduction.py_docs.md`](./reduction.py_docs.md)
- [`broadcast.py_docs.md`](./broadcast.py_docs.md)
- [`swish.py_docs.md`](./swish.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `attention.py_docs.md`
- **Keyword Index**: `attention.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

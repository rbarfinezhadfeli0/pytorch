# Documentation: `benchmarks/operator_benchmark/pt/qcat_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qcat_test.py`
- **Size**: 1,951 bytes (1.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq


"""Microbenchmarks for quantized Cat operator"""

# Configs for PT Cat operator
qcat_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "L", "dim"],
    attrs=[
        [256, 512, 1, 2, 0],
        [512, 512, 2, 1, 1],
    ],
    cross_product_configs={
        "contig": ("all", "one", "none"),
        "dtype": (torch.quint8, torch.qint8, torch.qint32),
    },
    tags=["short"],
)

qcat_configs_long = op_bench.cross_product_configs(
    M=[128, 1024],
    N=[128, 1024],
    K=[1, 2],
    L=[5, 7],
    dim=[0, 1, 2],
    contig=["all", "one", "none"],
    dtype=[torch.quint8],
    tags=["long"],
)


class QCatBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, L, dim, contig, dtype):
        f_input = (torch.rand(M, N, K) - 0.5) * 256
        self.qf = nnq.QFunctional()
        scale = 1.0
        zero_point = 0
        self.qf.scale = scale
        self.qf.zero_point = zero_point

        assert contig in ("none", "one", "all")
        q_input = torch.quantize_per_tensor(f_input, scale, zero_point, dtype)
        permute_dims = tuple(range(q_input.ndim - 1, -1, -1))
        q_input_non_contig = q_input.permute(permute_dims).contiguous()
        q_input_non_contig = q_input_non_contig.permute(permute_dims)
        if contig == "all":
            self.input = (q_input, q_input)
        elif contig == "one":
            self.input = (q_input, q_input_non_contig)
        elif contig == "none":
            self.input = (q_input_non_contig, q_input_non_contig)

        self.inputs = {"input": self.input, "dim": dim}
        self.set_module_name("qcat")

    def forward(self, input: list[torch.Tensor], dim: int):
        return self.qf.cat(input, dim=dim)


op_bench.generate_pt_test(qcat_configs_short + qcat_configs_long, QCatBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for quantized Cat operator"""# Configs for PT Cat operatorqcat_configs_short = op_bench.config_list(    attr_names=["M", "N", "K", "L", "dim"],    attrs=[        [256, 512, 1, 2, 0],        [512, 512, 2, 1, 1],    ],    cross_product_configs={        "contig": ("all", "one", "none"),        "dtype": (torch.quint8, torch.qint8, torch.qint32),    },    tags=["short"],)qcat_configs_long = op_bench.cross_product_configs(    M=[128, 1024],    N=[128, 1024],    K=[1, 2],    L=[5, 7],    dim=[0, 1, 2],    contig=["all", "one", "none"],    dtype=[torch.quint8],    tags=["long"],)class QCatBenchmark(op_bench.TorchBenchmarkBase):    def init(self, M, N, K, L, dim, contig, dtype):        f_input = (torch.rand(M, N, K) - 0.5) * 256        self.qf = nnq.QFunctional()        scale = 1.0        zero_point = 0        self.qf.scale = scale        self.qf.zero_point = zero_point        assert contig in ("none", "one", "all")        q_input = torch.quantize_per_tensor(f_input, scale, zero_point, dtype)        permute_dims = tuple(range(q_input.ndim - 1, -1, -1))        q_input_non_contig = q_input.permute(permute_dims).contiguous()        q_input_non_contig = q_input_non_contig.permute(permute_dims)        if contig == "all":            self.input = (q_input, q_input)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QCatBenchmark`

**Functions defined**: `init`, `forward`

**Key imports**: operator_benchmark as op_bench, torch, torch.ao.nn.quantized as nnq


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`
- `torch`
- `torch.ao.nn.quantized as nnq`


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
python benchmarks/operator_benchmark/pt/qcat_test.py
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

- **File Documentation**: `qcat_test.py_docs.md`
- **Keyword Index**: `qcat_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `benchmarks/operator_benchmark/pt/diag_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/diag_test.py`
- **Size**: 1,253 bytes (1.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for diag operator"""


# Configs for PT diag operator
diag_configs_short = op_bench.config_list(
    attr_names=["dim", "M", "N", "diagonal", "out"],
    attrs=[
        [1, 64, 64, 0, True],
        [2, 128, 128, -10, False],
        [1, 256, 256, 20, True],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


class DiagBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, dim, M, N, diagonal, out, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device)
            if dim == 2
            else torch.rand(M, device=device),
            "diagonal": diagonal,
            "out": out,
            "out_tensor": torch.tensor(
                (),
                device=device,
            ),
        }
        self.set_module_name("diag")

    def forward(self, input, diagonal: int, out: bool, out_tensor):
        if out:
            return torch.diag(input, diagonal=diagonal, out=out_tensor)
        else:
            return torch.diag(input, diagonal=diagonal)


op_bench.generate_pt_test(diag_configs_short, DiagBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for diag operator"""# Configs for PT diag operatordiag_configs_short = op_bench.config_list(    attr_names=["dim", "M", "N", "diagonal", "out"],    attrs=[        [1, 64, 64, 0, True],        [2, 128, 128, -10, False],        [1, 256, 256, 20, True],    ],    cross_product_configs={        "device": ["cpu", "cuda"],    },    tags=["short"],)class DiagBenchmark(op_bench.TorchBenchmarkBase):    def init(self, dim, M, N, diagonal, out, device):        self.inputs = {            "input": torch.rand(M, N, device=device)            if dim == 2            else torch.rand(M, device=device),            "diagonal": diagonal,            "out": out,            "out_tensor": torch.tensor(                (),                device=device,            ),        }        self.set_module_name("diag")    def forward(self, input, diagonal: int, out: bool, out_tensor):        if out:            return torch.diag(input, diagonal=diagonal, out=out_tensor)        else:            return torch.diag(input, diagonal=diagonal)op_bench.generate_pt_test(diag_configs_short, DiagBenchmark)if __name__ == "__main__":    op_bench.benchmark_runner.main()

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DiagBenchmark`

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
python benchmarks/operator_benchmark/pt/diag_test.py
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

- **File Documentation**: `diag_test.py_docs.md`
- **Keyword Index**: `diag_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

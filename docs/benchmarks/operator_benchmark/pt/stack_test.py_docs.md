# Documentation: `benchmarks/operator_benchmark/pt/stack_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/stack_test.py`
- **Size**: 2,675 bytes (2.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import random

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for Stack operator"""

# Configs for PT stack operator
stack_configs_static_runtime = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(20, 40), 5],
        [(1, 40), 5],
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(3))},
    tags=["static_runtime"],
)

stack_configs_short = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(1, 1, 1), 2],  # noqa: E241
        [(512, 512, 2), 2],  # noqa: E241
        [(128, 1024, 2), 2],  # noqa: E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},
    tags=["short"],
)

stack_configs_long = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(2**10, 2**10, 2), 2],  # noqa: E241
        [(2**10 + 1, 2**10 - 1, 2), 2],  # noqa: E226,E241
        [(2**10, 2**10, 2), 2],  # noqa: E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},
    tags=["long"],
)

# There is a different codepath on CUDA for >4 dimensions
stack_configs_multidim = op_bench.config_list(
    attr_names=["sizes", "N"],
    attrs=[
        [(2**6, 2**5, 2**2, 2**4, 2**5), 2],  # noqa: E241
        [(2**4, 2**5, 2**2, 2**4, 2**5), 8],  # noqa: E241
        [
            (2**3 + 1, 2**5 - 1, 2**2 + 1, 2**4 - 1, 2**5 + 1),
            17,
        ],  # noqa: E226,E241
    ],
    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(6))},
    tags=["multidim"],
)


class StackBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, sizes, N, dim, device):
        random.seed(42)
        inputs = []
        gen_sizes = []
        if type(sizes) is list and N == -1:
            gen_sizes = sizes
        else:
            for i in range(N):
                gen_sizes.append(
                    [
                        old_size() if callable(old_size) else old_size
                        for old_size in sizes
                    ]
                )

        for s in gen_sizes:
            inputs.append(torch.rand(s, device=device))
        result = torch.rand(gen_sizes[0], device=device)
        self.inputs = {"result": result, "inputs": inputs, "dim": dim}
        self.set_module_name("stack")

    def forward(self, result: torch.Tensor, inputs: list[torch.Tensor], dim: int):
        return torch.stack(inputs, dim=dim, out=result)


op_bench.generate_pt_test(
    stack_configs_static_runtime
    + stack_configs_short
    + stack_configs_long
    + stack_configs_multidim,
    StackBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for Stack operator"""# Configs for PT stack operatorstack_configs_static_runtime = op_bench.config_list(    attr_names=["sizes", "N"],    attrs=[        [(20, 40), 5],        [(1, 40), 5],    ],    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(3))},    tags=["static_runtime"],)stack_configs_short = op_bench.config_list(    attr_names=["sizes", "N"],    attrs=[        [(1, 1, 1), 2],  # noqa: E241        [(512, 512, 2), 2],  # noqa: E241        [(128, 1024, 2), 2],  # noqa: E241    ],    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},    tags=["short"],)stack_configs_long = op_bench.config_list(    attr_names=["sizes", "N"],    attrs=[        [(2**10, 2**10, 2), 2],  # noqa: E241        [(2**10 + 1, 2**10 - 1, 2), 2],  # noqa: E226,E241        [(2**10, 2**10, 2), 2],  # noqa: E241    ],    cross_product_configs={"device": ["cpu", "cuda"], "dim": list(range(4))},    tags=["long"],)# There is a different codepath on CUDA for >4 dimensionsstack_configs_multidim = op_bench.config_list(    attr_names=["sizes", "N"],    attrs=[        [(2**6, 2**5, 2**2, 2**4, 2**5), 2],  # noqa: E241        [(2**4, 2**5, 2**2, 2**4, 2**5), 8],  # noqa: E241        [            (2**3 + 1, 2**5 - 1, 2**2 + 1, 2**4 - 1, 2**5 + 1),

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StackBenchmark`

**Functions defined**: `init`, `forward`

**Key imports**: random, operator_benchmark as op_bench, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `random`
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
python benchmarks/operator_benchmark/pt/stack_test.py
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

- **File Documentation**: `stack_test.py_docs.md`
- **Keyword Index**: `stack_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

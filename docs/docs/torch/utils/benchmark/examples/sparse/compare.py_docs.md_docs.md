# Documentation: `docs/torch/utils/benchmark/examples/sparse/compare.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/benchmark/examples/sparse/compare.py_docs.md`
- **Size**: 6,380 bytes (6.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `torch/utils/benchmark/examples/sparse/compare.py`

## File Metadata

- **Path**: `torch/utils/benchmark/examples/sparse/compare.py`
- **Size**: 3,969 bytes (3.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""Example of Timer and Compare APIs:

$ python -m examples.sparse.compare
"""

import pickle
import sys
import time

import torch
import torch.utils.benchmark as benchmark_utils


class FauxTorch:
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element) -> None:
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    @property
    def sparse(self):
        return self.Sparse(self._real_torch, self._extra_ns_per_element)

    class Sparse:
        def __init__(self, real_torch, extra_ns_per_element) -> None:
            self._real_torch = real_torch
            self._extra_ns_per_element = extra_ns_per_element

        def extra_overhead(self, result):
            # time.sleep has a ~65 us overhead, so only fake a
            # per-element overhead if numel is large enough.
            size = sum(result.size())
            if size > 5000:
                time.sleep(size * self._extra_ns_per_element * 1e-9)
            return result

        def mm(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.sparse.mm(*args, **kwargs))

def generate_coo_data(size, sparse_dim, nnz, dtype, device):
    """
    Parameters
    ----------
    size : tuple
    sparse_dim : int
    nnz : int
    dtype : torch.dtype
    device : str
    Returns
    -------
    indices : torch.tensor
    values : torch.tensor
    """
    if dtype is None:
        dtype = 'float32'

    indices = torch.rand(sparse_dim, nnz, device=device)
    indices.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(indices))
    indices = indices.to(torch.long)
    # pyrefly: ignore [no-matching-overload]
    values = torch.rand([nnz, ], dtype=dtype, device=device)
    return indices, values

def gen_sparse(size, density, dtype, device='cpu'):
    sparse_dim = len(size)
    nnz = int(size[0] * size[1] * density)
    indices, values = generate_coo_data(size, sparse_dim, nnz, dtype, device)
    return torch.sparse_coo_tensor(indices, values, size, dtype=dtype, device=device)

def main() -> None:
    tasks = [
        ("matmul", "x @ y", "torch.sparse.mm(x, y)"),
        ("matmul", "x @ y + 0", "torch.sparse.mm(x, y) + zero"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": gen_sparse(size=size, density=density, dtype=torch.float32),
                "y": torch.rand(size, dtype=torch.float32),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 10)]
        for label, sub_label, stmt in tasks
        for density in [0.05, 0.1]
        for size in [(8, 8), (32, 32), (64, 64), (128, 128)]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Example of Timer and Compare APIs:$ python -m examples.sparse.compare

This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FauxTorch`, `Sparse`

**Functions defined**: `__init__`, `sparse`, `__init__`, `extra_overhead`, `mm`, `generate_coo_data`, `gen_sparse`, `main`

**Key imports**: pickle, sys, time, torch, torch.utils.benchmark as benchmark_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/benchmark/examples/sparse`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `pickle`
- `sys`
- `time`
- `torch`
- `torch.utils.benchmark as benchmark_utils`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/utils/benchmark/examples/sparse`):

- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`op_benchmark.py_docs.md`](./op_benchmark.py_docs.md)


## Cross-References

- **File Documentation**: `compare.py_docs.md`
- **Keyword Index**: `compare.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/utils/benchmark/examples/sparse`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/benchmark/examples/sparse`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/utils/benchmark/examples/sparse`):

- [`fuzzer.py_kw.md_docs.md`](./fuzzer.py_kw.md_docs.md)
- [`compare.py_kw.md_docs.md`](./compare.py_kw.md_docs.md)
- [`op_benchmark.py_docs.md_docs.md`](./op_benchmark.py_docs.md_docs.md)
- [`fuzzer.py_docs.md_docs.md`](./fuzzer.py_docs.md_docs.md)
- [`op_benchmark.py_kw.md_docs.md`](./op_benchmark.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `compare.py_docs.md_docs.md`
- **Keyword Index**: `compare.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

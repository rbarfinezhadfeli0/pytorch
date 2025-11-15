# Documentation: `docs/torch/utils/benchmark/examples/compare.py_docs.md`

## File Metadata

- **Path**: `docs/torch/utils/benchmark/examples/compare.py_docs.md`
- **Size**: 5,450 bytes (5.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `torch/utils/benchmark/examples/compare.py`

## File Metadata

- **Path**: `torch/utils/benchmark/examples/compare.py`
- **Size**: 2,931 bytes (2.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
"""Example of Timer and Compare APIs:

$ python -m examples.compare
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

    def extra_overhead(self, result):
        # time.sleep has a ~65 us overhead, so only fake a
        # per-element overhead if numel is large enough.
        numel = int(result.numel())
        if numel > 5000:
            time.sleep(numel * self._extra_ns_per_element * 1e-9)
        return result

    def add(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.add(*args, **kwargs))

    def mul(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.mul(*args, **kwargs))

    def cat(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.cat(*args, **kwargs))

    def matmul(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))


def main() -> None:
    tasks = [
        ("add", "add", "torch.add(x, y)"),
        ("add", "add (extra +0)", "torch.add(x, y + zero)"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": torch.ones((size, 4)),
                "y": torch.ones((1, 4)),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 5)]
        for label, sub_label, stmt in tasks
        for size in [1, 10, 100, 1000, 10000, 50000]
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

"""Example of Timer and Compare APIs:$ python -m examples.compare

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FauxTorch`

**Functions defined**: `__init__`, `extra_overhead`, `add`, `mul`, `cat`, `matmul`, `main`

**Key imports**: pickle, sys, time, torch, torch.utils.benchmark as benchmark_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/utils/benchmark/examples`, which is part of the **core PyTorch library**.



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

Files in the same folder (`torch/utils/benchmark/examples`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuzzer.py_docs.md`](./fuzzer.py_docs.md)
- [`spectral_ops_fuzz_test.py_docs.md`](./spectral_ops_fuzz_test.py_docs.md)
- [`simple_timeit.py_docs.md`](./simple_timeit.py_docs.md)
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

This file is part of the PyTorch framework located at `docs/torch/utils/benchmark/examples`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/utils/benchmark/examples`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/utils/benchmark/examples`):

- [`simple_timeit.py_kw.md_docs.md`](./simple_timeit.py_kw.md_docs.md)
- [`spectral_ops_fuzz_test.py_docs.md_docs.md`](./spectral_ops_fuzz_test.py_docs.md_docs.md)
- [`fuzzer.py_kw.md_docs.md`](./fuzzer.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`compare.py_kw.md_docs.md`](./compare.py_kw.md_docs.md)
- [`spectral_ops_fuzz_test.py_kw.md_docs.md`](./spectral_ops_fuzz_test.py_kw.md_docs.md)
- [`op_benchmark.py_docs.md_docs.md`](./op_benchmark.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`simple_timeit.py_docs.md_docs.md`](./simple_timeit.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `compare.py_docs.md_docs.md`
- **Keyword Index**: `compare.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

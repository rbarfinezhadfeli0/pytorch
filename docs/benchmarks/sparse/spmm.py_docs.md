# Documentation: `benchmarks/sparse/spmm.py`

## File Metadata

- **Path**: `benchmarks/sparse/spmm.py`
- **Size**: 3,889 bytes (3.80 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse
import sys

from utils import Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr

import torch


def test_sparse_csr(m, n, k, nnz, test_count):
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    csr = gen_sparse_csr((m, k), nnz)
    mat = torch.randn(k, n, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        csr.matmul(mat)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)


def test_sparse_coo(m, n, k, nnz, test_count):
    start_timer = Event(enable_timing=True)
    stop_timer = Event(enable_timing=True)

    coo = gen_sparse_coo((m, k), nnz)
    mat = torch.randn(k, n, dtype=torch.double)

    times = []
    for _ in range(test_count):
        start_timer.record()
        coo.matmul(mat)
        stop_timer.record()
        times.append(start_timer.elapsed_time(stop_timer))

    return sum(times) / len(times)


def test_sparse_coo_and_csr(m, n, k, nnz, test_count):
    start = Event(enable_timing=True)
    stop = Event(enable_timing=True)

    coo, csr = gen_sparse_coo_and_csr((m, k), nnz)
    mat = torch.randn((k, n), dtype=torch.double)

    times = []
    for _ in range(test_count):
        start.record()
        coo.matmul(mat)
        stop.record()
        times.append(start.elapsed_time(stop))

    coo_mean_time = sum(times) / len(times)

    times = []
    for _ in range(test_count):
        start.record()
        csr.matmul(mat)
        stop.record()
        times.append(start.elapsed_time(stop))

    csr_mean_time = sum(times) / len(times)

    return coo_mean_time, csr_mean_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMM")

    parser.add_argument("--format", default="csr", type=str)
    parser.add_argument("--m", default="1000", type=int)
    parser.add_argument("--n", default="1000", type=int)
    parser.add_argument("--k", default="1000", type=int)
    parser.add_argument("--nnz-ratio", "--nnz_ratio", default="0.1", type=float)
    parser.add_argument("--outfile", default="stdout", type=str)
    parser.add_argument("--test-count", "--test_count", default="10", type=int)

    args = parser.parse_args()

    if args.outfile == "stdout":
        outfile = sys.stdout
        need_close = False
    elif args.outfile == "stderr":
        outfile = sys.stderr
        need_close = False
    else:
        outfile = open(args.outfile, "a")
        need_close = True

    test_count = args.test_count
    m = args.m
    n = args.n
    k = args.k
    nnz_ratio = args.nnz_ratio

    nnz = int(nnz_ratio * m * k)
    if args.format == "csr":
        time = test_sparse_csr(m, n, k, nnz, test_count)
    elif args.format == "coo":
        time = test_sparse_coo(m, n, k, nnz, test_count)
    elif args.format == "both":
        time_coo, time_csr = test_sparse_coo_and_csr(m, nnz, test_count)

    if args.format == "both":
        print(
            "format=coo",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time_coo,
            file=outfile,
        )
        print(
            "format=csr",
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time_csr,
            file=outfile,
        )
    else:
        print(
            "format=",
            args.format,
            " nnz_ratio=",
            nnz_ratio,
            " m=",
            m,
            " n=",
            n,
            " k=",
            k,
            " time=",
            time,
            file=outfile,
        )
    if need_close:
        outfile.close()

```



## High-Level Overview


This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `test_sparse_csr`, `test_sparse_coo`, `test_sparse_coo_and_csr`

**Key imports**: argparse, sys, Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/sparse`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `sys`
- `utils`: Event, gen_sparse_coo, gen_sparse_coo_and_csr, gen_sparse_csr
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`benchmarks/sparse`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`triton_ops.py_docs.md`](./triton_ops.py_docs.md)
- [`test_csr.sh_docs.md`](./test_csr.sh_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`spmv.py_docs.md`](./spmv.py_docs.md)


## Cross-References

- **File Documentation**: `spmm.py_docs.md`
- **Keyword Index**: `spmm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

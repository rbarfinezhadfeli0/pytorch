# Documentation: `docs/test/bench_mps_ops.py_docs.md`

## File Metadata

- **Path**: `docs/test/bench_mps_ops.py_docs.md`
- **Size**: 9,847 bytes (9.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/bench_mps_ops.py`

## File Metadata

- **Path**: `test/bench_mps_ops.py`
- **Size**: 6,972 bytes (6.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: mps"]
# Collection of op level benchmarks for MPS
# Useful as reference tool when migrating ops from MPS to Metal
import itertools
import timeit
import warnings
from typing import Optional

import torch
from torch.utils.benchmark import Compare, Measurement, Timer


def bench_unary_op(func, x, label) -> Measurement:
    sync_cmd = "torch.mps.synchronize()" if "mps" in str(x.device) else ""
    t = Timer(
        stmt=f"f(x);{sync_cmd}",
        globals={"f": func, "x": x},
        language="python",
        timer=timeit.default_timer,
        sub_label=f"{func.__name__} ({str(x.dtype)})",
        description=label,
        env=torch.__version__,
    )
    return t.blocked_autorange()


def bench_binary_op(func, x, y, label) -> Measurement:
    sync_cmd = "torch.mps.synchronize()" if "mps" in str(x.device) else ""
    t = Timer(
        stmt=f"f(x, y);{sync_cmd}",
        globals={"f": func, "x": x, "y": y},
        language="python",
        timer=timeit.default_timer,
        sub_label=f"{func.__name__} ({str(x.dtype)}, {str(y.dtype)})",
        description=label,
        env=torch.__version__,
    )
    return t.blocked_autorange()


def bench_unary(
    unary_func, device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    x = torch.testing.make_tensor(1024, 1024, device=device, dtype=dtype)
    x_s = torch.testing.make_tensor(1024, 2048, device=device, dtype=dtype)[::, ::2]
    rc = []
    rc.append(bench_unary_op(unary_func, x, "dense"))
    rc.append(bench_unary_op(unary_func, x.t(), "transposed"))
    rc.append(bench_unary_op(unary_func, x_s, "strided"))
    rc.append(bench_unary_op(unary_func, x_s.t(), "strided + transposed"))
    return rc


def bench_binary(
    binary_func,
    device: str = "mps",
    dt_a: torch.dtype = torch.float32,
    dt_b: Optional[torch.dtype] = None,
) -> list[Measurement]:
    dt_b = dt_b if dt_b is not None else dt_a
    x = torch.testing.make_tensor(1024, 1024, device=device, dtype=dt_a)
    y = torch.testing.make_tensor(1024, 1024, device=device, dtype=dt_b)
    s = torch.testing.make_tensor((), device=device, dtype=dt_b)
    rc = []
    rc.append(bench_binary_op(binary_func, x, y, "dense-dense"))
    rc.append(bench_binary_op(binary_func, x.t(), y.t(), "transp-transp"))
    rc.append(bench_binary_op(binary_func, x, y.t(), "dense-transp"))
    rc.append(bench_binary_op(binary_func, x.t(), y, "transp-dense"))
    rc.append(bench_binary_op(binary_func, x, s, "dense-scalar"))
    rc.append(bench_binary_op(binary_func, x, y[0], "dense-bcast"))
    return rc


def check_eager_vs_compile(rc_c, rc_e, func, dtype):
    if not torch.allclose(rc_c, rc_e):
        mdiff = (rc_c - rc_e).abs().max()
        warnings.warn(
            f"Eager and compile reduction do not match for {func.__name__} and {dtype} max_diff={mdiff}",
            stacklevel=2,
        )


def bench_reduction(
    reduction_func, device: str = "mps", dtype: torch.dtype = torch.float32
) -> list[Measurement]:
    rc = []

    # Bench 2D with reduction over dim=0
    def f(t):
        return reduction_func(t, dim=0)

    f.__name__ = reduction_func.__name__
    f_c = torch.compile(f, dynamic=False, fullgraph=True)

    for size in (512, 1024, 2048, 4096):
        x = torch.testing.make_tensor(size, size, device=device, dtype=dtype)
        rc_c, rc_e = f(x), f_c(x)
        rc_c, rc_e = (rc_c[0], rc_e[0]) if isinstance(rc_c, tuple) else (rc_c, rc_e)
        check_eager_vs_compile(rc_c, rc_e, reduction_func, dtype)
        rc.append(bench_unary_op(f, x, f"eager-{size}x{size}"))
        rc.append(bench_unary_op(f_c, x, f"compile-{size}x{size}"))
    return rc


def bench_scan(
    scan_func,
    device: str = "mps",
    dtype: torch.dtype = torch.float32,
    with_indices: bool = False,
) -> list[Measurement]:
    rc = []

    # Bench cumsum along different dimensions
    for dim in [0, 1]:

        def f(t):
            return scan_func(t, dim=dim)

        f_c = torch.compile(f, dynamic=False, fullgraph=True)

        for size in (32, 128, 512, 1024):
            f.__name__ = f"{scan_func.__name__}-dim{dim}-{size}x{size}"
            f_c.__name__ = f.__name__
            x = torch.testing.make_tensor(size, size, device=device, dtype=dtype)
            rc_c, rc_e = f(x), f_c(x)
            if with_indices:
                check_eager_vs_compile(rc_c[0], rc_e[0], scan_func, dtype)
                check_eager_vs_compile(rc_c[1], rc_e[1], scan_func, dtype)
            else:
                check_eager_vs_compile(rc_c, rc_e, scan_func, dtype)
            rc.append(bench_unary_op(f, x, "eager"))
            rc.append(bench_unary_op(f_c, x, "compile"))

    # Bench 1D cumsum for different sizes
    def f_1d(t):
        return scan_func(t, dim=0)

    f_1d_c = torch.compile(f_1d, dynamic=False, fullgraph=True)

    for size in (100, 10000, 1000000):
        f_1d.__name__ = f"{scan_func.__name__}-1d-{size}"
        f_1d_c.__name__ = f_1d.__name__
        x = torch.testing.make_tensor(size, device=device, dtype=dtype)
        rc_c, rc_e = f_1d(x), f_1d_c(x)
        if with_indices:
            check_eager_vs_compile(rc_c[0], rc_e[0], scan_func, dtype)
            check_eager_vs_compile(rc_c[1], rc_e[1], scan_func, dtype)
        else:
            check_eager_vs_compile(rc_c, rc_e, scan_func, dtype)
        rc.append(bench_unary_op(f_1d, x, "eager"))
        rc.append(bench_unary_op(f_1d_c, x, "compile"))

    return rc


def main() -> None:
    dtypes = [torch.float16, torch.float32, torch.bfloat16]

    # Profile index ops
    B = 11
    rc = []
    for dtype, N in itertools.product(
        [torch.int8, torch.float16, torch.float32], [50, 100, 500, 1000, 2000]
    ):
        x = torch.testing.make_tensor((B, N, N), device="mps", dtype=dtype)
        y = torch.randint(0, B, (3,))
        rc.append(bench_binary_op(torch.Tensor.__getitem__, x, y, f"{B}x{N}x{N}"))
    Compare(rc).print()

    # Profile unary ops
    rc = []
    for op, dtype in itertools.product([torch.sqrt, torch.sin], dtypes):
        rc.extend(bench_unary(op, dtype=dtype))
    Compare(rc).print()

    # Profile reduction ops
    rc = []
    for op in [torch.sum, torch.max]:
        rc.extend(bench_reduction(op))
    Compare(rc).print()

    # Profile scan ops (cumsum)
    rc = []
    for dtype in dtypes:
        rc.extend(bench_scan(torch.cumsum, dtype=dtype))
    Compare(rc).print()

    # Profile scan with indices ops (cummin)
    rc = []
    for dtype in dtypes:
        rc.extend(bench_scan(torch.cummin, dtype=dtype, with_indices=True))
    Compare(rc).print()

    # Profile binary ops
    rc = []
    ops = [torch.fmax, torch.add]
    for op, dtype in itertools.product(ops, dtypes):
        rc.extend(bench_binary(op, dt_a=dtype))
        if dtype == torch.float32:
            rc.extend(bench_binary(op, dt_b=torch.float16))
    Compare(rc).print()


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 2**16
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `bench_unary_op`, `bench_binary_op`, `bench_unary`, `bench_binary`, `check_eager_vs_compile`, `bench_reduction`, `f`, `bench_scan`, `f`, `f_1d`, `main`

**Key imports**: itertools, timeit, warnings, Optional, torch, Compare, Measurement, Timer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `timeit`
- `warnings`
- `typing`: Optional
- `torch`
- `torch.utils.benchmark`: Compare, Measurement, Timer


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
python test/bench_mps_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `bench_mps_ops.py_docs.md`
- **Keyword Index**: `bench_mps_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/bench_mps_ops.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `bench_mps_ops.py_docs.md_docs.md`
- **Keyword Index**: `bench_mps_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `.ci/pytorch/smoke_test/check_gomp.py`

## File Metadata

- **Path**: `.ci/pytorch/smoke_test/check_gomp.py`
- **Size**: 2,372 bytes (2.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
import ctypes
import os
import sys
from pathlib import Path


def get_gomp_thread():
    """
    Retrieves the maximum number of OpenMP threads after loading the `libgomp.so.1` library
    and the `libtorch_cpu.so` library. It then queries the
    maximum number of threads available for OpenMP parallel regions using the
    `omp_get_max_threads` function.

    Returns:
        int: The maximum number of OpenMP threads available.

    Notes:
        - The function assumes the default path for `libgomp.so.1` on AlmaLinux OS.
        - The path to `libtorch_cpu.so` is constructed based on the Python executable's
          installation directory.
        - This function is specific to environments where PyTorch and OpenMP are used
          together and may require adjustments for other setups.
    """
    python_path = Path(sys.executable).resolve()
    python_prefix = (
        python_path.parent.parent
    )  # Typically goes to the Python installation root

    # Get the additional ABI flags (if any); it may be an empty string.
    abiflags = getattr(sys, "abiflags", "")

    # Construct the Python directory name correctly (e.g., "python3.13t").
    python_version = (
        f"python{sys.version_info.major}.{sys.version_info.minor}{abiflags}"
    )

    libtorch_cpu_path = (
        python_prefix
        / "lib"
        / python_version
        / "site-packages"
        / "torch"
        / "lib"
        / "libtorch_cpu.so"
    )

    # use the default gomp path of AlmaLinux OS
    libgomp_path = "/usr/lib64/libgomp.so.1"
    # if it does not exist, try Ubuntu path
    if not os.path.exists(libgomp_path):
        libgomp_path = f"/usr/lib/{os.uname().machine}-linux-gnu/libgomp.so.1"

    os.environ["GOMP_CPU_AFFINITY"] = "0-3"

    libgomp = ctypes.CDLL(libgomp_path)
    libgomp = ctypes.CDLL(libtorch_cpu_path)

    libgomp.omp_get_max_threads.restype = ctypes.c_int
    libgomp.omp_get_max_threads.argtypes = []

    omp_max_threads = libgomp.omp_get_max_threads()
    return omp_max_threads


def main():
    omp_max_threads = get_gomp_thread()
    print(
        f"omp_max_threads after loading libgomp.so and libtorch_cpu.so: {omp_max_threads}"
    )
    if omp_max_threads == 1:
        raise RuntimeError(
            "omp_max_threads is 1. Check whether libgomp.so is loaded twice."
        )


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""    Retrieves the maximum number of OpenMP threads after loading the `libgomp.so.1` library    and the `libtorch_cpu.so` library. It then queries the    maximum number of threads available for OpenMP parallel regions using the    `omp_get_max_threads` function.    Returns:        int: The maximum number of OpenMP threads available.    Notes:        - The function assumes the default path for `libgomp.so.1` on AlmaLinux OS.        - The path to `libtorch_cpu.so` is constructed based on the Python executable's          installation directory.        - This function is specific to environments where PyTorch and OpenMP are used          together and may require adjustments for other setups.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_gomp_thread`, `main`

**Key imports**: ctypes, os, sys, Path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/pytorch/smoke_test`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `ctypes`
- `os`
- `sys`
- `pathlib`: Path


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .ci/pytorch/smoke_test/check_gomp.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.ci/pytorch/smoke_test`):

- [`smoke_test.py_docs.md`](./smoke_test.py_docs.md)
- [`check_binary_symbols.py_docs.md`](./check_binary_symbols.py_docs.md)
- [`max_autotune.py_docs.md`](./max_autotune.py_docs.md)


## Cross-References

- **File Documentation**: `check_gomp.py_docs.md`
- **Keyword Index**: `check_gomp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

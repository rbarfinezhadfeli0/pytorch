# Documentation: `.ci/pytorch/smoke_test/check_binary_symbols.py`

## File Metadata

- **Path**: `.ci/pytorch/smoke_test/check_binary_symbols.py`
- **Size**: 4,577 bytes (4.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
import concurrent.futures
import distutils.sysconfig
import functools
import itertools
import os
import re
from pathlib import Path
from typing import Any


# We also check that there are [not] cxx11 symbols in libtorch
#
# To check whether it is using cxx11 ABI, check non-existence of symbol:
PRE_CXX11_SYMBOLS = (
    "std::basic_string<",
    "std::list",
)
# To check whether it is using pre-cxx11 ABI, check non-existence of symbol:
CXX11_SYMBOLS = (
    "std::__cxx11::basic_string",
    "std::__cxx11::list",
)
# NOTE: Checking the above symbols in all namespaces doesn't work, because
# devtoolset7 always produces some cxx11 symbols even if we build with old ABI,
# and CuDNN always has pre-cxx11 symbols even if we build with new ABI using gcc 5.4.
# Instead, we *only* check the above symbols in the following namespaces:
LIBTORCH_NAMESPACE_LIST = (
    "c10::",
    "at::",
    "caffe2::",
    "torch::",
)

# Patterns for detecting statically linked libstdc++ symbols
STATICALLY_LINKED_CXX11_ABI = [re.compile(r".*recursive_directory_iterator.*")]


def _apply_libtorch_symbols(symbols):
    return [
        re.compile(f"{x}.*{y}")
        for (x, y) in itertools.product(LIBTORCH_NAMESPACE_LIST, symbols)
    ]


LIBTORCH_CXX11_PATTERNS = _apply_libtorch_symbols(CXX11_SYMBOLS)

LIBTORCH_PRE_CXX11_PATTERNS = _apply_libtorch_symbols(PRE_CXX11_SYMBOLS)


@functools.lru_cache(100)
def get_symbols(lib: str) -> list[tuple[str, str, str]]:
    from subprocess import check_output

    lines = check_output(f'nm "{lib}"|c++filt', shell=True)
    return [x.split(" ", 2) for x in lines.decode("latin1").split("\n")[:-1]]


def grep_symbols(
    lib: str, patterns: list[Any], symbol_type: str | None = None
) -> list[str]:
    def _grep_symbols(
        symbols: list[tuple[str, str, str]], patterns: list[Any]
    ) -> list[str]:
        rc = []
        for _s_addr, _s_type, s_name in symbols:
            # Filter by symbol type if specified
            if symbol_type and _s_type != symbol_type:
                continue
            for pattern in patterns:
                if pattern.match(s_name):
                    rc.append(s_name)
                    continue
        return rc

    all_symbols = get_symbols(lib)
    num_workers = 32
    chunk_size = (len(all_symbols) + num_workers - 1) // num_workers

    def _get_symbols_chunk(i):
        return all_symbols[i * chunk_size : (i + 1) * chunk_size]

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        tasks = [
            executor.submit(_grep_symbols, _get_symbols_chunk(i), patterns)
            for i in range(num_workers)
        ]
        return functools.reduce(list.__add__, (x.result() for x in tasks), [])


def check_lib_statically_linked_libstdc_cxx_abi_symbols(lib: str) -> None:
    cxx11_statically_linked_symbols = grep_symbols(
        lib, STATICALLY_LINKED_CXX11_ABI, symbol_type="T"
    )
    num_statically_linked_symbols = len(cxx11_statically_linked_symbols)
    print(f"num_statically_linked_symbols (T): {num_statically_linked_symbols}")
    if num_statically_linked_symbols > 0:
        raise RuntimeError(
            f"Found statically linked libstdc++ symbols (recursive_directory_iterator): {cxx11_statically_linked_symbols[:100]}"
        )


def check_lib_symbols_for_abi_correctness(lib: str) -> None:
    print(f"lib: {lib}")
    cxx11_symbols = grep_symbols(lib, LIBTORCH_CXX11_PATTERNS)
    pre_cxx11_symbols = grep_symbols(lib, LIBTORCH_PRE_CXX11_PATTERNS)
    num_cxx11_symbols = len(cxx11_symbols)
    num_pre_cxx11_symbols = len(pre_cxx11_symbols)
    print(f"num_cxx11_symbols: {num_cxx11_symbols}")
    print(f"num_pre_cxx11_symbols: {num_pre_cxx11_symbols}")
    if num_pre_cxx11_symbols > 0:
        raise RuntimeError(
            f"Found pre-cxx11 symbols, but there shouldn't be any, see: {pre_cxx11_symbols[:100]}"
        )
    if num_cxx11_symbols < 100:
        raise RuntimeError("Didn't find enough cxx11 symbols")


def main() -> None:
    if "install_root" in os.environ:
        install_root = Path(os.getenv("install_root"))  # noqa: SIM112
    else:
        if os.getenv("PACKAGE_TYPE") == "libtorch":
            install_root = Path(os.getcwd())
        else:
            install_root = Path(distutils.sysconfig.get_python_lib()) / "torch"

    libtorch_cpu_path = str(install_root / "lib" / "libtorch_cpu.so")
    check_lib_symbols_for_abi_correctness(libtorch_cpu_path)
    check_lib_statically_linked_libstdc_cxx_abi_symbols(libtorch_cpu_path)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_apply_libtorch_symbols`, `get_symbols`, `grep_symbols`, `_grep_symbols`, `_get_symbols_chunk`, `check_lib_statically_linked_libstdc_cxx_abi_symbols`, `check_lib_symbols_for_abi_correctness`, `main`

**Key imports**: concurrent.futures, distutils.sysconfig, functools, itertools, os, re, Path, Any, check_output


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/pytorch/smoke_test`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `concurrent.futures`
- `distutils.sysconfig`
- `functools`
- `itertools`
- `os`
- `re`
- `pathlib`: Path
- `typing`: Any
- `subprocess`: check_output


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python .ci/pytorch/smoke_test/check_binary_symbols.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.ci/pytorch/smoke_test`):

- [`check_gomp.py_docs.md`](./check_gomp.py_docs.md)
- [`smoke_test.py_docs.md`](./smoke_test.py_docs.md)
- [`max_autotune.py_docs.md`](./max_autotune.py_docs.md)


## Cross-References

- **File Documentation**: `check_binary_symbols.py_docs.md`
- **Keyword Index**: `check_binary_symbols.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

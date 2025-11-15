# Documentation: `docs/tools/testing/discover_tests.py_docs.md`

## File Metadata

- **Path**: `docs/tools/testing/discover_tests.py_docs.md`
- **Size**: 8,202 bytes (8.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file is a **utility or tool script**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `tools/testing/discover_tests.py`

## File Metadata

- **Path**: `tools/testing/discover_tests.py`
- **Size**: 5,217 bytes (5.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


CPP_TEST_PREFIX = "cpp"
CPP_TEST_PATH = "build/bin"
CPP_TESTS_DIR = os.path.abspath(os.getenv("CPP_TESTS_DIR", default=CPP_TEST_PATH))
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_test_module(test: str) -> str:
    return test.split(".", maxsplit=1)[0]


def discover_tests(
    base_dir: Path = REPO_ROOT / "test",
    cpp_tests_dir: str | Path | None = None,
    blocklisted_patterns: list[str] | None = None,
    blocklisted_tests: list[str] | None = None,
    extra_tests: list[str] | None = None,
) -> list[str]:
    """
    Searches for all python files starting with test_ excluding one specified by patterns.
    If cpp_tests_dir is provided, also scan for all C++ tests under that directory. They
    are usually found in build/bin
    """

    def skip_test_p(name: str) -> bool:
        rc = False
        if blocklisted_patterns is not None:
            rc |= any(name.startswith(pattern) for pattern in blocklisted_patterns)
        if blocklisted_tests is not None:
            rc |= name in blocklisted_tests
        return rc

    # This supports symlinks, so we can link domain library tests to PyTorch test directory
    all_py_files = [
        Path(p) for p in glob.glob(f"{base_dir}/**/test_*.py", recursive=True)
    ]

    cpp_tests_dir = (
        f"{base_dir.parent}/{CPP_TEST_PATH}" if cpp_tests_dir is None else cpp_tests_dir
    )
    # CPP test files are located under pytorch/build/bin. Unlike Python test, C++ tests
    # are just binaries and could have any name, i.e. basic or atest
    all_cpp_files = [
        Path(p) for p in glob.glob(f"{cpp_tests_dir}/**/*", recursive=True)
    ]

    rc = [str(fname.relative_to(base_dir))[:-3] for fname in all_py_files]
    # Add the cpp prefix for C++ tests so that we can tell them apart
    rc.extend(
        [
            parse_test_module(f"{CPP_TEST_PREFIX}/{fname.relative_to(cpp_tests_dir)}")
            for fname in all_cpp_files
        ]
    )

    # Invert slashes on Windows
    if sys.platform == "win32":
        rc = [name.replace("\\", "/") for name in rc]
    rc = [test for test in rc if not skip_test_p(test)]
    if extra_tests is not None:
        rc += extra_tests
    return sorted(rc)


TESTS = discover_tests(
    cpp_tests_dir=CPP_TESTS_DIR,
    blocklisted_patterns=[
        "ao",
        "custom_backend",
        "custom_operator",
        "fx",  # executed by test_fx.py
        "jit",  # executed by test_jit.py
        "mobile",
        "onnx_caffe2",
        "package",  # executed by test_package.py
        "quantization",  # executed by test_quantization.py
        "autograd",  # executed by test_autograd.py
        "cpp_extensions/open_registration_extension/torch_openreg/tests",  # executed by test_openreg.py
    ],
    blocklisted_tests=[
        "test_bundled_images",
        "test_cpp_extensions_aot",
        "test_determination",
        "test_jit_fuser",
        "test_jit_simple",
        "test_jit_string",
        "test_kernel_launch_checks",
        "test_nnapi",
        "test_static_runtime",
        "test_throughput_benchmark",
        "distributed/bin/test_script",
        "distributed/elastic/multiprocessing/bin/test_script",
        "distributed/launcher/bin/test_script",
        "distributed/launcher/bin/test_script_init_method",
        "distributed/launcher/bin/test_script_is_torchelastic_launched",
        "distributed/launcher/bin/test_script_local_rank",
        "distributed/test_c10d_spawn",
        "distributions/test_transforms",
        "distributions/test_utils",
        "lazy/test_meta_kernel",
        "lazy/test_extract_compiled_graph",
        "test/inductor/test_aot_inductor_utils",
        "inductor/test_aoti_cross_compile_windows",
        "onnx/test_onnxscript_no_runtime",
        "onnx/test_pytorch_onnx_onnxruntime_cuda",
        "onnx/test_models",
        # These are not C++ tests
        f"{CPP_TEST_PREFIX}/CMakeFiles",
        f"{CPP_TEST_PREFIX}/CTestTestfile.cmake",
        f"{CPP_TEST_PREFIX}/Makefile",
        f"{CPP_TEST_PREFIX}/cmake_install.cmake",
        f"{CPP_TEST_PREFIX}/c10_intrusive_ptr_benchmark",
        f"{CPP_TEST_PREFIX}/example_allreduce",
        f"{CPP_TEST_PREFIX}/parallel_benchmark",
        f"{CPP_TEST_PREFIX}/protoc",
        f"{CPP_TEST_PREFIX}/protoc-3.13.0.0",
        f"{CPP_TEST_PREFIX}/torch_shm_manager",
        f"{CPP_TEST_PREFIX}/tutorial_tensorexpr",
    ],
    extra_tests=[
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "distributed/elastic/timer/api_test",
        "distributed/elastic/timer/local_timer_example",
        "distributed/elastic/timer/local_timer_test",
        "distributed/elastic/events/lib_test",
        "distributed/elastic/metrics/api_test",
        "distributed/elastic/utils/logging_test",
        "distributed/elastic/utils/util_test",
        "distributed/elastic/utils/distributed_test",
        "distributed/elastic/multiprocessing/api_test",
        "doctests",
        "test_autoload_enable",
        "test_autoload_disable",
        "test_openreg",
    ],
)


if __name__ == "__main__":
    print(TESTS)

```



## High-Level Overview

"""    Searches for all python files starting with test_ excluding one specified by patterns.    If cpp_tests_dir is provided, also scan for all C++ tests under that directory. They    are usually found in build/bin

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `parse_test_module`, `discover_tests`, `skip_test_p`

**Key imports**: annotations, glob, os, sys, Path


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/testing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `glob`
- `os`
- `sys`
- `pathlib`: Path


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python tools/testing/discover_tests.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`explicit_ci_jobs.py_docs.md`](./explicit_ci_jobs.py_docs.md)
- [`test_selections.py_docs.md`](./test_selections.py_docs.md)
- [`clickhouse.py_docs.md`](./clickhouse.py_docs.md)
- [`update_slow_tests.py_docs.md`](./update_slow_tests.py_docs.md)
- [`upload_artifacts.py_docs.md`](./upload_artifacts.py_docs.md)
- [`do_target_determination_for_s3.py_docs.md`](./do_target_determination_for_s3.py_docs.md)
- [`modulefinder_determinator.py_docs.md`](./modulefinder_determinator.py_docs.md)


## Cross-References

- **File Documentation**: `discover_tests.py_docs.md`
- **Keyword Index**: `discover_tests.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/testing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/testing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/tools/testing/discover_tests.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/tools/testing`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`upload_artifacts.py_docs.md_docs.md`](./upload_artifacts.py_docs.md_docs.md)
- [`test_selections.py_kw.md_docs.md`](./test_selections.py_kw.md_docs.md)
- [`modulefinder_determinator.py_docs.md_docs.md`](./modulefinder_determinator.py_docs.md_docs.md)
- [`explicit_ci_jobs.py_kw.md_docs.md`](./explicit_ci_jobs.py_kw.md_docs.md)
- [`test_selections.py_docs.md_docs.md`](./test_selections.py_docs.md_docs.md)
- [`clickhouse.py_kw.md_docs.md`](./clickhouse.py_kw.md_docs.md)
- [`discover_tests.py_kw.md_docs.md`](./discover_tests.py_kw.md_docs.md)
- [`modulefinder_determinator.py_kw.md_docs.md`](./modulefinder_determinator.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `discover_tests.py_docs.md_docs.md`
- **Keyword Index**: `discover_tests.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

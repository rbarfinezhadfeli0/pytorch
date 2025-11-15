# Documentation: `tools/testing/modulefinder_determinator.py`

## File Metadata

- **Path**: `tools/testing/modulefinder_determinator.py`
- **Size**: 6,149 bytes (6.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
from __future__ import annotations

import modulefinder
import os
import sys
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

# These tests are slow enough that it's worth calculating whether the patch
# touched any related files first. This list was manually generated, but for every
# run with --determine-from, we use another generated list based on this one and the
# previous test stats.
TARGET_DET_LIST = [
    # test_autograd.py is not slow, so it does not belong here. But
    # note that if you try to add it back it will run into
    # https://bugs.python.org/issue40350 because it imports files
    # under test/autograd/.
    "test_binary_ufuncs",
    "test_cpp_extensions_aot_ninja",
    "test_cpp_extensions_aot_no_ninja",
    "test_cpp_extensions_jit",
    "test_cpp_extensions_stream_and_event",
    "test_cpp_extensions_mtia_backend",
    "test_cuda",
    "test_cuda_primary_ctx",
    "test_dataloader",
    "test_determination",
    "test_futures",
    "test_jit",
    "test_jit_legacy",
    "test_jit_profiling",
    "test_linalg",
    "test_multiprocessing",
    "test_nn",
    "test_numpy_interop",
    "test_optim",
    "test_overrides",
    "test_pruning_op",
    "test_quantization",
    "test_reductions",
    "test_serialization",
    "test_shape_ops",
    "test_sort_and_select",
    "test_tensorboard",
    "test_testing",
    "test_torch",
    "test_utils",
    "test_view_ops",
]


_DEP_MODULES_CACHE: dict[str, set[str]] = {}


def should_run_test(
    target_det_list: list[str], test: str, touched_files: list[str], options: Any
) -> bool:
    test = parse_test_module(test)
    # Some tests are faster to execute than to determine.
    if test not in target_det_list:
        if options.verbose:
            print_to_stderr(f"Running {test} without determination")
        return True
    # HACK: "no_ninja" is not a real module
    if test.endswith("_no_ninja"):
        test = test[: (-1 * len("_no_ninja"))]
    if test.endswith("_ninja"):
        test = test[: (-1 * len("_ninja"))]

    dep_modules = get_dep_modules(test)

    for touched_file in touched_files:
        file_type = test_impact_of_file(touched_file)
        if file_type == "NONE":
            continue
        elif file_type == "CI":
            # Force all tests to run if any change is made to the CI
            # configurations.
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type == "UNKNOWN":
            # Assume uncategorized source files can affect every test.
            log_test_reason(file_type, touched_file, test, options)
            return True
        elif file_type in ["TORCH", "CAFFE2", "TEST"]:
            parts = os.path.splitext(touched_file)[0].split(os.sep)
            touched_module = ".".join(parts)
            # test/ path does not have a "test." namespace
            if touched_module.startswith("test."):
                touched_module = touched_module.split("test.")[1]
            if touched_module in dep_modules or touched_module == test.replace(
                "/", "."
            ):
                log_test_reason(file_type, touched_file, test, options)
                return True

    # If nothing has determined the test has run, don't run the test.
    if options.verbose:
        print_to_stderr(f"Determination is skipping {test}")

    return False


def test_impact_of_file(filename: str) -> str:
    """Determine what class of impact this file has on test runs.

    Possible values:
        TORCH - torch python code
        CAFFE2 - caffe2 python code
        TEST - torch test code
        UNKNOWN - may affect all tests
        NONE - known to have no effect on test outcome
        CI - CI configuration files
    """
    parts = filename.split(os.sep)
    if parts[0] in [".jenkins", ".circleci", ".ci"]:
        return "CI"
    if parts[0] in ["docs", "scripts", "CODEOWNERS", "README.md"]:
        return "NONE"
    elif parts[0] == "torch":
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "TORCH"
    elif parts[0] == "caffe2":
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "CAFFE2"
    elif parts[0] == "test":
        if parts[-1].endswith(".py") or parts[-1].endswith(".pyi"):
            return "TEST"

    return "UNKNOWN"


def log_test_reason(file_type: str, filename: str, test: str, options: Any) -> None:
    if options.verbose:
        print_to_stderr(
            f"Determination found {file_type} file {filename} -- running {test}"
        )


def get_dep_modules(test: str) -> set[str]:
    # Cache results in case of repetition
    if test in _DEP_MODULES_CACHE:
        return _DEP_MODULES_CACHE[test]

    test_location = REPO_ROOT / "test" / f"{test}.py"

    # HACK: some platforms default to ascii, so we can't just run_script :(
    finder = modulefinder.ModuleFinder(
        # Ideally exclude all third party modules, to speed up calculation.
        excludes=[
            "scipy",
            "numpy",
            "numba",
            "multiprocessing",
            "sklearn",
            "setuptools",
            "hypothesis",
            "llvmlite",
            "joblib",
            "email",
            "importlib",
            "unittest",
            "urllib",
            "json",
            "collections",
            # Modules below are excluded because they are hitting https://bugs.python.org/issue40350
            # Trigger AttributeError: 'NoneType' object has no attribute 'is_package'
            "mpl_toolkits",
            "google",
            "onnx",
            # Triggers RecursionError
            "mypy",
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        finder.run_script(str(test_location))
    dep_modules = set(finder.modules.keys())
    _DEP_MODULES_CACHE[test] = dep_modules
    return dep_modules


def parse_test_module(test: str) -> str:
    return test.split(".", maxsplit=1)[0]


def print_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `should_run_test`, `test_impact_of_file`, `log_test_reason`, `get_dep_modules`, `parse_test_module`, `print_to_stderr`

**Key imports**: annotations, modulefinder, os, sys, warnings, Path, Any


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/testing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `modulefinder`
- `os`
- `sys`
- `warnings`
- `pathlib`: Path
- `typing`: Any


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python tools/testing/modulefinder_determinator.py
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
- [`discover_tests.py_docs.md`](./discover_tests.py_docs.md)


## Cross-References

- **File Documentation**: `modulefinder_determinator.py_docs.md`
- **Keyword Index**: `modulefinder_determinator.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

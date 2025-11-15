# Documentation: `docs/torch/testing/_internal/dynamo_test_failures.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/dynamo_test_failures.py_docs.md`
- **Size**: 9,159 bytes (8.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/dynamo_test_failures.py`

## File Metadata

- **Path**: `torch/testing/_internal/dynamo_test_failures.py`
- **Size**: 5,471 bytes (5.34 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
"""
This file contains the list of tests that are known to fail under Dynamo

We generate xFailIfTorchDynamo* for all tests in `dynamo_expected_failures`
We generate skipIfTorchDynamo* for all tests in `dynamo_skips`
We generate runWithoutCompiledAutograd for all tests in `compiled_autograd_skips`

For an easier-than-manual way of generating and updating these lists,
see scripts/compile_tests/update_failures.py

If you're adding a new test, and it's failing PYTORCH_TEST_WITH_DYNAMO=1,
either add the appropriate decorators to your test or add skips for them
via test/dynamo_skips and test/dynamo_expected_failures.

*These are not exactly unittest.expectedFailure and unittest.skip. We'll
always execute the test and then suppress the signal, if necessary.
If your tests crashes, or is slow, please use @skipIfTorchDynamo instead.

The expected failure and skip files are located in test/dynamo_skips and
test/dynamo_expected_failures. They're individual files rather than a list so
git will merge changes easier.
"""

import logging
import os
import sys
from typing import Optional


def find_test_dir() -> Optional[str]:
    # Find the path to the dynamo expected failure and skip files.
    from os.path import abspath, basename, dirname, exists, join, normpath

    if sys.platform == "win32":
        return None

    # Check relative to this file (local build):
    test_dir = normpath(join(dirname(abspath(__file__)), "../../../test"))
    if exists(join(test_dir, "dynamo_expected_failures")):
        return test_dir

    # Check relative to __main__ (installed builds relative to test file):
    main = sys.modules["__main__"]
    file = getattr(main, "__file__", None)
    if file is None:
        # Generated files do not have a module.__file__
        return None
    test_dir = dirname(abspath(file))
    while dirname(test_dir) != test_dir:
        if basename(test_dir) == "test" and exists(
            join(test_dir, "dynamo_expected_failures")
        ):
            return test_dir
        test_dir = dirname(test_dir)

    # Not found
    return None


test_dir = find_test_dir()
if not test_dir:
    logger = logging.getLogger(__name__)
    logger.warning(
        "test/dynamo_expected_failures directory not found - known dynamo errors won't be skipped."
    )

# Tests that run without strict mode in PYTORCH_TEST_WITH_INDUCTOR=1.
# Please don't add anything to this list.
FIXME_inductor_non_strict = {
    "test_modules",
    "test_ops",
    "test_ops_gradients",
    "test_torch",
}

# We generate unittest.expectedFailure for all of the following tests
# when run under PYTORCH_TEST_WITH_DYNAMO=1.
# see NOTE [dynamo_test_failures.py] for more details
#
# This lists exists so we can more easily add large numbers of failing tests,
if test_dir is None:
    dynamo_expected_failures = set()
    dynamo_skips = set()

    inductor_expected_failures = set()
    inductor_skips = set()

    compiled_autograd_skips = set()
else:
    dynamo_failures_directory = os.path.join(test_dir, "dynamo_expected_failures")
    dynamo_skips_directory = os.path.join(test_dir, "dynamo_skips")

    dynamo_expected_failures = set(os.listdir(dynamo_failures_directory))
    dynamo_skips = set(os.listdir(dynamo_skips_directory))

    inductor_failures_directory = os.path.join(test_dir, "inductor_expected_failures")
    inductor_skips_directory = os.path.join(test_dir, "inductor_skips")

    inductor_expected_failures = set(os.listdir(inductor_failures_directory))
    inductor_skips = set(os.listdir(inductor_skips_directory))

    compiled_autograd_skips_directory = os.path.join(
        test_dir, "compiled_autograd_skips"
    )
    compiled_autograd_skips = set(os.listdir(compiled_autograd_skips_directory))

# TODO: due to case sensitivity problems, for now list these files by hand
extra_dynamo_skips = {
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_fake_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_inplace_t_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_T_cpu_float32",
    "TestProxyTensorOpInfoCPU.test_make_fx_symbolic_exhaustive_out_t_cpu_float32",
}
dynamo_skips = dynamo_skips.union(extra_dynamo_skips)


# verify some invariants
for test in (
    dynamo_expected_failures
    | dynamo_skips
    | inductor_expected_failures
    | inductor_skips
):
    if len(test.split(".")) != 2:
        raise AssertionError(f'Invalid test name: "{test}"')

dynamo_intersection = dynamo_expected_failures.intersection(dynamo_skips)
if len(dynamo_intersection) > 0:
    raise AssertionError(
        "there should be no overlap between dynamo_expected_failures "
        "and dynamo_skips, got " + str(dynamo_intersection)
    )

inductor_intersection = inductor_expected_failures.intersection(inductor_skips)
if len(inductor_intersection) > 0:
    raise AssertionError(
        "there should be no overlap between inductor_expected_failures "
        "and inductor_skips, got " + str(inductor_intersection)
    )

```



## High-Level Overview

"""This file contains the list of tests that are known to fail under DynamoWe generate xFailIfTorchDynamo* for all tests in `dynamo_expected_failures`We generate skipIfTorchDynamo* for all tests in `dynamo_skips`We generate runWithoutCompiledAutograd for all tests in `compiled_autograd_skips`For an easier-than-manual way of generating and updating these lists,see scripts/compile_tests/update_failures.pyIf you're adding a new test, and it's failing PYTORCH_TEST_WITH_DYNAMO=1,either add the appropriate decorators to your test or add skips for themvia test/dynamo_skips and test/dynamo_expected_failures.*These are not exactly unittest.expectedFailure and unittest.skip. We'llalways execute the test and then suppress the signal, if necessary.If your tests crashes, or is slow, please use @skipIfTorchDynamo instead.The expected failure and skip files are located in test/dynamo_skips andtest/dynamo_expected_failures. They're individual files rather than a list sogit will merge changes easier.

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `find_test_dir`

**Key imports**: logging, os, sys, Optional, abspath, basename, dirname, exists, join, normpath


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `os`
- `sys`
- `typing`: Optional
- `os.path`: abspath, basename, dirname, exists, join, normpath


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python torch/testing/_internal/dynamo_test_failures.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `dynamo_test_failures.py_docs.md`
- **Keyword Index**: `dynamo_test_failures.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python docs/torch/testing/_internal/dynamo_test_failures.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `dynamo_test_failures.py_docs.md_docs.md`
- **Keyword Index**: `dynamo_test_failures.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/test/test_cuda_nvml_based_avail.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_cuda_nvml_based_avail.py_docs.md`
- **Size**: 10,523 bytes (10.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_cuda_nvml_based_avail.py`

## File Metadata

- **Path**: `test/test_cuda_nvml_based_avail.py`
- **Size**: 7,258 bytes (7.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cuda"]

import multiprocessing
import os
import sys
import unittest
from unittest.mock import patch

import torch


# NOTE: Each of the tests in this module need to be run in a brand new process to ensure CUDA is uninitialized
# prior to test initiation.
with patch.dict(os.environ, {"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}):
    # Before executing the desired tests, we need to disable CUDA initialization and fork_handler additions that would
    # otherwise be triggered by the `torch.testing._internal.common_utils` module import
    from torch.testing._internal.common_utils import (
        instantiate_parametrized_tests,
        IS_JETSON,
        IS_WINDOWS,
        NoTest,
        parametrize,
        run_tests,
        TestCase,
    )

    # NOTE: Because `remove_device_and_dtype_suffixes` initializes CUDA context (triggered via the import of
    # `torch.testing._internal.common_device_type` which imports `torch.testing._internal.common_cuda`) we need
    # to bypass that method here which should be irrelevant to the parameterized tests in this module.
    torch.testing._internal.common_utils.remove_device_and_dtype_suffixes = lambda x: x

    TEST_CUDA = torch.cuda.is_available()
    if not TEST_CUDA:
        print("CUDA not available, skipping tests", file=sys.stderr)
        TestCase = NoTest  # type: ignore[misc, assignment] # noqa: F811


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestExtendedCUDAIsAvail(TestCase):
    SUBPROCESS_REMINDER_MSG = (
        "\n REMINDER: Tests defined in test_cuda_nvml_based_avail.py must be run in a process "
        "where there CUDA Driver API has not been initialized. Before further debugging, ensure you are either using "
        "run_test.py or have added --subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        super().setUp()
        torch.cuda._cached_device_count = (
            None  # clear the lru_cache on this method before our test
        )

    @staticmethod
    def in_bad_fork_test() -> bool:
        _ = torch.cuda.is_available()
        return torch.cuda._is_in_bad_fork()

    # These tests validate the behavior and activation of the weaker, NVML-based, user-requested
    # `torch.cuda.is_available()` assessment. The NVML-based assessment should be attempted when
    # `PYTORCH_NVML_BASED_CUDA_CHECK` is set to 1, reverting to the default CUDA Runtime API check otherwise.
    # If the NVML-based assessment is attempted but fails, the CUDA Runtime API check should be executed
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    @parametrize("nvml_avail", [True, False])
    @parametrize("avoid_init", ["1", "0", None])
    def test_cuda_is_available(self, avoid_init, nvml_avail):
        if IS_JETSON and nvml_avail and avoid_init == "1":
            self.skipTest("Not working for Jetson")
        patch_env = {"PYTORCH_NVML_BASED_CUDA_CHECK": avoid_init} if avoid_init else {}
        with patch.dict(os.environ, **patch_env):
            if nvml_avail:
                _ = torch.cuda.is_available()
            else:
                with patch.object(torch.cuda, "_device_count_nvml", return_value=-1):
                    _ = torch.cuda.is_available()
            with multiprocessing.get_context("fork").Pool(1) as pool:
                in_bad_fork = pool.apply(TestExtendedCUDAIsAvail.in_bad_fork_test)
            if os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1" and nvml_avail:
                self.assertFalse(
                    in_bad_fork, TestExtendedCUDAIsAvail.SUBPROCESS_REMINDER_MSG
                )
            else:
                assert in_bad_fork


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestVisibleDeviceParses(TestCase):
    def test_env_var_parsing(self):
        def _parse_visible_devices(val):
            from torch.cuda import _parse_visible_devices as _pvd

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": val}, clear=True):
                return _pvd()

        # rest of the string is ignored
        self.assertEqual(_parse_visible_devices("1gpu2,2ampere"), [1, 2])
        # Negatives abort parsing
        self.assertEqual(_parse_visible_devices("0, 1, 2, -1, 3"), [0, 1, 2])
        # Double mention of ordinal returns empty set
        self.assertEqual(_parse_visible_devices("0, 1, 2, 1"), [])
        # Unary pluses and minuses
        self.assertEqual(_parse_visible_devices("2, +3, -0, 5"), [2, 3, 0, 5])
        # Random string is used as empty set
        self.assertEqual(_parse_visible_devices("one,two,3,4"), [])
        # Random string is used as separator
        self.assertEqual(_parse_visible_devices("4,3,two,one"), [4, 3])
        # GPU ids are parsed
        self.assertEqual(_parse_visible_devices("GPU-9e8d35e3"), ["GPU-9e8d35e3"])
        # Ordinals are not included in GPUid set
        self.assertEqual(_parse_visible_devices("GPU-123, 2"), ["GPU-123"])
        # MIG ids are parsed
        self.assertEqual(_parse_visible_devices("MIG-89c850dc"), ["MIG-89c850dc"])

    def test_partial_uuid_resolver(self):
        from torch.cuda import _transform_uuid_to_ordinals

        uuids = [
            "GPU-9942190a-aa31-4ff1-4aa9-c388d80f85f1",
            "GPU-9e8d35e3-a134-0fdd-0e01-23811fdbd293",
            "GPU-e429a63e-c61c-4795-b757-5132caeb8e70",
            "GPU-eee1dfbc-0a0f-6ad8-5ff6-dc942a8b9d98",
            "GPU-bbcd6503-5150-4e92-c266-97cc4390d04e",
            "GPU-472ea263-58d7-410d-cc82-f7fdece5bd28",
            "GPU-e56257c4-947f-6a5b-7ec9-0f45567ccf4e",
            "GPU-1c20e77d-1c1a-d9ed-fe37-18b8466a78ad",
        ]
        self.assertEqual(_transform_uuid_to_ordinals(["GPU-9e8d35e3"], uuids), [1])
        self.assertEqual(
            _transform_uuid_to_ordinals(["GPU-e4", "GPU-9e8d35e3"], uuids), [2, 1]
        )
        self.assertEqual(
            _transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-1", "GPU-47"], uuids),
            [1, 7, 5],
        )
        # First invalid UUID aborts parsing
        self.assertEqual(
            _transform_uuid_to_ordinals(["GPU-123", "GPU-9e8d35e3"], uuids), []
        )
        self.assertEqual(
            _transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-123", "GPU-47"], uuids),
            [1],
        )
        # First ambiguous UUID aborts parsing
        self.assertEqual(
            _transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-e", "GPU-47"], uuids), [1]
        )
        # Duplicate UUIDs result in empty set
        self.assertEqual(
            _transform_uuid_to_ordinals(["GPU-9e8d35e3", "GPU-47", "GPU-9e8"], uuids),
            [],
        )

    def test_ordinal_parse_visible_devices(self):
        def _device_count_nvml(val):
            from torch.cuda import _device_count_nvml as _dc

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": val}, clear=True):
                return _dc()

        with patch.object(torch.cuda, "_raw_device_count_nvml", return_value=2):
            self.assertEqual(_device_count_nvml("1, 0"), 2)
            # Ordinal out of bounds aborts parsing
            self.assertEqual(_device_count_nvml("1, 5, 0"), 1)


instantiate_parametrized_tests(TestExtendedCUDAIsAvail)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestExtendedCUDAIsAvail`, `TestVisibleDeviceParses`

**Functions defined**: `setUp`, `in_bad_fork_test`, `test_cuda_is_available`, `test_env_var_parsing`, `_parse_visible_devices`, `test_partial_uuid_resolver`, `test_ordinal_parse_visible_devices`, `_device_count_nvml`

**Key imports**: multiprocessing, os, sys, unittest, patch, torch, from torch.testing._internal.common_utils import , of, _parse_visible_devices as _pvd, _transform_uuid_to_ordinals


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `multiprocessing`
- `os`
- `sys`
- `unittest`
- `unittest.mock`: patch
- `torch`
- `from torch.testing._internal.common_utils import `
- `of`
- `torch.cuda`: _parse_visible_devices as _pvd


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_cuda_nvml_based_avail.py
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

- **File Documentation**: `test_cuda_nvml_based_avail.py_docs.md`
- **Keyword Index**: `test_cuda_nvml_based_avail.py_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_cuda_nvml_based_avail.py_docs.md
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

- **File Documentation**: `test_cuda_nvml_based_avail.py_docs.md_docs.md`
- **Keyword Index**: `test_cuda_nvml_based_avail.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

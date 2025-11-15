# Documentation: `docs/test/test_cpp_extensions_mtia_backend.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_cpp_extensions_mtia_backend.py_docs.md`
- **Size**: 9,450 bytes (9.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_cpp_extensions_mtia_backend.py`

## File Metadata

- **Path**: `test/test_cpp_extensions_mtia_backend.py`
- **Size**: 5,734 bytes (5.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: mtia"]

import os
import tempfile
import unittest

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_LINUX,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_PRIVATEUSE1,
    TEST_XPU,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


# define TEST_ROCM before changing TEST_CUDA
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None


@unittest.skipIf(
    IS_ARM64 or not IS_LINUX or TEST_CUDA or TEST_PRIVATEUSE1 or TEST_ROCM or TEST_XPU,
    "Only on linux platform and mutual exclusive to other backends",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionMTIABackend(common.TestCase):
    """Tests MTIA backend with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()
        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def tearDown(self):
        super().tearDown()
        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def tearDownClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    @classmethod
    def setUpClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()
        build_dir = tempfile.mkdtemp()
        # Load the fake device guard impl.
        cls.module = torch.utils.cpp_extension.load(
            name="mtia_extension",
            sources=["cpp_extensions/mtia_extension.cpp"],
            build_directory=build_dir,
            extra_include_paths=[
                "cpp_extensions",
                "path / with spaces in it",
                "path with quote'",
            ],
            is_python_module=False,
            verbose=True,
        )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_get_device_module(self):
        device = torch.device("mtia:0")
        default_stream = torch.get_device_module(device).current_stream()
        self.assertEqual(
            default_stream.device_type, int(torch._C._autograd.DeviceType.MTIA)
        )
        print(torch._C.Stream.__mro__)
        print(torch.cuda.Stream.__mro__)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_basic(self):
        default_stream = torch.mtia.current_stream()
        user_stream = torch.mtia.Stream()
        self.assertEqual(torch.mtia.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        # Check mtia_extension.cpp, default stream id starts from 0.
        self.assertEqual(default_stream.stream_id, 0)
        self.assertNotEqual(user_stream.stream_id, 0)
        with torch.mtia.stream(user_stream):
            self.assertEqual(torch.mtia.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_context(self):
        mtia_stream_0 = torch.mtia.Stream(device="mtia:0")
        mtia_stream_1 = torch.mtia.Stream(device="mtia:0")
        print(mtia_stream_0)
        print(mtia_stream_1)
        with torch.mtia.stream(mtia_stream_0):
            current_stream = torch.mtia.current_stream()
            msg = f"current_stream {current_stream} should be {mtia_stream_0}"
            self.assertTrue(current_stream == mtia_stream_0, msg=msg)

        with torch.mtia.stream(mtia_stream_1):
            current_stream = torch.mtia.current_stream()
            msg = f"current_stream {current_stream} should be {mtia_stream_1}"
            self.assertTrue(current_stream == mtia_stream_1, msg=msg)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_stream_context_different_device(self):
        device_0 = torch.device("mtia:0")
        device_1 = torch.device("mtia:1")
        mtia_stream_0 = torch.mtia.Stream(device=device_0)
        mtia_stream_1 = torch.mtia.Stream(device=device_1)
        print(mtia_stream_0)
        print(mtia_stream_1)
        orig_current_device = torch.mtia.current_device()
        with torch.mtia.stream(mtia_stream_0):
            current_stream = torch.mtia.current_stream()
            self.assertTrue(torch.mtia.current_device() == device_0.index)
            msg = f"current_stream {current_stream} should be {mtia_stream_0}"
            self.assertTrue(current_stream == mtia_stream_0, msg=msg)
        self.assertTrue(torch.mtia.current_device() == orig_current_device)
        with torch.mtia.stream(mtia_stream_1):
            current_stream = torch.mtia.current_stream()
            self.assertTrue(torch.mtia.current_device() == device_1.index)
            msg = f"current_stream {current_stream} should be {mtia_stream_1}"
            self.assertTrue(current_stream == mtia_stream_1, msg=msg)
        self.assertTrue(torch.mtia.current_device() == orig_current_device)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_device_context(self):
        device_0 = torch.device("mtia:0")
        device_1 = torch.device("mtia:1")
        with torch.mtia.device(device_0):
            self.assertTrue(torch.mtia.current_device() == device_0.index)

        with torch.mtia.device(device_1):
            self.assertTrue(torch.mtia.current_device() == device_1.index)


if __name__ == "__main__":
    common.run_tests()

```



## High-Level Overview

"""Tests MTIA backend with C++ extensions."""    module = None    def setUp(self):        super().setUp()        # cpp extensions use relative paths. Those paths are relative to        # this file, so we'll change the working directory temporarily        self.old_working_dir = os.getcwd()        os.chdir(os.path.dirname(os.path.abspath(__file__)))    def tearDown(self):        super().tearDown()        # return the working directory (see setUp)        os.chdir(self.old_working_dir)    @classmethod    def tearDownClass(cls):        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCppExtensionMTIABackend`

**Functions defined**: `setUp`, `tearDown`, `tearDownClass`, `setUpClass`, `test_get_device_module`, `test_stream_basic`, `test_stream_context`, `test_stream_context_different_device`, `test_device_context`

**Key imports**: os, tempfile, unittest, torch, torch.testing._internal.common_utils as common, torch.utils.cpp_extension, CUDA_HOME, ROCM_HOME


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `tempfile`
- `unittest`
- `torch`
- `torch.testing._internal.common_utils as common`
- `torch.utils.cpp_extension`


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
python test/test_cpp_extensions_mtia_backend.py
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

- **File Documentation**: `test_cpp_extensions_mtia_backend.py_docs.md`
- **Keyword Index**: `test_cpp_extensions_mtia_backend.py_kw.md`
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
python docs/test/test_cpp_extensions_mtia_backend.py_docs.md
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

- **File Documentation**: `test_cpp_extensions_mtia_backend.py_docs.md_docs.md`
- **Keyword Index**: `test_cpp_extensions_mtia_backend.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

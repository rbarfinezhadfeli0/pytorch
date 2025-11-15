# Documentation: `docs/test/test_cpp_extensions_stream_and_event.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_cpp_extensions_stream_and_event.py_docs.md`
- **Size**: 7,164 bytes (7.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_cpp_extensions_stream_and_event.py`

## File Metadata

- **Path**: `test/test_cpp_extensions_stream_and_event.py`
- **Size**: 3,792 bytes (3.70 KB)
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
    TEST_MPS,
    TEST_PRIVATEUSE1,
    TEST_XPU,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


# define TEST_ROCM before changing TEST_CUDA
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None


# Since we use a fake MTIA device backend to test generic Stream/Event, device backends are mutual exclusive to each other.
# The test will be skipped if any of the following conditions are met:
@unittest.skipIf(
    IS_ARM64
    or not IS_LINUX
    or TEST_CUDA
    or TEST_XPU
    or TEST_MPS
    or TEST_PRIVATEUSE1
    or TEST_ROCM,
    "Only on linux platform and mutual exclusive to other backends",
)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionStreamAndEvent(common.TestCase):
    """Tests Stream and Event with C++ extensions."""

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
        src = f"{os.path.abspath(os.path.dirname(__file__))}/cpp_extensions/mtia_extension.cpp"
        cls.module = torch.utils.cpp_extension.load(
            name="mtia_extension",
            sources=[src],
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
    def test_stream_event(self):
        s = torch.Stream()
        self.assertTrue(s.device_type, int(torch._C._autograd.DeviceType.MTIA))
        e = torch.Event(enable_timing=True)
        e1 = torch.Event(enable_timing=True)
        e1.record()
        self.assertTrue(e.device.type, "mtia")
        # Should be nullptr by default
        self.assertTrue(e.event_id == 0)
        s.record_event(event=e)
        print(f"recorded event 1: {e}")
        self.assertTrue(e.event_id != 0)
        # The enable_timing of event created by record_event() is false
        e2 = s.record_event()
        print(f"recorded event 2: {e2}")
        self.assertTrue(e2.event_id != 0)
        self.assertTrue(e2.event_id != e.event_id)
        e.synchronize()
        e1.synchronize()
        e2.synchronize()
        time_elapsed = e.elapsed_time(e1)
        print(f"time elapsed between e and e1: {time_elapsed}")
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            time_elapsed = e.elapsed_time(e2)
        old_event_id = e.event_id
        e.record(stream=s)
        print(f"recorded event 1: {e}")
        self.assertTrue(e.event_id == old_event_id)


if __name__ == "__main__":
    common.run_tests()

```



## High-Level Overview

"""Tests Stream and Event with C++ extensions."""    module = None    def setUp(self):        super().setUp()        # cpp extensions use relative paths. Those paths are relative to        # this file, so we'll change the working directory temporarily        self.old_working_dir = os.getcwd()        os.chdir(os.path.dirname(os.path.abspath(__file__)))

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCppExtensionStreamAndEvent`

**Functions defined**: `setUp`, `tearDown`, `tearDownClass`, `setUpClass`, `test_stream_event`

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
python test/test_cpp_extensions_stream_and_event.py
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

- **File Documentation**: `test_cpp_extensions_stream_and_event.py_docs.md`
- **Keyword Index**: `test_cpp_extensions_stream_and_event.py_kw.md`
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
python docs/test/test_cpp_extensions_stream_and_event.py_docs.md
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

- **File Documentation**: `test_cpp_extensions_stream_and_event.py_docs.md_docs.md`
- **Keyword Index**: `test_cpp_extensions_stream_and_event.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

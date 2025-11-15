# Documentation: `test/dynamo/test_deviceguard.py`

## File Metadata

- **Path**: `test/dynamo/test_deviceguard.py`
- **Size**: 3,092 bytes (3.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import unittest
from unittest.mock import Mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.device_interface import CudaInterface, DeviceGuard
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU


class TestDeviceGuard(torch._dynamo.test_case.TestCase):
    """
    Unit tests for the DeviceGuard class using a mock DeviceInterface.
    """

    def setUp(self):
        super().setUp()
        self.device_interface = Mock()

        self.device_interface.exchange_device = Mock(return_value=0)
        self.device_interface.maybe_exchange_device = Mock(return_value=1)

    def test_device_guard(self):
        device_guard = DeviceGuard(self.device_interface, 1)

        with device_guard as _:
            self.device_interface.exchange_device.assert_called_once_with(1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.device_interface.maybe_exchange_device.assert_called_once_with(0)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    def test_device_guard_no_index(self):
        device_guard = DeviceGuard(self.device_interface, None)

        with device_guard as _:
            self.device_interface.exchange_device.assert_not_called()
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.device_interface.maybe_exchange_device.assert_not_called()
        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


@unittest.skipIf(not TEST_CUDA, "No CUDA available.")
class TestCUDADeviceGuard(torch._dynamo.test_case.TestCase):
    """
    Unit tests for the DeviceGuard class using a CudaInterface.
    """

    def setUp(self):
        super().setUp()
        self.device_interface = CudaInterface

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_device_guard(self):
        current_device = torch.cuda.current_device()

        device_guard = DeviceGuard(self.device_interface, 1)

        with device_guard as _:
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.assertEqual(torch.cuda.current_device(), current_device)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    def test_device_guard_no_index(self):
        current_device = torch.cuda.current_device()

        device_guard = DeviceGuard(self.device_interface, None)

        with device_guard as _:
            self.assertEqual(torch.cuda.current_device(), current_device)
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

"""    Unit tests for the DeviceGuard class using a mock DeviceInterface.

This Python file contains 4 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDeviceGuard`, `TestCUDADeviceGuard`

**Functions defined**: `setUp`, `test_device_guard`, `test_device_guard_no_index`, `setUp`, `test_device_guard`, `test_device_guard_no_index`

**Key imports**: unittest, Mock, torch, torch._dynamo.test_case, torch._dynamo.testing, CudaInterface, DeviceGuard, TEST_CUDA, TEST_MULTIGPU, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `unittest.mock`: Mock
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch._dynamo.device_interface`: CudaInterface, DeviceGuard
- `torch.testing._internal.common_cuda`: TEST_CUDA, TEST_MULTIGPU


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
python test/dynamo/test_deviceguard.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_deviceguard.py_docs.md`
- **Keyword Index**: `test_deviceguard.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

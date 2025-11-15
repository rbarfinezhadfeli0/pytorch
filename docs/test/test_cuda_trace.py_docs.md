# Documentation: `test/test_cuda_trace.py`

## File Metadata

- **Path**: `test/test_cuda_trace.py`
- **Size**: 4,232 bytes (4.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cuda"]

import sys
import unittest
import unittest.mock

import torch
import torch.cuda._gpu_trace as gpu_trace
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase


# NOTE: Each test needs to be run in a brand new process, to reset the registered hooks
# and make sure the CUDA streams are initialized for each test that uses them.

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaTrace(TestCase):
    def setUp(self):
        super().setUp()
        torch._C._activate_gpu_trace()
        self.mock = unittest.mock.MagicMock()

    def test_event_creation_callback(self):
        gpu_trace.register_callback_for_event_creation(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_event_deletion_callback(self):
        gpu_trace.register_callback_for_event_deletion(self.mock)

        event = torch.cuda.Event()
        event.record()
        event_id = event._as_parameter_.value
        del event
        self.mock.assert_called_once_with(event_id)

    def test_event_record_callback(self):
        gpu_trace.register_callback_for_event_record(self.mock)

        event = torch.cuda.Event()
        event.record()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )

    def test_event_wait_callback(self):
        gpu_trace.register_callback_for_event_wait(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.wait()
        self.mock.assert_called_once_with(
            event._as_parameter_.value, torch.cuda.default_stream().cuda_stream
        )

    def test_memory_allocation_callback(self):
        gpu_trace.register_callback_for_memory_allocation(self.mock)

        tensor = torch.empty(10, 4, device="cuda")
        self.mock.assert_called_once_with(tensor.data_ptr())

    def test_memory_deallocation_callback(self):
        gpu_trace.register_callback_for_memory_deallocation(self.mock)

        tensor = torch.empty(3, 8, device="cuda")
        data_ptr = tensor.data_ptr()
        del tensor
        self.mock.assert_called_once_with(data_ptr)

    def test_stream_creation_callback(self):
        gpu_trace.register_callback_for_stream_creation(self.mock)

        # see Note [HIP Lazy Streams]
        if torch.version.hip:
            user_stream = torch.cuda.Stream()
            with torch.cuda.stream(user_stream):
                torch.ones(5, device="cuda")
        else:
            torch.cuda.Stream()

        self.mock.assert_called()

    def test_device_synchronization_callback(self):
        gpu_trace.register_callback_for_device_synchronization(self.mock)

        torch.cuda.synchronize()
        self.mock.assert_called()

    def test_stream_synchronization_callback(self):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        stream = torch.cuda.Stream()
        stream.synchronize()
        self.mock.assert_called_once_with(stream.cuda_stream)

    def test_event_synchronization_callback(self):
        gpu_trace.register_callback_for_event_synchronization(self.mock)

        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        self.mock.assert_called_once_with(event._as_parameter_.value)

    def test_memcpy_synchronization(self):
        gpu_trace.register_callback_for_stream_synchronization(self.mock)

        tensor = torch.rand(5, device="cuda")
        tensor.nonzero()
        self.mock.assert_called_once_with(torch.cuda.default_stream().cuda_stream)

    def test_all_trace_callbacks_called(self):
        other = unittest.mock.MagicMock()
        gpu_trace.register_callback_for_memory_allocation(self.mock)
        gpu_trace.register_callback_for_memory_allocation(other)

        tensor = torch.empty(10, 4, device="cuda")
        self.mock.assert_called_once_with(tensor.data_ptr())
        other.assert_called_once_with(tensor.data_ptr())


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCudaTrace`

**Functions defined**: `setUp`, `test_event_creation_callback`, `test_event_deletion_callback`, `test_event_record_callback`, `test_event_wait_callback`, `test_memory_allocation_callback`, `test_memory_deallocation_callback`, `test_stream_creation_callback`, `test_device_synchronization_callback`, `test_stream_synchronization_callback`, `test_event_synchronization_callback`, `test_memcpy_synchronization`, `test_all_trace_callbacks_called`

**Key imports**: sys, unittest, unittest.mock, torch, torch.cuda._gpu_trace as gpu_trace, NoTest, run_tests, TEST_CUDA, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `unittest.mock`
- `torch`
- `torch.cuda._gpu_trace as gpu_trace`
- `torch.testing._internal.common_utils`: NoTest, run_tests, TEST_CUDA, TestCase


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
python test/test_cuda_trace.py
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

- **File Documentation**: `test_cuda_trace.py_docs.md`
- **Keyword Index**: `test_cuda_trace.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

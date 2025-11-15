# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_streams.py`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_streams.py`
- **Size**: 2,509 bytes (2.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestStream(TestCase):
    @skipIfTorchDynamo()
    def test_stream_create(self):
        stream = torch.Stream(device="openreg")
        self.assertEqual(stream.device_index, torch.openreg.current_device())

        stream = torch.Stream(device="openreg:1")
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream = torch.Stream(1)
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream1 = torch.Stream(
            stream_id=stream.stream_id,
            device_type=stream.device_type,
            device_index=stream.device_index,
        )
        self.assertEqual(stream, stream1)

    @skipIfTorchDynamo()
    def test_stream_context(self):
        with torch.Stream(device="openreg:1") as stream:
            self.assertEqual(torch.accelerator.current_stream(), stream)

    @skipIfTorchDynamo()
    def test_stream_switch(self):
        stream1 = torch.Stream(device="openreg:0")
        torch.accelerator.set_stream(stream1)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)

        stream2 = torch.Stream(device="openreg:1")
        torch.accelerator.set_stream(stream2)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream2)

    @skipIfTorchDynamo()
    def test_stream_synchronize(self):
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(True, stream.query())

        event = torch.Event()
        event.record(stream)
        stream.synchronize()
        self.assertEqual(True, stream.query())

    @skipIfTorchDynamo()
    def test_stream_repr(self):
        stream = torch.Stream(device="openreg:1")
        self.assertTrue(
            "torch.Stream device_type=openreg, device_index=1" in repr(stream)
        )

    @skipIfTorchDynamo()
    def test_stream_wait_stream(self):
        stream_1 = torch.Stream(device="openreg:0")
        stream_2 = torch.Stream(device="openreg:1")
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestStream`

**Functions defined**: `test_stream_create`, `test_stream_context`, `test_stream_switch`, `test_stream_synchronize`, `test_stream_repr`, `test_stream_wait_stream`, `test_stream_wait_event`

**Key imports**: torch, run_tests, skipIfTorchDynamo, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/tests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_streams.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/tests`):

- [`test_rng.py_docs.md`](./test_rng.py_docs.md)
- [`test_device.py_docs.md`](./test_device.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_storage.py_docs.md`](./test_storage.py_docs.md)
- [`test_misc.py_docs.md`](./test_misc.py_docs.md)
- [`test_memory.py_docs.md`](./test_memory.py_docs.md)
- [`test_autocast.py_docs.md`](./test_autocast.py_docs.md)
- [`test_ops.py_docs.md`](./test_ops.py_docs.md)
- [`test_event.py_docs.md`](./test_event.py_docs.md)
- [`test_autograd.py_docs.md`](./test_autograd.py_docs.md)


## Cross-References

- **File Documentation**: `test_streams.py_docs.md`
- **Keyword Index**: `test_streams.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_event.py`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_event.py`
- **Size**: 2,449 bytes (2.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestEvent(TestCase):
    @skipIfTorchDynamo()
    def test_event_create(self):
        event = torch.Event(device="openreg")
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = torch.Event(device="openreg:1")
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = torch.Event()
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        stream = torch.Stream(device="openreg:1")
        event = stream.record_event()
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, 1)
        self.assertNotEqual(event.event_id, 0)

    @skipIfTorchDynamo()
    def test_event_query(self):
        event = torch.Event()
        self.assertTrue(event.query())

        stream = torch.Stream(device="openreg:1")
        event = stream.record_event()
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_event_record(self):
        stream = torch.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)

        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)

        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
        stream = torch.Stream(device="openreg:1")

        event1 = torch.Event(device="openreg:1", enable_timing=True)
        event1.record(stream)
        event2 = torch.Event(device="openreg:1", enable_timing=True)
        event2.record(stream)

        stream.synchronize()
        self.assertTrue(event1.query())
        self.assertTrue(event2.query())

        ms = event1.elapsed_time(event2)
        self.assertTrue(ms > 0)

    @skipIfTorchDynamo()
    def test_event_wait_stream(self):
        stream1 = torch.Stream(device="openreg")
        stream2 = torch.Stream(device="openreg")

        event = stream1.record_event()
        stream2.wait_event(event)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestEvent`

**Functions defined**: `test_event_create`, `test_event_query`, `test_event_record`, `test_event_elapsed_time`, `test_event_wait_stream`

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
python test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_event.py
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
- [`test_autograd.py_docs.md`](./test_autograd.py_docs.md)


## Cross-References

- **File Documentation**: `test_event.py_docs.md`
- **Keyword Index**: `test_event.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

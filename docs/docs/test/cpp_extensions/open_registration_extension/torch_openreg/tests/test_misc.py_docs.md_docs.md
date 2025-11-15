# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_misc.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_misc.py_docs.md`
- **Size**: 9,469 bytes (9.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_misc.py`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_misc.py`
- **Size**: 6,113 bytes (5.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: PrivateUse1"]

import types
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestBackendModule(TestCase):
    def test_backend_module_name(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")
        # backend can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("openreg")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dev")

    def test_backend_module_registration(self):
        def generate_faked_module():
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("openreg", generate_faked_module())

    def test_backend_module_function(self):
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 2
        )


class TestBackendProperty(TestCase):
    def test_backend_generate_methods(self):
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        self.assertTrue(hasattr(torch.Tensor, "is_openreg"))
        self.assertTrue(hasattr(torch.Tensor, "openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.nn.Module, "openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "openreg"))

    def test_backend_tensor_methods(self):
        x = torch.empty(4, 4)
        self.assertFalse(x.is_openreg)

        y = x.openreg(torch.device("openreg"))
        self.assertTrue(y.is_openreg)
        z = x.openreg(torch.device("openreg:0"))
        self.assertTrue(z.is_openreg)
        n = x.openreg(0)
        self.assertTrue(n.is_openreg)

    @unittest.skip("Need to support Parameter in openreg")
    def test_backend_module_methods(self):
        class FakeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self):
                pass

        module = FakeModule()
        self.assertEqual(module.x.device.type, "cpu")
        module.openreg()  # type: ignore[misc]
        self.assertEqual(module.x.device.type, "openreg")

    @unittest.skip("Need to support untyped_storage in openreg")
    def test_backend_storage_methods(self):
        x = torch.empty(4, 4)

        x_cpu = x.storage()
        self.assertFalse(x_cpu.is_openreg)
        x_openreg = x_cpu.openreg()
        self.assertTrue(x_openreg.is_openreg)

        y = torch.empty(4, 4)

        y_cpu = y.untyped_storage()
        self.assertFalse(y_cpu.is_openreg)
        y_openreg = y_cpu.openreg()
        self.assertTrue(y_openreg.is_openreg)

    def test_backend_packed_sequence_methods(self):
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)


class TestTensorType(TestCase):
    def test_backend_tensor_type(self):
        dtypes_map = {
            torch.bool: "torch.openreg.BoolTensor",
            torch.double: "torch.openreg.DoubleTensor",
            torch.float32: "torch.openreg.FloatTensor",
            torch.half: "torch.openreg.HalfTensor",
            torch.int32: "torch.openreg.IntTensor",
            torch.int64: "torch.openreg.LongTensor",
            torch.int8: "torch.openreg.CharTensor",
            torch.short: "torch.openreg.ShortTensor",
            torch.uint8: "torch.openreg.ByteTensor",
        }

        for dtype, str in dtypes_map.items():
            x = torch.empty(4, 4, dtype=dtype, device="openreg")
            self.assertTrue(x.type() == str)

    # Note that all dtype-d Tensor objects here are only for legacy reasons
    # and should NOT be used.
    @skipIfTorchDynamo()
    def test_backend_type_methods(self):
        # Tensor
        tensor_cpu = torch.randn([8]).float()
        self.assertEqual(tensor_cpu.type(), "torch.FloatTensor")

        tensor_openreg = tensor_cpu.openreg()
        self.assertEqual(tensor_openreg.type(), "torch.openreg.FloatTensor")

        # Storage
        storage_cpu = tensor_cpu.storage()
        self.assertEqual(storage_cpu.type(), "torch.FloatStorage")

        tensor_openreg = tensor_cpu.openreg()
        storage_openreg = tensor_openreg.storage()
        self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        try:
            torch.openreg.FloatStorage = CustomFloatStorage()
            self.assertEqual(storage_openreg.type(), "torch.openreg.FloatStorage")

            # test custom int storage after defining FloatStorage
            tensor_openreg = tensor_cpu.int().openreg()
            storage_openreg = tensor_openreg.storage()
            self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")
        finally:
            torch.openreg.FloatStorage = None


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestBackendModule`, `TestBackendProperty`, `FakeModule`, `TestTensorType`, `CustomFloatStorage`

**Functions defined**: `test_backend_module_name`, `test_backend_module_registration`, `generate_faked_module`, `test_backend_module_function`, `test_backend_generate_methods`, `test_backend_tensor_methods`, `test_backend_module_methods`, `__init__`, `forward`, `test_backend_storage_methods`, `test_backend_packed_sequence_methods`, `test_backend_tensor_type`, `test_backend_type_methods`, `__module__`, `__name__`

**Key imports**: types, unittest, torch, run_tests, skipIfTorchDynamo, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/tests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `types`
- `unittest`
- `torch`
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_misc.py
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
- [`test_memory.py_docs.md`](./test_memory.py_docs.md)
- [`test_autocast.py_docs.md`](./test_autocast.py_docs.md)
- [`test_ops.py_docs.md`](./test_ops.py_docs.md)
- [`test_event.py_docs.md`](./test_event.py_docs.md)
- [`test_autograd.py_docs.md`](./test_autograd.py_docs.md)


## Cross-References

- **File Documentation**: `test_misc.py_docs.md`
- **Keyword Index**: `test_misc.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg/tests`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg/tests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_misc.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg/tests`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_streams.py_kw.md_docs.md`](./test_streams.py_kw.md_docs.md)
- [`test_streams.py_docs.md_docs.md`](./test_streams.py_docs.md_docs.md)
- [`test_storage.py_kw.md_docs.md`](./test_storage.py_kw.md_docs.md)
- [`test_rng.py_docs.md_docs.md`](./test_rng.py_docs.md_docs.md)
- [`test_memory.py_docs.md_docs.md`](./test_memory.py_docs.md_docs.md)
- [`test_misc.py_kw.md_docs.md`](./test_misc.py_kw.md_docs.md)
- [`test_rng.py_kw.md_docs.md`](./test_rng.py_kw.md_docs.md)
- [`test_autocast.py_docs.md_docs.md`](./test_autocast.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_misc.py_docs.md_docs.md`
- **Keyword Index**: `test_misc.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

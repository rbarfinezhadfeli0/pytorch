# Documentation: `test/test_extension_utils.py`

## File Metadata

- **Path**: `test/test_extension_utils.py`
- **Size**: 3,024 bytes (2.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: PrivateUse1"]
import sys

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class DummyPrivateUse1Module:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_autocast_enabled():
        return True

    @staticmethod
    def get_autocast_dtype():
        return torch.float16

    @staticmethod
    def set_autocast_enabled(enable):
        pass

    @staticmethod
    def set_autocast_dtype(dtype):
        pass

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16]


class TestExtensionUtils(TestCase):
    def tearDown(self):
        # Clean up
        backend_name = torch._C._get_privateuse1_backend_name()
        if hasattr(torch, backend_name):
            delattr(torch, backend_name)
        if f"torch.{backend_name}" in sys.modules:
            del sys.modules[f"torch.{backend_name}"]

    def test_external_module_register(self):
        # Built-in module
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("cuda", torch.cuda)

        # Wrong device type
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dummmy", DummyPrivateUse1Module)

        with self.assertRaises(AttributeError):
            torch.privateuseone.is_available()  # type: ignore[attr-defined]

        torch._register_device_module("privateuseone", DummyPrivateUse1Module)

        torch.privateuseone.is_available()  # type: ignore[attr-defined]

        # No supporting for override
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("privateuseone", DummyPrivateUse1Module)

    @skipIfTorchDynamo(
        "accelerator doesn't compose with privateuse1 : https://github.com/pytorch/pytorch/issues/166696"
    )
    def test_external_module_register_with_renamed_backend(self):
        torch.utils.rename_privateuse1_backend("foo")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "foo")

        with self.assertRaises(AttributeError):
            torch.foo.is_available()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("foo", DummyPrivateUse1Module)

        torch.foo.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("foo:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("foo:2")), 2)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DummyPrivateUse1Module`, `TestExtensionUtils`

**Functions defined**: `is_available`, `is_autocast_enabled`, `get_autocast_dtype`, `set_autocast_enabled`, `set_autocast_dtype`, `get_amp_supported_dtype`, `tearDown`, `test_external_module_register`, `test_external_module_register_with_renamed_backend`

**Key imports**: sys, torch, run_tests, skipIfTorchDynamo, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.testing._internal.common_utils`: run_tests, skipIfTorchDynamo, TestCase


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
python test/test_extension_utils.py
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

- **File Documentation**: `test_extension_utils.py_docs.md`
- **Keyword Index**: `test_extension_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

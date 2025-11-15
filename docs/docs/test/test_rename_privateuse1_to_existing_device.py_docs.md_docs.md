# Documentation: `docs/test/test_rename_privateuse1_to_existing_device.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_rename_privateuse1_to_existing_device.py_docs.md`
- **Size**: 4,783 bytes (4.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_rename_privateuse1_to_existing_device.py`

## File Metadata

- **Path**: `test/test_rename_privateuse1_to_existing_device.py`
- **Size**: 1,826 bytes (1.78 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: PrivateUse1"]

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


class TestRenamePrivateuseoneToExistingBackend(TestCase):
    @skipIfTorchDynamo(
        "TorchDynamo exposes https://github.com/pytorch/pytorch/issues/166696"
    )
    def test_external_module_register_with_existing_backend(self):
        torch.utils.rename_privateuse1_backend("maia")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "maia")

        with self.assertRaises(AttributeError):
            torch.maia.is_available()

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("maia", DummyPrivateUse1Module)

        torch.maia.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("maia:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("maia:2")), 2)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DummyPrivateUse1Module`, `TestRenamePrivateuseoneToExistingBackend`

**Functions defined**: `is_available`, `is_autocast_enabled`, `get_autocast_dtype`, `set_autocast_enabled`, `set_autocast_dtype`, `get_amp_supported_dtype`, `test_external_module_register_with_existing_backend`

**Key imports**: torch, run_tests, skipIfTorchDynamo, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



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
python test/test_rename_privateuse1_to_existing_device.py
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

- **File Documentation**: `test_rename_privateuse1_to_existing_device.py_docs.md`
- **Keyword Index**: `test_rename_privateuse1_to_existing_device.py_kw.md`
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
python docs/test/test_rename_privateuse1_to_existing_device.py_docs.md
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

- **File Documentation**: `test_rename_privateuse1_to_existing_device.py_docs.md_docs.md`
- **Keyword Index**: `test_rename_privateuse1_to_existing_device.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

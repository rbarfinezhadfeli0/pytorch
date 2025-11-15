# Documentation: `test/test_type_info.py`

## File Metadata

- **Path**: `test/test_type_info.py`
- **Size**: 5,288 bytes (5.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
# Owner(s): ["module: typing"]

from torch.testing._internal.common_utils import (
    load_tests,
    run_tests,
    set_default_dtype,
    TEST_NUMPY,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

import sys
import unittest

import torch


if TEST_NUMPY:
    import numpy as np


class TestDTypeInfo(TestCase):
    def test_invalid_input(self):
        for dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
            torch.bool,
        ]:
            with self.assertRaises(TypeError):
                _ = torch.iinfo(dtype)

        for dtype in [
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]:
            with self.assertRaises(TypeError):
                _ = torch.finfo(dtype)
            with self.assertRaises(RuntimeError):
                dtype.to_complex()

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_iinfo(self):
        for dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
            x = torch.zeros((2, 2), dtype=dtype)
            xinfo = torch.iinfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.iinfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.dtype, xninfo.dtype)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_finfo(self):
        for dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]:
            x = torch.zeros((2, 2), dtype=dtype)
            xinfo = torch.finfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.finfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.eps, xninfo.eps)
            self.assertEqual(xinfo.tiny, xninfo.tiny)
            self.assertEqual(xinfo.resolution, xninfo.resolution)
            self.assertEqual(xinfo.dtype, xninfo.dtype)
            if not dtype.is_complex:
                with set_default_dtype(dtype):
                    self.assertEqual(torch.finfo(dtype), torch.finfo())

        # Special test case for BFloat16 type
        x = torch.zeros((2, 2), dtype=torch.bfloat16)
        xinfo = torch.finfo(x.dtype)
        self.assertEqual(xinfo.bits, 16)
        self.assertEqual(xinfo.max, 3.38953e38)
        self.assertEqual(xinfo.min, -3.38953e38)
        self.assertEqual(xinfo.eps, 0.0078125)
        self.assertEqual(xinfo.tiny, 1.17549e-38)
        self.assertEqual(xinfo.tiny, xinfo.smallest_normal)
        self.assertEqual(xinfo.resolution, 0.01)
        self.assertEqual(xinfo.dtype, "bfloat16")
        with set_default_dtype(x.dtype):
            self.assertEqual(torch.finfo(x.dtype), torch.finfo())

        # Special test case for Float8_E5M2
        xinfo = torch.finfo(torch.float8_e5m2)
        self.assertEqual(xinfo.bits, 8)
        self.assertEqual(xinfo.max, 57344.0)
        self.assertEqual(xinfo.min, -57344.0)
        self.assertEqual(xinfo.eps, 0.25)
        self.assertEqual(xinfo.tiny, 6.10352e-05)
        self.assertEqual(xinfo.resolution, 1.0)
        self.assertEqual(xinfo.dtype, "float8_e5m2")

        # Special test case for Float8_E4M3FN
        xinfo = torch.finfo(torch.float8_e4m3fn)
        self.assertEqual(xinfo.bits, 8)
        self.assertEqual(xinfo.max, 448.0)
        self.assertEqual(xinfo.min, -448.0)
        self.assertEqual(xinfo.eps, 0.125)
        self.assertEqual(xinfo.tiny, 0.015625)
        self.assertEqual(xinfo.resolution, 1.0)
        self.assertEqual(xinfo.dtype, "float8_e4m3fn")

    def test_to_complex(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/124868
        # If reference count is leaked this would be a set of 10 elements
        ref_cnt = {sys.getrefcount(torch.float32.to_complex()) for _ in range(10)}

        self.assertLess(len(ref_cnt), 3)

        self.assertEqual(torch.float64.to_complex(), torch.complex128)
        self.assertEqual(torch.float32.to_complex(), torch.complex64)
        self.assertEqual(torch.float16.to_complex(), torch.complex32)

    def test_to_real(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/124868
        # If reference count is leaked this would be a set of 10 elements
        ref_cnt = {sys.getrefcount(torch.cfloat.to_real()) for _ in range(10)}

        self.assertLess(len(ref_cnt), 3)

        self.assertEqual(torch.complex128.to_real(), torch.double)
        self.assertEqual(torch.complex64.to_real(), torch.float32)
        self.assertEqual(torch.complex32.to_real(), torch.float16)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDTypeInfo`

**Functions defined**: `test_invalid_input`, `test_iinfo`, `test_finfo`, `test_to_complex`, `test_to_real`

**Key imports**: sys, unittest, torch, numpy as np


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `torch`
- `numpy as np`


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
python test/test_type_info.py
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


## Cross-References

- **File Documentation**: `test_type_info.py_docs.md`
- **Keyword Index**: `test_type_info.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

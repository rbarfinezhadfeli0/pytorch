# Documentation: `test/onnx/exporter/test_core.py`

## File Metadata

- **Path**: `test/onnx/exporter/test_core.py`
- **Size**: 5,729 bytes (5.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Unit tests for the _core module."""

from __future__ import annotations

import io
import os
import tempfile

import ml_dtypes
import numpy as np

import torch
from torch.onnx._internal.exporter import _core
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class TorchTensorTest(common_utils.TestCase):
    @common_utils.parametrize(
        "dtype, np_dtype",
        [
            (torch.bfloat16, ml_dtypes.bfloat16),
            (torch.bool, np.bool_),
            (torch.complex128, np.complex128),
            (torch.complex64, np.complex64),
            (torch.float16, np.float16),
            (torch.float32, np.float32),
            (torch.float64, np.float64),
            (torch.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            (torch.float8_e4m3fnuz, ml_dtypes.float8_e4m3fnuz),
            (torch.float8_e5m2, ml_dtypes.float8_e5m2),
            (torch.float8_e5m2fnuz, ml_dtypes.float8_e5m2fnuz),
            (torch.int16, np.int16),
            (torch.int32, np.int32),
            (torch.int64, np.int64),
            (torch.int8, np.int8),
            (torch.uint16, np.uint16),
            (torch.uint32, np.uint32),
            (torch.uint64, np.uint64),
            (torch.uint8, np.uint8),
            (torch.float4_e2m1fn_x2, ml_dtypes.float4_e2m1fn),
        ],
    )
    def test_numpy_returns_correct_dtype(self, dtype: torch.dtype, np_dtype):
        if dtype == torch.float4_e2m1fn_x2:
            tensor = _core.TorchTensor(torch.tensor([1], dtype=torch.uint8).view(dtype))
        else:
            tensor = _core.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.numpy().dtype, np_dtype)
        self.assertEqual(tensor.__array__().dtype, np_dtype)
        self.assertEqual(np.array(tensor).dtype, np_dtype)

    @common_utils.parametrize(
        "dtype",
        [
            torch.bfloat16,
            torch.bool,
            torch.complex128,
            torch.complex64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.uint8,
        ],
    )
    def test_tobytes(self, dtype: torch.dtype):
        tensor = _core.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.tobytes(), tensor.numpy().tobytes())

    def test_tobytes_float4(self):
        tensor = _core.TorchTensor(
            torch.tensor([1], dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        )
        self.assertEqual(tensor.tobytes(), b"\x01")


class TorchTensorToFileTest(common_utils.TestCase):
    def _roundtrip_file(self, tensor: _core.TorchTensor) -> bytes:
        expected = tensor.tobytes()
        # NamedTemporaryFile (binary)
        with tempfile.NamedTemporaryFile() as tmp:
            tensor.tofile(tmp)
            tmp.seek(0)
            data = tmp.read()
        self.assertEqual(data, expected)

        # Explicit path write using open handle
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bin.dat")
            with open(path, "wb") as f:
                tensor.tofile(f)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), expected)

        return expected

    def test_tofile_basic_uint8(self):
        tensor = _core.TorchTensor(torch.arange(10, dtype=torch.uint8))
        self._roundtrip_file(tensor)

    def test_tofile_float32(self):
        tensor = _core.TorchTensor(
            torch.arange(0, 16, dtype=torch.float32).reshape(4, 4)
        )
        self._roundtrip_file(tensor)

    def test_tofile_bfloat16(self):
        tensor = _core.TorchTensor(torch.arange(0, 8, dtype=torch.bfloat16))
        self._roundtrip_file(tensor)

    def test_tofile_float4_packed(self):
        # 3 packed bytes -> 6 logical float4 values (when unpacked), but we want packed bytes
        raw = torch.tensor([0x12, 0x34, 0xAB], dtype=torch.uint8)
        tensor = _core.TorchTensor(raw.view(torch.float4_e2m1fn_x2))
        expected = self._roundtrip_file(tensor)
        self.assertEqual(expected, bytes([0x12, 0x34, 0xAB]))

    def test_tofile_file_like_no_fileno(self):
        tensor = _core.TorchTensor(torch.arange(0, 32, dtype=torch.uint8))
        buf = io.BytesIO()
        tensor.tofile(buf)
        self.assertEqual(buf.getvalue(), tensor.tobytes())

    def test_tofile_text_mode_error(self):
        tensor = _core.TorchTensor(torch.arange(0, 4, dtype=torch.uint8))
        with tempfile.NamedTemporaryFile(mode="w") as tmp_text:
            path = tmp_text.name
            with open(path, "w") as f_text:
                with self.assertRaises(TypeError):
                    tensor.tofile(f_text)

    def test_tofile_non_contiguous(self):
        base = torch.arange(0, 64, dtype=torch.int32).reshape(8, 8)
        sliced = base[:, ::2]  # Stride in last dim -> non-contiguous
        self.assertFalse(sliced.is_contiguous())
        tensor = _core.TorchTensor(sliced)
        # Ensure bytes correspond to the contiguous clone inside implementation
        expected_manual = sliced.contiguous().numpy().tobytes()
        with tempfile.NamedTemporaryFile() as tmp:
            tensor.tofile(tmp)
            tmp.seek(0)
            data = tmp.read()
        self.assertEqual(data, expected_manual)
        self.assertEqual(tensor.tobytes(), expected_manual)


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Unit tests for the _core module."""from __future__ import annotationsimport ioimport osimport tempfileimport ml_dtypesimport numpy as npimport torchfrom torch.onnx._internal.exporter import _corefrom torch.testing._internal import common_utils@common_utils.instantiate_parametrized_testsclass TorchTensorTest(common_utils.TestCase):    @common_utils.parametrize(        "dtype, np_dtype",        [            (torch.bfloat16, ml_dtypes.bfloat16),            (torch.bool, np.bool_),            (torch.complex128, np.complex128),            (torch.complex64, np.complex64),            (torch.float16, np.float16),            (torch.float32, np.float32),            (torch.float64, np.float64),            (torch.float8_e4m3fn, ml_dtypes.float8_e4m3fn),            (torch.float8_e4m3fnuz, ml_dtypes.float8_e4m3fnuz),            (torch.float8_e5m2, ml_dtypes.float8_e5m2),            (torch.float8_e5m2fnuz, ml_dtypes.float8_e5m2fnuz),            (torch.int16, np.int16),            (torch.int32, np.int32),            (torch.int64, np.int64),            (torch.int8, np.int8),            (torch.uint16, np.uint16),            (torch.uint32, np.uint32),            (torch.uint64, np.uint64),            (torch.uint8, np.uint8),            (torch.float4_e2m1fn_x2, ml_dtypes.float4_e2m1fn),        ],    )    def test_numpy_returns_correct_dtype(self, dtype: torch.dtype, np_dtype):        if dtype == torch.float4_e2m1fn_x2:            tensor = _core.TorchTensor(torch.tensor([1], dtype=torch.uint8).view(dtype))        else:            tensor = _core.TorchTensor(torch.tensor([1], dtype=dtype))        self.assertEqual(tensor.numpy().dtype, np_dtype)

This Python file contains 2 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TorchTensorTest`, `TorchTensorToFileTest`

**Functions defined**: `test_numpy_returns_correct_dtype`, `test_tobytes`, `test_tobytes_float4`, `_roundtrip_file`, `test_tofile_basic_uint8`, `test_tofile_float32`, `test_tofile_bfloat16`, `test_tofile_float4_packed`, `test_tofile_file_like_no_fileno`, `test_tofile_text_mode_error`, `test_tofile_non_contiguous`

**Key imports**: annotations, io, os, tempfile, ml_dtypes, numpy as np, torch, _core, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `io`
- `os`
- `tempfile`
- `ml_dtypes`
- `numpy as np`
- `torch`
- `torch.onnx._internal.exporter`: _core
- `torch.testing._internal`: common_utils


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
python test/onnx/exporter/test_core.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/exporter`):

- [`test_capture_strategies.py_docs.md`](./test_capture_strategies.py_docs.md)
- [`test_building.py_docs.md`](./test_building.py_docs.md)
- [`test_hf_models_e2e.py_docs.md`](./test_hf_models_e2e.py_docs.md)
- [`test_verification.py_docs.md`](./test_verification.py_docs.md)
- [`test_dynamic_shapes.py_docs.md`](./test_dynamic_shapes.py_docs.md)
- [`test_small_models_e2e.py_docs.md`](./test_small_models_e2e.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test_ir_passes.py_docs.md`](./test_ir_passes.py_docs.md)
- [`test_tensors.py_docs.md`](./test_tensors.py_docs.md)


## Cross-References

- **File Documentation**: `test_core.py_docs.md`
- **Keyword Index**: `test_core.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

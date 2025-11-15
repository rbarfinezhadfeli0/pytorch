# Documentation: `test/onnx/exporter/test_verification.py`

## File Metadata

- **Path**: `test/onnx/exporter/test_verification.py`
- **Size**: 5,456 bytes (5.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]
"""Test the verification module."""

from __future__ import annotations

import json

import torch
from torch.onnx._internal.exporter import _verification
from torch.testing._internal import common_utils


class VerificationInfoTest(common_utils.TestCase):
    def test_from_tensors(self):
        # Test with tensors
        expected = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0, 3.0])
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor", expected, actual
        )
        self.assertEqual(verification_info.name, "test_tensor")
        self.assertEqual(verification_info.max_abs_diff, 0)
        self.assertEqual(verification_info.max_rel_diff, 0)
        torch.testing.assert_close(
            verification_info.abs_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)
        )
        torch.testing.assert_close(
            verification_info.rel_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)
        )
        self.assertEqual(verification_info.expected_dtype, torch.float32)
        self.assertEqual(verification_info.actual_dtype, torch.float32)

    def test_from_tensors_int(self):
        # Test with int tensors
        expected = torch.tensor([1])
        actual = 1
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor_int", expected, actual
        )
        self.assertEqual(verification_info.name, "test_tensor_int")
        self.assertEqual(verification_info.max_abs_diff, 0)
        self.assertEqual(verification_info.max_rel_diff, 0)
        torch.testing.assert_close(
            verification_info.abs_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)
        )
        torch.testing.assert_close(
            verification_info.rel_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)
        )
        self.assertEqual(verification_info.expected_dtype, torch.int64)
        self.assertEqual(verification_info.actual_dtype, torch.int64)

    def test_asdict(self):
        # Test the asdict method
        expected = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0, 3.0])
        verification_info = _verification.VerificationInfo.from_tensors(
            "test_tensor", expected, actual
        )
        asdict_result = verification_info.asdict()
        self.assertEqual(asdict_result["name"], "test_tensor")
        self.assertEqual(asdict_result["max_abs_diff"], 0)
        self.assertEqual(asdict_result["max_rel_diff"], 0)
        self.assertEqual(
            asdict_result["abs_diff_hist"],
            [
                [3.0] + [0.0] * 8,
                [0.0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 1000000.0],
            ],
        )
        self.assertEqual(
            asdict_result["rel_diff_hist"],
            [
                [3.0] + [0.0] * 8,
                [0.0, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 1000000.0],
            ],
        )
        self.assertEqual(asdict_result["expected_dtype"], "torch.float32")
        self.assertEqual(asdict_result["actual_dtype"], "torch.float32")
        # Ensure it can be round tripped as json
        json_str = json.dumps(asdict_result)
        loaded_dict = json.loads(json_str)
        self.assertEqual(loaded_dict, asdict_result)


class VerificationInterpreterTest(common_utils.TestCase):
    def test_interpreter_stores_correct_info(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = a + b
                return c - 1

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        assert onnx_program is not None
        interpreter = _verification._VerificationInterpreter(onnx_program)
        results = interpreter.run(args)
        torch.testing.assert_close(results, model(*args))
        verification_infos = interpreter.verification_infos
        self.assertEqual(len(verification_infos), 3)
        for info in verification_infos:
            self.assertEqual(info.max_abs_diff, 0)
            self.assertEqual(info.max_rel_diff, 0)


class VerificationFunctionsTest(common_utils.TestCase):
    def test_verify_onnx_program(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = a + b
                return c - 1, c

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        assert onnx_program is not None
        verification_infos = _verification.verify_onnx_program(
            onnx_program, args, compare_intermediates=False
        )
        self.assertEqual(len(verification_infos), 2)

    def test_verify_onnx_program_with_compare_intermediates_true(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = a + b
                return c - 1, c

        model = Model()
        args = (torch.tensor([1.0]), torch.tensor([2.0]))
        onnx_program = torch.onnx.export(model, args, dynamo=True, verbose=False)
        assert onnx_program is not None
        verification_infos = _verification.verify_onnx_program(
            onnx_program, args, compare_intermediates=True
        )
        self.assertEqual(len(verification_infos), 3)


if __name__ == "__main__":
    common_utils.run_tests()

```



## High-Level Overview

"""Test the verification module."""from __future__ import annotationsimport jsonimport torchfrom torch.onnx._internal.exporter import _verificationfrom torch.testing._internal import common_utilsclass VerificationInfoTest(common_utils.TestCase):    def test_from_tensors(self):        # Test with tensors        expected = torch.tensor([1.0, 2.0, 3.0])        actual = torch.tensor([1.0, 2.0, 3.0])        verification_info = _verification.VerificationInfo.from_tensors(            "test_tensor", expected, actual        )        self.assertEqual(verification_info.name, "test_tensor")        self.assertEqual(verification_info.max_abs_diff, 0)        self.assertEqual(verification_info.max_rel_diff, 0)        torch.testing.assert_close(            verification_info.abs_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)        )        torch.testing.assert_close(            verification_info.rel_diff_hist[0], torch.tensor([3.0] + [0.0] * 8)        )        self.assertEqual(verification_info.expected_dtype, torch.float32)        self.assertEqual(verification_info.actual_dtype, torch.float32)    def test_from_tensors_int(self):        # Test with int tensors        expected = torch.tensor([1])        actual = 1        verification_info = _verification.VerificationInfo.from_tensors(            "test_tensor_int", expected, actual        )        self.assertEqual(verification_info.name, "test_tensor_int")        self.assertEqual(verification_info.max_abs_diff, 0)        self.assertEqual(verification_info.max_rel_diff, 0)        torch.testing.assert_close(            verification_info.abs_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)        )        torch.testing.assert_close(            verification_info.rel_diff_hist[0], torch.tensor([1.0] + [0.0] * 8)        )        self.assertEqual(verification_info.expected_dtype, torch.int64)        self.assertEqual(verification_info.actual_dtype, torch.int64)

This Python file contains 6 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `VerificationInfoTest`, `VerificationInterpreterTest`, `Model`, `VerificationFunctionsTest`, `Model`, `Model`

**Functions defined**: `test_from_tensors`, `test_from_tensors_int`, `test_asdict`, `test_interpreter_stores_correct_info`, `forward`, `test_verify_onnx_program`, `forward`, `test_verify_onnx_program_with_compare_intermediates_true`, `forward`

**Key imports**: annotations, json, torch, _verification, common_utils


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx/exporter`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `json`
- `torch`
- `torch.onnx._internal.exporter`: _verification
- `torch.testing._internal`: common_utils


## Code Patterns & Idioms

### Common Patterns

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
python test/onnx/exporter/test_verification.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx/exporter`):

- [`test_capture_strategies.py_docs.md`](./test_capture_strategies.py_docs.md)
- [`test_building.py_docs.md`](./test_building.py_docs.md)
- [`test_hf_models_e2e.py_docs.md`](./test_hf_models_e2e.py_docs.md)
- [`test_dynamic_shapes.py_docs.md`](./test_dynamic_shapes.py_docs.md)
- [`test_small_models_e2e.py_docs.md`](./test_small_models_e2e.py_docs.md)
- [`test_core.py_docs.md`](./test_core.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`test_ir_passes.py_docs.md`](./test_ir_passes.py_docs.md)
- [`test_tensors.py_docs.md`](./test_tensors.py_docs.md)


## Cross-References

- **File Documentation**: `test_verification.py_docs.md`
- **Keyword Index**: `test_verification.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

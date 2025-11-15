# Documentation: test_verification.py

## File Metadata
- **Path**: `test/onnx/exporter/test_verification.py`
- **Size**: 5456 bytes
- **Lines**: 139
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 6 class(es): VerificationInfoTest, VerificationInterpreterTest, Model, VerificationFunctionsTest, Model, Model

### Functions
This file defines 9 function(s): test_from_tensors, test_from_tensors_int, test_asdict, test_interpreter_stores_correct_info, forward, test_verify_onnx_program, forward, test_verify_onnx_program_with_compare_intermediates_true, forward


## Key Components

The file contains 370 words across 139 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5456 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

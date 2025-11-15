# Documentation: test_aten_pow.py

## File Metadata
- **Path**: `test/jit/test_aten_pow.py`
- **Size**: 4444 bytes
- **Lines**: 105
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["oncall: jit"]

import torch
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class TestAtenPow(TestCase):
    def test_aten_pow_zero_negative_exponent(self):
        """
        1. Testing a = int, b = int
        """

        @torch.jit.script
        def fn_int_int(a: int, b: int):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_int_int(2, 1), 2**1)
        self.assertEqual(fn_int_int(2, 0), 2**0)
        self.assertEqual(fn_int_int(2, -2), 2 ** (-2))
        self.assertEqual(fn_int_int(-2, 2), (-2) ** 2)
        self.assertEqual(fn_int_int(-2, 0), (-2) ** 0)
        self.assertEqual(fn_int_int(-2, -2), (-2) ** (-2))
        self.assertEqual(fn_int_int(-2, -1), (-2) ** (-1))
        self.assertEqual(fn_int_int(0, 2), 0**1)
        self.assertEqual(fn_int_int(0, 0), 0**0)
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_int, 0, -2)

        """
        2. Testing a = int, b = float
        """

        @torch.jit.script
        def fn_int_float(a: int, b: float):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_int_float(2, 2.5), 2**2.5)
        self.assertEqual(fn_int_float(2, -2.5), 2 ** (-2.5))
        self.assertEqual(fn_int_float(2, -0.0), 2 ** (-0.0))
        self.assertEqual(fn_int_float(2, 0.0), 2 ** (0.0))
        self.assertEqual(fn_int_float(-2, 2.0), (-2) ** 2.0)
        self.assertEqual(fn_int_float(-2, -2.0), (-2) ** (-2.0))
        self.assertEqual(fn_int_float(-2, -3.0), (-2) ** (-3.0))
        self.assertEqual(fn_int_float(-2, -0.0), (-2) ** (-0.0))
        self.assertEqual(fn_int_float(-2, 0.0), (-2) ** (0.0))
        self.assertEqual(fn_int_float(0, 2.0), 0**2.0)
        self.assertEqual(fn_int_float(0, 0.5), 0**0.5)
        self.assertEqual(fn_int_float(0, 0.0), 0**0.0)
        self.assertEqual(fn_int_float(0, -0.0), 0 ** (-0.0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_float, 0, -2.5)

        """
        3. Testing a = float, b = int
        """

        @torch.jit.script
        def fn_float_int(a: float, b: int):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_float_int(2.5, 2), 2.5**2)
        self.assertEqual(fn_float_int(2.5, -2), 2.5 ** (-2))
        self.assertEqual(fn_float_int(2.5, -0), 2.5 ** (-0))
        self.assertEqual(fn_float_int(2.5, 0), 2.5**0)
        self.assertEqual(fn_float_int(-2.5, 2), 2.5**2)
        self.assertEqual(fn_float_int(-2.5, -2), (-2.5) ** (-2))
        self.assertEqual(fn_float_int(-2.5, -3), (-2.5) ** (-3))
        self.assertEqual(fn_float_int(-2.5, -0), (-2.5) ** (-0))
        self.assertEqual(fn_float_int(-2.5, 0), (-2.5) ** 0)
        self.assertEqual(fn_float_int(0.0, 2), 0**2)
        self.assertEqual(fn_float_int(0.0, 0), 0**0)
        self.assertEqual(fn_float_int(0.0, -0), 0 ** (-0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_int, 0.0, -2)

        """
        4. Testing a = float, b = float
        """

        @torch.jit.script
        def fn_float_float(a: float, b: float):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_float_float(2.5, 2.0), 2.5**2.0)
        self.assertEqual(fn_float_float(2.5, -2.0), 2.5 ** (-2.0))
        self.assertEqual(fn_float_float(2.5, -0.0), 2.5 ** (-0.0))
        self.assertEqual(fn_float_float(2.5, 0.0), 2.5**0.0)
        self.assertEqual(fn_float_float(-2.5, 2.0), 2.5**2.0)
        self.assertEqual(fn_float_float(-2.5, -2.0), (-2.5) ** (-2.0))
        self.assertEqual(fn_float_float(-2.5, -3.0), (-2.5) ** (-3.0))
        self.assertEqual(fn_float_float(-2.5, -0.0), (-2.5) ** (-0.0))
        self.assertEqual(fn_float_float(-2.5, 0.0), (-2.5) ** 0.0)
        self.assertEqual(fn_float_float(0.0, 2.0), 0.0**2.0)
        self.assertEqual(fn_float_float(0.0, 0.0), 0.0**0.0)
        self.assertEqual(fn_float_float(0.0, -0.0), 0.0 ** (-0.0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_float, 0.0, -2.0)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TestAtenPow

### Functions
This file defines 5 function(s): test_aten_pow_zero_negative_exponent, fn_int_int, fn_int_float, fn_float_int, fn_float_float


## Key Components

The file contains 370 words across 105 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4444 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

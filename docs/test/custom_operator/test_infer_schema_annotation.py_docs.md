# Documentation: test_infer_schema_annotation.py

## File Metadata
- **Path**: `test/custom_operator/test_infer_schema_annotation.py`
- **Size**: 7085 bytes
- **Lines**: 210
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: pt2-dispatcher"]
from __future__ import annotations

import typing
from typing import Optional, Union

import torch
from torch import Tensor, types
from torch.testing._internal.common_utils import run_tests, TestCase


if typing.TYPE_CHECKING:
    from collections.abc import Sequence


mutates_args = {}


class TestInferSchemaWithAnnotation(TestCase):
    def test_tensor(self):
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.clone() + y

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x, Tensor y) -> Tensor")

    def test_native_types(self):
        def foo_op(x: int) -> int:
            return x

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt x) -> SymInt")

        def foo_op_2(x: bool) -> bool:
            return x

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(bool x) -> bool")

        def foo_op_3(x: str) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(str x) -> SymInt")

        def foo_op_4(x: float) -> float:
            return x

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(float x) -> float")

    def test_torch_types(self):
        def foo_op_1(x: torch.types.Number) -> torch.types.Number:
            return x

        result = torch.library.infer_schema(foo_op_1, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_2(x: torch.dtype) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(ScalarType x) -> SymInt")

        def foo_op_3(x: torch.device) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(Device x) -> SymInt")

    def test_type_variants(self):
        def foo_op_1(x: typing.Optional[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_1, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        def foo_op_2(x: typing.Sequence[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_3(x: list[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_4(x: typing.Optional[typing.Sequence[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_5(x: typing.Optional[list[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_5, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_6(x: typing.Union[int, float, bool]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_6, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_7(x: typing.Union[int, bool, float]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_7, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

    def test_no_library_prefix(self):
        def foo_op(x: Tensor) -> Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_2(x: Tensor) -> torch.Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_3(x: torch.Tensor) -> Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_4(x: list[int]) -> types.Number:
            return x[0]

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> Scalar")

        def foo_op_5(x: Optional[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_5, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        def foo_op_6(x: Sequence[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_6, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_7(x: list[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_7, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_8(x: Optional[Sequence[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_8, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_9(x: Optional[list[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_9, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_10(x: Union[int, float, bool]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_10, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_11(x: Union[int, bool, float]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_11, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

    def test_unsupported_annotation(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation D. It is not a type.",
        ):

            def foo_op(x: D) -> Tensor:  # noqa: F821
                return torch.Tensor(x)

            torch.library.infer_schema(foo_op, mutates_args=mutates_args)

        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation E. It is not a type.",
        ):

            def foo_op_2(x: Tensor) -> E:  # noqa: F821
                return x

            torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)


if __name__ == "__main__":
    run_tests()

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TestInferSchemaWithAnnotation

### Functions
This file defines 35 function(s): test_tensor, foo_op, foo_op_2, test_native_types, foo_op, foo_op_2, foo_op_3, foo_op_4, test_torch_types, foo_op_1, foo_op_2, foo_op_3, test_type_variants, foo_op_1, foo_op_2, foo_op_3, foo_op_4, foo_op_5, foo_op_6, foo_op_7, test_no_library_prefix, foo_op, foo_op_2, foo_op_3, foo_op_4, foo_op_5, foo_op_6, foo_op_7, foo_op_8, foo_op_9


## Key Components

The file contains 551 words across 210 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 7085 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

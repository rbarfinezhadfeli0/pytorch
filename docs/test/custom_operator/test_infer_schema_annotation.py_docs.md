# Documentation: `test/custom_operator/test_infer_schema_annotation.py`

## File Metadata

- **Path**: `test/custom_operator/test_infer_schema_annotation.py`
- **Size**: 7,085 bytes (6.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
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


This Python file contains 1 class(es) and 35 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestInferSchemaWithAnnotation`

**Functions defined**: `test_tensor`, `foo_op`, `foo_op_2`, `test_native_types`, `foo_op`, `foo_op_2`, `foo_op_3`, `foo_op_4`, `test_torch_types`, `foo_op_1`, `foo_op_2`, `foo_op_3`, `test_type_variants`, `foo_op_1`, `foo_op_2`, `foo_op_3`, `foo_op_4`, `foo_op_5`, `foo_op_6`, `foo_op_7`

**Key imports**: annotations, typing, Optional, Union, torch, Tensor, types, run_tests, TestCase, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/custom_operator`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`
- `torch`
- `torch.testing._internal.common_utils`: run_tests, TestCase
- `collections.abc`: Sequence


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
python test/custom_operator/test_infer_schema_annotation.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/custom_operator`):

- [`my_custom_ops.py_docs.md`](./my_custom_ops.py_docs.md)
- [`test_custom_ops.cpp_docs.md`](./test_custom_ops.cpp_docs.md)
- [`model.py_docs.md`](./model.py_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`pointwise.py_docs.md`](./pointwise.py_docs.md)
- [`test_custom_ops.py_docs.md`](./test_custom_ops.py_docs.md)
- [`op.cpp_docs.md`](./op.cpp_docs.md)
- [`my_custom_ops2.py_docs.md`](./my_custom_ops2.py_docs.md)
- [`op.h_docs.md`](./op.h_docs.md)


## Cross-References

- **File Documentation**: `test_infer_schema_annotation.py_docs.md`
- **Keyword Index**: `test_infer_schema_annotation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

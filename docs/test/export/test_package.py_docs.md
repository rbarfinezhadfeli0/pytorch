# Documentation: `test/export/test_package.py`

## File Metadata

- **Path**: `test/export/test_package.py`
- **Size**: 2,853 bytes (2.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]
import unittest

import torch
from torch._dynamo.eval_frame import is_dynamo_supported
from torch.export import Dim
from torch.export.experimental import _ExportPackage
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestPackage(TestCase):
    def test_basic(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        self.assertEqual(
            package._exporter("fn", fn)(x),
            fn(x),
        )
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].fallbacks), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 0)

    def test_more_than_once(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn)
        exporter(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export .* more than once",
        ):
            exporter(x)

    def test_error(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        x = torch.randn(3, 2)
        package = _ExportPackage()
        exporter = package._exporter("fn", fn, fallback="error")
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot export fallback .* when fallback policy is set to 'error'",
        ):
            exporter(x)

    def test_overloads(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] == 4:
                    return x + 1
                elif x.shape[0] == 3:
                    return x - 1
                else:
                    return x + 2

        fn = Module()
        x = torch.randn(3, 2)
        x2 = torch.randn(4, 2)
        x3 = torch.randn(5, 2)

        def spec(self, x):
            assert x.shape[0] == 3

        def spec2(self, x):
            assert x.shape[0] == 4

        def spec3(self, x):
            assert x.shape[0] >= 5
            return {"x": (Dim("batch", min=5), Dim.STATIC)}

        package = _ExportPackage()
        exporter = (
            package._exporter("fn", fn)
            ._define_overload("spec", spec)
            ._define_overload("spec2", spec2)
            ._define_overload("spec3", spec3)
        )
        self.assertEqual(exporter(x), x - 1)
        self.assertEqual(exporter(x2), x2 + 1)
        self.assertEqual(exporter(x3), x3 + 2)
        self.assertEqual(len(package.methods), 1)
        self.assertEqual(len(package.methods["fn"].overloads), 3)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPackage`, `Module`

**Functions defined**: `test_basic`, `fn`, `test_more_than_once`, `fn`, `test_error`, `fn`, `test_overloads`, `forward`, `spec`, `spec2`, `spec3`

**Key imports**: unittest, torch, is_dynamo_supported, Dim, _ExportPackage, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._dynamo.eval_frame`: is_dynamo_supported
- `torch.export`: Dim
- `torch.export.experimental`: _ExportPackage
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/export/test_package.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_package.py_docs.md`
- **Keyword Index**: `test_package.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

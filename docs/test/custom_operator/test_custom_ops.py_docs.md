# Documentation: `test/custom_operator/test_custom_ops.py`

## File Metadata

- **Path**: `test/custom_operator/test_custom_ops.py`
- **Size**: 5,582 bytes (5.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: unknown"]

import sys
import tempfile
import unittest

from model import get_custom_op_library_path, Model

import torch
import torch._library.utils as utils
from torch import ops
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


torch.ops.import_module("pointwise")


class TestCustomOperators(TestCase):
    def setUp(self):
        super().setUp()
        self.library_path = get_custom_op_library_path()
        ops.load_library(self.library_path)

    def test_custom_library_is_loaded(self):
        self.assertIn(self.library_path, ops.loaded_libraries)

    def test_op_with_no_abstract_impl_pystub(self):
        x = torch.randn(3, device="meta")
        if utils.requires_set_python_module():
            with self.assertRaisesRegex(RuntimeError, "pointwise"):
                torch.ops.custom.tan(x)
        else:
            # Smoketest
            torch.ops.custom.tan(x)

    def test_op_with_incorrect_abstract_impl_pystub(self):
        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "pointwise"):
            torch.ops.custom.cos(x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    def test_dynamo_pystub_suggestion(self):
        x = torch.randn(3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return torch.ops.custom.asin(x)

        with self.assertRaisesRegex(
            RuntimeError,
            r"(?s)Operator does not support running with fake tensors.*you may need to `import nonexistent`",
        ):
            f(x)

    def test_abstract_impl_pystub_faketensor(self):
        from functorch import make_fx

        x = torch.randn(3, device="cpu")
        self.assertNotIn("my_custom_ops", sys.modules.keys())

        with self.assertRaises(
            torch._subclasses.fake_tensor.UnsupportedOperatorException
        ):
            gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)

        torch.ops.import_module("my_custom_ops")
        gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            """\
def forward(self, arg0_1):
    nonzero = torch.ops.custom.nonzero.default(arg0_1);  arg0_1 = None
    return nonzero
""".strip(),
            gm.code.strip(),
        )

    def test_abstract_impl_pystub_meta(self):
        x = torch.randn(3, device="meta")
        self.assertNotIn("my_custom_ops2", sys.modules.keys())
        with self.assertRaisesRegex(NotImplementedError, r"'my_custom_ops2'"):
            torch.ops.custom.sin.default(x)
        torch.ops.import_module("my_custom_ops2")
        torch.ops.custom.sin.default(x)

    def test_calling_custom_op_string(self):
        output = ops.custom.op2("abc", "def")
        self.assertLess(output, 0)
        output = ops.custom.op2("abc", "abc")
        self.assertEqual(output, 0)

    def test_calling_custom_op(self):
        output = ops.custom.op(torch.ones(5), 2.0, 3)
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 3)
        for tensor in output:
            self.assertTrue(tensor.allclose(torch.ones(5) * 2))

        output = ops.custom.op_with_defaults(torch.ones(5))
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].allclose(torch.ones(5)))

    def test_calling_custom_op_with_autograd(self):
        x = torch.randn((5, 5), requires_grad=True)
        y = torch.randn((5, 5), requires_grad=True)
        output = ops.custom.op_with_autograd(x, 2, y)
        self.assertTrue(output.allclose(x + 2 * y + x * y))

        go = torch.ones((), requires_grad=True)
        output.sum().backward(go, False, True)
        grad = torch.ones(5, 5)

        self.assertEqual(x.grad, y + grad)
        self.assertEqual(y.grad, x + grad * 2)

        # Test with optional arg.
        x.grad.zero_()
        y.grad.zero_()
        z = torch.randn((5, 5), requires_grad=True)
        output = ops.custom.op_with_autograd(x, 2, y, z)
        self.assertTrue(output.allclose(x + 2 * y + x * y + z))

        go = torch.ones((), requires_grad=True)
        output.sum().backward(go, False, True)
        self.assertEqual(x.grad, y + grad)
        self.assertEqual(y.grad, x + grad * 2)
        self.assertEqual(z.grad, grad)

    def test_calling_custom_op_with_autograd_in_nograd_mode(self):
        with torch.no_grad():
            x = torch.randn((5, 5), requires_grad=True)
            y = torch.randn((5, 5), requires_grad=True)
            output = ops.custom.op_with_autograd(x, 2, y)
            self.assertTrue(output.allclose(x + 2 * y + x * y))

    def test_calling_custom_op_inside_script_module(self):
        model = Model()
        output = model.forward(torch.ones(5))
        self.assertTrue(output.allclose(torch.ones(5) + 1))

    def test_saving_and_loading_script_module_with_custom_op(self):
        model = Model()
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually.
        with tempfile.NamedTemporaryFile() as file:
            file.close()
            model.save(file.name)
            loaded = torch.jit.load(file.name)

            output = loaded.forward(torch.ones(5))
            self.assertTrue(output.allclose(torch.ones(5) + 1))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCustomOperators`

**Functions defined**: `setUp`, `test_custom_library_is_loaded`, `test_op_with_no_abstract_impl_pystub`, `test_op_with_incorrect_abstract_impl_pystub`, `test_dynamo_pystub_suggestion`, `f`, `test_abstract_impl_pystub_faketensor`, `forward`, `test_abstract_impl_pystub_meta`, `test_calling_custom_op_string`, `test_calling_custom_op`, `test_calling_custom_op_with_autograd`, `test_calling_custom_op_with_autograd_in_nograd_mode`, `test_calling_custom_op_inside_script_module`, `test_saving_and_loading_script_module_with_custom_op`

**Key imports**: sys, tempfile, unittest, get_custom_op_library_path, Model, torch, torch._library.utils as utils, ops, IS_WINDOWS, run_tests, TestCase, nonexistent, make_fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/custom_operator`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `tempfile`
- `unittest`
- `model`: get_custom_op_library_path, Model
- `torch`
- `torch._library.utils as utils`
- `torch.testing._internal.common_utils`: IS_WINDOWS, run_tests, TestCase
- `nonexistent`
- `functorch`: make_fx


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/custom_operator/test_custom_ops.py
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
- [`test_infer_schema_annotation.py_docs.md`](./test_infer_schema_annotation.py_docs.md)
- [`pointwise.py_docs.md`](./pointwise.py_docs.md)
- [`op.cpp_docs.md`](./op.cpp_docs.md)
- [`my_custom_ops2.py_docs.md`](./my_custom_ops2.py_docs.md)
- [`op.h_docs.md`](./op.h_docs.md)


## Cross-References

- **File Documentation**: `test_custom_ops.py_docs.md`
- **Keyword Index**: `test_custom_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

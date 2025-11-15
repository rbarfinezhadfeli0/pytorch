# Documentation: `docs/test/jit/test_exception.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_exception.py_docs.md`
- **Size**: 9,028 bytes (8.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_exception.py`

## File Metadata

- **Path**: `test/jit/test_exception.py`
- **Size**: 5,970 bytes (5.83 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase


r"""
Test TorchScript exception handling.
"""


class TestException(TestCase):
    def test_pyop_exception_message(self):
        class Foo(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 10, kernel_size=5)

            @torch.jit.script_method
            def forward(self, x):
                return self.conv(x)

        foo = Foo()
        # testing that the correct error message propagates
        with self.assertRaisesRegex(
            RuntimeError, r"Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d"
        ):
            foo(torch.ones([123]))  # wrong size

    def test_builtin_error_messsage(self):
        with self.assertRaisesRegex(RuntimeError, "Arguments for call are not valid"):

            @torch.jit.script
            def close_match(x):
                return x.masked_fill(True)

        with self.assertRaisesRegex(
            RuntimeError,
            "This op may not exist or may not be currently supported in TorchScript",
        ):

            @torch.jit.script
            def unknown_op(x):
                torch.set_anomaly_enabled(True)
                return x

    def test_exceptions(self):
        cu = torch.jit.CompilationUnit(
            """
            def foo(cond):
                if bool(cond):
                    raise ValueError(3)
                return 1
        """
        )

        cu.foo(torch.tensor(0))
        with self.assertRaisesRegex(torch.jit.Error, "3"):
            cu.foo(torch.tensor(1))

        def foo(cond):
            a = 3
            if bool(cond):
                raise ArbitraryError(a, "hi")  # noqa: F821
                if 1 == 2:
                    raise ArbitraryError  # noqa: F821
            return a

        with self.assertRaisesRegex(RuntimeError, "undefined value ArbitraryError"):
            torch.jit.script(foo)

        def exception_as_value():
            a = Exception()
            print(a)

        with self.assertRaisesRegex(RuntimeError, "cannot be used as a value"):
            torch.jit.script(exception_as_value)

        @torch.jit.script
        def foo_no_decl_always_throws():
            raise RuntimeError("Hi")

        # function that has no declared type but always throws set to None
        output_type = next(foo_no_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "NoneType")

        @torch.jit.script
        def foo_decl_always_throws():
            # type: () -> Tensor
            raise Exception("Hi")  # noqa: TRY002

        output_type = next(foo_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "Tensor")

        def foo():
            raise 3 + 4

        with self.assertRaisesRegex(RuntimeError, "must derive from BaseException"):
            torch.jit.script(foo)

        # a escapes scope
        @torch.jit.script
        def foo():
            if 1 == 1:
                a = 1
            else:
                if 1 == 1:
                    raise Exception("Hi")  # noqa: TRY002
                else:
                    raise Exception("Hi")  # noqa: TRY002
            return a

        self.assertEqual(foo(), 1)

        @torch.jit.script
        def tuple_fn():
            raise RuntimeError("hello", "goodbye")

        with self.assertRaisesRegex(torch.jit.Error, "hello, goodbye"):
            tuple_fn()

        @torch.jit.script
        def no_message():
            raise RuntimeError

        with self.assertRaisesRegex(torch.jit.Error, "RuntimeError"):
            no_message()

    def test_assertions(self):
        cu = torch.jit.CompilationUnit(
            """
            def foo(cond):
                assert bool(cond), "hi"
                return 0
        """
        )

        cu.foo(torch.tensor(1))
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            cu.foo(torch.tensor(0))

        @torch.jit.script
        def foo(cond):
            assert bool(cond), "hi"

        foo(torch.tensor(1))
        # we don't currently validate the name of the exception
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            foo(torch.tensor(0))

    def test_python_op_exception(self):
        @torch.jit.ignore
        def python_op(x):
            raise Exception("bad!")  # noqa: TRY002

        @torch.jit.script
        def fn(x):
            return python_op(x)

        with self.assertRaisesRegex(
            RuntimeError, "operation failed in the TorchScript interpreter"
        ):
            fn(torch.tensor(4))

    def test_dict_expansion_raises_error(self):
        def fn(self):
            d = {"foo": 1, "bar": 2, "baz": 3}
            return {**d}

        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError, "Dict expansion "
        ):
            torch.jit.script(fn)

    def test_custom_python_exception(self):
        class MyValueError(ValueError):
            pass

        @torch.jit.script
        def fn():
            raise MyValueError("test custom exception")

        with self.assertRaisesRegex(
            torch.jit.Error, "jit.test_exception.MyValueError: test custom exception"
        ):
            fn()

    def test_custom_python_exception_defined_elsewhere(self):
        from jit.myexception import MyKeyError

        @torch.jit.script
        def fn():
            raise MyKeyError("This is a user defined key error")

        with self.assertRaisesRegex(
            torch.jit.Error,
            "jit.myexception.MyKeyError: This is a user defined key error",
        ):
            fn()


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

r"""Test TorchScript exception handling.

This Python file contains 3 class(es) and 28 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestException`, `Foo`, `MyValueError`

**Functions defined**: `test_pyop_exception_message`, `__init__`, `forward`, `test_builtin_error_messsage`, `close_match`, `unknown_op`, `test_exceptions`, `foo`, `foo`, `exception_as_value`, `foo_no_decl_always_throws`, `foo_decl_always_throws`, `foo`, `foo`, `tuple_fn`, `no_message`, `test_assertions`, `foo`, `foo`, `test_python_op_exception`

**Key imports**: torch, nn, TestCase, MyKeyError


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_utils`: TestCase
- `jit.myexception`: MyKeyError


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/jit/test_exception.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_exception.py_docs.md`
- **Keyword Index**: `test_exception.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/jit/test_exception.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_exception.py_docs.md_docs.md`
- **Keyword Index**: `test_exception.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

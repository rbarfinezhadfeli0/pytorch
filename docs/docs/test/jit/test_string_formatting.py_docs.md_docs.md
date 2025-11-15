# Documentation: `docs/test/jit/test_string_formatting.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_string_formatting.py_docs.md`
- **Size**: 9,782 bytes (9.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_string_formatting.py`

## File Metadata

- **Path**: `test/jit/test_string_formatting.py`
- **Size**: 6,290 bytes (6.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
from typing import List

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestStringFormatting(JitTestCase):
    def test_modulo_operator(self):
        def fn(dividend: int, divisor: int) -> int:
            return dividend % divisor

        self.checkScript(fn, (5, 2))

    def test_string_interpolation_with_string_placeholder_and_string_variable(self):
        def fn(arg1: str):
            return "%s in template" % arg1

        self.checkScript(fn, ("foo",))

    def test_string_interpolation_with_string_placeholder_and_format_string_variable(
        self,
    ):
        def fn(arg1: str):
            return arg1 % "foo"

        self.checkScript(fn, ("%s in template",))

    def test_string_interpolation_with_double_percent_in_string(self):
        def fn(arg1: str):
            return "%s in template %%" % arg1

        self.checkScript(fn, ("foo",))

    def test_string_interpolation_with_percent_in_string(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%s in template %" % arg1  # noqa: F501

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Incomplete format specifier", '"%s in template %" % arg1'
        ):
            fn("foo")

    def test_string_interpolation_with_string_placeholder_and_digit_variable(self):
        def fn(arg1: int) -> str:
            return "%s in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_digit_placeholder_and_digit_variable(self):
        def fn(arg1: int) -> str:
            return "%d in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_alternate_digit_placeholder(self):
        def fn(arg1: int) -> str:
            return "%i in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_digit_placeholder_and_string_variable(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%d in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%d requires a number for formatting, but got String",
            '"%d in template" % arg1',
        ):
            fn("1")

    def test_string_interpolation_with_exponent_placeholder_and_string_variable(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%e in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%e requires a number for formatting, but got String",
            '"%e in template" % arg1',
        ):
            fn("1")

    def test_string_interpolation_with_lowercase_exponent_placeholder_and_digit_variable(
        self,
    ):
        def fn(arg1: int) -> str:
            return "%e in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_capital_exponent_placeholder_and_digit_variable(
        self,
    ):
        def fn(arg1: int) -> str:
            return "%E in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_float_placeholder_and_float_variable(self):
        def fn(arg1: float) -> str:
            return "%f in template" % arg1

        self.checkScript(fn, (1.0,))

    def test_string_interpolation_with_float_placeholder_and_digit_variable(self):
        def fn(arg1: int) -> str:
            return "%f in template" % arg1

        self.checkScript(fn, (1,))

    def test_string_interpolation_with_char_placeholder_and_char_variable(self):
        def fn(arg1: str) -> str:
            return "%c in template" % arg1

        self.checkScript(fn, ("a",))

    def test_string_interpolation_with_char_placeholder_and_digit_variable(self):
        def fn(arg1: int) -> str:
            return "%c in template" % arg1

        self.checkScript(fn, (97,))

    def test_string_interpolation_with_char_placeholder_and_true_string_variable(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%c in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%c requires an int or char for formatting, but got String",
            '"%c in template" % arg1',
        ):
            fn("foo")

    def test_string_interpolation_with_multiple_placeholders(self):
        def fn(arg1: str, arg2: int, arg3: float) -> str:
            return "%s %d %f in template" % (arg1, arg2, arg3)

        self.checkScript(fn, ("foo", 1, 1))

    def test_string_interpolation_with_subscript(self):
        def fn(arg1: List[str]) -> str:
            return "%s in template" % arg1[0]

        self.checkScript(fn, (["foo", "bar"],))

    def test_string_interpolation_with_too_few_arguments(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%s %s in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Too few arguments for format string",
            '"%s %s in template" % arg1',
        ):
            fn("foo")

    def test_string_interpolation_with_too_many_arguments(self):
        @torch.jit.script
        def fn(arg1: str, arg2: str) -> str:
            return "%s in template" % (arg1, arg2)  # noqa: F507

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Too many arguments for format string",
            '"%s in template" % (arg1, arg2',
        ):
            fn("foo", "bar")

    def test_string_interpolation_with_unknown_format_specifier(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%a in template" % arg1  # noqa: F501

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "The specifier %a is not supported in TorchScript format strings",
            '"%a in template" % arg1',
        ):
            fn("foo")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 1 class(es) and 44 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestStringFormatting`

**Functions defined**: `test_modulo_operator`, `fn`, `test_string_interpolation_with_string_placeholder_and_string_variable`, `fn`, `test_string_interpolation_with_string_placeholder_and_format_string_variable`, `fn`, `test_string_interpolation_with_double_percent_in_string`, `fn`, `test_string_interpolation_with_percent_in_string`, `fn`, `test_string_interpolation_with_string_placeholder_and_digit_variable`, `fn`, `test_string_interpolation_with_digit_placeholder_and_digit_variable`, `fn`, `test_string_interpolation_with_alternate_digit_placeholder`, `fn`, `test_string_interpolation_with_digit_placeholder_and_string_variable`, `fn`, `test_string_interpolation_with_exponent_placeholder_and_string_variable`, `fn`

**Key imports**: os, sys, List, torch, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `typing`: List
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


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
python test/jit/test_string_formatting.py
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

- **File Documentation**: `test_string_formatting.py_docs.md`
- **Keyword Index**: `test_string_formatting.py_kw.md`
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

*No specific patterns automatically detected.*


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
python docs/test/jit/test_string_formatting.py_docs.md
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

- **File Documentation**: `test_string_formatting.py_docs.md_docs.md`
- **Keyword Index**: `test_string_formatting.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

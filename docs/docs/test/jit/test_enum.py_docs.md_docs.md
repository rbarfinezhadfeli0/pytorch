# Documentation: `docs/test/jit/test_enum.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_enum.py_docs.md`
- **Size**: 13,536 bytes (13.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_enum.py`

## File Metadata

- **Path**: `test/jit/test_enum.py`
- **Size**: 10,016 bytes (9.78 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
from enum import Enum
from typing import Any, List

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestEnum(JitTestCase):
    def test_enum_value_types(self):
        class IntEnum(Enum):
            FOO = 1
            BAR = 2

        class FloatEnum(Enum):
            FOO = 1.2
            BAR = 2.3

        class StringEnum(Enum):
            FOO = "foo as in foo bar"
            BAR = "bar as in foo bar"

        make_global(IntEnum, FloatEnum, StringEnum)

        @torch.jit.script
        def supported_enum_types(a: IntEnum, b: FloatEnum, c: StringEnum):
            return (a.name, b.name, c.name)

        FileCheck().check("IntEnum").check("FloatEnum").check("StringEnum").run(
            str(supported_enum_types.graph)
        )

        class TensorEnum(Enum):
            FOO = torch.tensor(0)
            BAR = torch.tensor(1)

        make_global(TensorEnum)

        def unsupported_enum_types(a: TensorEnum):
            return a.name

        # TODO: rewrite code so that the highlight is not empty.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Cannot create Enum with value type 'Tensor'", ""
        ):
            torch.jit.script(unsupported_enum_types)

    def test_enum_comp(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        FileCheck().check("aten::eq").run(str(enum_comp.graph))

        self.assertEqual(enum_comp(Color.RED, Color.RED), True)
        self.assertEqual(enum_comp(Color.RED, Color.GREEN), False)

    def test_enum_comp_diff_classes(self):
        class Foo(Enum):
            ITEM1 = 1
            ITEM2 = 2

        class Bar(Enum):
            ITEM1 = 1
            ITEM2 = 2

        make_global(Foo, Bar)

        @torch.jit.script
        def enum_comp(x: Foo) -> bool:
            return x == Bar.ITEM1

        FileCheck().check("prim::Constant").check_same("Bar.ITEM1").check(
            "aten::eq"
        ).run(str(enum_comp.graph))

        self.assertEqual(enum_comp(Foo.ITEM1), False)

    def test_heterogenous_value_type_enum_error(self):
        class Color(Enum):
            RED = 1
            GREEN = "green"

        make_global(Color)

        def enum_comp(x: Color, y: Color) -> bool:
            return x == y

        # TODO: rewrite code so that the highlight is not empty.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Could not unify type list", ""
        ):
            torch.jit.script(enum_comp)

    def test_enum_name(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_name(x: Color) -> str:
            return x.name

        FileCheck().check("Color").check_next("prim::EnumName").check_next(
            "return"
        ).run(str(enum_name.graph))

        self.assertEqual(enum_name(Color.RED), Color.RED.name)
        self.assertEqual(enum_name(Color.GREEN), Color.GREEN.name)

    def test_enum_value(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_value(x: Color) -> int:
            return x.value

        FileCheck().check("Color").check_next("prim::EnumValue").check_next(
            "return"
        ).run(str(enum_value.graph))

        self.assertEqual(enum_value(Color.RED), Color.RED.value)
        self.assertEqual(enum_value(Color.GREEN), Color.GREEN.value)

    def test_enum_as_const(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_const(x: Color) -> bool:
            return x == Color.RED

        FileCheck().check(
            "prim::Constant[value=__torch__.jit.test_enum.Color.RED]"
        ).check_next("aten::eq").check_next("return").run(str(enum_const.graph))

        self.assertEqual(enum_const(Color.RED), True)
        self.assertEqual(enum_const(Color.GREEN), False)

    def test_non_existent_enum_value(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        def enum_const(x: Color) -> bool:
            if x == Color.PURPLE:
                return True
            else:
                return False

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "has no attribute 'PURPLE'", "Color.PURPLE"
        ):
            torch.jit.script(enum_const)

    def test_enum_ivalue_type(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def is_color_enum(x: Any):
            return isinstance(x, Color)

        FileCheck().check(
            "prim::isinstance[types=[Enum<__torch__.jit.test_enum.Color>]]"
        ).check_next("return").run(str(is_color_enum.graph))

        self.assertEqual(is_color_enum(Color.RED), True)
        self.assertEqual(is_color_enum(Color.GREEN), True)
        self.assertEqual(is_color_enum(1), False)

    def test_closed_over_enum_constant(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        a = Color

        @torch.jit.script
        def closed_over_aliased_type():
            return a.RED.value

        FileCheck().check("prim::Constant[value={}]".format(a.RED.value)).check_next(
            "return"
        ).run(str(closed_over_aliased_type.graph))

        self.assertEqual(closed_over_aliased_type(), Color.RED.value)

        b = Color.RED

        @torch.jit.script
        def closed_over_aliased_value():
            return b.value

        FileCheck().check("prim::Constant[value={}]".format(b.value)).check_next(
            "return"
        ).run(str(closed_over_aliased_value.graph))

        self.assertEqual(closed_over_aliased_value(), Color.RED.value)

    def test_enum_as_module_attribute(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            def __init__(self, e: Color):
                super().__init__()
                self.e = e

            def forward(self):
                return self.e.value

        m = TestModule(Color.RED)
        scripted = torch.jit.script(m)

        FileCheck().check("TestModule").check_next("Color").check_same(
            'prim::GetAttr[name="e"]'
        ).check_next("prim::EnumValue").check_next("return").run(str(scripted.graph))

        self.assertEqual(scripted(), Color.RED.value)

    def test_string_enum_as_module_attribute(self):
        class Color(Enum):
            RED = "red"
            GREEN = "green"

        class TestModule(torch.nn.Module):
            def __init__(self, e: Color):
                super().__init__()
                self.e = e

            def forward(self):
                return (self.e.name, self.e.value)

        make_global(Color)
        m = TestModule(Color.RED)
        scripted = torch.jit.script(m)

        self.assertEqual(scripted(), (Color.RED.name, Color.RED.value))

    def test_enum_return(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def return_enum(cond: bool):
            if cond:
                return Color.RED
            else:
                return Color.GREEN

        self.assertEqual(return_enum(True), Color.RED)
        self.assertEqual(return_enum(False), Color.GREEN)

    def test_enum_module_return(self):
        class Color(Enum):
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            def __init__(self, e: Color):
                super().__init__()
                self.e = e

            def forward(self):
                return self.e

        make_global(Color)
        m = TestModule(Color.RED)
        scripted = torch.jit.script(m)

        FileCheck().check("TestModule").check_next("Color").check_same(
            'prim::GetAttr[name="e"]'
        ).check_next("return").run(str(scripted.graph))

        self.assertEqual(scripted(), Color.RED)

    def test_enum_iterate(self):
        class Color(Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def iterate_enum(x: Color):
            res: List[int] = []
            for e in Color:
                if e != x:
                    res.append(e.value)
            return res

        make_global(Color)
        scripted = torch.jit.script(iterate_enum)

        FileCheck().check("Enum<__torch__.jit.test_enum.Color>[]").check_same(
            "Color.RED"
        ).check_same("Color.GREEN").check_same("Color.BLUE").run(str(scripted.graph))

        # PURPLE always appears last because we follow Python's Enum definition order.
        self.assertEqual(scripted(Color.RED), [Color.GREEN.value, Color.BLUE.value])
        self.assertEqual(scripted(Color.GREEN), [Color.RED.value, Color.BLUE.value])

    # Tests that explicitly and/or repeatedly scripting an Enum class is permitted.
    def test_enum_explicit_script(self):
        @torch.jit.script
        class Color(Enum):
            RED = 1
            GREEN = 2

        torch.jit.script(Color)

    # Regression test for https://github.com/pytorch/pytorch/issues/108933
    def test_typed_enum(self):
        class Color(int, Enum):
            RED = 1
            GREEN = 2

        @torch.jit.script
        def is_red(x: Color) -> bool:
            return x == Color.RED


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 26 class(es) and 38 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestEnum`, `IntEnum`, `FloatEnum`, `StringEnum`, `TensorEnum`, `Color`, `Foo`, `Bar`, `Color`, `Color`, `Color`, `Color`, `Color`, `Color`, `Color`, `Color`, `TestModule`, `Color`, `TestModule`, `Color`

**Functions defined**: `test_enum_value_types`, `supported_enum_types`, `unsupported_enum_types`, `test_enum_comp`, `enum_comp`, `test_enum_comp_diff_classes`, `enum_comp`, `test_heterogenous_value_type_enum_error`, `enum_comp`, `test_enum_name`, `enum_name`, `test_enum_value`, `enum_value`, `test_enum_as_const`, `enum_const`, `test_non_existent_enum_value`, `enum_const`, `test_enum_ivalue_type`, `is_color_enum`, `test_closed_over_enum_constant`

**Key imports**: os, sys, Enum, Any, List, torch, FileCheck, raise_on_run_directly, JitTestCase, make_global


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `enum`: Enum
- `typing`: Any, List
- `torch`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase, make_global


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/jit/test_enum.py
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

- **File Documentation**: `test_enum.py_docs.md`
- **Keyword Index**: `test_enum.py_kw.md`
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
- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/jit/test_enum.py_docs.md
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

- **File Documentation**: `test_enum.py_docs.md_docs.md`
- **Keyword Index**: `test_enum.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

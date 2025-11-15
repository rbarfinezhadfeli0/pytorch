# Documentation: `docs/test/mobile/test_lite_script_type.py_docs.md`

## File Metadata

- **Path**: `docs/test/mobile/test_lite_script_type.py_docs.md`
- **Size**: 9,393 bytes (9.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/mobile/test_lite_script_type.py`

## File Metadata

- **Path**: `test/mobile/test_lite_script_type.py`
- **Size**: 6,133 bytes (5.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: mobile"]

import io
import unittest
from collections import namedtuple
from typing import NamedTuple

import torch
import torch.utils.bundled_inputs
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLiteScriptModule(TestCase):
    def test_typing_namedtuple(self):
        myNamedTuple = NamedTuple(  # noqa: UP014
            "myNamedTuple", [("a", list[torch.Tensor])]
        )

        class MyTestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor):
                p = myNamedTuple([a])
                return p

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input).a

        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)  # Error here
        mobile_module_result = mobile_module(sample_input).a
        torch.testing.assert_close(script_module_result, mobile_module_result)

    @unittest.skip("T137512434")
    def test_typing_dict_with_namedtuple(self):
        class Foo(NamedTuple):
            id: torch.Tensor

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo(torch.tensor(1))

            def forward(self, a: torch.Tensor):
                self.foo = Foo(a)
                re: dict[str, Foo] = {}
                re["test"] = Foo(a)
                return self.foo, re["test"]

        # The corresponding bytecode is
        # (8,
        #  ('__torch__.___torch_mangle_2.Bar.forward',
        #   (('instructions',
        #     (('STOREN', 1, 2),
        #      ('DROPR', 1, 0),
        #      ('DICT_CONSTRUCT', 0, 0),
        #      ('STORE', 3, 0),
        #      ('LOAD', 3, 0),
        #      ('LOADC', 1, 0),
        #      ('MOVE', 2, 0),
        #      ('NAMED_TUPLE_CONSTRUCT', 1, 1),
        #      ('OP', 0, 0),
        #      ('MOVE', 3, 0),
        #      ('LOADC', 1, 0),
        #      ('DICT_INDEX', 0, 0),
        #      ('LOADC', 0, 0),
        #      ('TUPLE_INDEX', 0, 0),
        #      ('RET', 0, 0))),
        #    ('operators', (('aten::_set_item', 'str', 3),)),
        #    ('constants', (0, 'test')),
        #    ('types',
        #     ('Dict[str,__torch__.Foo[NamedTuple, [[id, Tensor]]]]',
        #      '__torch__.Foo[NamedTuple, [[id, Tensor]]]')),
        #    ('register_size', 3)),
        #   (('arguments',
        #     ((('name', 'self'),
        #       ('type', '__torch__.___torch_mangle_2.Bar'),
        #       ('default_value', None)),
        #      (('name', 'a'), ('type', 'Tensor'), ('default_value', None)))),
        #    ('returns',
        #     ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())

        script_module_result = script_module(sample_input)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_typing_namedtuple_custom_classtype(self):
        class Foo(NamedTuple):
            id: torch.Tensor

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo(torch.tensor(1))

            def forward(self, a: torch.Tensor):
                self.foo = Foo(a)
                return self.foo

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())
        script_module_result = script_module(sample_input)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_return_collections_namedtuple(self):
        myNamedTuple = namedtuple("myNamedTuple", [("a")])

        class MyTestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor):
                return myNamedTuple(a)

        sample_input = torch.Tensor(1)
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input)
        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_nest_typing_namedtuple_custom_classtype(self):
        class Baz(NamedTuple):
            di: torch.Tensor

        class Foo(NamedTuple):
            id: torch.Tensor
            baz: Baz

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo(torch.tensor(1), Baz(torch.tensor(1)))

            def forward(self, a: torch.Tensor):
                self.foo = Foo(a, Baz(torch.tensor(1)))
                return self.foo

        sample_input = torch.tensor(5)
        script_module = torch.jit.script(Bar())
        script_module_result = script_module(sample_input)

        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        mobile_module_result = mobile_module(sample_input)
        torch.testing.assert_close(
            script_module_result.baz.di, mobile_module_result.baz.di
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 10 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLiteScriptModule`, `MyTestModule`, `Foo`, `Bar`, `Foo`, `Bar`, `MyTestModule`, `Baz`, `Foo`, `Bar`

**Functions defined**: `test_typing_namedtuple`, `forward`, `test_typing_dict_with_namedtuple`, `__init__`, `forward`, `test_typing_namedtuple_custom_classtype`, `__init__`, `forward`, `test_return_collections_namedtuple`, `forward`, `test_nest_typing_namedtuple_custom_classtype`, `__init__`, `forward`

**Key imports**: io, unittest, namedtuple, NamedTuple, torch, torch.utils.bundled_inputs, _load_for_lite_interpreter, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `unittest`
- `collections`: namedtuple
- `typing`: NamedTuple
- `torch`
- `torch.utils.bundled_inputs`
- `torch.jit.mobile`: _load_for_lite_interpreter
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/mobile/test_lite_script_type.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile`):

- [`test_upgrader_codegen.py_docs.md`](./test_upgrader_codegen.py_docs.md)
- [`test_quantize_fx_lite_script_module.py_docs.md`](./test_quantize_fx_lite_script_module.py_docs.md)
- [`test_upgrader_bytecode_table_example.cpp_docs.md`](./test_upgrader_bytecode_table_example.cpp_docs.md)
- [`test_lite_script_module.py_docs.md`](./test_lite_script_module.py_docs.md)
- [`test_upgraders.py_docs.md`](./test_upgraders.py_docs.md)
- [`test_bytecode.py_docs.md`](./test_bytecode.py_docs.md)


## Cross-References

- **File Documentation**: `test_lite_script_type.py_docs.md`
- **Keyword Index**: `test_lite_script_type.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/mobile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/mobile`, which is part of the **testing infrastructure**.



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
python docs/test/mobile/test_lite_script_type.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/mobile`):

- [`test_bytecode.py_kw.md_docs.md`](./test_bytecode.py_kw.md_docs.md)
- [`test_upgrader_codegen.py_kw.md_docs.md`](./test_upgrader_codegen.py_kw.md_docs.md)
- [`test_upgrader_bytecode_table_example.cpp_docs.md_docs.md`](./test_upgrader_bytecode_table_example.cpp_docs.md_docs.md)
- [`test_quantize_fx_lite_script_module.py_docs.md_docs.md`](./test_quantize_fx_lite_script_module.py_docs.md_docs.md)
- [`test_upgraders.py_docs.md_docs.md`](./test_upgraders.py_docs.md_docs.md)
- [`test_quantize_fx_lite_script_module.py_kw.md_docs.md`](./test_quantize_fx_lite_script_module.py_kw.md_docs.md)
- [`test_bytecode.py_docs.md_docs.md`](./test_bytecode.py_docs.md_docs.md)
- [`test_lite_script_module.py_kw.md_docs.md`](./test_lite_script_module.py_kw.md_docs.md)
- [`test_upgrader_bytecode_table_example.cpp_kw.md_docs.md`](./test_upgrader_bytecode_table_example.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_lite_script_type.py_docs.md_docs.md`
- **Keyword Index**: `test_lite_script_type.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

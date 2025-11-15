# Documentation: test_lite_script_type.py

## File Metadata
- **Path**: `test/mobile/test_lite_script_type.py`
- **Size**: 6133 bytes
- **Lines**: 169
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 10 class(es): TestLiteScriptModule, MyTestModule, Foo, Bar, Foo, Bar, MyTestModule, Baz, Foo, Bar

### Functions
This file defines 13 function(s): test_typing_namedtuple, forward, test_typing_dict_with_namedtuple, __init__, forward, test_typing_namedtuple_custom_classtype, __init__, forward, test_return_collections_namedtuple, forward, test_nest_typing_namedtuple_custom_classtype, __init__, forward


## Key Components

The file contains 400 words across 169 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6133 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

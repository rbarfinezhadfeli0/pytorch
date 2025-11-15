# Documentation: test_frame_init.py

## File Metadata
- **Path**: `test/dynamo/test_frame_init.py`
- **Size**: 4568 bytes
- **Lines**: 144
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch._C._dynamo.eval_frame import set_eval_frame
from torch._dynamo.types import ConvertFrameReturn, GuardedCode, wrap_guarded_code
from torch._guards import CompileId


def target_with_varkwargs(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    local = 1
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def varkwargs_code1(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # remove a local variable: local = 1
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def varkwargs_code2(arg1, /, positional_only_arg, *, keyword_only_arg, **kwargs):
    # introduce a local variable
    local1 = 0
    local2 = 1
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "keyword_only_arg": keyword_only_arg,
        "kwargs": kwargs,
    }


def target_with_varargs(arg1, /, positional_only_arg, *varargs, **kwargs):
    local = 1
    return {
        "local": local,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def varargs_code1(arg1, /, positional_only_arg, *varargs, **kwargs):
    # remove a local variable: local = 1
    return {
        "local": 1,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


def varargs_code2(arg1, /, positional_only_arg, *varargs, **kwargs):
    # introduce a local variable
    local1 = 0
    local2 = 1
    return {
        "local": local1 + local2,
        "arg1": arg1,
        "positional_only_arg": positional_only_arg,
        "varargs": varargs,
        "kwargs": kwargs,
    }


class FrameInitTests(torch._dynamo.test_case.TestCase):
    def test_frame_init(self):
        code_map1 = {
            target_with_varargs.__code__: varargs_code1.__code__,
            target_with_varkwargs.__code__: varkwargs_code1.__code__,
        }
        code_map2 = {
            target_with_varargs.__code__: varargs_code2.__code__,
            target_with_varkwargs.__code__: varkwargs_code2.__code__,
        }

        empty_guard_manager = torch._dynamo.guards.GuardManagerWrapper()

        def callback1(frame, cache_entry, frame_state):
            if frame.f_code in code_map1:
                transformed_code = code_map1[frame.f_code]
                return wrap_guarded_code(
                    GuardedCode(
                        transformed_code,
                        empty_guard_manager,
                        CompileId(
                            frame_id=None, frame_compile_id=0, compiled_autograd_id=0
                        ),
                    )
                )
            return ConvertFrameReturn()

        def callback2(frame, cache_entry, frame_state):
            if frame.f_code in code_map2:
                transformed_code = code_map2[frame.f_code]
                return wrap_guarded_code(
                    GuardedCode(
                        transformed_code,
                        empty_guard_manager,
                        CompileId(
                            frame_id=None, frame_compile_id=0, compiled_autograd_id=0
                        ),
                    )
                )
            return ConvertFrameReturn()

        for _ in [callback1, callback2]:
            torch._dynamo.reset()
            expected_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            expected_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            original = set_eval_frame(callback1)
            real_varargs_output = target_with_varargs(
                1, 2, 3, 4, name1=1, name2=2, name3=3
            )
            real_kwargs_output = target_with_varkwargs(
                1, 2, keyword_only_arg=1, name2=2, name3=3
            )
            self.assertEqual(real_varargs_output, expected_varargs_output)
            self.assertEqual(real_kwargs_output, expected_kwargs_output)
            set_eval_frame(original)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): FrameInitTests

### Functions
This file defines 9 function(s): target_with_varkwargs, varkwargs_code1, varkwargs_code2, target_with_varargs, varargs_code1, varargs_code2, test_frame_init, callback1, callback2


## Key Components

The file contains 323 words across 144 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4568 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

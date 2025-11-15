# Documentation: `test/dynamo/test_frame_init.py`

## File Metadata

- **Path**: `test/dynamo/test_frame_init.py`
- **Size**: 4,568 bytes (4.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
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


This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FrameInitTests`

**Functions defined**: `target_with_varkwargs`, `varkwargs_code1`, `varkwargs_code2`, `target_with_varargs`, `varargs_code1`, `varargs_code2`, `test_frame_init`, `callback1`, `callback2`

**Key imports**: torch, torch._dynamo.test_case, set_eval_frame, ConvertFrameReturn, GuardedCode, wrap_guarded_code, CompileId, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo.test_case`
- `torch._C._dynamo.eval_frame`: set_eval_frame
- `torch._dynamo.types`: ConvertFrameReturn, GuardedCode, wrap_guarded_code
- `torch._guards`: CompileId


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
python test/dynamo/test_frame_init.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_frame_init.py_docs.md`
- **Keyword Index**: `test_frame_init.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

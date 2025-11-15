# Documentation: `docs/test/dynamo/test_skip_guard_eval_unsafe.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_skip_guard_eval_unsafe.py_docs.md`
- **Size**: 6,862 bytes (6.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_skip_guard_eval_unsafe.py`

## File Metadata

- **Path**: `test/dynamo/test_skip_guard_eval_unsafe.py`
- **Size**: 3,884 bytes (3.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


def my_custom_function(x):
    return x + 1


class RunDiffGuardTests(torch._dynamo.test_case.TestCase):
    def test_bool_recompile(self):
        def fn(x, y, c):
            if c:
                return x * y
            else:
                return x + y

        opt_fn = torch.compile(fn, backend="inductor")
        x = 2 * torch.ones(4)
        y = 3 * torch.ones(4)

        ref1 = opt_fn(x, y, True)
        ref2 = opt_fn(x, y, False)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res2 = opt_fn(x, y, False)
            res1 = opt_fn(x, y, True)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_tensor_recompile(self):
        def fn(x, y):
            return x * y

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4, dtype=torch.float32)
        y = torch.randn(4, dtype=torch.float32)

        ref1 = opt_fn(x, y)

        x64 = torch.randn(4, dtype=torch.float64)
        y64 = torch.randn(4, dtype=torch.float64)
        ref2 = opt_fn(x64, y64)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res1 = opt_fn(x, y)
            res2 = opt_fn(x64, y64)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_post_recompile(self):
        class Foo:
            def __init__(self):
                self.a = 4
                self.b = 5

        foo = Foo()

        def fn(x):
            return x + foo.a + foo.b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        foo.a = 11
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            # Set it back to original value
            foo.a = 4
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

            foo.a = 11
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

        # Check that we are back to original behavior
        foo.b = 8
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)

    def test_fail_on_tensor_shape_change(self):
        def fn(dt):
            return dt["x"] + 1

        x = torch.randn(4)
        dt = {}
        dt["x"] = x
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(dt)

        with self.assertRaisesRegex(
            RuntimeError, "Recompilation triggered with skip_guard_eval_unsafe stance"
        ):
            with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                x = torch.randn(4, 4)
                dt["x"] = x
                opt_fn(dt)

    def test_cache_line_pickup(self):
        def fn(x, a=None, b=None):
            x = x * 3
            if a:
                x = x * 5
            if b:
                x = x * 7
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.ones(4)

        ref1 = opt_fn(x, a=None, b=None)
        ref2 = opt_fn(x, a=1, b=None)
        ref3 = opt_fn(x, a=1, b=1)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res1 = opt_fn(x, a=None, b=None)
            res2 = opt_fn(x, a=1, b=None)
            res3 = opt_fn(x, a=1, b=1)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref3, res3)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `RunDiffGuardTests`, `Foo`

**Functions defined**: `my_custom_function`, `test_bool_recompile`, `fn`, `test_tensor_recompile`, `fn`, `test_post_recompile`, `__init__`, `fn`, `test_fail_on_tensor_shape_change`, `fn`, `test_cache_line_pickup`, `fn`

**Key imports**: torch, torch._dynamo.test_case, torch._dynamo.testing, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/dynamo/test_skip_guard_eval_unsafe.py
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

- **File Documentation**: `test_skip_guard_eval_unsafe.py_docs.md`
- **Keyword Index**: `test_skip_guard_eval_unsafe.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
python docs/test/dynamo/test_skip_guard_eval_unsafe.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_skip_guard_eval_unsafe.py_docs.md_docs.md`
- **Keyword Index**: `test_skip_guard_eval_unsafe.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

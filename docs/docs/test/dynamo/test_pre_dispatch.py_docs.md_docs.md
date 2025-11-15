# Documentation: `docs/test/dynamo/test_pre_dispatch.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_pre_dispatch.py_docs.md`
- **Size**: 4,814 bytes (4.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_pre_dispatch.py`

## File Metadata

- **Path**: `test/dynamo/test_pre_dispatch.py`
- **Size**: 2,139 bytes (2.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case


class PreDispatchTests(torch._dynamo.test_case.TestCase):
    def test_no_grad_simple(self):
        def f(a):
            b = a.sin()
            with torch.no_grad():
                c = b.cos()
            return b * c.sin()

        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        a_ref = torch.randn(4, requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_enable_grad_and_no_grad(self):
        def f(a):
            b = a * 2
            with torch.no_grad():
                c = b * 3
                with torch.enable_grad():
                    d = c * 4
                e = d * 5
            return b + c + d + e

        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        a_ref = torch.randn(4, requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)

    def test_autocast_simple(self):
        def f(a):
            b = a * 2
            with torch.amp.autocast(device_type="cpu"):
                c = torch.matmul(b, b)
            return b + c

        f_compiled = torch.compile(f, backend="pre_dispatch_eager")

        a_ref = torch.randn(4, device="cpu", requires_grad=True)
        a_test = a_ref.detach().clone().requires_grad_(True)

        out_ref = f(a_ref)
        out_test = f_compiled(a_test)
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad, a_test.grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PreDispatchTests`

**Functions defined**: `test_no_grad_simple`, `f`, `test_enable_grad_and_no_grad`, `f`, `test_autocast_simple`, `f`

**Key imports**: torch, torch._dynamo, torch._dynamo.test_case, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo`
- `torch._dynamo.test_case`


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
python test/dynamo/test_pre_dispatch.py
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
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_pre_dispatch.py_docs.md`
- **Keyword Index**: `test_pre_dispatch.py_kw.md`
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
python docs/test/dynamo/test_pre_dispatch.py_docs.md
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

- **File Documentation**: `test_pre_dispatch.py_docs.md_docs.md`
- **Keyword Index**: `test_pre_dispatch.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

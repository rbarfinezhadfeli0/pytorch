# Documentation: `docs/test/dynamo/test_deque_reconstruct.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_deque_reconstruct.py_docs.md`
- **Size**: 5,519 bytes (5.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_deque_reconstruct.py`

## File Metadata

- **Path**: `test/dynamo/test_deque_reconstruct.py`
- **Size**: 2,607 bytes (2.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import collections
import contextlib

import torch
import torch._inductor.test_case


class TestDequeReconstruct(torch._inductor.test_case.TestCase):
    UNSET = object()

    @contextlib.contextmanager
    def set_deque_in_globals(self, value):
        prev = globals().pop("deque", self.UNSET)
        assert "deque" not in globals()

        try:
            if value is not self.UNSET:
                globals()["deque"] = value
            yield
        finally:
            if prev is self.UNSET:
                globals().pop("deque", None)
                assert "deque" not in globals()
            else:
                globals()["deque"] = prev

    def test_deque_reconstruct_not_in_globals(self):
        with self.set_deque_in_globals(self.UNSET):

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_in_globals(self):
        with self.set_deque_in_globals(collections.deque):
            # This does not emit a NameError
            dummy = deque([0, 1, 2], maxlen=2)  # noqa: F821
            self.assertIsInstance(dummy, collections.deque)
            self.assertEqual(list(dummy), [1, 2])

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))

    def test_deque_reconstruct_shallows_globals(self):
        with self.set_deque_in_globals(None):
            # This does not emit a NameError
            self.assertIsNone(deque)  # noqa: F821

            @torch.compile(backend="eager", fullgraph=True)
            def func(x):
                return collections.deque([x, x + 1, x + 2], maxlen=2)

            x = torch.randn(3, 4)
            out = func(x)
            self.assertIsInstance(out, collections.deque)
            self.assertEqual(out.maxlen, 2)
            self.assertEqual(out, collections.deque([x + 1, x + 2], maxlen=2))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDequeReconstruct`

**Functions defined**: `set_deque_in_globals`, `test_deque_reconstruct_not_in_globals`, `func`, `test_deque_reconstruct_in_globals`, `func`, `test_deque_reconstruct_shallows_globals`, `func`

**Key imports**: collections, contextlib, torch, torch._inductor.test_case, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `contextlib`
- `torch`
- `torch._inductor.test_case`
- `torch._dynamo.test_case`: run_tests


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
python test/dynamo/test_deque_reconstruct.py
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

- **File Documentation**: `test_deque_reconstruct.py_docs.md`
- **Keyword Index**: `test_deque_reconstruct.py_kw.md`
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
python docs/test/dynamo/test_deque_reconstruct.py_docs.md
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

- **File Documentation**: `test_deque_reconstruct.py_docs.md_docs.md`
- **Keyword Index**: `test_deque_reconstruct.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

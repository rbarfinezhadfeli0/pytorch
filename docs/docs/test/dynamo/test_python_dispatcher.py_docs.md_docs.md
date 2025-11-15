# Documentation: `docs/test/dynamo/test_python_dispatcher.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_python_dispatcher.py_docs.md`
- **Size**: 8,198 bytes (8.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_python_dispatcher.py`

## File Metadata

- **Path**: `test/dynamo/test_python_dispatcher.py`
- **Size**: 4,832 bytes (4.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter, EagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TEST_XPU


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class PythonDispatcherTests(torch._dynamo.test_case.TestCase):
    def test_dispatch_key1(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x = x + 1
            return torch._C._dispatch_keys(x)

        x = torch.randn(2, 3)
        self.assertTrue(fn(x).raw_repr() == torch._C._dispatch_keys(x + 1).raw_repr())

    def test_dispatch_key2(self):
        from torch.testing._internal.two_tensor import TwoTensor

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            x = x.sin()
            return torch._C._dispatch_keys(x)

        x = torch.randn(3)
        y = torch.randn(3)
        z = TwoTensor(x, y)
        self.assertTrue(fn(z).raw_repr() == torch._C._dispatch_keys(z.sin()).raw_repr())

    def test_dispatch_key3(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            key_set = torch._C._dispatch_tls_local_include_set()
            return torch.sin(x + 1), key_set

        x = torch.randn(2, 3)
        self.assertEqual(fn(x)[0], torch.sin(x + 1))
        self.assertTrue(
            fn(x)[1].raw_repr() == torch._C._dispatch_tls_local_include_set().raw_repr()
        )

    def test_dispatch_key4(self):
        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(x):
            key_set = torch._C._dispatch_tls_local_include_set()
            key_set = key_set | torch._C._dispatch_keys(x)
            key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
            if key_set.highestPriorityTypeId() == torch.DispatchKey.PythonDispatcher:
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(2, 3)
        self.assertEqual(fn(x), torch.sin(x - 1))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        sub: "f32[2, 3]" = l_x_ - 1;  l_x_ = None
        sin: "f32[2, 3]" = torch.sin(sub);  sub = None
        return (sin,)
""",  # NOQA: B950
        )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda or xpu")
    def test_dispatch_key_set_guard(self):
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, dks):
            if dks.has("CPU"):
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x1 = torch.randn(2, 3)
        dks1 = torch._C._dispatch_keys(x1)
        self.assertEqual(fn(x1, dks1), torch.sin(x1 + 1))
        self.assertEqual(counter.frame_count, 1)

        x2 = torch.randn(2, 3)
        dks2 = torch._C._dispatch_keys(x2)
        self.assertEqual(fn(x2, dks2), torch.sin(x2 + 1))
        # No recompile since the dispatch key set is the same though the tensor is different.
        self.assertEqual(counter.frame_count, 1)

        x3 = torch.randn(2, 3, device=device_type)
        dks3 = torch._C._dispatch_keys(x3)
        self.assertEqual(fn(x3, dks3), torch.sin(x3 - 1))
        # Re-compile since the dispatch key set is different.
        self.assertEqual(counter.frame_count, 2)

    def test_functorch_interpreter(self):
        counter = CompileCounter()

        def square_and_add(x, y):
            interpreter = (
                torch._functorch.pyfunctorch.retrieve_current_functorch_interpreter()
            )
            level = interpreter.level()
            if interpreter.key() == torch._C._functorch.TransformType.Vmap:
                return (x**2 + y) * level
            else:
                return x**2 * level

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, y):
            return torch.vmap(square_and_add)(x, y)

        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([10, 20, 30, 40])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 56]))
        self.assertEqual(counter.frame_count, 1)

        x = torch.tensor([1, 2, 3, 1])
        y = torch.tensor([10, 20, 30, 10])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 11]))
        # No recompile
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PythonDispatcherTests`, `GraphModule`

**Functions defined**: `test_dispatch_key1`, `fn`, `test_dispatch_key2`, `fn`, `test_dispatch_key3`, `fn`, `test_dispatch_key4`, `fn`, `forward`, `test_dispatch_key_set_guard`, `fn`, `test_functorch_interpreter`, `square_and_add`, `fn`

**Key imports**: unittest, torch, torch._dynamo.test_case, CompileCounter, EagerAndRecordGraphs, normalize_gm, TEST_CUDA, TEST_XPU, TwoTensor, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`: CompileCounter, EagerAndRecordGraphs, normalize_gm
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `torch.testing._internal.common_utils`: TEST_XPU
- `torch.testing._internal.two_tensor`: TwoTensor


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/dynamo/test_python_dispatcher.py
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

- **File Documentation**: `test_python_dispatcher.py_docs.md`
- **Keyword Index**: `test_python_dispatcher.py_kw.md`
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

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/dynamo/test_python_dispatcher.py_docs.md
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

- **File Documentation**: `test_python_dispatcher.py_docs.md_docs.md`
- **Keyword Index**: `test_python_dispatcher.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

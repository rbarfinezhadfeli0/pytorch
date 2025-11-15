# Documentation: `docs/test/dynamo/test_minifier.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_minifier.py_docs.md`
- **Size**: 11,953 bytes (11.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_minifier.py`

## File Metadata

- **Path**: `test/dynamo/test_minifier.py`
- **Size**: 7,871 bytes (7.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import unittest

import torch._dynamo
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfNNModuleInlined


requires_gpu = unittest.skipUnless(
    torch.cuda.is_available() or torch.xpu.is_available(), "requires cuda or xpu"
)


class MinifierTests(MinifierTestBase):
    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd (both CPU and CUDA/XPU)
    def _test_after_dynamo(self, device, backend, expected_error):
        run_code = f"""\
@torch.compile(backend={backend!r})
def inner(x):
    for _ in range(10):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(10):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("{device}"))
"""
        self._run_full_test(run_code, "dynamo", expected_error, isolate=False)

    def test_after_dynamo_cpu_compile_error(self):
        self._test_after_dynamo(
            "cpu", "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    def test_after_dynamo_cpu_runtime_error(self):
        self._test_after_dynamo(
            "cpu", "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    def test_after_dynamo_cpu_accuracy_error(self):
        self._test_after_dynamo(
            "cpu", "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    @requires_gpu
    def test_after_dynamo_cuda_compile_error(self, device):
        self._test_after_dynamo(
            device, "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    @requires_gpu
    def test_after_dynamo_cuda_runtime_error(self, device):
        self._test_after_dynamo(
            device, "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    @requires_gpu
    def test_after_dynamo_cuda_accuracy_error(self, device):
        self._test_after_dynamo(
            device, "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    def test_after_dynamo_non_leaf_compile_error(self):
        run_code = """\
@torch.compile(backend="non_leaf_compile_error_TESTING_ONLY")
def inner(x):
    return x + 1

inner(torch.randn(20, 20, requires_grad=True) + 1)
"""
        self._run_full_test(
            run_code, "dynamo", "TestingOnlyCompileError", isolate=False
        )

    # Ensure that the testing backends pass when relu is not present.
    def _test_after_dynamo_backend_passes(self, device, backend):
        @torch.compile(backend=backend)
        def inner(x):
            for _ in range(10):
                x = torch.sin(x)
            for _ in range(10):
                x = torch.cos(x)
            return x

        inner(torch.randn(20, 20).to(device))

    def test_after_dynamo_cpu_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", "relu_compile_error_TESTING_ONLY")

    def test_after_dynamo_cpu_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", "relu_runtime_error_TESTING_ONLY")

    def test_after_dynamo_cpu_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cpu", "relu_accuracy_error_TESTING_ONLY"
        )

    @requires_gpu
    def test_after_dynamo_cuda_compile_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_compile_error_TESTING_ONLY"
        )

    @requires_gpu
    def test_after_dynamo_cuda_runtime_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_runtime_error_TESTING_ONLY"
        )

    @requires_gpu
    def test_after_dynamo_cuda_accuracy_backend_passes(self, device):
        self._test_after_dynamo_backend_passes(
            device, "relu_accuracy_error_TESTING_ONLY"
        )

    # Test that a module with mixed cpu/(cuda|xpu) parts with an error after dynamo can be repro'd
    @skipIfNNModuleInlined()
    @requires_gpu
    def test_cpu_cuda_module_after_dynamo(self, device):
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = f"""\
class CpuCudaModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m_x = torch.nn.Linear(20, 20).to(device)
        self.m_y = torch.nn.Linear(20, 20)
        self.p_x = torch.nn.Parameter(torch.randn(20, 20).to(device))
        self.p_y = torch.nn.Parameter(torch.randn(20, 20))
        self.b_x = torch.nn.Buffer(torch.ones(20, 20).to(device))
        self.b_y = torch.nn.Buffer(torch.ones(20, 20))

    def forward(self, x, y):
        return self.m_x(x) + self.p_x + self.b_x, self.m_y(y) + self.p_y + self.b_y

mod = CpuCudaModule()

@torch.compile(backend={backend_name!r})
def inner(x1, y1):
    x2 = torch.randn(20, 20).to(device)
    y2 = torch.randn(20, 20)
    x3, y3 = mod(x1 + x2, y1 + y2)
    return torch.relu(x3.cpu() + y3)

inner(torch.randn(20, 20).to(device), torch.randn(20, 20))
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)

        self.assertExpectedInline(
            res.minifier_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.G__mod___m_x = Linear(in_features=20, out_features=20, bias=True).to(device)
        self.G__mod___m_y = Linear(in_features=20, out_features=20, bias=True)
        self.register_buffer('G__mod___b_x', torch.randn([20, 20], dtype=torch.float32).to(device))
        self.register_buffer('G__mod___b_y', torch.randn([20, 20], dtype=torch.float32))
        self.G__mod___p_x = torch.nn.Parameter(torch.randn([20, 20], dtype=torch.float32, device=device))
        self.G__mod___p_y = torch.nn.Parameter(torch.randn([20, 20], dtype=torch.float32))

    def forward(self, L_x1_ : torch.Tensor, L_y1_ : torch.Tensor):
        l_x1_ = L_x1_
        l_y1_ = L_y1_
        randn = torch.randn(20, 20)
        x2 = randn.to(device);  randn = None
        y2 = torch.randn(20, 20)
        add = l_x1_ + x2;  l_x1_ = x2 = None
        add_1 = l_y1_ + y2;  l_y1_ = y2 = None
        g__mod___m_x = self.G__mod___m_x(add);  add = None
        g__mod___p_x = self.G__mod___p_x
        add_2 = g__mod___m_x + g__mod___p_x;  g__mod___m_x = g__mod___p_x = None
        g__mod___b_x = self.G__mod___b_x
        x3 = add_2 + g__mod___b_x;  add_2 = g__mod___b_x = None
        g__mod___m_y = self.G__mod___m_y(add_1);  add_1 = None
        g__mod___p_y = self.G__mod___p_y
        add_4 = g__mod___m_y + g__mod___p_y;  g__mod___m_y = g__mod___p_y = None
        g__mod___b_y = self.G__mod___b_y
        y3 = add_4 + g__mod___b_y;  add_4 = g__mod___b_y = None
        cpu = x3.cpu();  x3 = None
        add_6 = cpu + y3;  cpu = y3 = None
        relu = torch.relu(add_6);  add_6 = None
        return (relu,)""",
        )

    # Test if we can actually get a minified graph
    def test_if_graph_minified(self):
        backend_name = "relu_compile_error_TESTING_ONLY"
        run_code = f"""\
@torch.compile(backend={backend_name!r})
def inner(x):
    for _ in range(20):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(20):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20))
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)

        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_19):
        x_20 = torch.relu(x_19);  x_19 = None
        return (x_20,)""",
        )


devices = ["cuda", "xpu", "cpu"]
instantiate_device_type_tests(
    MinifierTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

run_code = f"""\@torch.compile(backend={backend!r})def inner(x):    for _ in range(10):        x = torch.sin(x)    x = torch.relu(x)    for _ in range(10):        x = torch.cos(x)    return xinner(torch.randn(20, 20).to("{device}"))

This Python file contains 4 class(es) and 28 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MinifierTests`, `CpuCudaModule`, `Repro`, `Repro`

**Functions defined**: `_test_after_dynamo`, `inner`, `test_after_dynamo_cpu_compile_error`, `test_after_dynamo_cpu_runtime_error`, `test_after_dynamo_cpu_accuracy_error`, `test_after_dynamo_cuda_compile_error`, `test_after_dynamo_cuda_runtime_error`, `test_after_dynamo_cuda_accuracy_error`, `test_after_dynamo_non_leaf_compile_error`, `inner`, `_test_after_dynamo_backend_passes`, `inner`, `test_after_dynamo_cpu_compile_backend_passes`, `test_after_dynamo_cpu_runtime_backend_passes`, `test_after_dynamo_cpu_accuracy_backend_passes`, `test_after_dynamo_cuda_compile_backend_passes`, `test_after_dynamo_cuda_runtime_backend_passes`, `test_after_dynamo_cuda_accuracy_backend_passes`, `test_cpu_cuda_module_after_dynamo`, `__init__`

**Key imports**: unittest, torch._dynamo, MinifierTestBase, instantiate_device_type_tests, skipIfNNModuleInlined, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch._dynamo`
- `torch._dynamo.test_minifier_common`: MinifierTestBase
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_utils`: skipIfNNModuleInlined
- `torch._dynamo.test_case`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/dynamo/test_minifier.py
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

- **File Documentation**: `test_minifier.py_docs.md`
- **Keyword Index**: `test_minifier.py_kw.md`
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
python docs/test/dynamo/test_minifier.py_docs.md
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

- **File Documentation**: `test_minifier.py_docs.md_docs.md`
- **Keyword Index**: `test_minifier.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

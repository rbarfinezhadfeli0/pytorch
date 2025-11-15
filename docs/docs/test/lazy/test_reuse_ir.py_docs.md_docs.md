# Documentation: `docs/test/lazy/test_reuse_ir.py_docs.md`

## File Metadata

- **Path**: `docs/test/lazy/test_reuse_ir.py_docs.md`
- **Size**: 7,881 bytes (7.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/lazy/test_reuse_ir.py`

## File Metadata

- **Path**: `test/lazy/test_reuse_ir.py`
- **Size**: 4,769 bytes (4.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import unittest

import torch
import torch._lazy
import torch._lazy.config
import torch._lazy.ir_cache
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


torch._lazy.ts_backend.init()
torch._lazy.config.set_reuse_ir(True)


def get_test_device():
    return "cuda" if "LTC_TS_CUDA" in os.environ else "cpu"


@unittest.skipIf(IS_WINDOWS, "To be fixed")
class TestLazyReuseIr(TestCase):
    def testAdd(self):
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for _ in range(10):
            z += x + y

        for _ in range(10):
            z_lazy += x_lazy + y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 14
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSub(self):
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 8
        metrics.reset()
        torch._lazy.ir_cache.reset()

    def testAddSubFallback(self):
        torch._lazy.config.set_force_fallback("aten::sub")
        device = get_test_device()
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 8
        metrics.reset()
        torch._lazy.ir_cache.reset()
        torch._lazy.config.set_force_fallback("")

    def testBatchNorm(self):
        device = get_test_device()
        x = torch.randn(16, 3, 224, 224, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)

        for _ in range(10):
            # BatchNorm2d does extra checks on dimensions which SymInts don't support yet
            # so we call `torch.ops.aten.native_batch_norm` to bypass the checks.
            z, _, _ = torch.ops.aten.native_batch_norm(
                x, weight, bias, None, None, True, 0.1, 1e-5
            )
            z_legit, _, _ = torch.ops.aten._native_batch_norm_legit(
                x, weight, bias, True, 0.1, 1e-5
            )

        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        weight_lazy = weight.detach().clone().to(device=device)
        bias_lazy = bias.detach().clone().to(device=device)
        for _ in range(10):
            z_lazy, _, _ = torch.ops.aten.native_batch_norm(
                x_lazy, weight_lazy, bias_lazy, None, None, True, 0.1, 1e-5
            )
            z_legit_lazy, _, _ = torch.ops.aten._native_batch_norm_legit(
                x_lazy, weight_lazy, bias_lazy, True, 0.1, 1e-5
            )
            torch._lazy.mark_step()

        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        torch.testing.assert_close(z_legit.cpu(), z_legit_lazy.cpu())
        assert metrics.counter_value("IrNodeReused_torch::lazy::NativeBatchNorm") >= 7
        metrics.reset()
        torch._lazy.ir_cache.reset()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLazyReuseIr`

**Functions defined**: `get_test_device`, `testAdd`, `testAddSub`, `testAddSubFallback`, `testBatchNorm`

**Key imports**: os, unittest, torch, torch._lazy, torch._lazy.config, torch._lazy.ir_cache, torch._lazy.metrics as metrics, torch._lazy.ts_backend, IS_WINDOWS, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest`
- `torch`
- `torch._lazy`
- `torch._lazy.config`
- `torch._lazy.ir_cache`
- `torch._lazy.metrics as metrics`
- `torch._lazy.ts_backend`
- `torch.testing._internal.common_utils`: IS_WINDOWS, run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/lazy/test_reuse_ir.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/lazy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ts_opinfo.py_docs.md`](./test_ts_opinfo.py_docs.md)
- [`test_meta_kernel.py_docs.md`](./test_meta_kernel.py_docs.md)
- [`test_functionalization.py_docs.md`](./test_functionalization.py_docs.md)
- [`test_generator.py_docs.md`](./test_generator.py_docs.md)
- [`test_bindings.py_docs.md`](./test_bindings.py_docs.md)
- [`test_extract_compiled_graph.py_docs.md`](./test_extract_compiled_graph.py_docs.md)
- [`test_step_closures.py_docs.md`](./test_step_closures.py_docs.md)
- [`test_debug_util.py_docs.md`](./test_debug_util.py_docs.md)


## Cross-References

- **File Documentation**: `test_reuse_ir.py_docs.md`
- **Keyword Index**: `test_reuse_ir.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/lazy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/lazy/test_reuse_ir.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/lazy`):

- [`test_reuse_ir.py_kw.md_docs.md`](./test_reuse_ir.py_kw.md_docs.md)
- [`test_extract_compiled_graph.py_kw.md_docs.md`](./test_extract_compiled_graph.py_kw.md_docs.md)
- [`test_step_closures.py_kw.md_docs.md`](./test_step_closures.py_kw.md_docs.md)
- [`test_step_closures.py_docs.md_docs.md`](./test_step_closures.py_docs.md_docs.md)
- [`test_ts_opinfo.py_kw.md_docs.md`](./test_ts_opinfo.py_kw.md_docs.md)
- [`test_meta_kernel.py_kw.md_docs.md`](./test_meta_kernel.py_kw.md_docs.md)
- [`test_bindings.py_kw.md_docs.md`](./test_bindings.py_kw.md_docs.md)
- [`test_ts_opinfo.py_docs.md_docs.md`](./test_ts_opinfo.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_reuse_ir.py_docs.md_docs.md`
- **Keyword Index**: `test_reuse_ir.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

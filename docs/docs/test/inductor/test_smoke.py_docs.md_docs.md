# Documentation: `docs/test/inductor/test_smoke.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_smoke.py_docs.md`
- **Size**: 4,963 bytes (4.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_smoke.py`

## File Metadata

- **Path**: `test/inductor/test_smoke.py`
- **Size**: 1,836 bytes (1.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import logging
import unittest

import torch
import torch._logging
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(1, 6)
        self.l2 = torch.nn.Linear(6, 1)

    def forward(self, x=None):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


def _test_f(x):
    return x * x


class SmokeTest(TestCase):
    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_mlp(self):
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )

        mlp = torch.compile(MLP().to(GPU_TYPE))
        for _ in range(3):
            mlp(torch.randn(1, device=GPU_TYPE))

        # set back to defaults
        torch._logging.set_logs()

    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_compile_decorator(self):
        @torch.compile
        def foo(x):
            return torch.sin(x) + x.min()

        @torch.compile(mode="reduce-overhead")
        def bar(x):
            return x * x

        for _ in range(3):
            foo(torch.full((3, 4), 0.7, device=GPU_TYPE))
            bar(torch.rand((2, 2), device=GPU_TYPE))

    def test_compile_invalid_options(self):
        with self.assertRaises(RuntimeError):
            torch.compile(_test_f, mode="ha")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if IS_LINUX and HAS_GPU:
        if (not HAS_CUDA_AND_TRITON) or torch.cuda.get_device_properties(0).major <= 5:
            run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MLP`, `SmokeTest`

**Functions defined**: `__init__`, `forward`, `_test_f`, `test_mlp`, `test_compile_decorator`, `foo`, `bar`, `test_compile_invalid_options`

**Key imports**: logging, unittest, torch, torch._logging, TestCase, IS_LINUX, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `unittest`
- `torch`
- `torch._logging`
- `torch._inductor.test_case`: TestCase
- `torch.testing._internal.common_utils`: IS_LINUX


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
python test/inductor/test_smoke.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_smoke.py_docs.md`
- **Keyword Index**: `test_smoke.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



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
python docs/test/inductor/test_smoke.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_smoke.py_docs.md_docs.md`
- **Keyword Index**: `test_smoke.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

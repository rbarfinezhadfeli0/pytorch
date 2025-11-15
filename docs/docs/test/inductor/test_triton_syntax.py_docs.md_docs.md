# Documentation: `docs/test/inductor/test_triton_syntax.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_triton_syntax.py_docs.md`
- **Size**: 4,922 bytes (4.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_triton_syntax.py`

## File Metadata

- **Path**: `test/inductor/test_triton_syntax.py`
- **Size**: 1,898 bytes (1.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


class TestTritonSyntacticallyValid(TestCase):
    @requires_gpu()
    def test_triton_sqrt(self):
        # https://github.com/pytorch/pytorch/issues/142328
        import math

        import torch.nn as nn

        def newtonschulz5(G, steps: int, eps=1e-7):
            assert len(G.shape) == 2
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.to(
                torch.bfloat16
                if torch.cuda.is_bf16_supported(including_emulation=False)
                else torch.float16
            )
            X /= X.norm() + eps  # ensure top singular value <= 1
            if G.size(0) > G.size(1):
                X = X.T
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X
            if G.size(0) > G.size(1):
                X = X.T
            return X

        @torch.compile(backend="inductor")
        def scaled_newton_schulz(G, steps: int):
            shape = G.shape
            dtype = G.dtype
            G = G.reshape(shape[0], -1)
            G = newtonschulz5(G, steps)
            G = G.reshape(shape).type(dtype)
            G = G * math.sqrt(max(1, shape[0] / G[0].numel()))
            return G

        model = nn.Sequential(
            nn.Linear(16, 16, bias=False),
            nn.Linear(16, 32, bias=False),
        ).to(device=torch.device(GPU_TYPE))

        loss = model(torch.randn(4, 16, device=torch.device(GPU_TYPE))).sum()
        loss.backward()

        scaled_newton_schulz(model[0].weight.grad, 6)
        scaled_newton_schulz(model[1].weight.grad, 6)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTritonSyntacticallyValid`

**Functions defined**: `test_triton_sqrt`, `newtonschulz5`, `scaled_newton_schulz`

**Key imports**: torch, TestCase, GPU_TYPE, HAS_GPU, requires_gpu, math, torch.nn as nn, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.test_case`: TestCase
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU, requires_gpu
- `math`
- `torch.nn as nn`


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
python test/inductor/test_triton_syntax.py
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

- **File Documentation**: `test_triton_syntax.py_docs.md`
- **Keyword Index**: `test_triton_syntax.py_kw.md`
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
python docs/test/inductor/test_triton_syntax.py_docs.md
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

- **File Documentation**: `test_triton_syntax.py_docs.md_docs.md`
- **Keyword Index**: `test_triton_syntax.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

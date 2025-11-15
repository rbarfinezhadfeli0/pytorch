# Documentation: `docs/test/inductor/test_helion_kernels.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_helion_kernels.py_docs.md`
- **Size**: 5,537 bytes (5.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_helion_kernels.py`

## File Metadata

- **Path**: `test/inductor/test_helion_kernels.py`
- **Size**: 2,369 bytes (2.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import torch
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_HELION, requires_helion


if HAS_HELION:
    import helion
    import helion.language as hl


class HelionTests(TestCase):
    @requires_helion()
    def test_add_kernel(self):
        @helion.kernel(config=helion.Config(block_sizes=[1, 2]))
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # match pytorch broadcasting rules
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty(
                x.shape,
                # match type promotion of torch.add
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            # tile will be a tuple of blocks
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return add(x, y)

        x = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(4, 8, device=GPU_TYPE, dtype=torch.float16)

        out = add(x, y)
        compiled_add = torch.compile(f, fullgraph=True, backend="inductor")
        compiled_out = compiled_add(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(compiled_out, x + y)

    @requires_helion()
    def test_softmax_view_reshape(self):
        @helion.kernel(config={"block_size": 1})
        def softmax(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1).view(tile_n, 1)
                exp = torch.exp(values - amax)
                sum_exp = torch.reshape(torch.sum(exp, dim=1), [tile_n, 1])
                out[tile_n, :] = exp / sum_exp
            return out

        x = torch.randn([1024, 1024], device=GPU_TYPE, dtype=torch.float16)
        result = softmax(x)
        self.assertEqual(
            result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
        )


instantiate_parametrized_tests(HelionTests)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HelionTests`

**Functions defined**: `test_add_kernel`, `add`, `f`, `test_softmax_view_reshape`, `softmax`

**Key imports**: torch, run_tests, TestCase, instantiate_parametrized_tests, GPU_TYPE, HAS_HELION, requires_helion, helion, helion.language as hl


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.common_utils`: instantiate_parametrized_tests
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_HELION, requires_helion
- `helion`
- `helion.language as hl`


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
python test/inductor/test_helion_kernels.py
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

- **File Documentation**: `test_helion_kernels.py_docs.md`
- **Keyword Index**: `test_helion_kernels.py_kw.md`
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
python docs/test/inductor/test_helion_kernels.py_docs.md
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

- **File Documentation**: `test_helion_kernels.py_docs.md_docs.md`
- **Keyword Index**: `test_helion_kernels.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/test/inductor/test_needs_exact_strides.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_needs_exact_strides.py_docs.md`
- **Size**: 6,437 bytes (6.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_needs_exact_strides.py`

## File Metadata

- **Path**: `test/inductor/test_needs_exact_strides.py`
- **Size**: 3,219 bytes (3.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import torch
import torch.utils._pytree as pytree
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class TestNeedsExactStrides(InductorTestCase):
    @parametrize("dtype", [torch.float, torch.float8_e8m0fnu])
    def test_custom_op(self, dtype):
        device = "cuda"  # float8_e8m0fnu errors on "cpu"
        x = torch.ones(4, 4, 2, 2, device=device, dtype=torch.float8_e8m0fnu)
        other = torch.ones(4, 4, 2, 2, device=device, dtype=torch.float8_e8m0fnu)

        class _CustomPass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        class TestPassed(RuntimeError):
            pass

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                assert x.transpose(2, 3).is_contiguous()
                assert y.is_contiguous()
                return x.float() + y.float()

            def meta(x, y):
                return x.float() + y.float()

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            def f(x, other):
                return torch.ops.mylib.add_op.default(x.transpose(2, 3), other)

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                try:
                    f_compile = torch.compile(f, fullgraph=True)
                    f_compile(x, other)
                except TestPassed:
                    pass
                assert called


instantiate_parametrized_tests(TestNeedsExactStrides)

if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")

```



## High-Level Overview


This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNeedsExactStrides`, `_CustomPass`, `TestPassed`

**Functions defined**: `test_custom_op`, `__init__`, `__call__`, `_`, `decomp`, `impl2`, `impl`, `meta`, `f`

**Key imports**: torch, torch.utils._pytree as pytree, run_tests, TestCase as InductorTestCase, HAS_CUDA_AND_TRITON, config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.utils._pytree as pytree`
- `torch._inductor.test_case`: run_tests, TestCase as InductorTestCase
- `torch.testing._internal.inductor_utils`: HAS_CUDA_AND_TRITON
- `torch._inductor`: config


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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
python test/inductor/test_needs_exact_strides.py
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

- **File Documentation**: `test_needs_exact_strides.py_docs.md`
- **Keyword Index**: `test_needs_exact_strides.py_kw.md`
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
- **Error Handling**: Includes exception handling


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
python docs/test/inductor/test_needs_exact_strides.py_docs.md
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

- **File Documentation**: `test_needs_exact_strides.py_docs.md_docs.md`
- **Keyword Index**: `test_needs_exact_strides.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

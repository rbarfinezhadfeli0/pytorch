# Documentation: `docs/test/inductor/test_deterministic.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_deterministic.py_docs.md`
- **Size**: 6,939 bytes (6.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_deterministic.py`

## File Metadata

- **Path**: `test/inductor/test_deterministic.py`
- **Size**: 3,630 bytes (3.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import contextlib
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    IS_BIG_GPU,
)


@instantiate_parametrized_tests
class DeterministicTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._exit_stack.close()
        super().tearDown()

    def test_use_deterministic_algorithsm(self):
        old_val = torch.are_deterministic_algorithms_enabled()
        try:
            for new_val in [True, False, True]:
                torch.use_deterministic_algorithms(new_val, warn_only=True)
                self.assertEqual(inductor_config.deterministic, new_val)
        finally:
            torch.use_deterministic_algorithms(old_val, warn_only=True)

    @parametrize("deterministic", [False, True])
    def test_mm_padding(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2049, 2049], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @inductor_config.patch(max_autotune=True)
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_max_autotune(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2048, 2048], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] == 0)
            else:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] > 0)

    def test_pointwise_coordesc_tuning(self):
        @torch.compile(mode="max-autotune")
        def f(x):
            return x + 1

        x = torch.randn(2048, device=GPU_TYPE)
        self.assertEqual(f(x), x + 1)

        self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @parametrize("deterministic", [False, True])
    def test_reduction_coordesc_tuning(self, deterministic):
        with inductor_config.patch(
            deterministic=deterministic, coordinate_descent_tuning=True
        ):

            @torch.compile()
            def foo(x):
                return x.sum(dim=-1)

            inp = torch.rand([2048, 2048], device=GPU_TYPE)

            out = foo(inp)
            self.assertEqual(out, inp.sum(dim=-1))

            if deterministic:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DeterministicTest`

**Functions defined**: `setUp`, `tearDown`, `test_use_deterministic_algorithsm`, `test_mm_padding`, `foo`, `test_max_autotune`, `foo`, `test_pointwise_coordesc_tuning`, `f`, `test_reduction_coordesc_tuning`, `foo`

**Key imports**: contextlib, unittest, torch, torch._inductor.config as inductor_config, counters, run_tests, TestCase, fresh_cache


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `unittest`
- `torch`
- `torch._inductor.config as inductor_config`
- `torch._dynamo.utils`: counters
- `torch._inductor.test_case`: run_tests, TestCase
- `torch._inductor.utils`: fresh_cache


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
python test/inductor/test_deterministic.py
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

- **File Documentation**: `test_deterministic.py_docs.md`
- **Keyword Index**: `test_deterministic.py_kw.md`
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
python docs/test/inductor/test_deterministic.py_docs.md
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

- **File Documentation**: `test_deterministic.py_docs.md_docs.md`
- **Keyword Index**: `test_deterministic.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

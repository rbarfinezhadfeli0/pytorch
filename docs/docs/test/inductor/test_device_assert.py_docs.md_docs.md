# Documentation: `docs/test/inductor/test_device_assert.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_device_assert.py_docs.md`
- **Size**: 6,474 bytes (6.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_device_assert.py`

## File Metadata

- **Path**: `test/inductor/test_device_assert.py`
- **Size**: 3,203 bytes (3.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


@instantiate_parametrized_tests
class TestTorchDeviceAssertTrigger(TestCase):
    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_throw(self, backend):
        def func():
            a = torch.tensor([1.0, -2.0], device="cpu")
            result = torch.all(a > 0)
            assert result, "should throw"

        def func_inline():
            a = torch.tensor([1.0, -2.0], device="cpu")
            assert torch.all(a > 0), "should throw"

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            torch._dynamo.reset()
            f_c = torch.compile(func, backend=backend)
            f_c()

        with self.assertRaisesRegex(RuntimeError, "should throw"):
            torch._dynamo.reset()
            f_c = torch.compile(func_inline, backend=backend)
            f_c()

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_assert_should_not_throw(self, backend):
        def func():
            a = torch.tensor([1.0, 2.0], device="cpu")
            result = torch.all(a > 0)
            assert result, "should throw"

        def func_inline():
            a = torch.tensor([1.0, 2.0], device="cpu")
            assert torch.all(a > 0), "should throw"

        torch._dynamo.reset()
        f_c = torch.compile(func, backend=backend)
        f_c()

        torch._dynamo.reset()
        f_c = torch.compile(func_inline, backend=backend)
        f_c()

    @requires_cuda_and_triton
    @skipIfRocm
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_assert_fusion(self):
        torch._logging.set_logs(inductor_metrics=True)

        def func():
            a = torch.tensor([1.0, 2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"

        torch._dynamo.reset()
        f_c = torch.compile(func, backend="inductor")
        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)
        f_c()
        self.assertEqual(metrics.generated_kernel_count, 1)
        torch._logging.set_logs()

    @requires_cuda_and_triton
    @skipIfRocm
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_run_assert_triton(self):
        @torch.compile(backend="inductor")
        def fn():
            a = torch.tensor([1.0, 2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"

        def should_not_throw(fn):
            try:
                fn()
                return True
            except Exception:
                return False

        self.assertEqual(should_not_throw(fn), True)

        _, code = run_and_get_code(fn)
        self.assertEqual(code[0].count("tl.device_assert"), 1)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTorchDeviceAssertTrigger`

**Functions defined**: `test_assert_should_throw`, `func`, `func_inline`, `test_assert_should_not_throw`, `func`, `func_inline`, `test_assert_fusion`, `func`, `test_run_assert_triton`, `fn`, `should_not_throw`

**Key imports**: torch, torch._inductor.config, metrics, run_tests, TestCase, run_and_get_code, requires_cuda_and_triton


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.config`
- `torch._inductor`: metrics
- `torch._inductor.test_case`: run_tests, TestCase
- `torch._inductor.utils`: run_and_get_code
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/inductor/test_device_assert.py
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

- **File Documentation**: `test_device_assert.py_docs.md`
- **Keyword Index**: `test_device_assert.py_kw.md`
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
python docs/test/inductor/test_device_assert.py_docs.md
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

- **File Documentation**: `test_device_assert.py_docs.md_docs.md`
- **Keyword Index**: `test_device_assert.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

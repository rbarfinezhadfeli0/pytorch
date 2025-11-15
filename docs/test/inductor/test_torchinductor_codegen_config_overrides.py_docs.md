# Documentation: `test/inductor/test_torchinductor_codegen_config_overrides.py`

## File Metadata

- **Path**: `test/inductor/test_torchinductor_codegen_config_overrides.py`
- **Size**: 4,755 bytes (4.64 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. This file handles **configuration or setup**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import importlib
from collections.abc import Callable
from typing import Any, Optional
from unittest import skipIf

import torch
import torch.utils._pytree as pytree
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


importlib.import_module("filelock")


@instantiate_parametrized_tests
class CodegenInductorTest(InductorTestCase):
    def run_and_compare(
        self,
        func: Callable[..., Any],
        *args,
        compile_kwargs: Optional[dict] = None,
        config_patches: Optional[dict] = None,
        atol: float | None = 1e-05,
        rtol: float | None = 1e-08,
    ):
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}
        if config_patches is None:
            config_patches = {}

        def flatten_tensors(tensors):
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        with config.patch(config_patches):
            compiled = torch.compile(func, backend="inductor", **compile_kwargs)
            result, code = run_and_get_code(compiled, *args)

        # Check numerical accuracy
        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            self.assertTrue(torch.allclose(ref, actual, atol=atol, rtol=rtol))

        return result, code

    def count_code(self, substr: str, code: list[str], expected: Optional[int]):
        count = sum(prog.count(substr) for prog in code)
        if expected is not None:
            self.assertEqual(count, expected)

    @parametrize("force_pointwise_cat", [False, True])
    def test_force_pointwise_cat(self, force_pointwise_cat: bool):
        def func(a, b):
            return torch.cat([a + 1, b + 2], dim=0)

        a = torch.randn(1024, device=torch.device("cpu"))
        b = torch.randn(1024, device=torch.device("cpu"))
        config_patches = {
            "force_pointwise_cat": force_pointwise_cat,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )

        reinterpret_call = (
            "= reinterpret_tensor_wrapper("
            if config.cpp_wrapper
            else "= reinterpret_tensor("
        )
        if force_pointwise_cat:
            self.count_code(reinterpret_call, code, 0)
        else:
            self.count_code(reinterpret_call, code, 2)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    def test_cse_make_block_ptr_reduction(self):
        def func(a, b):
            tmp0 = a * b
            tmp1 = a + b
            c = tmp0 + tmp1
            return c.sum(dim=0)

        config_patches = {
            "triton.use_block_ptr": True,
            "triton.tile_reductions": True,
            "triton.prefer_nd_tiling": True,
            "triton.max_tiles": 3,
            "split_reductions": False,
        }
        a = torch.randn((512, 4096), device=torch.device(GPU_TYPE))
        b = torch.randn((512, 4096), device=torch.device(GPU_TYPE))
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
            atol=1e-4,
        )
        self.count_code("= tl.make_block_ptr(in_ptr", code, 2)
        self.count_code("= tl.load(block_ptr", code, 2)

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Triton is not available for MPS")
    def test_kernel_fusion_thresholds(self):
        def func(a, b):
            tmp0 = a + 1
            tmp1 = tmp0 + 2
            tmp2 = tmp1 + 3
            tmp3 = tmp2 + b
            return tmp0, tmp2, tmp3

        a = torch.randn(1024, device=torch.device(GPU_TYPE))
        b = torch.randn(1024, device=torch.device(GPU_TYPE))
        config_patches = {
            "max_fusion_size": 1,
            "realize_reads_threshold": 1,
            "realize_opcount_threshold": 1,
            "inplace_buffers": False,
        }
        _, code = self.run_and_compare(
            func,
            a,
            b,
            config_patches=config_patches,
        )
        self.count_code("@triton.jit", code, 3)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or HAS_CPU:
        run_tests(needs="filelock")

```



## High-Level Overview

"""        Runs the module through Inductor, comparing to eager reference.

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CodegenInductorTest`

**Functions defined**: `run_and_compare`, `flatten_tensors`, `count_code`, `test_force_pointwise_cat`, `func`, `test_cse_make_block_ptr_reduction`, `func`, `test_kernel_fusion_thresholds`, `func`

**Key imports**: importlib, Callable, Any, Optional, skipIf, torch, torch.utils._pytree as pytree, config, TestCase as InductorTestCase, run_and_get_code, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `collections.abc`: Callable
- `typing`: Any, Optional
- `unittest`: skipIf
- `torch`
- `torch.utils._pytree as pytree`
- `torch._inductor`: config
- `torch._inductor.test_case`: TestCase as InductorTestCase
- `torch._inductor.utils`: run_and_get_code


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/inductor/test_torchinductor_codegen_config_overrides.py
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

- **File Documentation**: `test_torchinductor_codegen_config_overrides.py_docs.md`
- **Keyword Index**: `test_torchinductor_codegen_config_overrides.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

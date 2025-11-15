# Documentation: `test/inductor/test_cutedsl_grouped_mm.py`

## File Metadata

- **Path**: `test/inductor/test_cutedsl_grouped_mm.py`
- **Size**: 5,346 bytes (5.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]


import unittest

import torch
from torch import Tensor
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import ensure_cute_available
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@unittest.skipIf(
    not (ensure_cute_available() and is_datacenter_blackwell_arch()),
    "CuTeDSL library or Blackwell device not available",
)
@instantiate_parametrized_tests
class TestCuTeDSLGroupedGemm(InductorTestCase):
    def _get_inputs(
        self,
        group_size: int,
        M_hint: int,
        K: int,
        N: int,
        device: str,
        dtype: torch.dtype,
        alignment: int = 16,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # --- Random, tile-aligned M sizes ---
        M_sizes = (
            torch.randint(1, (M_hint // alignment) + 1, (group_size,), dtype=torch.int)
            * alignment
        )

        M_total = torch.sum(M_sizes).item()

        # --- Construct input tensors ---
        A = torch.randn(int(M_total), K, dtype=dtype, device=device) * 0.1
        B = torch.randn((group_size, K, N), dtype=dtype, device=device) * 0.01

        # --- Build offsets (no leading zero, strictly increasing) ---
        offsets = torch.cumsum(M_sizes, dim=0).to(dtype=torch.int32, device=device)

        return (A, B, offsets)

    @parametrize("group_size", (2, 8))
    @parametrize("M_hint", (256, 1024))
    @parametrize("K", (64, 128))
    @parametrize("N", (128, 256))
    def test_grouped_gemm_basic(self, group_size: int, M_hint: int, K: int, N: int):
        device = "cuda"
        dtype = torch.bfloat16

        A, B, offsets = self._get_inputs(group_size, M_hint, K, N, device, dtype)

        def grouped_gemm_fn(A_packed, B_batched, offs):
            return torch._grouped_mm(A_packed, B_batched, offs=offs)

        # Eager execution
        c_eager = grouped_gemm_fn(A, B, offsets)

        # Test with Cute backend
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "test_configs.autotune_choice_name_regex": "cutedsl",
                "autotune_fallback_to_aten": False,
            }
        ):
            grouped_gemm_compiled = torch.compile(
                grouped_gemm_fn, backend="inductor", dynamic=False
            )
            c_compiled = grouped_gemm_compiled(A, B, offsets)

        self.assertEqual(c_eager.dtype, dtype)
        self.assertEqual(c_compiled.dtype, dtype)
        torch.testing.assert_close(c_eager, c_compiled)

    @parametrize("layout_A", ("contiguous", "offset", "padded", "view"))
    @parametrize("layout_B", ("contiguous", "broadcasted"))
    def test_grouped_gemm_assorted_layouts(
        self,
        layout_A: str,
        layout_B: str,
    ):
        device = "cuda"
        dtype = torch.bfloat16

        G, K, N = 8, 64, 128
        M_sizes = [128] * G
        sum_M = sum(M_sizes)
        offsets = torch.tensor(
            [sum(M_sizes[: i + 1]) for i in range(G)], dtype=torch.int32, device=device
        )

        A_base = torch.randn(sum_M, K, device=device, dtype=dtype)
        A = A_base

        if layout_A == "offset":
            # allocate bigger buffer than needed, use nonzero storage offset
            storage = torch.randn(sum_M * K + 512, device=device, dtype=dtype)
            offset = 128  # skip first 128 elements
            A = torch.as_strided(storage[offset:], (sum_M, K), (K, 1))
        elif layout_A == "padded":
            # simulate row pitch > K (row_stride = K + pad)
            row_pitch = K + 8
            storage = torch.randn(sum_M * row_pitch, device=device, dtype=dtype)
            A = torch.as_strided(storage, (sum_M, K), (row_pitch, 1))
        elif layout_A == "view":
            A_storage = torch.randn(sum_M * K, device=device, dtype=dtype)
            A = A_storage.view(sum_M, K)
            assert A._base is not None
            assert A.shape == (sum_M, K)

        B = torch.randn((G, K, N), dtype=dtype, device=device) * 0.01

        if layout_B == "broadcasted":
            # Broadcast B across groups (zero stride along G)
            B = B[0].expand(G, K, N)
            assert B.stride(0) == 0

        def grouped_gemm_fn(A_packed, B_batched, offs):
            return torch._grouped_mm(A_packed, B_batched, offs=offs)

        # --- eager ---
        c_eager = grouped_gemm_fn(A, B, offsets)

        # --- compiled (CUTE backend) ---
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "test_configs.autotune_choice_name_regex": "cutedsl",
                "autotune_fallback_to_aten": False,
            }
        ):
            grouped_gemm_compiled = torch.compile(
                grouped_gemm_fn, backend="inductor", dynamic=False
            )
            c_compiled = grouped_gemm_compiled(A, B, offsets)

        self.assertEqual(c_eager.dtype, dtype)
        self.assertEqual(c_compiled.dtype, dtype)
        torch.testing.assert_close(c_eager, c_compiled)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCuTeDSLGroupedGemm`

**Functions defined**: `_get_inputs`, `test_grouped_gemm_basic`, `grouped_gemm_fn`, `test_grouped_gemm_assorted_layouts`, `grouped_gemm_fn`

**Key imports**: unittest, torch, Tensor, config, is_datacenter_blackwell_arch, run_tests, TestCase as InductorTestCase, ensure_cute_available


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._inductor`: config
- `torch._inductor.codegen.cuda.cuda_env`: is_datacenter_blackwell_arch
- `torch._inductor.test_case`: run_tests, TestCase as InductorTestCase
- `torch._inductor.utils`: ensure_cute_available


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
python test/inductor/test_cutedsl_grouped_mm.py
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

- **File Documentation**: `test_cutedsl_grouped_mm.py_docs.md`
- **Keyword Index**: `test_cutedsl_grouped_mm.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

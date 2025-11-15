# Documentation: `docs/test/inductor/test_subgraph_choice.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_subgraph_choice.py_docs.md`
- **Size**: 9,695 bytes (9.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_subgraph_choice.py`

## File Metadata

- **Path**: `test/inductor/test_subgraph_choice.py`
- **Size**: 6,012 bytes (5.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor.ir import Buffer, FixedLayout, FlexibleLayout
from torch._inductor.lowering import register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfXpu, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


def decomposeK(a, b, kPartitions):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    B = k // kPartitions
    a_reshaped = torch.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
    b_reshaped = b.reshape(B, kPartitions, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    result_fp32 = result.to(torch.float32)
    reduced_buf = torch.sum(result_fp32, 0)
    return reduced_buf.to(a.dtype)


class TestSubgraphChoice(TestCase):
    def setUp(self):
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "decompose_k not supported on ROCm")
    def test_subgraph_decompose_k(self):
        from torch._inductor.kernel.mm import aten_mm
        from torch._inductor.kernel.mm_common import mm_args

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @torch.library.custom_op("mylib::matmul_decompose", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)

            choices = [aten_mm.bind((mat1, mat2), layout)]

            kPartitions = 256

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            # Test benchmarking against aten
            autotune_select_algorithm("test_subgraph_choice", choices, [a, b], layout)

            # Only return decomposeK case for codegen
            choices = [choices[1]]
            return autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )

        a_in = torch.randn(
            mat1_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            mat2_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose(mat1, mat2)

        compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

        res = compiled_func(a_in, b_in)

        # Check same results of compiled result and regular torch.mm
        torch.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)

    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "decompose_k not supported on ROCm")
    def test_subgraph_freeze_layout(self):
        from torch._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = torch.randn(
            (M, K), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            (K, N), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )

        @torch.library.custom_op("mylib::matmul_decompose_padding", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose_padding)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            mat1_layout = mat1.layout
            assert isinstance(mat1_layout, FlexibleLayout)
            mat1_stride = mat1_layout.stride

            choices = []

            kPartitions = 2

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            choice = choices[0]
            assert isinstance(mat1.layout, FixedLayout)

            # Creating the subgraph choice should have frozen the layout
            # We ensure padding so the stride should differ
            assert mat1.layout.stride != mat1_stride

            for example_stride, layout_stride in zip(
                choice.example_inputs[0].stride(), mat1.layout.stride
            ):
                # Example inputs should have same stride as current layout
                assert example_stride == layout_stride

            return autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("torch._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSubgraphChoice`

**Functions defined**: `decomposeK`, `setUp`, `_create_buffer`, `test_subgraph_decompose_k`, `matmul_decompose`, `_`, `_`, `func`, `test_subgraph_freeze_layout`, `matmul_decompose`, `_`, `_`, `func`

**Key imports**: unittest, mock, MagicMock, torch, Buffer, FixedLayout, FlexibleLayout, register_lowering, autotune_select_algorithm, run_tests, TestCase, skipIfXpu, TEST_WITH_ROCM, GPU_TYPE, HAS_CPU, HAS_GPU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `unittest.mock`: MagicMock
- `torch`
- `torch._inductor.ir`: Buffer, FixedLayout, FlexibleLayout
- `torch._inductor.lowering`: register_lowering
- `torch._inductor.select_algorithm`: autotune_select_algorithm
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.common_utils`: skipIfXpu, TEST_WITH_ROCM
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_CPU, HAS_GPU
- `torch._inductor.kernel.mm`: aten_mm
- `torch._inductor.kernel.mm_common`: mm_args


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/inductor/test_subgraph_choice.py
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

- **File Documentation**: `test_subgraph_choice.py_docs.md`
- **Keyword Index**: `test_subgraph_choice.py_kw.md`
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
python docs/test/inductor/test_subgraph_choice.py_docs.md
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

- **File Documentation**: `test_subgraph_choice.py_docs.md_docs.md`
- **Keyword Index**: `test_subgraph_choice.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

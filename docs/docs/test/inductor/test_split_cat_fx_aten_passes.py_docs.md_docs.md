# Documentation: `docs/test/inductor/test_split_cat_fx_aten_passes.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_split_cat_fx_aten_passes.py_docs.md`
- **Size**: 18,070 bytes (17.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_split_cat_fx_aten_passes.py`

## File Metadata

- **Path**: `test/inductor/test_split_cat_fx_aten_passes.py`
- **Size**: 14,225 bytes (13.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_cuda_and_triton


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False


class TestSplitCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        cat = torch.ops.aten.cat.default([x, y], 1)
        split = torch.ops.aten.split.Tensor(cat, 32, 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        cat_1 = torch.ops.aten.cat.default(
            [
                getitem,
                getitem_1,
                getitem_2,
                getitem_3,
                getitem_4,
                getitem_5,
                getitem_6,
                getitem_7,
            ],
            1,
        )
        cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        return torch.ops.aten.cat.default([cat_1, cat_2], 1)


class TestSplitCatSingular(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        cat = torch.ops.aten.cat.default([x, y], 1)
        split = torch.ops.aten.split.Tensor(cat, 32, 1)
        getitem = split[0]
        cat_1 = torch.ops.aten.cat.default(
            [getitem],
            1,
        )
        cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        return torch.ops.aten.cat.default([cat_1, cat_2], 1)


class TestSplitCatPartial(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ):
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
            x1,
            [
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
            ],
            1,
        )
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(
            x2, [96, 96, 96, 96], 1
        )
        getitem_71 = split_with_sizes_1[0]
        getitem_72 = split_with_sizes_1[1]
        getitem_73 = split_with_sizes_1[2]
        getitem_74 = split_with_sizes_1[3]
        getitem_75 = split_with_sizes_1[4]
        getitem_76 = split_with_sizes_1[5]
        getitem_77 = split_with_sizes_1[10]
        getitem_78 = split_with_sizes_1[11]
        getitem_79 = split_with_sizes_1[12]
        getitem_80 = split_with_sizes_1[13]
        getitem_81 = split_with_sizes_1[14]
        getitem_82 = split_with_sizes_1[15]
        getitem_83 = split_with_sizes_1[16]
        getitem_84 = split_with_sizes_1[17]
        getitem_85 = split_with_sizes_2[0]
        getitem_86 = split_with_sizes_2[1]
        getitem_87 = split_with_sizes_2[2]
        getitem_88 = split_with_sizes_2[3]

        cat = torch.ops.aten.cat.default(
            [
                z,
                getitem_71,
                getitem_72,
                getitem_73,
                getitem_74,
                getitem_75,
                getitem_76,
                getitem_82,
                getitem_83,
                getitem_84,
                y,
                getitem_77,
                getitem_78,
                getitem_79,
                getitem_80,
                getitem_81,
                y,
                getitem_85,
                getitem_86,
                getitem_87,
                getitem_88,
                z,
            ],
            1,
        )
        return cat


class TestMoveViewAferCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
            x, [1, 1, 1, 1, 1, 1, 1]
        )
        getitem_71 = split_with_sizes_1[0]
        getitem_72 = split_with_sizes_1[1]
        getitem_73 = split_with_sizes_1[2]
        getitem_74 = split_with_sizes_1[3]
        getitem_75 = split_with_sizes_1[4]
        getitem_76 = split_with_sizes_1[5]
        getitem_77 = split_with_sizes_1[6]
        view_1 = torch.ops.aten.view.default(getitem_71, [8, 96])
        view_2 = torch.ops.aten.view.default(getitem_72, [8, 96])
        view_3 = torch.ops.aten.view.default(getitem_73, [8, 96])
        view_4 = torch.ops.aten.view.default(getitem_74, [8, 96])
        view_5 = torch.ops.aten.view.default(getitem_75, [8, 96])
        view_6 = torch.ops.aten.view.default(getitem_76, [8, 96])
        view_7 = torch.ops.aten.view.default(getitem_77, [8, 96])
        clone = torch.ops.aten.clone.default(view_1)

        cat = torch.ops.aten.cat.default(
            [
                view_1,
                view_2,
                view_3,
                view_4,
                view_5,
                view_6,
                view_7,
            ],
            1,
        )
        cat_1 = torch.ops.aten.cat.default([clone, cat], 1)
        return torch.cat([clone, cat_1], 1)


class TestSelectCat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        select = torch.ops.aten.select.int(x, 1, 0)
        select_1 = torch.ops.aten.select.int(x, 1, 1)
        select_2 = torch.ops.aten.select.int(x, 1, 2)
        select_3 = torch.ops.aten.select.int(x, 1, 3)
        select_4 = torch.ops.aten.select.int(x, 1, 4)
        select_5 = torch.ops.aten.select.int(x, 1, 5)
        cat = torch.ops.aten.cat.default(
            [select, select_1, select_2, select_3, select_4, select_5], 1
        )
        cat1 = torch.ops.aten.cat.default(
            [select, select_1, select_2, select_3, select_4], 1
        )
        cat2 = torch.ops.aten.cat.default([select, select_2, select_4], 1)
        select_6 = torch.ops.aten.select.int(y, 1, 0)
        select_7 = torch.ops.aten.select.int(y, 1, 1)
        select_8 = torch.ops.aten.select.int(y, 1, 2)
        select_9 = torch.ops.aten.select.int(y, 1, 3)
        select_10 = torch.ops.aten.select.int(y, 1, 4)
        cat3 = torch.ops.aten.cat.default(
            [select_6, select_7, select_8, select_9, select_10], 1
        )
        return cat, cat1, cat2, cat3


class TestSplitCatAten(TestCase):
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "split_cat_aten_pass": {"threshold_to_cat": 5},
        },
    )
    def test_split_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 32, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSplitCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 5)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

        inputs = [
            torch.randn(1024, 96 * 21, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 96 * 4, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 96, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 96, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSplitCatPartial()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 3)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "split_cat_aten_pass": {"threshold_to_cat": 5},
        },
    )
    def test_split_cat_post_grad_singular(self):
        counters.clear()
        inputs = [
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 32, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSplitCatSingular()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 4)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 0)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "select_cat_aten_pass": {},
        },
    )
    def test_select_cat_post_grad(self):
        counters.clear()
        inputs = [
            torch.randn(1024, 6, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 6, 128, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestSelectCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 4)
        self.assertEqual(counters["inductor"]["select_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "move_view_after_cat_aten_pass": {},
        },
    )
    def test_move_view_after_cat_aten(self):
        counters.clear()
        inputs = [
            torch.randn(7, 8, 96, device=torch.device(device=GPU_TYPE)),
        ]
        module = TestMoveViewAferCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 4)
        self.assertEqual(counters["inductor"]["move_view_after_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()


class TestSplitCatAtenNormalizationPasses(TestCase):
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
        },
    )
    def test_split_aten_normalization(self):
        def arg_only_size_same(x):
            return torch.ops.aten.split.Tensor(x, 300, 1)

        def arg_only_size_different(x):
            return torch.ops.aten.split.Tensor(x, 320, 1)

        args = [
            torch.randn(4096, 300),
        ]
        for fn, expected_split_norm_count in [
            (arg_only_size_same, 1),
            (arg_only_size_different, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["normalization_aten_pass"],
                expected_split_norm_count,
                msg=f"for {fn}",
            )
            counters.clear()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 7 class(es) and 21 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSplitCat`, `TestSplitCatSingular`, `TestSplitCatPartial`, `TestMoveViewAferCat`, `TestSelectCat`, `TestSplitCatAten`, `TestSplitCatAtenNormalizationPasses`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `compare_dict_tensors`, `compare_pred`, `compare_parameters`, `compare_gradients`, `test_split_cat_post_grad`, `test_split_cat_post_grad_singular`, `test_select_cat_post_grad`, `test_move_view_after_cat_aten`, `test_split_aten_normalization`, `arg_only_size_same`

**Key imports**: torch, torch._inductor, counters, run_tests, TestCase, GPU_TYPE, requires_cuda_and_triton, deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor`
- `torch._dynamo.utils`: counters
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.inductor_utils`: GPU_TYPE
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton
- `deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/inductor/test_split_cat_fx_aten_passes.py
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

- **File Documentation**: `test_split_cat_fx_aten_passes.py_docs.md`
- **Keyword Index**: `test_split_cat_fx_aten_passes.py_kw.md`
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
python docs/test/inductor/test_split_cat_fx_aten_passes.py_docs.md
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

- **File Documentation**: `test_split_cat_fx_aten_passes.py_docs.md_docs.md`
- **Keyword Index**: `test_split_cat_fx_aten_passes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

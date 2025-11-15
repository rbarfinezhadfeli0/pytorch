# Documentation: `docs/test/inductor/test_mix_order_reduction.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_mix_order_reduction.py_docs.md`
- **Size**: 17,840 bytes (17.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_mix_order_reduction.py`

## File Metadata

- **Path**: `test/inductor/test_mix_order_reduction.py`
- **Size**: 14,094 bytes (13.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import same
from torch._inductor import metrics, utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestBase(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    def check_numeric(self, f, args, tol=1e-3):
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertTrue(same(ref, act, tol=tol))


class SkipPatternTest(TestBase):
    """
    Illustate the cases that we skip mix-order reduction. We skip in cases
    like when the outer reduction is followed by a pointwise that load
    the un-reduced tensor.
    """

    @inductor_config.patch(split_reductions=False)
    def test_dimension_too_close(self):
        """
        Skip if the two reduction size are too close.
        We require one reduction dimension to be much larger so we can split
        that dimension and make it efficient.
        """

        def f(x):
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0)
            return out1, out2

        x = torch.randn(768, 768, device=GPU_TYPE)
        torch.compile(f)(x)
        self.assertEqual(2, metrics.generated_kernel_count)

    @inductor_config.patch(split_reductions=False)
    def test_skip_if_outer_reduction_followed_by_full_pointwise(self):
        """
        Skip for now if the outer reduction is followed by a pointwise node
        accessing the original tensor. Accessing the reduced tensor is fine
        (e.g. to support torch.mean).
        """

        def f(x):
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0, keepdim=True) + x
            return out1, out2

        x = torch.randn(32768, 768, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(0, metrics.codegen_mix_order_reduction)


@instantiate_parametrized_tests
class MixOrderReductionTest(TestBase):
    @parametrize(
        "name",
        [
            "sum",
            "prod",
            "mean",
        ],
    )
    @parametrize("swap", (False, True))
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768), (32, 1024, 768)))
    def test_mix_order_reduction(self, name, swap, split_reductions, shape):
        # torch.prod does not accept tuple for dim argument
        if name == "prod" and len(shape) == 3:
            self.skipTest("Invalid combination")

        def f(x):
            def outer_red():
                if len(shape) == 3:
                    return reduction_fn(x, dim=(0, 1))
                else:
                    assert len(shape) == 2
                    return reduction_fn(x, dim=0)

            if swap:
                return outer_red(), reduction_fn(x, dim=-1)
            else:
                return reduction_fn(x, dim=-1), outer_red()

        reduction_fn = getattr(torch, name)
        dtype = torch.float
        x = torch.randn(shape, dtype=dtype, device=GPU_TYPE)

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = f(x)
        act = opt_f(x)

        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_xmask(self):
        """
        Make sure xmask is setup properly
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=0), x.sum(dim=1)

        M, N = 32768 + 1023, 768
        EXTRA_ROW = 1
        buf = torch.randn(M + EXTRA_ROW, N, device=GPU_TYPE)
        x = buf[:M, :]
        # make sure wrong xmask error loud if read excess elements
        buf[M:, :] = 1000000

        opt_f = torch.compile(
            f,
            options={
                "triton.mix_order_reduction_initial_xblock": 2,
            },
        )

        ref = f(x)
        act = opt_f(x)

        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_avoid_non_coalesced_access(self):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, y):
            return (x + y).sum(dim=-1), x.sum(dim=(0, 1))

        x = torch.randn(128, 256, 768, device=GPU_TYPE)
        y = torch.randn(128, 768, 256, device=GPU_TYPE).transpose(1, 2)
        self.check_numeric(f, (x, y))

        # we skip mix order reduction for such kernel since
        # we force XBLOCK to be 1, the access to tensor y would be
        # very inefficient.
        # TODO: support XBLOCK larger than 1. But in that case, we
        # would have bigger restriction on rnumel to avoid exploding
        # shared memory.
        self.assertEqual(metrics.codegen_mix_order_reduction, 0)

    @inductor_config.patch(coordinate_descent_tuning=True)
    def test_XBLOCK_coordest_tuning(self):
        """
        We should skip XBLOCK coordinate descent tuning for
        mix order reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=-1), x.sum(dim=0)

        x = torch.randn(32768, 256, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(metrics.codegen_mix_order_reduction, 1)

    @inductor_config.patch(unroll_reductions_threshold=1)
    def test_3layer_split_reduction(self):
        """
        Use a larger M and smaller N to trigger a 3 layer split reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=-1), x.sum(dim=0)

        x = torch.randn(32768 * 256, 2, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        # We don't do mix order reduction for split redutions
        # with more than 2 layers
        self.assertEqual(metrics.codegen_mix_order_reduction, 0)

    def test_independent_split_size(self):
        """
        Make sure mix order reduction can pick the split size it wants
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=-1), x.sum(dim=0)

        def check_one_split_size(split_size):
            torch._dynamo.reset()

            with inductor_config.patch(
                "triton.mix_order_reduction_split_size", split_size
            ):
                self.check_numeric(f, (x,))
                self.assertEqual(
                    inductor_config.triton.mix_order_reduction,
                    metrics.codegen_mix_order_reduction,
                )

                _, (code,) = utils.run_and_get_code(torch.compile(f), x)
                self.assertTrue(f"'RSPLIT_SIZE': {split_size}" in code)

        x = torch.randn(32768, 768, dtype=torch.float, device=GPU_TYPE)

        check_one_split_size(8)
        check_one_split_size(16)

    @inductor_config.patch(split_reductions=False)
    def test_non_contiguous_input(self):
        def f(x):
            return x.sum(dim=-1), x.sum(dim=[0, 1])

        x = torch.randn(1024, 32, 768, dtype=torch.float, device=GPU_TYPE).permute(
            1, 0, 2
        )
        self.check_numeric(f, (x,))
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @inductor_config.patch(split_reductions=False)
    def test_multi_workspace_allocation(self):
        def f(x, y):
            return x.sum(dim=0), x.sum(dim=1), y.sum(dim=0), y.sum(dim=1)

        x = torch.randn(4096 * 64, 32, device=GPU_TYPE)
        y = torch.randn(4098 * 64, 34, device=GPU_TYPE)

        self.check_numeric(f, (x, y))
        expected_mix_order_reduction = (
            0 if not inductor_config.triton.mix_order_reduction else 2
        )
        self.assertEqual(
            expected_mix_order_reduction, metrics.codegen_mix_order_reduction
        )

    @parametrize(
        "wdtype",
        [
            torch.bfloat16,  # extra down cast for dw is needed
            torch.float,
        ],
    )
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 2048), (32768, 768), (32768 + 1023, 768)))
    @parametrize("max_autotune", (False, True))
    @parametrize("initial_xblock", (1, 2))
    def test_rms_norm_bwd(
        self, wdtype, split_reductions, shape, max_autotune, initial_xblock
    ):
        # max_autotune can be slow and cost resource, trim down the tests
        # for max autotune
        if max_autotune and not (
            wdtype == torch.bfloat16
            and not split_reductions
            and shape in ((32768, 768), (32769, 768))
            and initial_xblock == 1
            and inductor_config.triton.mix_order_reduction
        ):
            self.skipTest("Skip non-critical tests to save resources.")

        def f(x, w, eps):
            orig_dtype = x.dtype

            x = x.float()
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)
            return y

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        torch.manual_seed(1337)

        # M, N = 1152 * 500, 384
        M, N = shape
        x = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
                "triton.mix_order_reduction_initial_xblock": initial_xblock,
                **(
                    {
                        "max_autotune": True,
                        "coordinate_descent_tuning": True,
                    }
                    if max_autotune
                    else {}
                ),
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize(
        "wbdtype",
        [
            torch.bfloat16,  # extra down cast for dw/db is needed
            torch.float,
        ],
    )
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768)))
    def test_layer_norm_bwd_with_bias(self, wbdtype, split_reductions, shape):
        def f(x, w, b, eps):
            return F.layer_norm(x, x.shape[-1:], w.float(), b.float(), eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            b.grad = None
            out = f(x, w, b, eps)
            out.backward(dy)
            return x.grad, w.grad, b.grad

        # M, N = 1152 * 500, 384
        M, N = shape
        xdtype = torch.float
        x = torch.randn(M, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768)))
    def test_layer_norm_bwd_no_bias(self, split_reductions, shape):
        def f(x, w, eps):
            return F.layer_norm(x, x.shape[-1:], w, bias=None, eps=eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        # M, N = 1152 * 500, 384
        M, N = shape
        xdtype = torch.float
        wbdtype = torch.float
        x = torch.randn(M, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )


@inductor_config.patch(
    "triton.mix_order_reduction", not inductor_config.triton.mix_order_reduction
)
class NoMixOrderReductionTest(MixOrderReductionTest):
    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()

```



## High-Level Overview

"""    Illustate the cases that we skip mix-order reduction. We skip in cases    like when the outer reduction is followed by a pointwise that load    the un-reduced tensor.

This Python file contains 4 class(es) and 33 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestBase`, `SkipPatternTest`, `MixOrderReductionTest`, `NoMixOrderReductionTest`

**Functions defined**: `setUp`, `check_numeric`, `test_dimension_too_close`, `f`, `test_skip_if_outer_reduction_followed_by_full_pointwise`, `f`, `test_mix_order_reduction`, `f`, `outer_red`, `test_xmask`, `f`, `test_avoid_non_coalesced_access`, `f`, `test_XBLOCK_coordest_tuning`, `f`, `test_3layer_split_reduction`, `f`, `test_independent_split_size`, `f`, `check_one_split_size`

**Key imports**: torch, torch._inductor.config as inductor_config, torch.nn.functional as F, same, metrics, utils, run_tests, TestCase, GPU_TYPE, HAS_GPU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor.config as inductor_config`
- `torch.nn.functional as F`
- `torch._dynamo.utils`: same
- `torch._inductor`: metrics, utils
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU


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
python test/inductor/test_mix_order_reduction.py
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

- **File Documentation**: `test_mix_order_reduction.py_docs.md`
- **Keyword Index**: `test_mix_order_reduction.py_kw.md`
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
python docs/test/inductor/test_mix_order_reduction.py_docs.md
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

- **File Documentation**: `test_mix_order_reduction.py_docs.md_docs.md`
- **Keyword Index**: `test_mix_order_reduction.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

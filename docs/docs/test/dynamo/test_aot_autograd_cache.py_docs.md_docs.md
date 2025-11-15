# Documentation: `docs/test/dynamo/test_aot_autograd_cache.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_aot_autograd_cache.py_docs.md`
- **Size**: 54,960 bytes (53.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_aot_autograd_cache.py`

## File Metadata

- **Path**: `test/dynamo/test_aot_autograd_cache.py`
- **Size**: 93,464 bytes (91.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import copy
import os
import shutil
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch._aot_autograd
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    autograd_cache_key,
    BypassAOTAutogradCache,
    sanitize_gm_for_cache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._guards import TracingContext
from torch._inductor import config as inductor_config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.runtime.triton_compat import tl, triton
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch._subclasses import FakeTensorMode
from torch.compiler._cache import CacheArtifactManager
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_cuda import SM80OrLater, TEST_MULTIGPU
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfWindows,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_triton
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.testing._internal.two_tensor import TwoTensor


def aot_eager_regional_inductor():
    """
    Regional inductor backend for AOT autograd.
    Uses regional_inductor as both forward and backward compiler.
    """
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    return aot_autograd(
        fw_compiler=regional_inductor,
        bw_compiler=regional_inductor,
    )


def saved_tensors_hooks_to_gm(
    pack_fn,
    unpack_fn,
    pack_cache_hash=None,
    unpack_cache_hash=None,
    symbolic_tracing=True,
    inp_fn=None,
):
    if symbolic_tracing:
        pack_gm = torch.fx.symbolic_trace(pack_fn)
        unpack_gm = torch.fx.symbolic_trace(unpack_fn)
    else:
        from functorch import make_fx

        if inp_fn:
            inp = inp_fn()
        else:
            inp = torch.randn(2, 3)
            torch._dynamo.mark_dynamic(inp, 0)
            torch._dynamo.mark_dynamic(inp, 1)
        pack_out = pack_fn(inp)
        pack_gm = make_fx(pack_fn)(inp)
        unpack_gm = make_fx(unpack_fn)(pack_out)

    def set_manual_hash(g, manual_hash):
        for node in g.nodes:
            if node.meta and node.meta.get("is_wrapped", False):
                node.meta["user_cache_hash"] = manual_hash

    if pack_cache_hash:
        set_manual_hash(pack_gm.graph, pack_cache_hash)
    if unpack_cache_hash:
        set_manual_hash(unpack_gm.graph, unpack_cache_hash)
    return pack_gm, unpack_gm


def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
):
    amax = amax.to(torch.float64)
    res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=1e-12)
    res = res.to(torch.float32)
    return res


# Must be at module level to use fx.wrap
@torch.fx.wrap
def _pack_fp8_with_scale_wrap(x):
    if not x.dtype.is_floating_point:
        return x

    amax = torch.max(torch.abs(x))
    scale = amax_to_scale(amax, torch.float8_e5m2)
    x_scaled = x.to(torch.float32) * scale
    x_fp8 = x_scaled.to(torch.float8_e5m2)
    return x.dtype, scale, x_fp8


@torch.fx.wrap
def _unpack_fp8_with_scale_wrap(x):
    if isinstance(x, torch.Tensor):
        return x

    dtype, scale, x_fp8 = x
    y = x_fp8.to(torch.float32) / scale
    return y.to(dtype)


@instantiate_parametrized_tests
class AOTAutogradCacheTests(InductorTestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        counters.clear()
        self._clear_all_caches()

    def _clear_all_caches(self):
        """
        Clear every cache, including AOTAutogradCache and FXCache
        """
        torch._inductor.codecache.FxGraphCache.clear()
        AOTAutogradCache.clear()
        CacheArtifactManager.clear()
        self._clear_dynamo_and_codecache()

    def _clear_dynamo_and_codecache(self):
        """
        Clear unrelated caches, like dynamo and PyCodeCache
        """
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

    @requires_triton()
    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    def test_cache_hot_load(self, device, dtype, dynamic):
        """
        Verify that we can populate and hot load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0

        if functorch_config.bundled_autograd_cache:
            self.assertEqual(len(cache_info.inductor_artifacts), 0)
        else:
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 4)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(len(cache_info.inductor_artifacts), 0)
            else:
                self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_weak_dynamic_indices"):
                del a._dynamo_weak_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 4)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_basic(self):
        """
        Verify the interactions between FXGraphCache and AOTAutogradCache.
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_vmap(self):
        """
        make
        """

        def fn(x, y):
            f = lambda x, y: (x * y + 1).sum(dim=0)  # noqa: E731
            vmapped = torch.vmap(f)(x, y)
            return vmapped.sum(dim=0)

        x = torch.randn(25, requires_grad=True)
        y = torch.randn(25, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(x, y), compiled_fn(x2, y2))
        fn(x, y).sum().backward()
        compiled_fn(x2, y2).sum().backward()
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset all tensors
        x = torch.randn(25, requires_grad=True)
        y = torch.randn(25, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(x, y), compiled_fn(x2, y2))
        fn(x, y).sum().backward()
        compiled_fn(x2, y2).sum().backward()
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_multi_graph_specialization(self):
        """
        Verify multi graph specializations all cache hit
        """

        def fn(x):
            return x * 5

        a = torch.randn(5)
        a8 = torch.randn(8)
        a16 = torch.randn(16)
        torch._dynamo.mark_dynamic(
            a,
            0,
            specialize_on=[
                lambda x: x == 8,
                lambda x: x == 16,
            ],
        )

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        compiled_fn(a)
        compiled_fn(a8)
        compiled_fn(a16)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

        self._clear_dynamo_and_codecache()

        # A second call should hit on all 3 graphs
        compiled_fn(a)
        compiled_fn(a8)
        compiled_fn(a16)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_symbol_specialization(self):
        """
        Verify the symbol specializations don't cause cache miss.
        """

        def fn(x, y, z):
            return (torch.randn(5) + x + y, z * torch.randn(1))

        a = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(a, 0)
        b = torch.rand(5)
        c = torch.randn(6)
        torch._dynamo.maybe_mark_dynamic(c, 0)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        compiled_fn(a, b, c)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # A second call should hit even if a new dimension is marked as dynamic
        # that is later specialized as part of tracing.
        a = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(a, 0)
        b = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(b, 0)
        c = torch.randn(6)
        torch._dynamo.maybe_mark_dynamic(c, 0)
        self._clear_dynamo_and_codecache()

        compiled_fn(a, b, c)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @functorch_config.patch({"enable_autograd_cache": True})
    def test_aot_runtime_trace_joint(self):
        @torch.compile(backend="inductor")
        def f(x):
            tmp = x.sin()
            s0 = tmp.shape[0]
            return tmp.expand(s0, s0)

        x_a = torch.randn(4, requires_grad=True)
        x = TwoTensor(x_a, x_a.clone())
        out = f(x)
        out.sum().backward()

        self._clear_dynamo_and_codecache()
        out = f(x)
        out.sum().backward()

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @skipIfWindows(
        msg="Known issue: Window can't delete loaded modules, so we can't clear module cache."
    )
    def test_clear_fx_graph_cache(self):
        """
        Verify the interactions between FXGraphCache and AOTAutogradCache.
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear FX graph cache: second call should also be a miss
        self._clear_dynamo_and_codecache()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        if functorch_config.bundled_autograd_cache:
            # Bundled AutogradCache doesn't care if FxGraphCache is cleared
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        else:
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            # We save again into the cache
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @requires_triton()
    def test_non_bundled_to_bundled_config_change(self):
        if functorch_config.bundled_autograd_cache:
            raise unittest.SkipTest("BundledAutogradCache is already enabled")

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25, device=GPU_TYPE)
        b = torch.rand(5, 5, device=GPU_TYPE)

        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Now turn on bundled autograd cache, see that we successfully save again
        with functorch_config.patch({"bundled_autograd_cache": True}):
            torch._dynamo.reset()
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "view_replay_for_aliased_outputs": True}
    )
    def test_view_replay(self):
        def fn(a):
            tmp = a.detach()
            a.mul_(2)
            return a, tmp

        with torch.autograd._force_original_view_tracking(True):
            compiled_fn = torch.compile(fn)

        def run_and_check(miss, hit, bypass):
            self._clear_dynamo_and_codecache()

            inp = torch.rand(2, 3)
            compiled_inp = inp.clone().detach()

            with torch.autograd._force_original_view_tracking(True):
                out = fn(inp)
                compiled_out = compiled_fn(compiled_inp)

            self.assertEqual(out, compiled_out)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], miss)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], hit)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], bypass)

        run_and_check(miss=1, hit=0, bypass=0)
        run_and_check(miss=1, hit=1, bypass=0)
        run_and_check(miss=1, hit=2, bypass=0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_invoke_subgraph(self):
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def gn(x, y):
            return x + y

        @torch.compile
        def fn(x, y):
            return gn(x, y) + gn(x, y)

        a = torch.randn(25)
        b = torch.randn(25)

        fn(a, b)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    @parametrize("fn_select", ("tag_activation_checkpoint", "allow_in_graph"))
    def test_unsafe_mark_cacheable(self, fn_select):
        if fn_select == "tag_activation_checkpoint":
            from torch.utils.checkpoint import checkpoint

            def gn(x, y, z=None):
                a = torch.matmul(x, y)
                if z is not None:
                    return torch.matmul(a, z)
                return a

            @torch.compile
            def fn(x, y, z):
                return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))

            fn_name = "torch.ops.higher_order.tag_activation_checkpoint"
        else:
            assert fn_select == "allow_in_graph"

            @torch._dynamo.allow_in_graph
            class AllowInGraphFunc(torch.autograd.Function):
                @staticmethod
                def forward(_, x):
                    torch._dynamo.graph_break()
                    return x.sin()

            @torch.compile
            def fn(x, y, z):
                return AllowInGraphFunc.apply(x)

            fn_name = "torch._dynamo.variables.misc.trampoline_autograd_apply"

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = torch.randn(4, 4)
        args = (x, y, z)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            r".*BypassAOTAutogradCache: Unsupported call_function target .*",
        ):
            fn(*args)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

        self._clear_dynamo_and_codecache()

        if fn_select == "allow_in_graph":
            # TODO: Fix allow in graph
            raise unittest.SkipTest(
                "Allow in graph produces an unserializable cache artifact"
            )

        with inductor_config.patch(
            "unsafe_marked_cacheable_functions", {fn_name: "key1"}
        ):
            fn(*args)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

            self._clear_dynamo_and_codecache()

            fn(*args)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

        self._clear_dynamo_and_codecache()
        with inductor_config.patch(
            "unsafe_marked_cacheable_functions", {fn_name: "key2"}
        ):
            fn(*args)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

            self._clear_dynamo_and_codecache()

            fn(*args)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

        # On second try with same key, it should hit once more
        with inductor_config.patch(
            "unsafe_marked_cacheable_functions", {fn_name: "key1"}
        ):
            self._clear_dynamo_and_codecache()

            fn(*args)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 3)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_fx_graph_cache_off(self):
        """
        Should not use cache if FXGraphCache is not enabled
        """

        def fn(x, y):
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear FX graph cache: second call should also be a miss
        self._clear_dynamo_and_codecache()

        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    @dynamo_config.patch("compiled_autograd", True)
    def test_compiled_autograd_bypass(self):
        # Need to make the compiled autograd graph serializable
        def fn(a, b):
            out = a.cos() + b
            loss = out.sum()
            ga, gb = torch.autograd.grad(loss, inputs=[a, b])

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="inductor")
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "BypassAOTAutogradCache: Unsupported call_function target torch._dynamo.compiled_autograd.ops.validate_outputs",
        ):
            compiled_fn(a, b)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @dynamo_config.patch("compiled_autograd", True)
    def test_inference_graph_cache_hit_with_compiled_autograd_enabled(self):
        def fn(a, b):
            out = a.cos() + b
            return out.sum()

        a = torch.randn(25)
        b = torch.randn(25)
        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_miss(self):
        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.sin()
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device="cuda", requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        result.sum().backward()
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        class MyAutogradFunction(torch.autograd.Function):  # noqa: F811
            # Change the function slightly
            @staticmethod
            def forward(ctx, x):
                y = x.cos()
                ctx.save_for_backward(y)
                ctx.foo = x.sin()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        # Clear dynamo and run again. Should be a cache miss.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function(self):
        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.sin()
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device="cuda", requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        result.sum().backward()
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_with_custom_triton_kernel(self):
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.ops.test.my_triton_op(x)
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)
        result.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_with_custom_triton_kernel_cache_invalidation(
        self,
    ):
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        class MyAutogradFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.ops.test.my_triton_op(x)
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        a3 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)
        result.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Now modify the source code of my_jit by redefining it
        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)  # Changed from +1 to +2

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        # Clear dynamo and run again. Should be a cache miss due to modified source code.
        counters.clear()
        self._clear_dynamo_and_codecache()
        compiled_fn = torch.compile(fn, backend="inductor")

        result = compiled_fn(a3)
        # Assert that after changing the source code, the cache no longer hits
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(fn(a3), result)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_cache_invalidation(self):
        from torch._library import capture_triton

        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            capture_triton(my_jit)[1,](y)
            return y

        def fn(a):
            return torch.ops.test.my_triton_op(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # Redefine the triton op

        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Second run should still miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @requires_cuda_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @unittest.expectedFailure  # Currently ops that call other ops does not properly invalidate cache
    def test_triton_op_cache_multiple_ops_invalidation(self):
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @triton.jit
        def my_jit2(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            torch._library.capture_triton(my_jit2)[1,](y)
            return y

        @torch._library.triton_op("test::my_triton_op2", mutates_args=())
        def my_triton_op2(x: torch.Tensor) -> torch.Tensor:
            y = x.clone().detach_().requires_grad_(True)
            torch.ops.test.my_triton_op(y)
            return y

        def fn(a):
            return torch.ops.test.my_triton_op2(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # Redefine the triton op

        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            torch._library.capture_triton(my_jit2)[1,](y)
            return y

        @torch._library.triton_op("test::my_triton_op2", mutates_args=())
        def my_triton_op2(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            y = x.clone().detach_().requires_grad_(True)
            torch.ops.test.my_triton_op(y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Second run should still miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_autograd_lazy_backward(self):
        """
        Lazily compile the backward, and lazily save to cache
        """

        def fn(a, b):
            return a.cos() + b

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear dynamo and run again. Should be a cache miss still, because backward hasn't run
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Now let's run the backward
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and rerun everything, now there should be a cache hit
        self._clear_dynamo_and_codecache()
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_autograd_no_dynamo_trace_backward(self):
        """
        Test that dynamo does not trace into the backward compiled function,
        even on cache hit.
        """
        torch._dynamo.eval_frame.clear_dynamo_tls()

        @torch.compile
        def fn(x):
            # Calls x.sum().backward() during forward execution of fn
            (x_grad,) = torch.autograd.grad(x.sum(), x)
            return x_grad

        a = torch.randn(10, 10, requires_grad=True, device="cpu")
        result = fn(a)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        # Backward of `sum` will run during execution of graph break
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        traced_frame_infos = copy.deepcopy(
            torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos
        )

        torch._dynamo.reset()
        torch._dynamo.eval_frame.clear_dynamo_tls()
        result2 = fn(a)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        new_traced_frame_infos = torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos
        self.assertEqual(result, result2)
        # Dynamo should trace exactly the same frames on cache hit
        self.assertEqual(traced_frame_infos, new_traced_frame_infos)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_function(self):
        """
        Tests autograd cache hits
        """

        def fn(a, b):
            return a.sin() + b

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset a
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python docs/test/dynamo/test_aot_autograd_cache.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_aot_autograd_cache.py_docs.md_docs.md`
- **Keyword Index**: `test_aot_autograd_cache.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

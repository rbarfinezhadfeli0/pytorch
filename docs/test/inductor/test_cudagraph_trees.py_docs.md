# Documentation: `test/inductor/test_cudagraph_trees.py`

## File Metadata

- **Path**: `test/inductor/test_cudagraph_trees.py`
- **Size**: 178,797 bytes (174.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
# ruff: noqa: F841
import contextlib
import functools
import gc
import importlib
import itertools
import re
import sys
import unittest
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch._dynamo.config as dynamo_config
import torch.nn as nn
from torch._dynamo.backends.debugging import aot_eager_decomp_partition_with_mode
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.cudagraph_trees import cudagraphify_impl as tree_cudagraphify_impl
from torch._inductor.cudagraph_utils import FunctionID
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.immutable_collections import immutable_dict
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    skipIfRocm,
    TEST_CUDA_GRAPH,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("functorch")
importlib.import_module("filelock")


aten = torch.ops.aten
requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from io import StringIO


def get_compile_fn(backend):
    if backend == "cudagraphs":
        return functools.partial(torch.compile, backend="cudagraphs")
    else:
        return functools.partial(torch.compile, mode="reduce-overhead")


class capture_stderr(list):
    """
    Replace sys.stderr with a temporary StringIO
    """

    def __enter__(self):
        self.sys_stderr = sys.stderr
        self.stringio = StringIO()
        sys.stderr = self.stringio
        return self

    def __exit__(self, *args):
        self.append(str(self.stringio.getvalue()))
        del self.stringio
        sys.stderr = self.sys_stderr


def cdata(t):
    return t.untyped_storage()._cdata


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()


if HAS_CUDA_AND_TRITON:

    def get_all_cudagraph_segments():
        segments = torch.cuda.memory_snapshot()
        return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]

    def all_live_blocks():
        blocks_addrs = []
        for segment in get_all_cudagraph_segments():
            addr = segment["address"]
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    blocks_addrs.append(addr)
                addr += block["size"]

        return blocks_addrs

    def all_live_block_count():
        return len(all_live_blocks())

    class CudaGraphTreeTests(TestCase):
        def setUp(self):
            super().setUp()
            self.graph_stack = contextlib.ExitStack()
            self.graph_stack.enter_context(
                config.patch(
                    {
                        "triton.cudagraphs": True,
                        "triton.cudagraph_trees": True,
                        "triton.fast_path_cudagraph_asserts": True,  # too slow
                        "triton.slow_path_cudagraph_asserts": True,
                    }
                )
            )
            self.graph_stack.enter_context(
                dynamo_config.patch(automatic_dynamic_shapes=True)
            )
            self.device_idx = torch.rand([0], device="cuda").device.index
            warnings.filterwarnings("ignore")

        def tearDown(self):
            super().tearDown()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            self.graph_stack.close()

            self.assertIsNone(self.get_manager())
            self.assertEqual(all_live_block_count(), 0)
            self.assertEqual(len(get_all_cudagraph_segments()), 0)
            warnings.resetwarnings()

        def get_manager(self, device_index=None):
            return torch._inductor.cudagraph_trees.get_container(
                device_index if device_index else self.device_idx
            ).tree_manager

        def get_roots(self):
            return self.get_manager().get_roots()

        def curr_node(self):
            return self.get_manager().current_node

        def get_root_children(self):
            return [root.num_descendants() for root in self.get_roots()]

        def cudagraphify_impl(
            self, *args, is_inference=True, is_backward=False, **kwargs
        ):
            return tree_cudagraphify_impl(
                *args,
                **kwargs,
                device_index=self.device_idx,
                is_inference=is_inference,
                is_backward=is_backward,
            )

        @staticmethod
        def run_twc(fn, *args, **kwargs):
            fn(*args, **kwargs)
            return fn(*args, **kwargs)

        def num_checkpoints(self):
            return self.get_manager().debug_checkpointing_counter

        def test_run_simple(self):
            def foo(x):
                return x * x * x

            foo_opt = torch.compile(foo)
            ones = torch.ones([4, 4], device="cuda")
            zeros = torch.zeros([5, 5], device="cuda")
            self.run_twc(foo_opt, ones)
            self.run_twc(foo_opt, zeros)
            self.assertEqual(self.get_root_children(), [0, 0])

        def check_rng(self):
            @torch.compile(mode="reduce-overhead")
            def foo():
                return torch.rand([20])

            torch.manual_seed(0)

            out = foo()
            out2 = foo()
            out3 = foo()

            torch.manual_seed(0)

            self.assertEqual(out, foo())
            self.assertEqual(out2, foo())
            self.assertEqual(out3, foo())

        @torch._inductor.config.patch("fallback_random", True)
        def test_rng_trees(self):
            self.check_rng()

        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        @torch._inductor.config.patch("fallback_random", True)
        def test_rng_non_trees(self):
            self.check_rng()

        def test_mutation_reinplaced(self):
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def forward(self, input, other, out):
                    input = torch.logical_xor(input=input, other=other, out=out)
                    return input

            x = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32).cuda()
            y = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32).cuda()
            z = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float16).cuda()

            model = Model().cuda()
            eag = model(x, y, z)
            with capture_stderr() as captured_output:
                opt = torch.compile(model.forward, mode="reduce-overhead")(x, y, z)

            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check("torch.logical_xor").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @requires_multigpu()
        @parametrize("backend", ("inductor", "cudagraphs"))
        def test_multiple_devices_msg(self, backend):
            def foo(x, y):
                return (x + 1, y + 2)

            foo = get_compile_fn(backend)(foo)
            with capture_stderr() as captured_output:
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            if torch._inductor.config.graph_partition:
                # graph partition splits on cpu ops
                self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
            else:
                FileCheck().check(
                    "skipping cudagraphs due to cpu device (arg1_1). Found from"
                ).check("y + 2").run(captured_output[0])
                self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

            with capture_stderr() as captured_output:
                foo(
                    torch.ones([10], device="cuda:0"), torch.ones([10], device="cuda:1")
                )

            FileCheck().check("skipping cudagraphs due to multiple devices").run(
                captured_output[0]
            )
            self.assertEqual(
                counters["inductor"]["cudagraph_skips"],
                1 if torch._inductor.config.graph_partition else 2,
            )

        @torch._inductor.config.patch("triton.cudagraph_skip_dynamic_graphs", True)
        def test_skip_symbolic(self):
            @torch.compile(dynamic=True)
            def foo(x, y):
                return x + y

            with capture_stderr() as captured_output:
                foo(torch.rand([10], device="cuda"), torch.rand([10], device="cuda"))

            FileCheck().check(
                "skipping cudagraphs due to graph with symbolic shapes inputs"
            ).check("x + y").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_on_inp(self, backend):
            def foo(x):
                x.add_(2)
                return x

            foo = get_compile_fn(backend)(foo)

            def inp():
                return torch.ones([10], device="cuda")

            with capture_stderr() as captured_output:
                foo(inp())

            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check(".add_(2)").run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

            # mutation on inp doesn't hit cudagraphs
            self.assertEqual(len(self.get_manager().roots), 0)

            # mutation on parameters/buffers hits cudagraphs
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.buf = torch.ones([10], device="cuda")

                def forward(self, x):
                    self.buf.add_(x)
                    return self.buf + x

            def foo(mod, x):
                return mod(x)

            foo = get_compile_fn(backend)(foo)
            mod = Mod()
            mod2 = Mod()

            for _ in range(3):
                self.assertEqual(foo(mod, inp()), mod2(inp()))
                self.assertEqual(mod.buf, mod2.buf)

            self.assertIsNotNone(self.get_manager())

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", False)
        def test_mutation_cudagraph_managed_tensors_config(self, backend):
            def foo(x):
                return x + 1

            def mut(x):
                x.add_(2)
                return x

            def non_mut(x):
                return x.add(2)

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            with capture_stderr() as captured_output:
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.rand([4], device="cuda")

                    tmp = foo(inp)
                    mut_out = mut(tmp)
                    self.assertEqual(mut_out, non_mut(foo(inp)))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(captured_output[0])

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensors(self, backend):
            def foo(x):
                return x + 1

            def mut(x):
                x.add_(2)
                return x

            def non_mut(x):
                return x.add(2)

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            with capture_stderr() as captured_output:
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.rand([4], device="cuda")

                    tmp = foo(inp)
                    mut_out = mut(tmp)
                    self.assertEqual(mut_out, non_mut(foo(inp)))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                0,
                exactly=True,
            ).run(captured_output[0])
            self.assertTrue("cudagraph_skips" not in counters["inductor"])

            torch.compiler.cudagraph_mark_step_begin()
            inp = torch.rand([4], device="cuda")
            tmp = foo(inp)
            mut_inp = tmp.clone()
            # in this case, what previously a mutated cudagraph managed tensor is no longer,
            # now its an input from eager we should fallback to inductor without cudagraphs
            with capture_stderr() as captured_output:
                mut(mut_inp)
            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check("x.add_(2)").run(captured_output[0])
            self.assertEqual(mut_inp, non_mut(foo(inp)))
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensor_warn(self, backend):
            def foo(x):
                return x.add_(1)

            def fee(y, z):
                return z.add(3)

            def inp():
                return torch.rand([4], device="cuda")

            foo = get_compile_fn(backend)(foo)
            fee = get_compile_fn(backend)(fee)

            with capture_stderr() as captured_output:
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    fee(inp(), foo(inp()))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensor_warn_only_once(self, backend):
            def foo(x):
                return x + 1

            def mut(x):
                x.add_(2)
                return x

            def inp():
                return torch.rand([4], device="cuda")

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            with capture_stderr() as captured_output:
                # Should warn for current_node=None
                mut(inp())

                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    tmp = foo(inp())
                    mut(tmp)  # should not warn

                mut_inp = tmp.clone()
                mut(mut_inp)  # should not warn since mut has warned

            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(captured_output[0])
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        def test_index_put(self):
            def fn(x, y, z):
                x = torch.zeros_like(x)
                return x.index_put_([y], z, True)

            fn_c = torch.compile(mode="reduce-overhead")(fn)

            for i in range(3):

                def args():
                    x = torch.zeros((512, 512), dtype=torch.bool, device="cuda")
                    y = torch.arange(512, dtype=torch.int64, device="cuda")
                    z = torch.ones((512, 512), dtype=torch.bool, device="cuda")
                    return x, y, z

                if i == 0:
                    out, code = run_and_get_code(fn_c, *args())
                    FileCheck().check("aten.index_put_").check_same("True").run(code[0])
                else:
                    out = fn_c(*args())

                self.assertEqual(fn(*args()), out)

        def test_function_compiled_multiple_times(self):
            def foo(x):
                y = foo2(x)
                y2 = foo2(y)
                return y + y2

            def foo2(x):
                torch._dynamo.graph_break()
                return x * x * x

            foo_opt = torch.compile(foo)
            ones = torch.ones([4, 4], device="cuda")
            foo(ones)
            foo_opt(ones)
            foo_opt(ones)
            self.assertEqual(foo_opt(ones), foo(ones))
            # paths
            children = self.get_root_children()
            # one root with two children
            self.assertEqual(children, [2])

        def test_end_recording_early(self):
            def foo(x):
                y = x * x * x
                torch._dynamo.graph_break()
                z = x + y
                return z

            @torch.compile
            def foo2(x):
                return x + 4

            foo_opt = torch.compile(foo)

            for _ in range(3):
                out = foo_opt(torch.ones([4, 4], device="cuda"))
                del out

                # when I tried inducing separate recordings via graph break,
                # the frame kept interfering by keeping outputs alive
                # this isn't great by simulates the logic.
                from torch._dynamo.mutation_guard import GenerationTracker

                GenerationTracker.generation -= 1

                out = foo2(torch.ones([4, 4], device="cuda"))
                del out

            foo_opt(torch.ones([4, 4], device="cuda"))

            # Two separate traces - one has a child, one doesn't
            self.assertEqual(self.get_root_children(), [1, 0])

        def test_execution_into_recording(self):
            def foo(x):
                y = x + x

                if y.sum() > 0:
                    return y + 10
                else:
                    return y - 10

            foo_opt = torch.compile(foo)
            inp = torch.zeros([4, 4], dtype=torch.float, device="cuda")
            self.assertEqual(foo_opt(inp), foo(inp))
            self.assertEqual(foo_opt(inp), foo(inp))

            inp.add_(1)
            out_eager = foo(inp)
            out_warmup = foo_opt(inp)
            self.assertEqual(out_warmup, out_eager)
            # warmup should be have storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)

            out_live = foo_opt(inp)
            self.assertEqual(out_live, out_eager)

            # should be in recording mode, with storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)
            # warmup should have been freed
            del out_warmup
            # should be in recording mode, with storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)

            del out_live
            self.assertEqual(all_live_block_count(), 0)

            out = foo_opt(inp)
            self.assertEqual(foo(inp), out)

            # should be in execution mode
            self.assertEqual(all_live_block_count(), 0)

        def test_forward_with_skipped_cudagraphed_backward(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            for _ in range(3):
                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)

                with config.patch(always_complex_memory_overlap_TESTING_ONLY=True):
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # we should not have cudagraph'd the backwards
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_cache_hit_forward_miss_backward(self):
            # Test that we don't cache cudagraphs, skipping cudagraphs on backward on a cache miss

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            # Run forwards, fx graph should cache miss
            for _ in range(3):
                torch._dynamo.reset()
                counters.clear()
                FxGraphCache.clear()
                AOTAutogradCache.clear()

                with config.patch(always_complex_memory_overlap_TESTING_ONLY=True):
                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

                    # Reset dynamo and related caches except for FXGraphCache
                    torch._dynamo.reset()
                    # Forwards should be a cache hit now, we still skip cudagraphs
                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

                    # Run backward without complex memory overlap being set

                # Run the backward without complex memory overlap reason
                # cache should miss, but cudagraphs should not run
                # because forward skipped it
                back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                out.backward(back_inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)

            # Run it one more time, this time AOTAutogradCache will hit
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            torch._dynamo.reset()
            inp = torch.rand([20, 20], device="cuda", requires_grad=True)
            out = foo(inp)
            back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
            out.backward(back_inp)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

            # we should not have cudagraph'd anything
            assert self.get_manager() is None

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        @requires_multigpu()
        def test_cached_boxed_forward_device_index(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            # Run with device index 1 so that we can see
            # on a cache hit we stay on device index 1
            with torch.cuda._DeviceGuard(1):
                torch.cuda.set_device(1)

                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                # Compile the backward and save to cache
                back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                out.backward(back_inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

                # Reset dynamo and rerun a few times
                for i in range(3):
                    torch._dynamo.reset()

                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    # Should cache hit each time; boxed_forward_device_index should still be set properly to 1
                    self.assertEqual(
                        counters["aot_autograd"]["autograd_cache_hit"], i + 1
                    )
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # After everything, we should have cudagraphs on device 1
            self.assertTrue(self.get_manager(device_index=0) is None)
            self.assertFalse(self.get_manager(device_index=1) is None)

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_backward_gets_cached_cudagraphs(self):
            # We pass cpu tensors to foo and save that into the cache
            # On a subsequent run in a new process, cudagraphs should be
            # disabled properly on both forward and backwards runs.

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                return x * x * x

            torch._dynamo.reset()
            counters.clear()
            FxGraphCache.clear()
            AOTAutogradCache.clear()

            # Use cpu device to disable cudagraphs during compilation
            inp = torch.rand([20, 20], device="cpu", requires_grad=True)
            out = foo(inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)

            # Run again on new process
            torch._dynamo.reset()

            # Forward and backward should also disable cudagraphs without compilation
            inp = torch.rand([20, 20], device="cpu", requires_grad=True)
            out = foo(inp)
            # AOTAutogradCache will load the forward and the backward from cache immediately, so fx_graph_cache_hit will equal 2
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            torch._dynamo.reset()

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)

            # we should not have cudagraph'd anything
            assert self.get_manager() is None

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_cached_forward_backward(self):
            counters.clear()
            AOTAutogradCache.clear()
            FxGraphCache.clear()

            @torch.compile
            def foo(x):
                torch.manual_seed(0)
                y = x * 2
                return torch.sin(y) * torch.nn.functional.dropout(x, p=0.4)

            inp = torch.rand([4, 4], requires_grad=True, device="cuda")
            inp2 = inp.detach().clone().requires_grad_(True)
            out = foo(inp)

            out.sum().backward()

            self.assertEqual(self.get_root_children(), [1])

            # the three saved tensors should die in the backward
            # we kept alive the output
            self.assertEqual(self.curr_node().expected_dead_indices_before_graph, [])
            if torch._inductor.config.graph_partition:
                self.assertEqual(
                    self.curr_node().expected_dead_indices_after_graph,
                    [(0, 0), (0, 2)],
                )
            else:
                self.assertEqual(
                    self.curr_node().expected_dead_indices_after_graph,
                    [(0, 1), (0, 2)],
                )
            self.assertFalse(self.get_manager().new_graph_id().id == 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)

            # Reset dynamo and rerun. We should see a cache hit now
            torch._dynamo.reset()

            out2 = foo(inp2)
            out2.sum().backward()
            self.assertEqual(out, out2)
            self.assertEqual(inp.grad, inp2.grad)

            self.assertEqual(self.get_root_children(), [1])
            self.assertFalse(self.get_manager().new_graph_id().id == 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        def test_forward_backward_not_called(self, backend):
            def foo(x, y):
                x_out = x * x * x
                torch._dynamo.graph_break()
                y_out = y * y * y
                return x_out, y_out

            foo = get_compile_fn(backend)(foo)

            for _ in range(3):
                inps = [
                    torch.rand([20, 20], requires_grad=True, device="cuda")
                    for _ in range(2)
                ]
                x_out, y_out = foo(inps[0], inps[1])
                x_out.sum().backward()

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            # we should not have cudagraph'd the y backward
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 3)

        def _test_unaligned_static_input_impl(self, expected_clones):
            def fn(x, y):
                return (x + y,)

            def get_aligned_inputs():
                return [torch.rand([5, 5], device="cuda") for _ in range(2)]

            mod = make_fx(fn)(*get_aligned_inputs())

            mode = torch._subclasses.FakeTensorMode()

            with mode:
                inps = [torch.rand([6, 5], device="cuda")[1:] for _ in range(2)]

            compiled_f = compile_fx_inner(
                mod, inps, static_input_idxs=[0], cudagraphs=True
            )

            def get_unaligned_inputs():
                return [torch.rand([6, 5], device="cuda")[1:] for _ in range(2)]

            class CloneCounterMode(TorchDispatchMode):
                def __init__(self) -> None:
                    self.count = 0

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = {} if kwargs is None else kwargs
                    self.count += func is torch.ops.aten.clone.default
                    return func(*args, **kwargs)

            for _ in range(3):
                with CloneCounterMode() as m:
                    compiled_f(get_unaligned_inputs())
                    self.assertEqual(m.count, expected_clones)

                    compiled_f(get_aligned_inputs())
                    self.assertEqual(m.count, expected_clones)

        def test_unaligned_static_input_trees(self):
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        def test_unaligned_static_input_non_trees(self):
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_unaligned_static_input_no_cudagraphs(self):
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("implicit_fallbacks", True)
        def test_graph_partition_custom_rule(self):
            def get_num_partitions(code):
                code = "".join(code)
                found = re.search(r"partitions=\[(.*)\]", code)
                assert found is not None
                partitions = found.group(1)
                num_partitions = len([p for p in partitions.split(",") if p])
                return num_partitions

            @torch.library.custom_op("mylib::bar", mutates_args=())
            def bar(x: torch.Tensor, flag: int) -> torch.Tensor:
                return x.clone()

            @bar.register_fake
            def _(x, flag):
                return x.clone()

            def f(x, flag):
                x = x + 1
                x = bar(x, flag)
                x = x + 1
                return x

            x = torch.randn(2, device="cuda")
            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x, True)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 1)

            @torch.library.custom_op("mylib::baz", mutates_args=())
            def baz(x: torch.Tensor) -> torch.Tensor:
                return x.clone()

            @baz.register_fake
            def _(x):
                return x.clone()

            # custom_should_partition_ops takes effect which lead to 2 partitions
            torch._inductor.config.custom_should_partition_ops = ["mylib::baz"]

            def f(x):
                x = x + 1
                x = baz(x)
                x = x + 1
                return x

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 2)

            # update the config should NOT force recompile
            torch._inductor.config.custom_should_partition_ops = []
            with torch.compiler.set_stance("fail_on_recompile"):
                f_compiled(x)

            # run_and_get_code forces recompile. Now we should cache miss, recompile, and
            # only have 1 partition.
            _, code = run_and_get_code(f_compiled, x)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 1)

            # test that op_overload name takes effect which lead to 2 partitions
            torch._inductor.config.custom_should_partition_ops = ["mylib::baz.default"]

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 2)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("implicit_fallbacks", True)
        def test_graph_partition_with_memory_plan_reuse(self):
            BATCH_SIZE = 16
            MLP_SIZE = 128
            HIDDEN_SIZE = 128
            RANDOM_SEED = 0

            @torch.library.custom_op(
                "silly::attention",
                mutates_args=["out"],
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def attention(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
            ) -> None:
                out.copy_(q + k + v)

            @attention.register_fake
            def _(q, k, v, out):
                return None

            class ParentModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x

            class Attention(torch.nn.Module):
                def __init__(self, mlp_size: int, hidden_size: int) -> None:
                    super().__init__()
                    self.pre_attn = torch.nn.Linear(mlp_size, hidden_size, bias=False)
                    self.post_attn = torch.nn.Linear(hidden_size, mlp_size, bias=False)
                    self.rms_norm_weight = torch.nn.Parameter(torch.ones(hidden_size))

                def rms_norm_ref(self, x: torch.Tensor) -> torch.Tensor:
                    x_f32 = x.float()
                    return (
                        x_f32
                        * torch.rsqrt(
                            torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6
                        )
                        * self.rms_norm_weight
                    ).to(x.dtype)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.pre_attn(x)
                    x = self.rms_norm_ref(x)
                    attn_output = torch.empty_like(x)
                    torch.ops.silly.attention(x, x, x, attn_output)
                    x = attn_output
                    x = self.rms_norm_ref(x)
                    x = self.post_attn(x)
                    return x

            class CompiledAttention(torch.nn.Module):
                def __init__(
                    self,
                    *,
                    mlp_size: int,
                    hidden_size: int,
                ) -> None:
                    super().__init__()
                    self.attn = Attention(mlp_size, hidden_size)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.attn(x)

            class CompiledAttentionTwo(CompiledAttention):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.attn(x) + x

            class SimpleModelWithTwoGraphs(ParentModel):
                def __init__(
                    self,
                    *,
                    mlp_size: int,
                    hidden_size: int,
                ) -> None:
                    super().__init__()
                    self.attn_one = CompiledAttention(
                        mlp_size=mlp_size,
                        hidden_size=hidden_size,
                    )
                    self.attn_two = CompiledAttentionTwo(
                        mlp_size=mlp_size,
                        hidden_size=hidden_size,
                    )

                    self.hidden_states = torch.zeros((BATCH_SIZE, MLP_SIZE)).cuda()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    bsz = x.shape[0]
                    # CUDAGraph expects same tensor addresses for each run
                    self.hidden_states[:bsz].copy_(x)
                    x = self.attn_one(self.hidden_states[:bsz])
                    self.hidden_states[:bsz].copy_(x)
                    x = self.attn_two(self.hidden_states[:bsz])
                    return x

            eager_model = (
                SimpleModelWithTwoGraphs(
                    mlp_size=MLP_SIZE,
                    hidden_size=HIDDEN_SIZE,
                )
                .eval()
                .cuda()
            )

            compiled_model = torch.compile(eager_model, mode="reduce-overhead")

            inputs = torch.randn(BATCH_SIZE, MLP_SIZE).cuda()

            for _ in range(3):
                eager_out = eager_model(inputs)
                compiled_out = compiled_model(inputs)
                self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        def test_graph_partition_gc(self):
            def _test_dummy():
                def foo(x):
                    return x + 1

                foo = torch.compile(foo)
                for _ in range(3):
                    foo(torch.randn(2, 3, device="cuda"))

            _test_dummy()
            gc.collect()
            self.assertIsNone(self.get_manager())

        def test_sparsity(self):
            def foo(view_6, buf31):
                return aten._sparse_coo_tensor_with_dims_and_tensors(
                    1,
                    1,
                    [1000000, 64],
                    view_6,
                    buf31,
                    dtype=torch.float32,
                    layout=torch.sparse_coo,
                    device="cuda",
                    pin_memory=None,
                )

            foo_opt = torch.compile(foo)

            view_6 = torch.zeros([1, 102397], dtype=torch.int64, device="cuda")
            buf31 = torch.rand([102397, 64], device="cuda")

            for _ in range(3):
                self.assertEqual(foo_opt(view_6, buf31), foo(view_6, buf31))

        def test_accumulate_multiple_recordings(self):
            def foo(x):
                y = x + x + x
                torch._dynamo.graph_break()
                if y.sum() <= 0:
                    return y
                else:
                    return y * 10

            foo_opt = torch.compile(foo)

            # two separate compilations & recordings
            out1 = self.run_twc(foo_opt, torch.zeros([5], device="cuda"))

            # out1 gets manually freed
            out2 = self.run_twc(foo_opt, torch.zeros([6], device="cuda"))

            self.assertEqual(all_live_block_count(), 1)

            out3 = self.run_twc(foo_opt, torch.ones([5], device="cuda"))

            self.assertEqual(out3, foo(torch.ones([5], device="cuda")))

            self.assertEqual(all_live_block_count(), 1)
            del out1, out2
            self.assertEqual(all_live_block_count(), 1)

            del out3
            gc.collect()
            self.assertEqual(all_live_block_count(), 0)

        @torch._inductor.config.patch("freezing", True)
        def test_constant_output(self):
            class Mod(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(
                        torch.tensor([float(i) for i in range(10)], device="cuda")
                    )

                def forward(self, inp):
                    return self.param, self.param[0:2], inp + 2

            inp = torch.tensor([2], device="cuda")
            m = Mod()
            with torch.no_grad():
                out_eager = m(inp)

                m_comp = torch.compile(m)
                for _ in range(3):
                    self.assertEqual(out_eager, m_comp(inp))

        def test_live_outputs_multiple_graphs(self):
            def foo(x):
                x = x + x + x
                y = x + 1
                torch._dynamo.graph_break()
                z = x * x
                if z.sum() > 0:
                    return y + 1
                else:
                    return y

            foo_opt = torch.compile(foo)

            self.run_twc(foo_opt, torch.zeros([5], device="cuda"))
            self.assertEqual(self.num_checkpoints(), 0)
            out = self.run_twc(foo_opt, torch.ones([5], device="cuda"))

            self.assertEqual(all_live_block_count(), 1)

            del out
            self.assertEqual(all_live_block_count(), 0)

            # we need to checkpoint from function to warmup y + 1,
            # and then again to record it
            self.assertEqual(self.num_checkpoints(), 2)

        def test_expanded_inputs(self):
            x = torch.rand(1, 512, device="cuda").expand(4, 512)

            def foo(x):
                return x + 4 + torch.ones([4, 512], device="cuda")

            foo_opt = torch.compile()(foo)

            for _ in range(3):
                self.assertEqual(foo_opt(x), foo(x))

            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_dies_between_checkpoint(self):
            def foo(args):
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            foo_cg(inp_list)
            foo_cg([inp])

            out1, out2 = foo_cg([inp])
            inp = [out1]

            del out1, out2

            def foo2(args):
                x = args[0]
                args.clear()
                return [x * x * x]

            self.assertEqual(self.num_checkpoints(), 0)
            foo2_cg = self.cudagraphify_impl(foo2, inp, ())

            x = foo2_cg(inp)[0]

            self.assertEqual(self.num_checkpoints(), 1)
            # out2 dies between the previous recording and the new one,
            # need to be manually deallocated after the checkpoint

            self.assertEqual(all_live_block_count(), 1)
            del x
            self.assertEqual(all_live_block_count(), 0)

        def test_aliased_storage_single_weakref(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                x = x * 20
                x_alias = x[0]
                y = x * 10
                y_alias = y[0]
                torch._dynamo.graph_break()
                ind = torch.tensor(4, device="cuda")
                x_alias2 = x[ind:]
                y_alias2 = y[ind:]
                return x, x_alias, x_alias2, y_alias, y_alias2

            for _ in range(4):
                outs = foo(torch.rand([20, 20], device="cuda"))

                ptr_to_ref = {
                    out.untyped_storage().data_ptr(): out.untyped_storage()._cdata
                    for out in outs
                }

                self.assertEqual(len(ptr_to_ref), 2)
                for out in outs:
                    self.assertEqual(
                        ptr_to_ref[out.untyped_storage().data_ptr()],
                        out.untyped_storage()._cdata,
                    )
                del outs
                del out

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def test_aliasing_static_ref(self):
            class Mod(torch.nn.Linear):
                def forward(self, x):
                    return self.weight.T @ x, self.weight.T, self.weight[0:4]

            m = Mod(10, 10).cuda()

            @torch.compile(mode="reduce-overhead")
            def foo(mod, x):
                return mod(x)

            @torch.compile(mode="reduce-overhead")
            def foo2(x):
                return x[2:]

            param_c = cdata(m.weight)
            for _ in range(3):
                x = torch.rand([10, 10], device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out1, alias_1, alias_2 = foo(m, x)
                self.assertEqual(len({param_c, cdata(alias_1), cdata(alias_2)}), 1)

                out2 = foo2(out1)
                out2.sum().backward()
                self.assertEqual(cdata(out1), cdata(out2))
                m.weight.grad = None
                m.bias.grad = None

            node = sel
```



## High-Level Overview


This Python file contains 31 class(es) and 466 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `capture_stderr`, `TestCase`, `CudaGraphTreeTests`, `Model`, `Mod`, `CloneCounterMode`, `ParentModel`, `Attention`, `CompiledAttention`, `CompiledAttentionTwo`, `SimpleModelWithTwoGraphs`, `Mod`, `Mod`, `AliasMod`, `Goo`, `Foo`, `TestModule`, `TestModule`, `Foo`, `Foo`

**Functions defined**: `get_compile_fn`, `__enter__`, `__exit__`, `cdata`, `setUpClass`, `tearDownClass`, `setUp`, `tearDown`, `get_all_cudagraph_segments`, `all_live_blocks`, `all_live_block_count`, `setUp`, `tearDown`, `get_manager`, `get_roots`, `curr_node`, `get_root_children`, `cudagraphify_impl`, `run_twc`, `num_checkpoints`

**Key imports**: contextlib, functools, gc, importlib, itertools, re, sys, unittest, warnings, defaultdict


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `functools`
- `gc`
- `importlib`
- `itertools`
- `re`
- `sys`
- `unittest`
- `warnings`
- `collections`: defaultdict
- `collections.abc`: Mapping, Sequence
- `torch`
- `torch._dynamo.config as dynamo_config`
- `torch.nn as nn`
- `torch._dynamo.backends.debugging`: aot_eager_decomp_partition_with_mode
- `torch._dynamo.utils`: counters
- `torch._functorch._aot_autograd.autograd_cache`: AOTAutogradCache
- `torch._inductor`: config
- `torch._inductor.codecache`: FxGraphCache
- `torch._inductor.compile_fx`: compile_fx_inner
- `torch._inductor.cudagraph_trees`: cudagraphify_impl as tree_cudagraphify_impl
- `torch._inductor.cudagraph_utils`: FunctionID
- `torch._inductor.test_case`: TestCase as InductorTestCase
- `torch._inductor.utils`: run_and_get_code
- `torch._ops`: OpOverload
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.fx.immutable_collections`: immutable_dict
- `torch.testing`: FileCheck
- `torch.testing._internal.common_cuda`: TEST_MULTIGPU
- `torch.testing._internal.inductor_utils`: HAS_CUDA_AND_TRITON


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_cudagraph_trees.py
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

- **File Documentation**: `test_cudagraph_trees.py_docs.md`
- **Keyword Index**: `test_cudagraph_trees.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

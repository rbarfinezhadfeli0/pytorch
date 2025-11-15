# Documentation: `test/test_jit_fuser_te.py`

## File Metadata

- **Path**: `test/test_jit_fuser_te.py`
- **Size**: 109,795 bytes (107.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["NNC"]
# ruff: noqa: F841

import contextlib
import math
import operator
import os
import sys
import unittest
import warnings

import torch
import torch.nn.functional as F
from torch.testing import FileCheck


# these needs to be set before `common_utils`
# infers `GRAPH_EXECUTOR`.
# this file **requires** these settings
# and setting them after `GRAPH_EXECUTOR` is
# inferred erroneously runs or skips
# some tests
torch._C._jit_set_profiling_executor(True)
torch._C._get_graph_executor_optimize(True)

if __name__ == "__main__":
    from torch.testing._internal.common_utils import parse_cmd_line_args

    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from itertools import combinations, permutations, product
from textwrap import dedent

from jit.test_fuser_common import TestFuserCommon  # noqa: F401
from test_jit import (
    backward_graph,
    get_lstm_inputs,
    get_milstm_inputs,
    LSTMCellC,
    LSTMCellF,
    LSTMCellS,
    MiLSTMCell,
)

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    IS_FBCODE,
    ProfilingMode,
    run_tests,
    skipIfTorchDynamo,
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn
from torch.testing._internal.jit_utils import (
    clone_inputs,
    get_traced_sample_variant_pairs,
    JitTestCase,
    NoTracerWarnContextManager,
    RUN_CUDA,
    RUN_CUDA_HALF,
    RUN_CUDA_MULTI_GPU,
    set_fusion_group_inlining,
    TensorExprTestOptions,
    warmup_backward,
)


FUSION_GROUP = "prim::TensorExprGroup"
LLVM_ENABLED = torch._C._llvm_enabled()

autograd_check_set = {
    "aten::__is__",
    "prim::AutogradAllNonZero",
    "prim::AutogradAllZero",
    "prim::ListConstruct",
}


def strip_profiling_nodes(nodes):
    profiling_opcodes = {"prim::BailoutTemplate", "prim::BailOut"}
    return [n for n in nodes if n.kind() not in profiling_opcodes]


def warmup_forward(f, *args, profiling_count=2):
    for _ in range(profiling_count):
        results = f(*args)

    return results


@contextlib.contextmanager
def texpr_reductions_enabled():
    old = torch._C._jit_set_texpr_reductions_enabled(True)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_reductions_enabled(old)


@contextlib.contextmanager
def texpr_enable_strategy(strategy):
    old = torch._C._jit_set_fusion_strategy(strategy)
    try:
        yield
    finally:
        torch._C._jit_set_fusion_strategy(old)


@contextlib.contextmanager
def inline_fusion_groups():
    old_inlining = torch._C._debug_get_fusion_group_inlining()
    torch._C._debug_set_fusion_group_inlining(True)
    try:
        yield
    finally:
        torch._C._debug_set_fusion_group_inlining(old_inlining)


class TestTEFuser(JitTestCase):
    def setUp(self):
        super().setUp()
        self.tensorexpr_options = TensorExprTestOptions()

        # note: `self.dynamic_shapes` instantiated in specialization of class
        # defined below

        fusion_strategy = [("DYNAMIC", 20)] if self.dynamic_shapes else [("STATIC", 20)]
        self.old_fusion_strategy = torch._C._jit_set_fusion_strategy(fusion_strategy)

        self.devices = ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
        self.int_dtypes = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        self.fp_dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ]
        self.dtypes = self.int_dtypes + self.fp_dtypes

    def tearDown(self):
        self.tensorexpr_options.restore()
        torch._C._jit_set_fusion_strategy(self.old_fusion_strategy)
        super().tearDown()

    def assertAllFused(self, graph, except_for=None):
        except_for = except_for if except_for is not None else set()
        # TODO - upstream
        guards = (
            "prim::TypeCheck",
            "prim::RequiresGradCheck",
            "prim::TensorExprDynamicGuard",
        )
        guard_found = False

        def autodiff_guard(node):
            if node.kind() != "aten::all":
                return False
            inps = list(node.inputs())
            if len(inps) != 1 or inps[0].node().kind() != "prim::ListConstruct":
                return False
            li_inps = list(inps[0].node().inputs())
            for li_inp in li_inps:
                if li_inp.node().kind() in (
                    "prim::AutogradAllNonZero",
                    "prim::AutogradAllZero",
                ):
                    return True
            return False

        def is_guard(node):
            return node.kind() in guards or autodiff_guard(node)

        for node in graph.block().nodes():
            if node.kind() == "prim::Constant":
                continue
            if is_guard(node):
                self.assertFalse(guard_found)
                guard_found = True
                continue
            if node.kind() in except_for:
                continue
            if node.kind() == "prim::If":
                self.assertTrue(is_guard(node.prev()))
                continue
            self.assertTrue(False, "Found unexpected node:" + node.kind())

        self.assertTrue(guard_found)

    def assertLastGraphAllFused(self):
        self.assertAllFused(torch.jit.last_executed_optimized_graph())

    def findFusionGroups(self, graph):
        result = []
        for n in graph.nodes():
            if n.kind() == FUSION_GROUP:
                result.append(n.g("Subgraph"))
                continue
            for block in n.blocks():
                result += self.findFusionGroups(block)
        return result

    def test_typecheck(self):
        a = torch.ones(1)

        def fused_kernel(a, b):
            return (a + b) * 2.0

        scripted = self.checkScript(fused_kernel, (a, a))
        graph = scripted.graph_for(a, a)
        # double check we fused
        fusion_groups = self.findFusionGroups(graph)
        self.assertEqual(len(fusion_groups), 1)
        # we use a bigger tensor now (size 2)
        # if we won't trigger a recompilation
        # we will still create a tensor up to (size 1)
        # if the type check fails
        a = torch.ones(2)
        # shape changed if we don't trigger recompilation
        # we would compute the wrong result silently
        self.assertEqual(scripted(a, a), fused_kernel(a, a))

    def test_sum_simple(self):
        def func(x):
            x2 = x * x
            return x2.sum()

        with texpr_reductions_enabled():
            a = torch.tensor(list(range(15)), dtype=torch.float, device="cpu")
            a = a.reshape(5, 3)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_nop(self):
        pass

    def test_sum_dim(self):
        def func(x):
            return x.sum((0,)) * 2

        def func_neg(x):
            return x.sum((-2,)) * 2

        with texpr_reductions_enabled():
            a = torch.tensor(list(range(15)), dtype=torch.float, device="cpu")
            a = a.reshape(5, 3)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()
            scripted = self.checkScript(func_neg, (a,))
            self.assertLastGraphAllFused()

    def test_sum_keepdim_cast(self):
        def func(x):
            return x.sum((0,), keepdim=True, dtype=torch.double) * 2

        with texpr_reductions_enabled():
            a = torch.tensor(list(range(15)), dtype=torch.float, device="cpu")
            a = a.reshape(5, 3)

            self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_abs(self):
        for device in self.devices:

            def func(x):
                return x.abs() * 2

            a = torch.randn(5, device=device)
            scripted = self.checkScript(func, (a,))
            self.assertLastGraphAllFused()

    def test_unsqueeze_size_calculation(self):
        for device in self.devices:

            def foo(b, d):
                x = d.unsqueeze(1)
                y = x * 42.0
                z = b + y
                r = z / 42.0
                return r

            inputs = (
                torch.rand(20, 28, device=device, requires_grad=True),
                torch.rand(20, device=device),
            )
            scripted = self.checkScript(foo, inputs)
            self.assertAllFused(scripted.graph_for(*inputs))

    def test_zero_element_tensors(self):
        for device in self.devices:

            def decode(sin_t, cos_t):
                theta = torch.atan2(sin_t.float(), cos_t.float())
                return theta

            sin = torch.zeros(0, device=device)
            cos = torch.zeros(0, device=device)
            inputs = [sin, cos]
            ge = self.checkScript(decode, inputs)

    def test_arg_configurations_smoke(self):
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        # A smoke test to make sure we won't use the same kernel for contiguous
        # and non-contiguous arguments.
        # TODO: add optionally enabled debug counters to the fuser to verify
        #       that we really can tell the difference between configurations
        for device in self.devices:

            def f(x, y):
                z1, z2 = (x + y).chunk(2, dim=1)
                return z1 * z2

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)
            traced_f = torch.jit.trace(f, (x, y))
            self.assertEqual(traced_f(x.t().contiguous(), y), traced_f(x.t(), y))

    def test_broadcast(self):
        for device in self.devices:

            def scaleshift(x, scale, shift):
                return x * scale + shift

            inputs = [
                torch.randn(4, 4, dtype=torch.float, device=device),
                torch.randn(4, dtype=torch.float, device=device),
                torch.randn(4, dtype=torch.float, device=device),
            ]
            self.checkScript(scaleshift, inputs)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_HALF, "no half support")
    @unittest.skipIf(
        GRAPH_EXECUTOR != ProfilingMode.LEGACY, "no half support with profiling on"
    )
    def test_cuda_half(self):
        x = torch.randn(4, 4, dtype=torch.half, device="cuda")
        y = torch.randn(4, 4, dtype=torch.half, device="cuda")

        funcs = [self.fn_test_comparison_gt_lt, self.fn_test_relu, self.fn_test_exp]

        # Note: Non fused inputs must be float to prevent loss of precision
        inputs = (x.float(), y.float())
        fusion_inputs = (x, y)
        for fn in funcs:
            local_inputs = [t.clone().requires_grad_() for t in inputs]
            local_fusion_inputs = [t.clone().requires_grad_() for t in fusion_inputs]

            # Verifies outputs
            fusion = torch.jit.trace(fn, local_fusion_inputs, check_trace=False)
            outputs = fn(*local_inputs)
            fusion_outputs = fusion(*local_fusion_inputs)
            outputs_half = [t.half() for t in outputs]
            self.assertEqual(outputs_half, fusion_outputs)

            # Verifies gradients
            for output, fusion_output in zip(outputs_half, fusion_outputs):
                grads = torch.autograd.grad(
                    output.float().sum(),
                    local_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                fusion_grads = torch.autograd.grad(
                    fusion_output.sum(),
                    local_fusion_inputs,
                    allow_unused=True,
                    retain_graph=True,
                )
                grads_half = [t.half() for t in grads]
                self.assertEqual(grads_half, fusion_grads)

    def test_checks_cat_inputs(self):
        # single fusion node causes error
        with set_fusion_group_inlining(True):
            for device in self.devices:
                # We shouldn't treat cat nodes as broadcasting. All their inputs
                # need to be checked for having the same map size, before we can
                # run the kernel.
                def f(x, y):
                    return torch.cat([x + 2 * x + x**2, y + 4 * y + y**3], dim=0)

                # NOTE: y is broadcastable to x, but output of f(x, y) should have
                # shape 3x4, and not 4x4.
                x = torch.randn(2, 4, dtype=torch.float, device=device)
                y = torch.randn(1, 4, dtype=torch.float, device=device)

                scripted = self.checkScript(f, (x, y))
                self.assertEqual(scripted(x, y).shape, (3, 4))
                self.assertAllFused(scripted.graph_for(x, y))

    def test_chunk(self):
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def fn(x):
                a, b, c = x.chunk(3, 1)
                return a * b + c

            inputs = [torch.randn(10, 6, dtype=torch.float, device=device)]

            self.checkScript(fn, inputs)
            self.assertLastGraphAllFused()

    def test_chunk_correctness(self):
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def chunk_4_0(x):
                x0, x1, x2, x3 = x.chunk(4, 0)
                return x0 + x1 + x2 + x3

            def chunk_4_1(x):
                x0, x1, x2, x3 = x.chunk(4, 1)
                return x0 + x1 + x2 + x3

            def chunk_4_last(x):
                x0, x1, x2, x3 = x.chunk(4, 2)
                return x0 + x1 + x2 + x3

            fns = [chunk_4_0, chunk_4_1, chunk_4_last]
            tensors = [
                # splitSize = 1
                torch.randn(4, 4, 4, dtype=torch.float, device=device),
                # contiguous case
                torch.randn(12, 8, 16, dtype=torch.float, device=device),
                # non-contiguous case
                torch.randn(12, 8, 16, dtype=torch.float, device=device).transpose(
                    1, 2
                ),
            ]

            for tensor in tensors:
                for fn in fns:
                    self.checkScript(fn, [tensor])
                    self.assertLastGraphAllFused()

    def test_chunk_distributes(self):
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def f(x, y):
                z1, z2 = (x + y).chunk(2, dim=1)
                return z1 * z2

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            graph = ge.graph_for(x, y)
            # XXX: The old fuser does broadcast_tensors but the new fuser doesn't.
            # FileCheck().check("broadcast_tensors").check('with ' + FUSION_GROUP + '_') \
            #     .check_count('ConstantChunk', 2, exactly=True).run(str(graph))
            FileCheck().check("with " + FUSION_GROUP + "_").check_count(
                "ConstantChunk", 1, exactly=True
            ).run(str(graph))

    def test_chunk_motion_deduplicates_inputs(self):
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:

            def func1(x):
                z = x * x
                z0, z1 = z.chunk(2)
                return z0 * z1

            def func2(x):
                z = x * x * x
                z0, z1 = z.chunk(2)
                return z0 * z1

            inputs = [torch.tensor([1.1, 1.2], device=device, dtype=torch.float)]
            for func in [func1, func2]:
                self.checkScript(func, inputs)
                self.assertLastGraphAllFused()

    def test_chunk_multiple(self):
        if self.dynamic_shapes:
            self.skipTest("TODO: chunk dynamic shapes")

        for device in self.devices:
            # The arguments are intentionally used out of order as a test to see
            # if the fusion compiler adds extra args in the correct order
            def fn(s, x, y, z):
                z1, z2 = z.chunk(2, 2)
                x1, x2, x3 = x.chunk(3, 1)
                y1, y2 = y.chunk(2, 0)
                return s + x1 + x2 + x3 + y1 + y2 + z1 + z2

            inputs = [
                torch.randn(5, 2, 3, dtype=torch.float, device=device),
                torch.randn(5, 6, 3, dtype=torch.float, device=device),
                torch.randn(10, 2, 3, dtype=torch.float, device=device),
                torch.randn(5, 2, 6, dtype=torch.float, device=device),
            ]

            ge = self.checkScript(fn, inputs)
            self.assertAllFused(ge.graph_for(*inputs))

    def test_minmax(self):
        for device in self.devices:

            def tmax(a, b):
                return torch.max(2 * a, b)

            def tmin(a, b):
                return torch.min(2 * a, b)

            a = torch.randn(4, 4, dtype=torch.float)
            b = torch.randn(4, 4, dtype=torch.float)
            nan = torch.tensor(float("nan"), dtype=torch.float)

            for f, inputs, device in product(
                (tmax, tmin), ([a, b], [a, nan], [b, nan]), self.devices
            ):
                inputs = [t.to(device) for t in inputs]
                s = self.checkScript(f, inputs)
                self.assertAllFused(s.graph_for(*inputs))

    def test_clamp(self):
        for device in self.devices:

            def func2(a, b):
                return torch.clamp(a + b, min=0, max=2)

            def funcInf(a, b):
                return torch.clamp(a + b, min=0, max=float("inf"))

            def funcNegInf(a, b):
                return torch.clamp(a + b, min=float("-inf"), max=0)

            def funcOptMin(a, b):
                return torch.clamp(a + b, max=2)

            def funcOptMax(a, b):
                return torch.clamp(a + b, min=0)

            a = torch.randn(4, 4, dtype=torch.float, device=device, requires_grad=True)
            b = torch.randn(4, 4, dtype=torch.float, device=device)
            nan = torch.tensor(float("nan"), dtype=torch.float, device=device)

            funcs = (func2, funcInf, funcNegInf, funcOptMin, funcOptMax)
            for f, inputs in product(funcs, [[a, b], [a, nan]]):
                inp1, inp2 = inputs
                s = self.checkScript(f, (inp1, inp2), profiling=ProfilingMode.PROFILING)
                self.assertAllFused(
                    s.graph_for(inp1, inp2),
                    except_for={"aten::size", "aten::_size_if_not_equal"},
                )
                c = s(inp1, inp2)
                with enable_profiling_mode_for_profiling_tests():
                    warmup_backward(c.sum())
                graph = backward_graph(s)
                self.assertAllFused(
                    graph,
                    except_for={"aten::Float", "aten::_grad_sum_to_size"}.union(
                        autograd_check_set
                    ),
                )

    def test_clamp_double(self):
        for device in self.devices:

            def clamp_double(x, eta: float):
                return 1 - x.clamp(eta, 1 - eta)

            x = torch.tensor([1.0, 1.0], dtype=torch.double, device=device)
            eta = 1e-9
            s = self.checkScript(
                clamp_double,
                (x, eta),
                profiling=ProfilingMode.PROFILING,
                atol=1e-10,
                rtol=1e-5,
            )
            self.assertAllFused(s.graph_for(x, eta), except_for={"aten::sub"})

    def test_clamp_int(self):
        for device in self.devices:

            def clamp_int(x, eta: int):
                return x.clamp(0, eta)

            x = torch.tensor([1, 1], device=device)
            eta = 1 << 32
            s = self.checkScript(clamp_int, (x, eta), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(s.graph_for(x, eta))

    def test_add_bool(self):
        sizes = [(1,), (2,), (4, 4)]
        for device, size in product(self.devices, sizes):

            def f(x, y, z):
                return x + y + z

            x = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            y = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            z = torch.randint(0, 2, size, dtype=torch.bool, device=device)
            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_mul_bool(self):
        for device in self.devices:

            def f(x, y, z):
                return x * y * z

            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)

            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_div_bool(self):
        for device in self.devices:

            def f(x, y, z):
                return (x + y) / z

            x = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            y = torch.randint(0, 2, (4, 4), dtype=torch.bool, device=device)
            z = torch.ones_like(x, dtype=torch.bool, device=device)

            ge = self.checkTrace(f, (x, y, z), inputs_require_grads=False)
            self.assertAllFused(ge.graph_for(x, y, z))

    def test_bitwise_ops(self):
        def apply(fn):
            return lambda x, y, z: fn(fn(x, y), z)

        binary_ops = [
            operator.__and__,
            operator.__or__,
            operator.__xor__,
            operator.__lshift__,
            operator.__rshift__,
        ]
        devices = self.devices
        for dtype, op, device in product(self.int_dtypes, binary_ops, devices):
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_minmax_int_ops(self):
        def apply(fn):
            return lambda x, y, z: fn(fn(x, y), z)

        binary_ops = [torch.min, torch.max]
        devices = self.devices
        for dtype, op, device in product(self.int_dtypes, binary_ops, devices):
            try:
                x = self.data_for(dtype, device)
                y = self.data_for(dtype, device)
                z = self.data_for(dtype, device)
                fn = apply(op)
                ref = fn(x, y, z)
            except Exception:
                # If eager mode doesn't support a dtype/op/device combo,
                # neither does the fuser.  Catch everything to avoid needing to
                # guess what errors might be thrown by eager.
                continue
            try:
                t = torch.jit.trace(fn, (x, y, z))
                self.assertEqual(ref, t(x, y, z))
                self.assertAllFused(t.graph_for(x, y, z))
            except Exception as e:
                raise RuntimeError(
                    " ".join(["Failed:", str(dtype), op.__name__, device])
                ) from e

    def test_comparison_eq_ne(self):
        for device in self.devices:

            def f(x, y):
                mask = (x == 0).type_as(x)
                z = x * mask + y
                mask = (x != 0).type_as(x)
                z = z * mask + y
                return z

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    @staticmethod
    def fn_test_comparison_gt_lt(x, y):
        mask = (x > 0).type_as(x)
        z = x * mask + y
        mask = (x < 0).type_as(x)
        z = z * mask + y
        return z

    def test_comparison_gt_lt(self):
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(self.fn_test_comparison_gt_lt, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_comparison_ge_le(self):
        for device in self.devices:

            def f(x, y):
                mask = (x >= 0).type_as(x)
                z = x * mask + y
                mask = (x <= 0).type_as(x)
                z = z * mask + y
                return z

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(f, (x, y))
            self.assertAllFused(ge.graph_for(x, y))
            x.requires_grad_(True)
            y.requires_grad_(True)
            self.assertAllFused(
                ge.graph_for(x, y),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    def test_addcmul(self):
        for device in self.devices:
            t = torch.randn(1, 4, dtype=torch.float, device=device)
            t1 = torch.randn(4, 1, dtype=torch.float, device=device)
            t2 = torch.randn(1, 4, dtype=torch.float, device=device)

            def foo(t, t1, t2):
                return t.addcmul(t + 1, t2, value=0.1)

            ge = self.checkTrace(foo, (t, t1, t2), allow_unused=True)
            graph = ge.graph_for(t, t1, t2)
            fusion_groups = self.findFusionGroups(graph)
            self.assertEqual(len(fusion_groups), 1)
            FileCheck().check("aten::add(").check("aten::addcmul(").run(
                str(fusion_groups[0])
            )

    # TODO: We leak CUDA memory here because the traced graph holds onto a
    # constant-ified tensor. Since the Python-global CompilationUnit is alive
    # until the end of the process, the memory is effectively leaked.
    # Removed `_cuda` suffix from this test which disables leak-checking.
    # If this is a real problem, we'll need to revisit Torchscript Function
    # lifetimes in Python.
    def test_lerp(self):
        for device in self.devices:
            start = torch.randn(4, 1, dtype=torch.float, device=device)
            end = torch.randn(1, 4, dtype=torch.float, device=device)
            weight = torch.tensor(0.5, dtype=torch.float, device=device)

            # scalar weight overload
            def foo_weight_scalar(start, end):
                return torch.lerp(start + 1, end, 0.5)

            # tensor weight overload
            def foo_weight_tensor(start, end):
                return torch.lerp(start + 1, end, weight)

            ge_weight_scalar = self.checkTrace(foo_weight_scalar, (start, end))
            graph = ge_weight_scalar.graph_for(start, end)
            self.assertAllFused(graph)

            # TODO: uncomment when TE enables support for scalar tensors
            # ge_weight_tensor = self.checkTrace(foo_weight_tensor, (start, end))
            # graph = ge_weight_tensor.graph_for(start, end)
            # self.assertAllFused(graph)

    def test_concat(self):
        # disabling concat causes error with single concat node
        with set_fusion_group_inlining(True):
            for device in self.devices:
                hx = torch.randn(3, 20, dtype=torch.float, device=device)
                cx = torch.randn(3, 20, dtype=torch.float, device=device)

                def foo(hx, cx):
                    return torch.cat((hx + cx, hx * cx))

                ge = self.checkTrace(foo, (hx, cx))
                graph = ge.graph_for(hx, cx)
                self.assertAllFused(graph)
                # XXX: TE fuser can handle concats in a fusion group.
                # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    def test_remove_output_used_only_in_size(self):
        for device in self.devices:

            def test_fuse(a, b):
                c = a + b
                d = c + b
                return d

            scripted_f = torch.jit.script(test_fuse)
            x = torch.ones(1, requires_grad=True, device=device)
            y = torch.ones(1, requires_grad=True, device=device)
            warmup_forward(scripted_f, x, y, profiling_count=3)
            g = scripted_f.graph_for(x, y)
            diff_nodes = g.findAllNodes("prim::DifferentiableGraph")
            self.assertEqual(len(diff_nodes), 1)
            g = diff_nodes[0].g("Subgraph")
            if_nodes = [n for n in g.nodes() if n.kind() == "prim::If"]
            self.assertEqual(len(if_nodes), 1)

            # the if node and the fusion group inside it should only have one output
            self.assertEqual(len(list(if_nodes[0].outputs())), 1)

    def test_concat_invariant(self):
        for device in self.devices:
            # Invariant: the output of prim::FusedConcat may
            # not be an input to any node inside the FusionGroup.
            def fn(x, y, z):
                x1 = x + y
                y1 = x - y
                w = torch.cat([x1, y1])
                return w + z

            x = torch.randn(2, 2, dtype=torch.float, device=device)
            y = torch.randn(2, 2, dtype=torch.float, device=device)
            z = torch.randn(4, 2, dtype=torch.float, device=device)
            ge = self.checkTrace(fn, (x, y, z))
            graph = ge.graph_for(x, y, z)
            self.assertAllFused(graph, except_for={"aten::add"})
            # XXX: TE fuser can handle concats inside a fusion group.
            # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    @staticmethod
    def fn_test_exp(x, y):
        return (x + 0.5 * y).exp()

    def test_exp(self):
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(self.fn_test_exp, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_threshold(self):
        for device in self.devices:

            def f(x):
                return torch.threshold(x, 0, -10) + x + x + x

            x = torch.tensor([-1, -0.5, 0, 1, 2, 3], device=device)
            scripted = self.checkScript(f, (x,))
            self.assertAllFused(scripted.graph_for(x))

    def test_scalar_arg(self):
        for device in self.devices:

            def fn_test_scalar_arg(x: torch.Tensor, p: float) -> torch.Tensor:
                return p * (x * x + x)

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            p = 3
            scripted = self.checkScript(fn_test_scalar_arg, (x, p))
            self.assertAllFused(scripted.graph_for(x, p))

            x.requires_grad_(True)

            # use another function otherwise we will bailout
            # and won't be able to do fused checks
            def fn_test_scalar_arg_requires_grad(
                x: torch.Tensor, p: float
            ) -> torch.Tensor:
                return p * (x * x + x)

            scripted = torch.jit.script(fn_test_scalar_arg_requires_grad)
            out = scripted(x, p)
            out = scripted(x, p)
            out = scripted(x, p)
            self.assertAllFused(
                scripted.graph_for(x, p),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_fusion_reuse_multi_gpu(self):
        def fn(x, y):
            return x * y * x * y

        inputs_cpu = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float),
        ]
        inputs_cuda0 = [x.cuda(0) for x in inputs_cpu]
        inputs_cuda1 = [y.cuda(1) for y in inputs_cpu]

        # Should not crash; these should compile different kernels.
        ge = self.checkScript(fn, inputs_cpu)
        self.assertAllFused(ge.graph_for(*inputs_cpu))
        ge(*inputs_cuda0)
        ge(*inputs_cuda1)

    # TODO: we're currently not checking 'device' in the type info when pulling
    # nodes into a fusion group. We should fix that and re-enable this test.
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_kernel_cache_multi_gpu(self):
        def not_fusible(x):
            return x

        def fn(x, y, z):
            x_out = x * x * x * x * x  # fusion: lambda x. x * x * x * x * x
            y_out = y * y * y * y * y
            z_out = z * z * z * z * z
            return not_fusible(x_out), not_fusible(y_out), not_fusible(z_out)

        inputs = [
            torch.randn(4, 4, dtype=torch.float),
            torch.randn(4, 4, dtype=torch.float, device="cuda:0"),
            torch.randn(4, 4, dtype=torch.float, device="cuda:1"),
        ]

        prev_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()

        # There are 3 FusionGroups. Because they have the same graph, they
        # should reuse the same KernelSpec in the KernelSpec cache.
        ge = self.checkScript(fn, inputs)
        self.assertGraphContainsExactly(ge.graph_for(*inputs), FUSION_GROUP, 3, True)
        new_cache_size = torch._C._jit_debug_fuser_num_cached_kernel_specs()
        # XXX: This assumes that the same kernel isn't already used by another test
        # FIXME: Use the TE fuser's way of querying the cache.
        # self.assertEqual(new_cache_size - prev_cache_size, 1)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_nonzero_device_cuda(self):
        device = "cuda:" + str(1)
        x = torch.tensor([0.4], dtype=torch.float, device=device)
        y = torch.tensor([0.7], dtype=torch.float, device=device)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y) + x))

        ge = self.checkTrace(doit, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    def test_lstm(self):
        for device in self.devices:
            inputs = get_lstm_inputs(device, training=True)
            module = self.checkScript(LSTMCellS, inputs)
            self.assertAllFused(
                module.graph_for(inputs), except_for={"prim::TupleConstruct"}
            )

    def test_lstm_concat(self):
        # single fusion node causes error
        with set_fusion_group_inlining(True):
            for device in self.devices:
                inputs = get_lstm_inputs(device)
                ge = self.checkTrace(LSTMCellC, inputs)
                graph = ge.graph_for(*inputs)
                except_nodes = {"prim::TupleConstruct", "aten::linear"}
                # TODO... Chunk
                if self.dynamic_shapes:
                    except_nodes = except_nodes.union(
                        {"aten::add", "prim::ConstantChunk"}
                    )
                self.assertAllFused(ge.graph_for(*inputs), except_for=except_nodes)
                # XXX: TE fuser can handle concats inside a fusion group.
                # FileCheck().check("FusedConcat").check_next("return").run(str(graph))

    def test_lstm_gates_permutations(self):
        for device in self.devices:
            # lstm has gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh.
            # Test that any permutation of this will still result in one FusionGroup.
            choices = ["x.mm(w_ih.t())", "hx.mm(w_hh.t())", "b_ih", "b_hh"]
            template = dedent(
                """
            def cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
                gates = {} + {} + {} + {}
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                return ingate * forgetgate * cellgate * outgate
            """
            )
            for permutation in permutations(choices, len(choices)):
                code = template.format(*permutation)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                fusion_group_len = 2 if self.dynamic_shapes else 1
                inputs = get_lstm_inputs(device, training=False)
                self.assertEqual(cu.cell(*inputs), scope["cell"](*inputs))
                forward_graph = cu.cell.graph_for(*inputs)
                self.assertGraphContainsExactly(
                    forward_graph, FUSION_GROUP, fusion_group_len
                )

    # TODO: Fuser doesn't work at all when inputs require grad. Fix that
    def test_lstm_traced(self):
        for device in self.devices:
            inputs = get_lstm_inputs(device)
            ge = self.checkTrace(LSTMCellF, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertEqual(len(fusion_groups), fusion_group_len)
            f = FileCheck()
            if not self.dynamic_shapes:
                f.check("Chunk")
            f.check("aten::sigmoid").check("aten::tanh").run(
                str(fusion_groups[0 if not self.dynamic_shapes else 1])
            )

    def test_milstm(self):
        if self.dynamic_shapes:
            self.skipTest("don't run conv with dynamic shapes")

        for device in self.devices:
            inputs = get_milstm_inputs(device, training=True)
            module = self.checkScript(MiLSTMCell, inputs)
            forward_graph = module.graph_for(*inputs)
            # TODO: chunk
            fusion_group_len = 2 if self.dynamic_shapes else 1
            self.assertGraphContainsExactly(
                forward_graph, FUSION_GROUP, fusion_group_len, consider_subgraphs=True
            )
            FileCheck().check("DifferentiableGraph").check("TupleConstruct").check_next(
                "return"
            ).check(FUSION_GROUP).run(str(forward_graph))
            hy, cy = module(*inputs)
            warmup_backward((hy + cy).sum())

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_cuda(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ["d"]

            def __init__(self) -> None:
                super().__init__()
                self.d = torch.device("cuda")

            @torch.jit.script_method
            def create(self, x):
                return x * x + x + torch.rand_like(x)

        x = torch.zeros([3, 4, 5], dtype=torch.float, device="cuda")
        m = M()
        out1 = m.create(x)
        out2 = m.create(x)
        self.assertNotEqual(out1, out2)
        self.assertTrue(torch.all(out1 >= 0))
        self.assertTrue(torch.all(out1 < 1))
        self.assertTrue(torch.all(out2 >= 0))
        self.assertTrue(torch.all(out2 < 1))
        self.assertAllFused(m.create.graph_for(x))

    @staticmethod
    def fn_test_relu(x, y):
        return F.relu(x + 0.5 * y)

    def test_relu(self):
        for device in self.devices:
            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(self.fn_test_relu, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    def test_erf(self):
        for device in self.devices:
            # only enabled on gpu
            if device == "cpu":
                continue

            def fn_test_erf(x):
                return F.relu(torch.erf(x) - torch.erfc(x))

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(ge.graph_for(x))
            x.requires_grad_(True)
            ge = self.checkScript(fn_test_erf, (x,), profiling=ProfilingMode.PROFILING)
            self.assertAllFused(
                ge.graph_for(x),
                except_for=(
                    "aten::size",
                    "prim::BroadcastSizes",
                    "aten::_size_if_not_equal",
                ),
            )

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_broadcast_cuda(self):
        def fn_test_rand(x, y):
            r = torch.rand_like(y)
            return r * x + x

        # If using profiling, a different function is needed to test different
        # shapes, or we'll use a cached script.
        def fn_test_rand2(x, y):
            r = torch.rand_like(y)
            return r * x * x

        x = torch.randn(4, 4, dtype=torch.float, device="cuda")
        y = torch.randn(4, 4, dtype=torch.float, device="cuda")
        script_f = torch.jit.script(fn_test_rand)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertAllFused(script_f.graph_for(x, y))
        x.requires_grad_(True)
        out = script_f(x, y)
        self.assertAllFused(
            script_f.graph_for(x, y),
            except_for=(
                "aten::size",
                "prim::BroadcastSizes",
                "aten::_size_if_not_equal",
            ),
        )

        # test that broadcasting random produces correct results
        x = torch.ones(4, 4, dtype=torch.float, device="cuda")
        y = torch.ones(4, dtype=torch.float, device="cuda")
        script_f = torch.jit.script(fn_test_rand2)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertEqual(out[0, :] + torch.zeros(4, 4, device="cuda"), out)

    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    @unittest.skip("rand_like is not supported yet")
    def test_rand_diamond(self):
        def fn_test_diamond(x, y):
            r = torch.rand_like(y)
            a = x + r
            b = y - r
            return a + b

        x = torch.randn(4, 4, dtype=torch.float, device="cuda")
        y = torch.randn(4, 4, dtype=torch.float, device="cuda")
        script_f = torch.jit.script(fn_test_diamond)
        warmup_forward(script_f, x, y)
        out = script_f(x, y)
        self.assertEqual(out, x + y)

    def test_scalar(self):
        def fn(x, y):
            return 2 * x + y

        x = torch.tensor(0.1, dtype=torch.float, device="cpu")
        y = torch.tensor(1, dtype=torch.float, device="cpu")
        ge = self.checkScript(fn, (x, y))
        self.assertAllFused(ge.graph_for(x, y))

    def test_inlined_optimized_graph(self):
        @torch.jit.script
        def foo(x):
            return torch.relu(x + x)

        for _ in range(3):
            foo(torch.rand([4, 4]))

        for _ in range(3):
            foo(torch.rand([10]))

        for _ in range(3):
            foo(torch.rand([2, 2, 2]))

        g = torch.jit.last_executed_optimized_graph()

        FileCheck().check_count("prim::If", 1, exactly=True).check(
            "prim::TensorExpr"
        ).run(g)
        torch._C._jit_pass_inline(g)
        f = FileCheck()
        for _ in range(3):
            f.check("prim::If").check("prim::TensorExpr")
        f.run(g)

    def test_small_constant(self):
        for device in self.devices:

            def fn_test_small_constant(x, y):
                return (1e-8 * x + 5e-9 * y) * 1e8

            x = torch.randn(4, 4, dtype=torch.float, device=device)
            y = torch.randn(4, 4, dtype=torch.float, device=device)

            ge = self.checkTrace(fn_test_small_constant, (x, y))
            self.assertAllFused(ge.graph_for(x, y))

    # Currently we don't pull constants into fusion groups, because in some
    # cases it could remove the constant from the original graph and now our
    # fusion group needs to return that constant for its other users.
    # Instead of never pulling constants into the fusion group, we should just
    # be more careful at how we rewrite its users.
    # TODO: fix that and reenable the test.
    def test_tensor_scalar_ops(self):
        for device in self.devices:

            def should_fuse(x):
                z = 3.0
                y = x + z
                return x * y

            def should_fuse_scalar(x, z):
                y = x + int(z)
                return x * y

            inputs = [torch.randn(2, 2, dtype=torch.float, device=device)]
            ge = self.checkScript(should_fuse, inputs)
            graph = ge.graph_for(*inputs)
            fusion_groups = self.findFusionGroups(graph)
            self.assertEqual(len(fusion_groups), 1)
            FileCheck().check("aten::add").check("aten::mul").run(str(fusion_groups[0]))

            inputs = [
                torch.randn(2, 2, dtype=torch.float, device=device),
                torch.tensor(3.0, dtype=torch.float, device=device),
            ]
            ge = self.checkScript(should_fuse_scalar, inputs)
            # Check that the fused graph computes correct results when the scalar
            # input changes.
            inputs = [
                torch.randn(2, 2, dtype=torch.float, device=device),
                torch.tensor(7.0, dtype=torch.float, device=device),
            ]
            self.assertEqual(ge(*inputs), should_fuse_scalar(*inputs))
            # The TE fuser supports fusion of non-constant scalars
            self.assertGraphContainsExactly(
                ge.graph_for(*inputs), FUSION_GROUP, 1, consider_subgraphs=True
            )

    def test_where_and_typing(self):
        for device in self.devices:

            def f(x, y):
                mask = x > y
                res = torch.where(mask, x, y)
                return mask, res

            x = torch.randn(4, 4, dtype=torch.double, device=device)
            y = torch.randn(4, 4, dtype=torch.double, device=device)

            script_f = self.checkScript(f, (x, y))
            self.assertAllFused(
                script_f.graph_for(x, y), except_for={"prim::TupleConstruct"}
            )

    def test_disabled(self):
        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(False)

        def fn(a):
            return a**2 + a

        x = torch.randn(4, dtype=torch.float, device="cpu")
        s = self.checkScript(fn, (x,))
        g = s.graph_for(x)
        self.assertEqual(len(self.findFusionGroups(g)), 0)

        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)

    def data_for(self, dtype, device="cuda", size=None):
        if size is None:
            v = torch.arange(1, 3, dtype=torch.float, device=device)
        else:
            v = torch.rand(*size, device=device)
        if dtype == torch.bool:
            return v > 2
        elif dtype in [torch.qint8, torch.quint8, torch.qint32]:
            return torch.quantize_per_tensor(v, 0.1, 1, dtype=dtype)
        else:
            return v.to(dtype)

    def test_torch_to(self):
        # test no op
        @torch.jit.script
        def foo(x):
            return x.to(torch.float)

        foo(torch.tensor([3.0], dtype=torch.float))
        foo(torch.tensor([3.0], dtype=torch.float))
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # test not fusing non-const inputs
        @torch.jit.script
        def foo(x, dtype: int):
            return x.to(dtype)

        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # test not fusing to_pinned inputs
        @torch.jit.script
        def foo(x, dtype: int):
            return x.to(pin_memory=True)

        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        foo(torch.tensor([3.0], dtype=torch.float), torch.int)
        FileCheck().check_not("TensorExpr").run(
            torch.jit.last_executed_optimized_graph()
        )

        # test across-device not supported
        if torch.cuda.is_available():

            @torch.jit.script
            def foo(x):
                return x.to(device="cuda")

            foo(torch.tensor([3.0], dtype=torch.float))
            foo(torch.tensor([3.0], dtype=torch.float))
            FileCheck().check_not("TensorExpr").run(
                torch.jit.last_executed_optimized_graph()
            )

        sizes = [(1, 4), (4, 4)]
        # reuses cast impl, smaller dtype set for faster test
        dtypes = [
            torch.bool,
            torch.i
```



## High-Level Overview


This Python file contains 11 class(es) and 260 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTEFuser`, `M`, `MyMod`, `TestTEFuserStatic`, `TestTEFuserDynamic`, `TestNNCOpInfoParent`, `TestNNCOpInfo`, `TestLoopnestRandomizationParent`, `TestLoopnestRandomization`

**Functions defined**: `strip_profiling_nodes`, `warmup_forward`, `texpr_reductions_enabled`, `texpr_enable_strategy`, `inline_fusion_groups`, `setUp`, `tearDown`, `assertAllFused`, `autodiff_guard`, `is_guard`, `assertLastGraphAllFused`, `findFusionGroups`, `test_typecheck`, `fused_kernel`, `test_sum_simple`, `func`, `test_nop`, `test_sum_dim`, `func`, `func_neg`

**Key imports**: contextlib, math, operator, os, sys, unittest, warnings, torch, torch.nn.functional as F, FileCheck


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `math`
- `operator`
- `os`
- `sys`
- `unittest`
- `warnings`
- `torch`
- `torch.nn.functional as F`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: parse_cmd_line_args
- `itertools`: combinations, permutations, product
- `textwrap`: dedent
- `jit.test_fuser_common`: TestFuserCommon  
- `torch.testing._internal.common_jit`: JitCommonTestCase
- `torch.testing._internal.common_methods_invocations`: op_db
- `torch.testing._internal.jit_metaprogramming_utils`: create_traced_fn
- `functools`: partial


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_jit_fuser_te.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_jit_fuser_te.py_docs.md`
- **Keyword Index**: `test_jit_fuser_te.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: test_aotdispatch.py

## File Metadata
- **Path**: `test/functorch/test_aotdispatch.py`
- **Size**: 333734 bytes
- **Lines**: 8767
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["oncall: pt2"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import unittest
import warnings
from collections.abc import Callable
from contextlib import ContextDecorator, ExitStack, nullcontext
from functools import partial, wraps
from typing import Any, Optional, Union
from unittest.mock import patch

from common_utils import (
    decorate,
    decorateForModules,
    saved_tensors_hooks_to_gm,
    skip,
    skipOps,
    xfail,
)

import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
from functorch import grad, jacrev, make_fx, vjp, vmap
from functorch.compile import (
    aot_function,
    aot_module,
    aot_module_simplified,
    compiled_function,
    compiled_module,
    default_decompositions,
    default_partition,
    get_aot_compilation_context,
    make_boxed_compiler,
    make_boxed_func,
    memory_efficient_fusion,
    min_cut_rematerialization_partition,
    nnc_jit,
    nop,
)
from functorch.experimental import control_flow
from torch._decomp import decomposition_table
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._functorch.aot_autograd import (
    _aot_export_function,
    aot_export_joint_simple,
    aot_export_module,
    SerializableAOTDispatchCompiler,
)
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.codecache import compiled_fx_graph_hash
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.output_code import MockFXGraphCacheOutput
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.utils.rnn import PackedSequence
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    compare_equal_outs_and_grads,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86,
    outs_and_grads,
    parametrize,
    run_tests,
    skipIfRocm,
    TEST_MKL,
    TestCase,
    xfail_inherited_tests,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_tensor import ConstantExtraMetadataTensor
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.optests import (
    _test_aot_autograd_forwards_backwards_helper,
    aot_autograd_check,
)
from torch.testing._internal.subclasses import WrapperSubclass
from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode
from torch.utils._python_dispatch import TorchDispatchMode


USE_TORCHVISION = False
try:
    import torchvision

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )

USE_NETWORKX = False
try:
    import networkx  # noqa: F401

    USE_NETWORKX = True
except ImportError:
    warnings.warn("Some tests use networkx but it was not installed", UserWarning)

# NB: numpy is a testing dependency!


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


@torch.fx.wrap
def _pack_fp8_wrap(x):
    if not x.dtype.is_floating_point:
        return x

    if type(x) is not torch.Tensor:
        # Check only during compilation
        # Test calls hooks to get reference output
        ctx = torch._functorch._aot_autograd.graph_compile._get_saved_tensor_hook_context()
        assert ctx["_fw_graph"] is not None
        assert ctx["_bw_graph"] is not None
        assert ctx["_node"] is not None

    return (x.dtype, x.to(torch.float8_e5m2))


@torch.fx.wrap
def _unpack_fp8_wrap(x):
    if isinstance(x, torch.Tensor):
        return x

    dtype, tensor = x
    if type(tensor) is not torch.Tensor:
        # Check only during compilation
        # Test calls hooks to get reference output
        ctx = torch._functorch._aot_autograd.graph_compile._get_saved_tensor_hook_context()
        assert ctx["_fw_graph"] is not None
        assert ctx["_bw_graph"] is not None
        assert ctx["_node"] is not None
    return tensor.to(dtype)


def pack_fp8(x):
    return _pack_fp8_wrap(x)


def unpack_fp8(packed):
    return _unpack_fp8_wrap(packed)


def pack_fp8_with_scale(x):
    return _pack_fp8_with_scale_wrap(x)


def unpack_fp8_with_scale(packed):
    return _unpack_fp8_with_scale_wrap(packed)


class AOTTestCase(TestCase):
    pass


class TestPythonKey(AOTTestCase):
    def test_make_fx(self, device):
        def f(x):
            return torch.sin(x)

        inp = torch.randn(3)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_grad(self, device):
        def f(x):
            return torch.sin(x).sum()

        inp = torch.randn(3)
        f = grad(f)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_scalar_device(self, device):
        def f(a, b):
            return a + b

        inps = [torch.randn(3, device=device), torch.tensor(5)]
        fx_f = make_fx(f)(*inps)
        self.assertEqual(fx_f(*inps), f(*inps))

    def test_make_fx_vmap(self, device):
        def f(x):
            return torch.sin(x)

        inp = torch.randn(5, 3)
        f = vmap(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_jacrev(self, device):
        def f(x):
            return x.sin().sum()

        inp = torch.randn(3)
        f = jacrev(jacrev(f))
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_vjp(self, device):
        def f(x):
            return torch.sin(x).sum()

        primals = torch.randn(3)
        _, vjp_fn = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    def test_make_fx_functionalize(self, device):
        from functorch.experimental import functionalize

        def fn(a):
            a = a * 2
            a.relu_()
            return a

        a = torch.randn(3, device=device)
        symbolic_gm = torch.fx.symbolic_trace(fn)
        includes_method_relu_ = any(
            str(n.target) == "relu_" for n in symbolic_gm.graph.nodes
        )
        self.assertTrue(includes_method_relu_)
        # Also verifies fix for https://github.com/pytorch/pytorch/issues/84570
        gm = make_fx(functionalize(symbolic_gm))(a)
        includes_aten_relu = any(
            n.target == torch.ops.aten.relu.default for n in gm.graph.nodes
        )
        self.assertTrue(includes_aten_relu)

    def test_make_fx_no_decompose(self, device):
        # FIXME
        return self.skipTest("error: maximum recursion reached")

        def f(x):
            return torch.tanh(x).sum()

        fx_f = make_fx(grad(f))(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}

        self.assertEqual(torch.ops.aten.tanh_backward in ops, True)

        fx_f = make_fx(grad(f), decomposition_table)(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}
        self.assertEqual(torch.ops.aten.tanh_backward in ops, False)

    def test_nnc_jit(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(3)
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_scalar(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(())
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_pytrees(self, device):
        def f(x):
            return [torch.sin(x[0])]

        jit_f = nnc_jit(f)

        inp = [torch.randn(3)]
        self.assertEqual(jit_f(inp), f(inp))

    def test_external_calls(self, device):
        def f(a, b):
            return torch.mv(a, b)

        jit_f = nnc_jit(f)
        inp = [torch.randn(3, 3), torch.randn(3)]
        self.assertEqual(jit_f(*inp), f(*inp))

    def test_nnc_passthrough(self, device):
        def f(x, y):
            return x + y, y

        inp = (torch.randn(3), torch.randn(3))
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

        def f(x):
            x["a"] = x["a"] * 2
            return x

        inp = ({"a": torch.randn(3), "b": torch.randn(3)},)
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self, device):
        mod = torchvision.models.resnet18()

        def f(x):
            out = mod(x)
            out.sum().backward()
            return [a.grad for a in mod.parameters()]

        inp = torch.randn(3, 3, 250, 250, requires_grad=True)
        grads = f(inp)

        mod.zero_grad()
        mod(inp).sum().backward()
        grads2 = [a.grad for a in mod.parameters()]
        self.assertEqual(grads, grads2)


def get_base(t):
    return t._base if t._is_view() else t


def is_in_base(t, maybe_tensors):
    t_base = get_base(t)
    for maybe_tensor in maybe_tensors:
        if isinstance(maybe_tensor, torch.Tensor):
            if t_base is get_base(maybe_tensor):
                return True
    return False


def skipIfDynamoInput(reason):
    """
    Skip TestAOTAutograd if running with dynamo input
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if isinstance(self, TestAOTAutogradWithDynamo):
                self.skipTest(
                    f"Skipping {self._testMethodName} in TestAOTAutogradWithDynamo because {reason}"
                )
            else:
                func(self, *args, **kwargs)

        return wrapper

    return decorator


class TestAOTAutograd(AOTTestCase):
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Optional[Callable]],
        decompositions: Optional[dict],
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        """
        Runs aot_autograd with the specified settings on f.
        """
        if isinstance(f, nn.Module):
            compiled_f = aot_module(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        else:
            compiled_f = aot_function(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        return compiled_f

    # test_mutation will:
    # - Ensure that inputs are non-leaves, so our graphs can mutate them
    # - try to mutate outputs of the graph (to ensure that autograd meta is set properly on outputs)
    @patch("functorch.compile.config.debug_assert", True)
    def verify_aot_autograd(
        self,
        f,
        inp_: Union[Callable, list[Any]],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: Optional[dict] = None,
        dynamic: bool = False,
        # Only active when inp_ is Callable.
        # TODO: probably consolidate all tests to make inp a Callable.
        make_inputs_subclasses: bool = False,
    ):
        def make_inputs(inp_):
            # Some tests pass in a callable for inp, to generate the inputs
            # (useful if we want to generate complicated aliasing inputs)
            if isinstance(inp_, Callable):
                inp_callable = inp_
                # The callable should return a tuple of f_inputs, f_graph_inputs
                # (The idea is that we might want to compile a function with the graph inputs,
                # but test autograd backprop all the way through the actual inputs)
                with TwoTensorMode() if make_inputs_subclasses else nullcontext():
                    inp, graph_inps = inp_callable()
            else:
                inp = []
                # Our input clones need to mimic when inputs are duplicates of one another
                dupes_map = {}
                for i, x in enumerate(inp_):
                    if x in dupes_map:
                        x_dupe_idx = dupes_map[x]
                        inp.append(inp[x_dupe_idx])
                    else:
                        dupes_map[x] = i
                        if not isinstance(x, torch.Tensor):
                            x_copy = x
                        else:
                            x_copy = x.detach().clone().requires_grad_(x.requires_grad)
                            if x.requires_grad and not x.is_leaf:
                                x_copy = x_copy.clone()

                        inp.append(x_copy)

                if test_mutation:
                    # For graphs where we mutate inputs, need our test to make sure inputs aren't leaves
                    graph_inps = [x.add(1) for x in inp]
                else:
                    graph_inps = inp

            return inp, graph_inps

        def check_results(
            ref_results,
            test_results,
            ref_graph_inps,
            test_graph_inps,
            ref_inp,
            test_inp,
        ):
            ref_out, ref_grad = ref_results
            test_out, test_grad = test_results
            self.assertEqual(ref_grad, test_grad)
            if isinstance(ref_out, torch.Tensor):
                self.assertTrue(isinstance(test_out, torch.Tensor))
                ref_out, test_out = [ref_out], [test_out]
            for ref_o, test_o in zip(ref_out, test_out):
                if isinstance(ref_o, torch.Tensor):
                    self.assertEqual(ref_o.requires_grad, test_o.requires_grad)
                    self.assertEqual(ref_o.is_leaf, test_o.is_leaf)
                    ref_is_view_of_non_interm = is_in_base(
                        ref_o, ref_graph_inps
                    ) or is_in_base(ref_o, ref_out)
                    test_is_view_of_non_interm = is_in_base(
                        test_o, test_graph_inps
                    ) or is_in_base(test_o, test_out)
                    self.assertEqual(
                        ref_is_view_of_non_interm, test_is_view_of_non_interm
                    )
                    self.assertEqual(ref_o, test_o)
                    if test_mutation:
                        # This tests that autograd meta is set properly on the output we can
                        # mutate it.
                        ref_o.add_(2)
                        test_o.add_(2)
                        self.assertEqual(ref_o, test_o)
                        # Reverse the modification
                        ref_o.sub_(2)
                        test_o.sub_(2)
                        self.assertEqual(ref_o, test_o)
            for ref_i, test_i in zip(ref_inp, test_inp):
                if isinstance(ref_i, torch.Tensor):
                    self.assertEqual(ref_i.requires_grad, test_i.requires_grad)
                self.assertEqual(ref_i, test_i)

        for keep_input_mutations in [True] if keep_inp_mutations else [True, False]:
            inp, graph_inps = make_inputs(inp_)
            test_inp, test_graph_inps = make_inputs(inp_)
            fw_graph_cell = [None]
            compiled_f = self.run_autograd(
                f, fw_graph_cell, decompositions, keep_input_mutations, dynamic
            )
            ref_results = outs_and_grads(f, graph_inps, inp)
            test_results = outs_and_grads(compiled_f, test_graph_inps, test_inp)

            check_results(
                ref_results, test_results, graph_inps, test_graph_inps, inp, test_inp
            )
            if isinstance(self, TestAOTAutogradWithCache):
                # When testing with cache, run compiled_f a second time
                cached_inp, cached_graph_inps = make_inputs(inp_)
                cached_results = outs_and_grads(
                    compiled_f, cached_graph_inps, cached_inp
                )
                check_results(
                    ref_results,
                    cached_results,
                    graph_inps,
                    cached_graph_inps,
                    inp,
                    cached_inp,
                )

        return fw_graph_cell[0]

    def test_non_tensor_and_none_inputs(self):
        # int, None, Tensor
        def f(a, b, c):
            return a * c

        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=True)]
        self.verify_aot_autograd(f, inp)
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=False)]
        self.verify_aot_autograd(f, inp)

    def test_single_output(self):
        def f(a, b):
            return a + b

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output(self):
        def f(a, b):
            return a + b, a - b

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output_list(self):
        def f(a, b):
            return [a + b, a - b]

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    # Test for bug occurring at the intersection of fake tensors & functionalization.
    def test_squeeze_mutation(self):
        def f(a):
            b = a.clone().squeeze(-1)
            b.add_(1.0)
            return a + b

        inp = [torch.randn(3, 1, requires_grad=True)]
        self.verify_aot_autograd(f, inp, dynamic=True)
        inp = [torch.randn(3, 1, requires_grad=False)]
        self.verify_aot_autograd(f, inp, dynamic=True)

    def test_complex_linear(self):
        # https://github.com/pytorch/pytorch/issues/93424
        inp = [torch.randn(1, 10, 10, dtype=torch.complex64)]

        class F(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 10, dtype=torch.complex64)

            def forward(self, x):
                return self.linear(x).sum().abs()

        self.verify_aot_autograd(F(), inp)

    def test_embedding_bag_view_dynamic(self):
        # Backwards pass tries to wrap a sparse tensor in a FunctionalTensorWrapper;
        # test that this works even though the sparse tensor has no storage.

        class F(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(100, 8, sparse=True)

            def forward(self, x, y):
                return self.emb(x, y).view(-1)

        x = torch.arange(3)
        y = torch.arange(3)
        self.verify_aot_autograd(F(), [x, y], dynamic=False)
        self.verify_aot_autograd(F(), [x, y], dynamic=True)

    def test_input_mutation_simple(self):
        def f(a):
            a.mul_(2)
            return a * 3

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        # Things to note:
        # - the extra clone is because we need to pass the pre-mutated input to grad(),
        #   but autograd operates above functionalization so we need to manually clone.
        #   Hopefully backends can optimize this easily.
        # - The extra return arg is because the compiled forward returns (mutated inputs + outputs)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, 3)
    return (mul, mul_1)""",
        )

    def test_input_mutation_set__input_mutation(self):
        def f(a):
            b = torch.arange(9, dtype=a.dtype).reshape(3, 3)
            with torch.no_grad():
                a.set_(b)
            return a * b

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)

    def test_set__steals_view_chain(self):
        def f(a, b):
            a_ = a.mul(2)
            b_ = b.mul(2)
            b_slice = b_[1].view(3, 3)
            # a_clone should inherit the view chain from b_slice
            a_.set_(b_slice)
            # Also mutates b_,
            a_.view(-1).mul_(2)
            return a_ * b_slice

        inp = [
            torch.ones(3, 3, requires_grad=False),
            torch.zeros(3, 9, requires_grad=False),
        ]
        self.verify_aot_autograd(f, inp, keep_inp_mutations=True)

    def _compile_autocast(self, device, *, forward_autocast):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x) -> Tensor")
            m.impl("foo", torch.clone, "CompositeExplicitAutograd")

            def autocast(x):
                return x + 1

            m.impl("foo", autocast, "AutocastCPU")
            m.impl("foo", autocast, "AutocastCUDA")

            foo = torch.ops.mylib.foo.default

            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return foo(x)

                @staticmethod
                def backward(ctx, grad):
                    (x,) = ctx.saved_tensors
                    return grad * foo(x)

            def fn(x):
                with torch.amp.autocast(device, enabled=False):
                    return Foo.apply(x)

            x = torch.tensor(0.0, device=device, requires_grad=True)
            if forward_autocast:
                with (
                    torch.amp.autocast(device),
                    torch._dynamo.config.patch(recompile_limit=999),
                ):
                    out = torch.compile(fn, fullgraph=True, backend="aot_eager")(x)
            else:
                with torch._dynamo.config.patch(recompile_limit=999):
                    out = torch.compile(fn, fullgraph=True, backend="aot_eager")(x)
            (grad,) = torch.autograd.grad(out, x)
            return out, grad

    @torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    def test_backward_pass_autocast_on(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            out, grad = self._compile_autocast(device, forward_autocast=True)
            self.assertEqual(out, torch.zeros_like(out))
            self.assertEqual(grad, torch.ones_like(grad))

    @torch._functorch.config.patch(backward_pass_autocast="off")
    def test_backward_pass_autocast_off(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            out, grad = self._compile_autocast(device, forward_autocast=True)
            self.assertEqual(out, torch.zeros_like(out))
            self.assertEqual(grad, torch.zeros_like(grad))

    @torch._functorch.config.patch(backward_pass_autocast="off")
    def test_backward_pass_autocast_custom(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            with torch._functorch.config.patch(
                backward_pass_autocast=[{"device_type": device}]
            ):
                out, grad = self._compile_autocast(device, forward_autocast=False)
                self.assertEqual(out, torch.zeros_like(out))
                self.assertEqual(grad, torch.ones_like(grad))

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__and_data_mutation_good(self):
        def f(a, b):
            # The data mutation happens *after* the set_(). This is ok (see the graph below)
            with torch.no_grad():
                a.set_(b)
                b.mul_(2)
            return a + b

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        inp = [
            torch.ones(3, 3, requires_grad=False),
            torch.zeros(3, 3, requires_grad=False),
        ]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # Important things to note:
        # - "return a.set_(b)" desugars into "return b"
        # - Both a and b are recorded as experiencing mutations,
        #   which is why we see "b_updated" (output of the mul) twice in the graph outputs.
        #   a is recorded as both a data mutation and a metadata mutation (due to set_ swapping its storage).
        # - the runtime epilogue for a is "a.set_(mul)"
        # - the runtime epilogue for b is "b.copy_(mul)"
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_2, 2)
    add = torch.ops.aten.add.Tensor(mul, mul)
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, mul);  primals_1 = set_ = None
    copy_ = torch.ops.aten.copy_.default(primals_2, mul);  primals_2 = mul = copy_ = None
    return (add,)""",
        )

    # This is a (hopefully) extremely rare case that is difficult to handle,
    # so we ban it.
    # https://github.com/pytorch/pytorch/issues/126236
    # https://github.com/pytorch/pytorch/pull/126113
    @xfailIfTorchDynamo
    def test_set__and_data_mutation_bad(self):
        def f(a):
            a_view = a.view(-1)
            tmp = torch.ones(3, 3, requires_grad=True)
            # Now, any mutations on either tmp
            # will be tracked as graph input mutations.
            with torch.no_grad():
                a.set_(tmp)
                # BAD: a_view is now detached from every graph input,
                # so we won't recognize that this caused an input mutation!
                a_view.mul_(2)
            return a + tmp

        inp = [torch.ones(3, 3, requires_grad=True)]
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__not_allowed(self):
        def f(a, b):
            with torch.no_grad():
                a.set_(b)
            # Mutating a will change a's grad_fn, which requires us to replay the mutation outside of the graph.
            # We currently ban this today, when the input also received a set_() input mutation.
            a.mul_(2)
            return a + b

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        with self.assertRaisesRegex(
            AssertionError, "but the input has other mutations that we cannot"
        ):
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    def test_input_mutation_set__nop(self):
        def f(a):
            b = torch.arange(9, dtype=a.dtype)
            a_old = torch.ops.aten.alias.default(a)
            with torch.no_grad():
                a.set_(b)
                a.set_(a_old)
            return a + b.reshape(3, 3)

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # Things to note:
        # - There are no set_() calls in the graph (we functionalize a.set_(b) into "b")
        # - There is only **1** graph output. We properly realized that the two set_() calls
        #   undo each other, and so effectively no inputs are mutated.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    arange = torch.ops.aten.arange.default(9, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    alias = torch.ops.aten.alias.default(primals_1);  primals_1 = None
    view = torch.ops.aten.view.default(arange, [3, 3]);  arange = None
    add = torch.ops.aten.add.Tensor(alias, view);  alias = view = None
    return (add,)""",
        )

    def test_input_mutation_simple_with_none_and_nontensor(self):
        # Tensor, None, int
        def f(a, b, c):
            return a * c

        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = [torch.ones(3, 3, requires_grad=req_grad), None, 3]
            out_ref = f(*inp)
            out_test = f_compiled(*inp)
            self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/93363
    def test_mutates_input_noncontiguous(self):
        def f(a):
            a.add_(1)
            return ()

        f_compiled = aot_function(f, nop)
        ref = torch.ones(4, requires_grad=True) + 0
        ref_view = ref[0::2]

        test = torch.ones(4, requires_grad=True) + 0
        test_view = test[0::2]

        out_ref = f(ref_view)  # noqa: F841
        out_test = f_compiled(test_view)  # noqa: F841
        self.assertEqual(ref, test)

    def test_input_mutation_modifies_autograd_meta_of_aliases(self):
        def f(a):
            a.mul_(2)
            out = a + 1
            return out.detach()

        x_ref = torch.ones(3, 3, requires_grad=True).clone()
        x_ref_view = x_ref.view(3, 3)

        x_test = torch.ones(3, 3, requires_grad=True).clone()
        x_test_view = x_test.view(3, 3)

        f_compiled = aot_function(f, nop, keep_inference_input_mutations=True)
        f(x_ref)
        f_compiled(x_test)
        # f will mutate aliases of the input, including its autograd metadata!
        # y.grad_fn is AsStridedBackward
        self.assertEqual(x_ref_view, x_test_view)
        self.assertEqual(x_ref_view._version, x_test_view._version)
        self.assertEqual(x_ref_view.grad_fn.__class__, x_test_view.grad_fn.__class__)
        # Test the actual gradients are correct
        (x_ref * x_ref_view).sum().backward()
        (x_test * x_test_view).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(x_ref_view.grad, x_test_view.grad)

    def test_nested_subclasses(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)
        aaaa = TwoTensor(aa, aa2)
        out = f(aaaa)
        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, TwoTensor))
        self.assertTrue(isinstance(out.b, TwoTensor))
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        out.sum().backward()
        self.assertTrue(isinstance(aaaa.grad, TwoTensor))
        self.assertTrue(isinstance(aaaa.grad.a, TwoTensor))
        self.assertTrue(isinstance(aaaa.grad.b, TwoTensor))

    def test_nested_subclasses_non_nested_grad(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        new_aa = TwoTensor(a3, a4)
        aa = TwoTensor(a, a2)

        aa2 = aa.detach().clone().requires_grad_()
        aaaa = TwoTensor(aa, aa2)
        out = f(new_aa)
        new_out = out + aaaa
        with self.assertRaisesRegex(
            RuntimeError,
            """
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
""",  # noqa: F541
        ):
            new_out.sum().backward()

    def test_nested_subclasses_non_homogenous(self):
        def f(x):
            x_elem = x.elem
            x_metadata = x.constant_attribute
            return x_metadata * x_elem * x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)
        custom_aa = ConstantExtraMetadataTensor(aa)
        custom_aa.constant_attribute = 6
        custom_aa2 = ConstantExtraMetadataTensor(aa2)
        custom_aa2.constant_attribute = 6

        out_eager = f(custom_aa)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_aa2)

        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.b, ConstantExtraMetadataTensor))
        self.assertTrue(torch.allclose(out_eager, out))

        out_eager.sum().backward()
        out.sum().backward()

        self.assertTrue(torch.allclose(custom_aa.grad, custom_aa2.grad))
        self.assertTrue(isinstance(custom_aa2.grad, TwoTensor))
        self.assertTrue(isinstance(custom_aa2.grad.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(custom_aa2.grad.b, ConstantExtraMetadataTensor))

    def test_subclasses_mixed(self):
        def f(x, y):
            x_metadata = x.constant_attribute
            out_a = x_metadata * x * y.a
            out_b = x * y.a * y.b
            return TwoTensor(out_a, out_b)

        a = torch.ones(4, requires_grad=False)
        a2 = a.clone()
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 5
        custom_a2 = ConstantExtraMetadataTensor(a2)
        custom_a2.constant_attribute = 5

        b = torch.ones(4, requires_grad=False)
        b2 = b.clone()
        b3 = b.clone()
        b4 = b.clone()
        bb = TwoTensor(b, b2)
        bb2 = TwoTensor(b3, b4)

        out_eager = f(custom_a, bb)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_a2, bb2)

        self.assertTrue(torch.allclose(out_eager, out))
        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.b, ConstantExtraMetadataTensor))

    def test_subclasses_mixed_mode(self):
        def f(x):
            return x.sin().cos()

        class AddConstantMetadataMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if ConstantExtraMetadataTensor not in types:
                    out = ConstantExtraMetadataTensor(out)
                    out.constant_attribute = 5
                return out

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)

        with AddConstantMetadataMode():
            out_eager = f(aa)

        compiled_f = torch.compile(f, backend="aot_eager")

        with AddConstantMetadataMode():
            out = compiled_f(aa2)

        self.assertTrue(isinstance(out, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.elem, TwoTensor))
        self.assertTrue(torch.allclose(out_eager, out))

        out_eager.sum().backward()
        out.sum().backward()

        self.assertTrue(torch.allclose(aa.grad, aa2.grad))
        self.assertTrue(isinstance(aa2.grad, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(aa2.grad.elem, TwoTensor))

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    def test_custom_tensor_metadata(self):
        def f(x):
            x_elem = x.elem
            x_elem_elem = x_elem.elem
            x_elem_metadata = x_elem.constant_attribute
            return x * x_elem * x_elem_elem * x_elem_metadata

        a = torch.ones(4, requires_grad=True)
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 6
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        custom_aa.constant_attribute = 4

        custom_aa_compile = custom_aa.detach().clone().requires_grad_()
        custom_aa_compile.elem.constant_attribute = 6
        out_eager = f(custom_aa)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_aa_compile)

        self.assertTrue(torch.allclose(out_eager, out))

        out.sum().backward()

        self.assertTrue(isinstance(custom_aa_compile.grad, ConstantExtraMetadataTensor))
        self.assertTrue(
            isinstance(custom_aa_compile.grad.elem, ConstantExtraMetadataTensor)
        )

    def test_nested_subclasses_complicated_inps(self):
        def f(x, y, z):
            temp = x + y
            temp_plain = x.a + y.b
            res = temp.sum() + temp_plain.sum()
            return x.sin().cos() + res

        x = torch.ones(4, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        xx = TwoTensor(x, x2)
        xx2 = xx.detach().clone().requires_grad_()

        x_nested = TwoTensor(xx, xx2)
        x_nested_compile = x_nested.detach().clone().requires_grad_()

        y_nested = x_nested.detach().clone().requires_grad_()
        y_nested_compile = y_nested.detach().clone().requires_grad_()

        z = x.detach().clone().requires_grad_()
        z_compile = z.detach().clone().requires_grad_()

        out_eager = f(x_nested, y_nested, z)
        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(x_nested_compile, y_nested_compile, z_compile)
        self.assertTrue(torch.allclose(out_eager, out))

        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, TwoTensor))
        self.assertTrue(isinstance(out.b, TwoTensor))
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        out.sum().backward()
        out_eager.sum().backward()

        self.assertTrue(isinstance(x_nested_compile.grad, TwoTensor))
        self.assertTrue(isinstance(x_nested_compile.grad.a, TwoTensor))
        self.assertTrue(isinstance(x_nested_compile.grad.b, TwoTensor))

        self.assertTrue(isinstance(y_nested_compile.grad, TwoTensor))
        self.assertTrue(isinstance(y_nested_compile.grad.a, TwoTensor))
        self.assertTrue(isinstance(y_nested_compile.grad.b, TwoTensor))

        self.assertTrue(torch.allclose(x_nested_compile.grad.a.a, x_nested.grad.a.a))
        self.assertTrue(torch.allclose(x_nested_compile.grad.a.b, x_nested.grad.a.b))
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.a, y_nested.grad.a.a))
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.b, y_nested.grad.a.b))

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    def test_nested_subclasses_complicated_inps_mixed(self):
        def f(x, y):
            y_elem = y.elem
            y_elem_elem = y_elem.elem
            y_elem_metadata = y_elem.constant_attribute
            return y * y_elem * y_elem_elem * y_elem_metadata + x

        x = torch.ones(4, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        xx = TwoTensor(x, x2)
        xx2 = xx.detach().clone().requires_grad_()

        x_nested = TwoTensor(xx, xx2)
        x_nested_compile = x_nested.detach().clone().requires_grad_()

        a = torch.ones(4, requires_grad=True)
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 6
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        custom_aa.constant_attribute = 4

        custom_aa_compile = custom_aa.detach().clone().requires_grad_()
        custom_aa_compile.constant_attribute = 4
        custom_aa_compile.elem.constant_attribute = 6

        compiled_f = torch.compile(f, backend="aot_eager")
        out_eager = f(x_nested, custom_aa)
        out = compiled_f(x_nested_compile, custom_aa_compile)
        self.assertTrue(torch.allclose(out_eager, out))

        out.sum().backward()
        out_eager.sum().backward()

        self.assertTrue(torch.allclose(x_nested_compile.grad, x_nested.grad))
        self.assertTrue(torch.allclose(custom_aa_compile.grad, custom_aa.grad))

    def test_composite_impl_compile(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, a):
                return self.linear(a)

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    t = torch.ops.aten.t.default(primals_1);  primals_1 = None
    addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = None
    return (addmm, primals_3, t)""",
        )

        with torch.inference_mode():
            fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
            inp = [torch.ones(3, 3, requires_grad=False)]
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = arg2_1 = t = None
    return (addmm,)""",
            )

    def test_outputs_are_aliased(self):
        # Tensor, None, int
        def f(a):
            b = a.mul(2)
            c = b.view(-1)
            return b, c

        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = torch.ones(3, requires_grad=req_grad)
            out_ref = f(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            # Try mutating one of the outputs, which is aliased.
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            # Assert that the aliasing relationship was preserved
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_input_mutation_is_output(self):
        def f(a):
            a.mul_(2)
            return a

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    return (mul, mul)""",
        )

    def test_input_mutation_multiple(self):
        def f(a, b, c):
            a.mul_(2)
            c.mul_(2)
            return a + b + c

        def create_inp(req_grad):
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None
    add = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
    add_1 = torch.ops.aten.add.Tensor(add, mul_1);  add = None
    return (mul, mul_1, add_1)""",
        )

    def test_input_mutation_return(self):
        def f(a, b):
            return torch.sin(a, out=b)

        inp = [torch.randn(3, 3), torch.ones(3, 3)]

        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, sin);  arg1_1 = sin = None
    return (copy_,)""",
        )

    def test_input_mutation_metadata(self):
        def f(a, b):
            a.transpose_(1, 0)
            return a + b

        def create_inp(req_grad):
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

    @parametrize("backend", ["aot_eager", "inductor"])
    @parametrize("view_replay_for_aliased_outputs", [False, True])
    @parametrize("dynamic_shapes", [False, True])
    def test_alias_of_intermediate_detach(
        self, backend, view_replay_for_aliased_outputs, dynamic_shapes
    ):
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            view_replay_for_aliased_outputs,
        ):

            def fn(x):
                x = x + 1
                a = x.transpose(0, 1)
                return a.detach(), a

            def inp_fn():
                t = torch.ones(3, 3, requires_grad=True)
                if dynamic_shapes:
                    torch._dynamo.mark_dynamic(t, 0)
                    torch._dynamo.mark_dynamic(t, 1)
                return t

            x_ref = inp_fn()
            y_ref = fn(x_ref)

            x = inp_fn()
            y = torch.compile(fn, backend=backend, fullgraph=True)(x)
            self.assertEqual(y_ref, y)
            y0, y1 = y
            self.assertFalse(y0.requires_grad)
            self.assertTrue(y1.requires_grad)
            # Check that detach and diff view points to the same intermediate tensor storage
            self.assertEqual(y0.data_ptr(), y1.data_ptr())
            self.assertTrue(y1._is_view())

            sum(y_ref).sum().backward()
            sum(y).sum().backward()
            self.assertEqual(x_ref.grad, x.grad)

    def test_input_mutation_storage_resize_up(self):
        def f(a):
            torch.ops.inductor.resize_storage_bytes_(a, 32)
            # float32, 4 bytes per element, 32 bytes == 8 elements
            with torch.no_grad():
                a.copy_(torch.ones(8))
            return a + 1

        inp = torch.zeros(8, requires_grad=True)
        # Input starts with zero-size-storage
        inp.untyped_storage().resize_(0)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)
        # Final functionalized graph has two mutation ops:
        # (1) a resize_() to resize input tensor up
        # (2) a copy_() to fill in the resized input with valid data
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1):
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 32);  resize_storage_bytes_ = None
    ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    copy = torch.ops.aten.copy.default(primals_1, ones);  ones = None
    add = torch.ops.aten.add.Tensor(copy, 1)
    copy_ = torch.ops.aten.copy_.default(primals_1, copy);  primals_1 = copy = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_storage_resize_down(self):
        def f(a):
            out = a.sin()
            torch.ops.inductor.resize_storage_bytes_(a, 0)
            return out

        inp = torch.zeros(8, requires_grad=True)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)
        # Final functionalized graph has one mutation ops:
        # (1) a resize_() to resize input tensor down
        # Even though there was technically a "data mutation" on the input (from a.copy_()),
        # We don't include it in the graph since the final input size has zero storage
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1):
    sin = torch.ops.aten.sin.default(primals_1)
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 0);  resize_storage_bytes_ = None
    return (sin, primals_1)""",
        )

    #     def test_input_mutation_storage_resize_up_down(self):
    #         def f(a):
    #             torch.ops.inductor.resize_storage_bytes_(a, 32)
    #             # float32, 4 bytes per element, 32 bytes == 8 elements
    #             with torch.no_grad():
    #                 a.copy_(torch.ones(8))
    #             out = a.sin()
    #             torch.ops.inductor.resize_storage_bytes_(a, 0)
    #             return out

    #         inp = torch.zeros(8, requires_grad=True)
    #         # Input starts with zero-size-storage
    #         inp.untyped_storage().resize_(0)

    #         fw_graph_cell = [None]
    #         compiled_f = aot_function(
    #             f,
    #             fw_compiler=make_boxed_compiler(
    #                 partial(extract_graph, graph_cell=fw_graph_cell)
    #             ),
    #             bw_compiler=nop,
    #             decompositions={},
    #             keep_inference_input_mutations=True,
    #             dynamic=False,
    #         )
    #         out = compiled_f(inp)
    #         # Final graph has two interesting properties:
    #         # (1) no resizes in the functional graph, since the two resizes cancel out
    #         #     and the final size is zero
    #         # (2) no copy_ in the functional graph, even though we copied data into the input,
    #         #     because the input has no storage at the end of graph execution (so no data to copy)
    #         self.assertExpectedInline(
    #             fw_graph_cell[0].code.strip(),
    #             """\
    # def forward(self, primals_1):
    #     ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    #     copy = torch.ops.aten.copy.default(primals_1, ones);  primals_1 = ones = None
    #     sin = torch.ops.aten.sin.default(copy)
    #     return [sin, copy]""",
    #         )

    # skipped after confirming with @yf225 and @bdhirsh
    @unittest.skipIf(
        True,
        "using set_ unsafely and PT2 FSDP2 no longer uses set_ as used in this test",
    )
    def test_input_mutation_storage_resize_down_and_set_(self):
        # Meant to mimic ppFSDP
        class TracableCreateParameter(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, placeholder):
                assert not tensor.requires_grad
                return placeholder.set_(tensor)

            @staticmethod
            def backward(ctx, grad):
                return None, grad  # grad flows to placeholder

        def f(dummy_param, param_shard):
            # simulate allgather
            with torch.no_grad():
                allgather_param = torch.cat([param_shard, param_shard])
            # simulate propagating grad state through dummy param, using data of allgather param
            dummy_param_with_grad_state = TracableCreateParameter.apply(  # noqa: F841
                allgather_param, dummy_param
            )
            out = dummy_param.sin()
            # Resize out dummy param, which now has the allgather data
            torch.ops.inductor.resize_storage_bytes_(dummy_param, 0)
            return out

        # Simulates the local shard of our param
        param_shard = torch.zeros(8, requires_grad=True)
        # The dummy, zero-sized allgathered param that autograd will actually compute gradients on
        dummy_param = torch.zeros(16, requires_grad=True)
        dummy_param.untyped_storage().resize_(0)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(dummy_param, param_shard)
        # Important stuff to point out:
        # (1) We save cat for backward (input to the sin()).
        #     While the original code was dummy_param.sin(),
        #     dummy_param actually contains the `cat` tensor due to the set_() call
        # (2) We emit a cat.resize_storage_(0) in the graph.
        #     After the set_(), cat is the actually data of dummy_param, which is what we call resize_() on
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    cat = torch.ops.aten.cat.default([primals_2, primals_2]);  primals_2 = None
    sin = torch.ops.aten.sin.default(cat)
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(cat, 0);  resize_storage_bytes_ = None
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, cat);  primals_1 = set_ = None
    return (sin, cat)""",
        )

    def test_input_mutation_storage_resize_before_set_(self):
        def f(a):
            with torch.no_grad():
                torch.ops.inductor.resize_storage_bytes_(a, 0)
                a.set_(torch.ones(2))

        inp = torch.zeros(8, requires_grad=True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)

    # def test_input_mutation_storage_resize_not_supported(self):
    #     def f(a):
    #         a.mul_(2)
    #         torch.ops.inductor.resize_storage_bytes_(a, 0)
    #         return a

    #     inp = torch.zeros(8, requires_grad=True)

    #     with self.assertRaisesRegex(
    #         AssertionError, "the input has other mutations that we cannot"
    #     ):
    #         compiled_f = aot_function(
    #             f,
    #             fw_compiler=nop,
    #             bw_compiler=nop,
    #             decompositions={},
    #             keep_inference_input_mutations=True,
    #             dynamic=False,
    #         )
    #         out = compiled_f(inp)

    def test_input_output_aliase_custom_autograd_function(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gx):
                return gx * 0.5

        def f(x):
            return Foo.apply(x)

        inp = [torch.ones(2, 2, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)

    def test_input_mutation_requires_grad_detach(self):
        # Here, "a" requires grad, and gets mutated, so we append a copy_() to the end of the graph.
        # Its mutation doesn't take part in autograd though, because we mutated a detach'd view.
        # Need to make sure that this copy_() doesn't error, and doesn't participate in autograd either.
        def f(a):
            a.detach().mul_(2)
            return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)
        inp = [torch.ones(4, requires_grad=True)]
        # test_mutation=True will first do some compute on inp, so it is no longer an autograd leaf
        # by the time it becomes a graph input. Good to test both cases.
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_hidden_from_autograd_aliasing(self):
        def f(a):
            a_alias = a.view(-1)
            with torch.no_grad():
                a_alias.mul_(2)
            return a + 1

        inp = [torch.ones(4, requires_grad=True)]
        # The important bit: we detected that the input mutation is safe
        # to include **inside** the graph, since it was under no_grad
        # (so all we need to do is use mark_dirty() on the input to bump the VC)
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    view = torch.ops.aten.view.default(primals_1, [-1])
    mul = torch.ops.aten.mul.Tensor(view, 2);  view = None
    view_1 = torch.ops.aten.view.default(mul, [4]);  mul = None
    add = torch.ops.aten.add.Tensor(view_1, 1)
    copy_ = torch.ops.aten.copy_.default(primals_1, view_1);  primals_1 = view_1 = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad(self):
        def f(a):
            with torch.no_grad():
                a.mul_(2)
            return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        # Even though the input requires_grad, we expect the keep the input mutation in the graph
        # (Even though this is a training graph!)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 3)
    copy_ = torch.ops.aten.copy_.default(primals_1, mul);  primals_1 = mul = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad_inference_graph(self):
        def f(a):
            with torch.no_grad():
                a.mul_(2)
                return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        # Even though the input requires_grad, we expect the keep the input mutation in the graph
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 3)
    copy_ = torch.ops.aten.copy_.default(arg0_1, mul);  arg0_1 = mul = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad_detach_mixed(self):
        # Perform a mix of mutations on a:
        # 1 normal, 1 in no_grad, 1 on a detach'd tensor.
        # Only the first should participate in gradient computation.
        def f(a):
            a.detach().mul_(2)
            a.mul_(3)
            with torch.no_grad():
                a.mul_(4)
            return a + 5

        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_metadata2(self):
        def f(a):
            a.transpose_(1, 0)
            a.mul_(2)
            return a + 1

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_batchnorm(self):
        def f(inpt, weight, bias, running_mean, running_var):
            # This is additionally a good test, because the input tensors that we mutate
            # are *also* saved for backwards.
            # This tests that what we save for the backward is actually cloned inputs,
            # and not the original inputs that got mutated.
            return torch._native_batch_norm_legit(
                inpt, weight, bias, running_mean, running_var, True, 0.5, 1e-5
            )

        def create_inp(req_grad):
            return [
                torch.ones(2, 5, 5, 5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5),
                torch.ones(5),
            ]

        from torch._decomp import get_decompositions

        # This simulates what inductor does (running the fw + bw decompositions)
        decompositions = get_decompositions(
            [
                torch.ops.aten._native_batch_norm_legit_functional,
                torch.ops.aten.native_batch_norm_backward,
            ]
        )
        self.verify_aot_autograd(
            f, create_inp(True), test_mutation=True, decompositions=decompositions
        )
        self.verify_aot_autograd(
            f, create_inp(False), test_mutation=True, decompositions=decompositions
        )

    def test_batchnorm_inference(self):
        inp = [
            torch.ones(2, 5, 5, 5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5),
            torch.ones(5),
        ]

        m = torch.nn.BatchNorm2d(4, 4)
        m.eval()
        fw_graph_cell = [None]
        inp = torch.ones(4, 4, 4, 4)
        fw_graph_cell = [None]
        compiled_m = aot_module(
            m,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp = torch.ones(4, 4, 4, 4)
        with torch.no_grad():
            compiled_m(inp)
        # expectation: there are no copy_() calls in the decomposed batch norm when running under training=False (eval mode)
        code = fw_graph_cell[0].code.strip()
        self.assertTrue("copy_" not in str(code))

    def test_input_output_view_simple(self):
        def f(a):
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # Outputs that alias inputs are pulled out of the graph entirely, so we don't compile anything here
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
    return (view,)""",
        )

    def test_input_output_view_mutate_multiple(self):
        def f(a, b, c):
            a.mul_(2)
            c.mul_(3)
            return b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # The original function returned two outputs, both of which aliased inputs.
        # We expect two outputs in the functional graph, a_updated and c_updated.
        # The actual aliased outputs themselves aren't in the compiled forward graph;
        # Instead, they're generated outside of  the graph.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 3);  clone_1 = None
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    view_2 = torch.ops.aten.view.default(mul_1, [2, 2])
    return (mul, mul_1, view, view_2)""",
        )

    def test_input_output_view_metadata_mutate_multiple(self):
        def f(a, b, c):
            b.mul_(3)
            c.t_()
            return a.view(2, 2), b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # Important thing to check here: of the three inputs:
        # Only the b.mul_(3) should show up in the graph (we functionalize it and return it).
        # Everything else that does not show up in the graph includes:
        # - The metadata mutation on c (we do it outside the graph)
        # - All 3 original fw outputs, which are aliases of inputs (we regenerate them outside of the graph)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_2);  primals_2 = None
    view = torch.ops.aten.view.default(primals_3, [2, 2]);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 3);  clone = None
    t = torch.ops.aten.t.default(view);  view = None
    view_1 = torch.ops.aten.view.default(primals_1, [2, 2]);  primals_1 = None
    view_3 = torch.ops.aten.view.default(t, [2, 2])
    view_4 = torch.ops.aten.view.default(mul, [2, 2])
    return (mul, t, view_1, view_4, view_3)""",
        )

    def test_input_mutation_and_output_view(self):
        def f(a):
            a.add_(1)
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # Here, total # of outputs is 1 because:
        # - num_mutated_inps = 1 (a_updated)
        # - num_fw_outputs = 0 (the output is an alias of the input, so we move it outside the compiled fw)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    view_1 = torch.ops.aten.view.default(add, [-1])
    return (add, view_1)""",
        )

    def test_input_mutation_output_view_multiple(self):
        def f(a, b, c, d):
            b.transpose_(1, 0)
            c.add_(1)
            return d + 1, b.diagonal(), a + c

        def create_inp(req_grad):
            return [
                torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
                .view(2, 2)
                .add(1),
                torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
                .view(2, 2)
                .add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(primals_4, 1);  primals_4 = None
    diagonal = torch.ops.aten.diagonal.default(transpose)
    add_2 = torch.ops.aten.add.Tensor(primals_1, add);  primals_1 = None
    return (transpose, add, add_1, diagonal, add_2)""",
        )

    def test_output_aliases_intermediate_single(self):
        def f(a):
            out = torch.mul(a, 3)
            return out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # In AOTAutograd, we are obligated to make the compiled forward directly return `out`,
        # and reconstruct `out.view(-1)` as a fresh output.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    return (view,)""",
        )

    def test_output_aliases_input_multi_output_view_should_raise_autograd_error(self):
        def f1(a):
            return list(a.unbind(0))

        f1_compiled = aot_function(f1, nop)

        inp1 = torch.ones(3, 3, requires_grad=True).clone()
        inp2 = torch.ones(3, 3, requires_grad=True).clone()
        inp3 = torch.ones(3, 3, requires_grad=True).clone()

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            out_test1 = f1_compiled(inp1)
            # This raises a runtime error from autograd in eager mode
            out_test1[0].mul_(2)

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            out_test2 = f1_compiled(inp2)
            inp2.mul_(2)
            # In eager mode, if we mutate a tensor, any multi-output-view aliases
            # get their grad_fn replaced with error nodes, so accessing grad_fn should error
            out_test2[0].grad_fn

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            f1_compiled(inp3)
            out_test1[0].detach().mul_(2)
            # The above case also applies to detached aliases (they turn the multi-output-view
            # alias's grad_fns into error nodes)
            out_test2[0].grad_fn

    def test_output_aliases_input_multi_output_view(self):
        # All aliased outs are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f1(a):
            return list(a.unbind(0))

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)

        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # Several of the outputs are from multi-output views.
        # However: they are part of the same alias set as "a", and "a.view(out.shape)",
        # which are both user-visible.
        # AOTAutograd will not try to be smart here and hide the aliasing relationships from autograd.
        # Instead, it will perform its "output aliases input" logic, and regenerate all aliases.
        def f3(a):
            return *list(a.unbind(0)), a.view(a.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)

        inp_ref_clone = inp_ref.clone()
        inp_clone = inp.clone()
        out_ref = f3(inp_ref_clone)
        out_test = f3_compiled(inp_clone)
        self.assertTrue(all("UnbindBackward" in str(o.grad_fn) for o in out_test[:3]))

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        # Also mutate the input, which should affect the aliased output.
        inp_ref_clone.view(-1).mul_(3)
        inp_clone.view(-1).mul_(3)
        # Do backward
        (inp_ref + out_ref[-1]).sum().backward()
        (inp + out_test[-1]).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_multi_output_view(self):
        # All aliased outs are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f1(a):
            out = torch.mul(a, 3)
            return list(out.unbind(0))

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)

        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # All aliased outs but one are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f2(a):
            out = torch.mul(a, 3)
            return *list(out.unbind(0)), out

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f2_compiled = aot_function(f2, nop)

        out_ref = f2(inp_ref)
        out_test = f2_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # All aliased outs but one are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f3(a):
            out = torch.mul(a, 3)
            return *list(out.unbind(0)), out.view(out.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)

        out_ref = f3(inp_ref)
        out_test = f3_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # There are 5 outputs that all alias each other.
        # 3 of them come from multi-output views, but the other 3 are "ordinary" aliases.
        # Therefore, AOTAutograd will not attempt the multi-output-view optimization,
        # and apply the intermediate_base logic to all aliases.
        # (In theory we could probably get AOTAutograd to only apply the intermediate base
        # logic to the last 2 outputs and not the first 3. We should probably
        # just do the graph partitioning defined in this doc instead though).
        # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
        def f4(a):
            out = torch.mul(a, 3)
            # also return the graph intermediate directly,
            # which will force AOTAutograd to do the "intermediate base" logic.
            # (Why? The user can mutate "out", which should change the autograd metadata
            #  of the other aliased outputs)
            return *list(out.unbind(0)), out, out.view(out.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f4_compiled = aot_function(f4, nop)

        out_ref = f4(inp_ref)
        out_test = f4_compiled(inp)
        # Mutate the last output of f4 (autograd will allow this, since it is not a multi-output view,
        # as long as *only* the non-multi-output views participate in the backward)
        # Note: We could probably try to hide **only** the multi-output views from autograd here
        # and only do the intermediate base logic for the last two aliases.
        # Longer term solution of graph partitioning is probably cleaner though (see the note).
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)

        out_ref_sum = out_ref[-1] + out_ref[-2]
        out_test_sum = out_test[-1] + out_test[-2]
        out_ref_sum.sum().backward()
        out_test_sum.sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_mutation_linear(self):
        def f(x):
            return (x + 1).view(-1)

        inp = [torch.ones(3, 3, requires_grad=True)]
        # use inductor's decomps (which will e.g. turn _unsafe_view() into view())
        from torch._inductor.decomposition import decompositions

        f_compiled = aot_function(f, nop, decompositions=decompositions)

        out_ref = f(*inp)
        out_test = f_compiled(*inp)

        out_ref.mul_(2)
        out_test.mul_(2)
        self.assertEqual(out_ref, out_test)

    def test_output_aliases_intermediate_no_grad(self):
        def f(a, b):
            out = torch.mul(a, 3)
            # First output is an alias of an intermediate that doesn't require grad
            return out.view(-1), b.add(1)

        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # important bit: we don't bother generating an intermediate base as an output in the graph,
        # because the intermediate base itself didn't require gradients.
        # (the only problematic case is when both the base and the aliasesed output require gradients).
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return (view, add)""",
        )

    def test_output_aliases_intermediate_returned_multiple_times(self):
        def f(a):
            out = torch.mul(a, 3)
            out_view = out.view(-1)
            return out, out_view, out

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate these two output views in the epilogue.
            return out.view(-1), out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    view_1 = torch.ops.aten.view.default(mul, [-1])
    return (view, view_1, mul)""",
        )

    def test_output_aliases_intermediate_and_returned(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out.view(-1), out

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    return (view, mul)""",
        )

    def test_output_aliases_intermediate_and_returned_flipped(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out, out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    return (mul, view)""",
        )

    def test_output_aliases_intermediate_and_returned_different_grad(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out.view(-1), out, out[0].detach()

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    select = torch.ops.aten.select.int(mul, 0, 0)
    detach = torch.ops.aten.detach.default(select);  select = None
    return (view, mul, detach)""",
        )

    def test_output_aliases_intermediate_inplace_view(self):
        def f(a):
            out = torch.mul(a, 3)
            out.t_()
            return out

        # TODO: fix this test.
        # See https://github.com/pytorch/pytorch/issues/90507
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_inplace_view_with_detach(self):
        def f(a):
            out = torch.mul(a, 3)
            out.t_()
            out.detach_()
            # Thanks to the detach_() AOT Autograd doesn't need to do anything.
            # `out` will show up as having OutputType.non_alias,
            # and ._is_view() == False
            return out, a + 1

        inp = [torch.ones(2, 4, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    t = torch.ops.aten.t.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(primals_1, 1);  primals_1 = None
    return (t, add)""",
        )

    def test_output_aliases_intermediate_inplace_view_and_view(self):
        def f(a):
            out = torch.mul(a, 3)
            out_view = out.unsqueeze(0)
            out.t_()
            out_view2 = out.unsqueeze(0)
            return out_view, out, out_view2

        inp = [torch.ones(2, 4, requires_grad=True)]  # noqa: F841

        # TODO: fix this test.
        # See <github issue link>
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple_mixed(self):
        def f(a):
            out1 = torch.mul(a, 3)
            out2 = torch.mul(a, 4)
            # AOTAutograd should manually generate these two output views in the epilogue.
            return out1.view(-1), out2.transpose(1, 0), out1.transpose(1, 0)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    mul_1 = torch.ops.aten.mul.Tensor(primals_1, 4);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    transpose = torch.ops.aten.transpose.int(mul_1, 1, 0);  mul_1 = None
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    return (view, transpose, transpose_1, mul)""",
        )

    def test_output_all_alias_types(self):
        # There are 3 types of aliasing that require us to return metadata in the compiled fw:
        # (1) outputs that are views of inputs
        # (2) outputs that are views of intermediates
        # (3) inputs that get metadata mutations
        # test all 3 of them here
        def f(a):
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.squeeze(), tmp.transpose(1, 0), a.unsqueeze(0)

        def inp_callable(req_grad):
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # TODO: make this test run with dynamic shapes so it is more meaningful
        # metadata output order: (a_updated_meta, out1_meta, out2_meta, out3_meta)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    view = torch.ops.aten.view.default(primals_1, [1, 2, 4]);  primals_1 = None
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    mul = torch.ops.aten.mul.Tensor(transpose, 2)
    squeeze = torch.ops.aten.squeeze.default(mul)
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0)
    return (transpose, squeeze, transpose_1, unsqueeze, mul)""",
        )

    @parametrize("req_grad", [False, True])
    def test_subclass_metadata_mutation(self, req_grad):
        def f(a):
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.transpose(1, 0)

        def inp_callable(req_grad):
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]

        # See https://github.com/pytorch/pytorch/issues/114975
        with self.assertRaisesRegex(
            RuntimeError,
            "Metadata mutations are currently not allowed on tensor subclasses",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=req_grad),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

    def test_input_data_and_metadata_mutation(self):
        def f(a):
            a.t_()
            a[0].mul_(2)
            return a.view(a.shape)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    t = torch.ops.aten.t.default(clone)
    select = torch.ops.aten.select.int(t, 0, 0);  t = None
    mul = torch.ops.aten.mul.Tensor(select, 2);  select = None
    t_1 = torch.ops.aten.t.default(clone);  clone = None
    select_scatter = torch.ops.aten.select_scatter.default(t_1, mul, 0, 0);  t_1 = mul = None
    t_2 = torch.ops.aten.t.default(select_scatter);  select_scatter = None
    t_4 = torch.ops.aten.t.default(t_2)
    t_6 = torch.ops.aten.t.default(t_2);  t_2 = None
    view_1 = torch.ops.aten.view.default(t_6, [3, 3]);  t_6 = None
    return (t_4, view_1)""",
        )

    def test_view_and_inplace_view(self):
        def f(a, b):
            a.t_()
            return b.view(b.shape), a.view(a.shape)

        def create_inp(req_grad):
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [3, 3]);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(t, [3, 3])
    return (t, view, view_1)""",
        )

    def test_view_detach(self):
        def f(a):
            tmp = a.detach()
            a.mul_(2)
            return a, tmp

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_inplace_requires_grad_true(self):
        def f(a, b):
            a.requires_grad_(True)
            return a.mul(3), b.mul(4)

        inp = [
            # First inp doesn't require grad, but we switch it on
            torch.ones(3, 3, requires_grad=False),
            torch.ones(3, 3, requires_grad=True),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 4);  primals_2 = None
    return (mul, mul_1)""",
        )

    # This is a torture test:
    # a and b get turned into a synthetic base in the compiled graph
    # One gets a data mutation, the other gets a metadata mutation.
    # We need to make sure that the metadata mutation gets propagated
    # back to the original input.
    @skipIfDynamoInput("Dynamo removes runtime error")
    def test_input_data_and_metadata_mutation_aliases_other_input(self):
        # a and b are aliased
        def f(a, b):
            a.mul_(2)
            b.t_()
            return a.mul(b)

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=True),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

    # https://github.com/pytorch/pytorch/issues/106456
    def test_input_mutation_noncontiguous(self):
        def f(a):
            a.mul_(2)
            return a + 1

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            # create a non-contiguous view to pass as an input to the compiler
            inp = x[:, 0]
            return [base], [inp]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_backward_mutation_data(self):
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                # bw mutation
                x.mul_(2)
                return grad_output.clone()

        def f(a, b):
            out = BwMutation.apply(b)
            return a * out

        inp_no_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        # Mutation on buffer that does not require grad during the backward is allowed
        self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp_grad, t

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 84 class(es): AOTTestCase, TestPythonKey, TestAOTAutograd, F, F, Foo, AddConstantMetadataMode, Foo, TracableCreateParameter, Foo, BwMutation, FwBwMutation, FwBwMutation, BwMutation, BwMutation, F, F, F, CustomFn, MyModel

### Functions
This file defines 833 function(s): amax_to_scale, _pack_fp8_with_scale_wrap, _unpack_fp8_with_scale_wrap, _pack_fp8_wrap, _unpack_fp8_wrap, pack_fp8, unpack_fp8, pack_fp8_with_scale, unpack_fp8_with_scale, test_make_fx, f, test_make_fx_grad, f, test_scalar_device, f, test_make_fx_vmap, f, test_make_fx_jacrev, f, test_make_fx_vjp, f, test_make_fx_functionalize, fn, test_make_fx_no_decompose, f, test_nnc_jit, f, test_nnc_scalar, f, test_nnc_pytrees


## Key Components

The file contains 26879 words across 8767 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 333734 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

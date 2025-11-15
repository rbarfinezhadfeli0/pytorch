# Documentation: `docs/test/functorch/test_aotdispatch.py_docs.md`

## File Metadata

- **Path**: `docs/test/functorch/test_aotdispatch.py_docs.md`
- **Size**: 54,903 bytes (53.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/functorch/test_aotdispatch.py`

## File Metadata

- **Path**: `test/functorch/test_aotdispatch.py`
- **Size**: 333,734 bytes (325.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
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
                    torc
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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
python docs/test/functorch/test_aotdispatch.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/functorch`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_aot_joint_with_descriptors.py_kw.md_docs.md`](./test_aot_joint_with_descriptors.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_eager_transforms.py_docs.md_docs.md`](./test_eager_transforms.py_docs.md_docs.md)
- [`functorch_additional_op_db.py_kw.md_docs.md`](./functorch_additional_op_db.py_kw.md_docs.md)
- [`test_ac_knapsack.py_docs.md_docs.md`](./test_ac_knapsack.py_docs.md_docs.md)
- [`common_utils.py_kw.md_docs.md`](./common_utils.py_kw.md_docs.md)
- [`test_logging.py_kw.md_docs.md`](./test_logging.py_kw.md_docs.md)
- [`test_rearrange.py_kw.md_docs.md`](./test_rearrange.py_kw.md_docs.md)
- [`test_dims.py_kw.md_docs.md`](./test_dims.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_aotdispatch.py_docs.md_docs.md`
- **Keyword Index**: `test_aotdispatch.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

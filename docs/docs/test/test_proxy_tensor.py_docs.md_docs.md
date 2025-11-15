# Documentation: `docs/test/test_proxy_tensor.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_proxy_tensor.py_docs.md`
- **Size**: 54,784 bytes (53.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_proxy_tensor.py`

## File Metadata

- **Path**: `test/test_proxy_tensor.py`
- **Size**: 83,521 bytes (81.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: ProxyTensor"]
# ruff: noqa: F841

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch._dynamo
import unittest
import warnings
import operator
from collections.abc import Iterable
from torch.nn.utils import stateless
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, skip, xfail, skipOps
from torch._subclasses.fake_tensor import DynamicOutputShapeException, DataDependentOutputException, FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._decomp import decomposition_table
from torch.fx.experimental.symbolic_shapes import (
    eval_guards, bind_symbols, fx_placeholder_vals, fx_placeholder_targets,
    guard_int, GuardOnDataDependentSymNode
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.common_device_type import ops
import torch.testing._internal.optests as optests
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import make_fx, DecompositionInterpreter, get_isolated_graphmodule
from torch.utils._pytree import tree_map
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch import nn
import torch._functorch.config
import re

import functools
import itertools
from pathlib import Path

aten = torch.ops.aten

HAS_CUDA = torch.cuda.is_available()


def strip_end(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


def show_guards(gm):
    names = [strip_end(n, "_1") for n in fx_placeholder_targets(gm)]
    return "\n".join(
        gm.shape_env.produce_guards(fx_placeholder_vals(gm), names, _simplified=True, input_contexts=None)
    )


def process_failures():
    """
    Takes file containing failures like

    FAILED test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive___getitem___cpu_float32 - RuntimeError: aten.size.default - couldn't find symbolic meta function/decomposition  # noqa: B950

    and processes them into a list of opinfo xfails
    """
    failures = [i.strip() for i in Path('pytest_failures').read_text().splitlines()]

    def process_failure_string(s, matcher):
        out = re.search(matcher, s)
        return out.groups()

    SYMBOLIC_TRACE_MATCH = r'exhaustive_(.*)_cpu.*: (.*)'
    failures = [process_failure_string(s, SYMBOLIC_TRACE_MATCH) for s in failures]

    def create_normalized_name(op):
        if op.variant_test_name == '':
            s = op.name
        else:
            s = f"{op.name}.{op.variant_test_name}"
        return s.replace('.', '_')

    remap_opinfo = {create_normalized_name(op): (op.name, op.variant_test_name) for op in op_db}

    print("symbolic_tensor_failures = {")
    for failure, reason in failures:
        print(f"    xfail{remap_opinfo[failure]},  # {reason}")
    print("}")


USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)


def _create_new_input(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype != torch.float:
        return x + 1
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    else:
        return torch.rand_like(x)

"""
Delays a cos being executed on the unwraptensor until its used. Simulates a CommTensor used
"""
class UnwrapTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )
        r._tensor = tensor
        return r

    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"UnwrapTensor({self._tensor})"

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            ret = e
            if isinstance(e, UnwrapTensor):
                ret = e._tensor.cos()

            return ret

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

class TestGenericProxyTensor(TestCase):
    # WARNING: if any of your inputs are index tensors, DO NOT use this
    # function
    def _test(self, f, inps):
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(*inps)
        new_inps = tree_map(_create_new_input, inps)
        r1 = fx_f(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    def test_pre_dispatch_mode_stack(self):
        def f(a):
            b = torch.ones(4, 4)
            return torch.matmul(a, b)
        # We expect to see matmul in the trace - it should NOT be decomposed into mm.
        # Also, torch.ones() doesn't show up in the trace.
        # This is annoying but expected: ones() never dispatches to the Autograd dispatch key,
        # so our mode never sees it - it goes directly to the BackendSelect key.
        inp = torch.ones(4, 4)
        # Test that make_fx(pre_dispatch=True) clears caches properly.
        from torch._dispatch.python import enable_python_dispatcher
        with enable_python_dispatcher():
            out1 = f(inp)
        fx_g = make_fx(f, pre_dispatch=True)(inp)
        self.assertExpectedInline(fx_g.code.strip(), """\
def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 4], device = device(type='cpu'), pin_memory = False)
    matmul = torch.ops.aten.matmul.default(a_1, ones);  a_1 = ones = None
    return matmul""")

    def test_pre_dispatch_linear(self):
        def f(a, b, c):
            return torch.nn.functional.linear(a, b, c)
        a = torch.ones(4, 4)
        b = torch.ones(4, 4)
        c = torch.ones(4)
        fx_g = make_fx(f, pre_dispatch=True)(a, b, c)
        out1 = f(a, b, c)
        out2 = fx_g(a, b, c)
        self.assertEqual(out1, out2)

    def test_pre_dispatch_no_grad(self):
        def f(a):
            b = a.sin()
            torch.set_grad_enabled(False)
            c = b.cos()
            torch.set_grad_enabled(True)
            return b + c.sin()
        a1 = torch.randn(4, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        a_tmp = a1.detach().clone().requires_grad_(True)
        fx_g = make_fx(f, pre_dispatch=True)(a_tmp)
        out1 = f(a1)
        out2 = fx_g(a2)
        self.assertEqual(out1, out2)
        out1.sum().backward()
        out2.sum().backward()
        self.assertEqual(a1.grad, a2.grad)

    def test_make_fx_simple(self):
        def f(x):
            return torch.sin(x)
        self._test(f, (torch.randn(3),))

    def test_scalar_device(self, device='cpu'):
        def f(a, b):
            return a + b
        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])

    def test_isolated_graphmodule(self):
        def is_any_sum(gm):
            return any(node.target == torch.ops.aten.sum.default for node in gm.graph.nodes)

        def is_any_digamma(gm):
            return any(node.target == torch.ops.aten.digamma.default for node in gm.graph.nodes)

        def is_any_sigmoid(gm):
            return any(node.target == torch.ops.aten.sigmoid.default for node in gm.graph.nodes)

        def inner(x):
            return torch.sum(x)

        def f(x):
            gm = get_isolated_graphmodule(inner, (x,), {})
            self.assertTrue(is_any_sum(gm))
            return x + torch.randn(x.shape)

        # get_isolated_graphmodule uses make_fx internally that shouldn't be traced
        # by the outer make_fx call
        traced = make_fx(f)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))

        # When factory functions are used, they should not be traced
        # by the outer make_fx call
        def inner_with_factory():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((10, 10), val).sum()

        def f1(x):
            gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2(x):
            gm = get_isolated_graphmodule(f1, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify nested make_fx calls don't make factory functions to be leaked
        # into the outer graph. Verify that `make_fx`` itself does not leak its execution.
        def f2(x):
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify that the `forward`` function of a graph module produced as a
        # side effect of an interior `make_fx` is still traced
        def f3(x):
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            # `gm.forward`` is still traced
            return torch.digamma(gm(x))

        traced = make_fx(f3)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertTrue(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify interaction with non-ProxyTensor modes
        from torch.testing._internal.logging_tensor import LoggingTensorMode

        def f1_logging(x):
            with LoggingTensorMode():
                gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2_logging(x):
            with LoggingTensorMode(), LoggingTensorMode():
                gm = get_isolated_graphmodule(f1_logging, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2_logging)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify interaction with another tensor subclass
        # This case currently doesn't work and should raise an error
        # See: https://github.com/pytorch/pytorch/pull/81764#issuecomment-1200472068
        from torch.testing._internal.logging_tensor import LoggingTensor

        def f1_logging_tensor(x):
            gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2_logging_tensor(x):
            x = LoggingTensor(x)
            gm = get_isolated_graphmodule(f1_logging_tensor, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2_logging_tensor)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))  # this fails, sigmoid is traced with LoggingTensor
        self.assertTrue(is_any_digamma(traced))

    # See https://github.com/pytorch/pytorch/issues/97541
    def test_empty_like_doesnt_burn_in_defaults(self):
        def f(x):
            return torch.empty_like(x)
        out = make_fx(f)(torch.randn(3))
        self.assertExpectedInline(out.code.strip(), """\
def forward(self, x_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False);  x_1 = None
    return empty_like""")

    def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
        def f(x):
            y = x.new_zeros(x.size())
            y.copy_(x)
            return y

        def _new_zeros_decomp(inp, size, dtype=None, layout=None, device=None, pin_memory=None):
            return torch.zeros(size, dtype=inp.dtype, device=inp.device)

        factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

        # When new_zeros() decomposes into torch.zero(), we expect ProxyTensorMode
        # to still be (re-entrantly) enabled, so that the `torch.zero()` call
        # returns a ProxyTensor.
        out = make_fx(f, decomposition_table=factory_func_decomp)(torch.ones(2))
        self.assertExpectedInline(out.code, """\



def forward(self, x_1):
    zeros = torch.ops.aten.zeros.default([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy_
    """)

    def test_make_fx_reentrant_dispatch(self):
        def f(x):
            return torch.ops.aten.norm.Scalar(x, 2.0)

        def norm_decomp(x, p=2.0):
            if p != 2.0:
                raise RuntimeError("can't handle with p != 2")
            return torch.sqrt(torch.sum(torch.square(x)))

        decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

        traced = make_fx(f, decomposition_table=decomp, tracing_mode=self.tracing_mode)(torch.rand(3))

        for n in traced.graph.nodes:
            self.assertTrue("square" not in str(n.target))
            self.assertTrue("norm" not in str(n.target))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self):
        mod = torchvision.models.resnet18()

        # An old version of this test called the module directly.  This works
        # for tracing_mode == "real", but for fake tensors, we also have to
        # ensure that the parameters and buffers get wrapped in fake tensors
        # because free fake tensors are not supported.  Fortunately functional_call
        # does precisely this for us.
        def f(x, params, buffers):
            for p in params.values():
                p.grad = None
            loss = torch.func.functional_call(mod, {**params, **buffers}, (x,)).sum()
            # I could have done this with the functional API, but there is
            # plenty of exercising this; I want to show mutating API still
            # works
            loss.backward()
            return [p.grad for p in params.values()]

        inp = torch.randn(3, 3, 250, 250)
        self._test(f, [inp, dict(mod.named_parameters()), dict(mod.named_buffers())])

    def test_varargs(self):
        def f(*args):
            return sum(args)

        self._test(f, [torch.randn(2), torch.randn(2)])

    def test_proxy_tensor(self):
        def f_grad(x):
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        def f_backward(x):
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        for f in [f_grad, f_backward]:
            self._test(f, [torch.randn(3, requires_grad=True)])

    def test_pickle_issue89626(self):
        import pickle
        x = torch.randn(2)
        make_fx(lambda x: x * 2, tracing_mode=self.tracing_mode)(x)
        pickle.dumps(x)

    def test_inplace_metadata(self):
        def f(x):
            x = x.clone()
            x.unsqueeze_(-1)
            assert x.shape[-1] == 1
            return x

        self._test(f, [torch.randn(5)])

    def test_mode_tracing_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        # default behavior should trace factory functions
        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))
        self.assertTrue(
            any(
                node.target == aten.randn.default
                for node in traced.graph.nodes
            )
        )

    def test_pre_dispatch_functionalization(self):
        def f(x):
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                y = y + x_unwrapped
                y.mul_(5)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1):
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    add = torch.ops.aten.add.Tensor(matmul, x_1);  matmul = x_1 = None
    mul = torch.ops.aten.mul.Tensor(add, 5);  add = None
    return mul""")

    def test_pre_dispatch_functionalization_view_op(self):
        def f(x):
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                x_unwrapped = x_unwrapped.transpose(1, 0)
                y = y + x_unwrapped
                y = y.view(2, 8)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1):
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    transpose = torch.ops.aten.transpose.int(x_1, 1, 0);  x_1 = None
    add = torch.ops.aten.add.Tensor(matmul, transpose);  matmul = transpose = None
    view = torch.ops.aten.view.default(add, [2, 8]);  add = None
    return view""")

    def test_val_metadata_mutation(self):
        def f(x):
            y = x.clone()
            y.unsqueeze_(0)
            return y

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3, requires_grad=True))
        self.assertEqual([
            tuple(node.meta['val'].shape)
            for node in traced.graph.nodes
            if 'val' in node.meta
        ], [(3,), (3,), (1, 3)])

    def test_make_fx_overloads(self):
        def f(x):
            return x.cos() + torch.randn(x.shape)

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))

        self.assertTrue(all(isinstance(node.target, torch._ops.OpOverload)
                            for node in traced.graph.nodes if node.op == 'call_function'))

    def test_tensor_constants(self):
        def f():
            val = torch.tensor(float('inf'))
            return torch.full((100, 100), val)

        self._test(f, [])

    def test_allclose(self):
        def f(a, b):
            return torch.allclose(a, b)

        def test_f():
            make_fx(f, tracing_mode=self.tracing_mode)(
                torch.zeros(3), torch.zeros(3)
            )

        if self.tracing_mode != "real":
            self.assertRaises(DataDependentOutputException, test_f)
        else:
            self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_proxy_tensor_mut(self):
        def f():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        self.assertEqual(g(), f())
        # In case we mutated shared state in the g graph!
        self.assertEqual(g(), f())

    def test_constant_unbind(self):
        def f():
            val = torch.tensor([2])
            r, = torch.unbind(val, 0)
            return r.item()

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        self.assertEqual(g(), f())

    def test_constant_blowup(self):
        def f():
            val = torch.tensor([2])
            blowup = val.repeat(1000)
            return bool(blowup.sum().item() == 2)

        def test_f():
            make_fx(f, tracing_mode=self.tracing_mode)()

        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_random(self):
        def f():
            val = torch.tensor([2.0])
            val.normal_()
            return bool(val.item() == 2.1)

        def test_f():
            make_fx(f, tracing_mode=self.tracing_mode)()

        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_decomposition_interpreter(self):
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        fx_module = make_fx(fn, tracing_mode=self.tracing_mode, decomposition_table=None)(x)

        found_silu = False
        for n in fx_module.graph.nodes:
            if n.target == torch.ops.aten.silu or n.target == torch.ops.aten.silu.default:
                found_silu = True

        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        silu_decomp_table = {torch.ops.aten.silu.default: decomposition_table[torch.ops.aten.silu.default]}
        DecompositionInterpreter(
            fx_module,
            new_graph=new_graph,
            decomposition_table=silu_decomp_table,
        ).run(x)

        decomposed_module = torch.fx.GraphModule(fx_module, new_graph)

        for n in decomposed_module.graph.nodes:
            self.assertTrue(n.target != torch.ops.aten.silu)
            self.assertTrue(n.target != torch.ops.aten.silu.default)

        self.assertEqual(fx_module(x), decomposed_module(x))

    def test_make_fx_model_fwd_bwd(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(x, params):
            out = torch.func.functional_call(model, params, x).sum()
            out.backward()
            return list(params.values())
        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params)
        # fx may change the order of parameters in list, so using set() to compare
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

    def test_make_fx_model_double_param(self):
        class Emformer(torch.nn.Module):
            def __init__(
                self,
                input_dim: int = 256,
            ) -> None:
                super().__init__()

                self.layer_norm = torch.nn.LayerNorm(input_dim)

            def forward(mod_self, x):  # noqa: B902
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                y = mod_self.layer_norm(x)
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                z = mod_self.layer_norm(y)
                return z


        gm = make_fx(Emformer())(torch.randn(16, 1, 256))
        ops = {n.target for n in gm.graph.nodes if n.op == 'call_function'}
        self.assertEqual(len(ops), 2)


    def test_make_fx_model_fwd_bwd_wgtupdate(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(args, params, buffers):
            for p in params.values():
                p.grad = None
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = torch.func.functional_call(model, params_and_buffers, args)
            out.sum().backward()
            return [p - 1e-4 * p.grad for p in params.values()]

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params, buffers)
        # fx may change the order of parameters in list, so using set() to compare
        # also there is a numerical difference in results so changing atol from 1e-08 to 1e-03
        self.assertTrue(
            torch.allclose(fx_f(input, params, buffers)[0], f(input, params, buffers)[0], atol=1e-03)
            or
            torch.allclose(fx_f(input, params, buffers)[0], f(input, params, buffers)[1], atol=1e-03)
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params, buffers)[1], f(input, params, buffers)[0], atol=1e-03)
            or
            torch.allclose(fx_f(input, params, buffers)[1], f(input, params, buffers)[1], atol=1e-03)
        )

    def test_trace_subclasses(self):
        def f1(x):
            x = UnwrapTensor(x)
            y = x * 2
            return y

        def f2(x):
            wrapped = UnwrapTensor(x)
            y = x * wrapped
            return y

        inp = [torch.randn(5)]
        self._test(f1, inp)
        self._test(f2, inp)

    def test_partial_decomp(self):
        def f(a, b, c):
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y
        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        fx_g = make_fx(f)(*inps)

        def addmm(a, b, c, beta=1, alpha=1):
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        decomposed_fx = make_fx(f, decomposition_table={aten.addmm.default: addmm})(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2)
        self.assertEqual(len([n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]), 1)

    def test_decomp_of_capture(self):
        val = torch.randn(5)

        def f(x):
            return x.t() + val.t()

        def nop(x):
            return x.cos()

        traced = make_fx(f, decomposition_table={torch.ops.aten.t.default: nop})(torch.randn(5))
        self.assertEqual(len([n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]), 0)


    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_amp_cache(self):
        layer = torch.nn.Conv2d(3, 3, 3).cuda()

        def f(x, w):
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)

        inp = torch.randn(4, 3, 10, 10, device='cuda')
        with torch.autocast('cuda'):
            out_graph = make_fx(f)(inp, layer.weight).graph
            out_graph2 = make_fx(f)(inp, layer.weight).graph

        self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))
        for a, b in zip(out_graph.nodes, out_graph2.nodes):
            self.assertEqual(a.op, b.op)

    def test_strides(self):
        def f(x):
            self.assertTrue(x.is_contiguous())
            self.assertFalse(x.is_contiguous(memory_format=torch.channels_last))
            x = x.permute(0, 3, 1, 2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
            return x
        make_fx(f)(torch.randn(2, 3, 4, 5))

        def f(x):
            self.assertTrue(x.is_contiguous())
            y = x[:, 1]
            self.assertFalse(y.is_contiguous())
            y = x[:, ::2]
            self.assertFalse(y.is_contiguous())
            return x.cos()

        make_fx(f)(torch.randn(2, 3, 4, 5))

    def test_pr_86917(self):
        # Tests the issue brought up here https://github.com/pytorch/pytorch/pull/86917#issuecomment-1283155344
        def f(a, b):
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_T244632748(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + (x.shape[0] * 2)

        mod = TestModule()
        sample = torch.randn((5, 5)).to("cuda")
        dim0 = torch.export.Dim.DYNAMIC(max=100)
        dynamic_shapes = {"x": (dim0, torch.export.Dim.STATIC)}
        ep = torch.export.export(mod, (sample,), dynamic_shapes=dynamic_shapes)
        gm = ep.module()
        symint = list(gm.graph.nodes)[3].meta["val"]
        list(gm.graph.nodes)[3].replace_all_uses_with(symint)
        gm.graph.eliminate_dead_code()

        inductor_fx = torch._inductor.aot_compile(
            gm, (sample,), options={"fx_wrapper": True, "compile_threads": 1}
        )


class TestGenericProxyTensorReal(TestGenericProxyTensor):
    tracing_mode = "real"


class TestGenericProxyTensorFake(TestGenericProxyTensor):
    tracing_mode = "fake"


class TestGenericProxyTensorSymbolic(TestGenericProxyTensor):
    tracing_mode = "symbolic"


del TestGenericProxyTensor


class TestRealProxyTensor(TestCase):
    def test_error_on_data_dependent_ops(self):
        def f():
            x = torch.randn([])
            y = torch.randn([])
            assert torch.allclose(x * y, y * x)
            z = float(x)
            z2 = float(y)

        # Smoke tests
        make_fx(f, _error_on_data_dependent_ops=False)()
        make_fx(f, pre_dispatch=True, _error_on_data_dependent_ops=False)()

class TestFakeProxyTensor(TestCase):
    def test_issue82547(self):
        x = nn.Parameter(torch.randn(3, 3))

        def f():
            return torch.ops.aten.t.default(x)
        self.assertRaisesRegex(Exception, "Please convert all Tensors", lambda: make_fx(f, tracing_mode="fake")())

        class A(torch.Tensor):
            pass

        x = A(torch.randn(3, 3))
        self.assertRaisesRegex(TypeError, "Multiple dispatch failed", lambda: make_fx(f, tracing_mode="fake")())

    def test_use_fake_and_tensor(self):
        def f(x, y):
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        g = make_fx(f, tracing_mode="fake")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

    def test_free_fake(self):
        def f(x):
            return torch.add(x, y)

        with FakeTensorMode() as fake_mode:
            y = torch.randn(2)
            make_fx(f, tracing_mode="real")(torch.randn(2))

    def test_fused_adam(self):
        # See https://github.com/pytorch/pytorch/issues/99356
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
            (new_params, _, _, _, _) = aten._fused_adam.default(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )

            for p, new_p in zip(params, new_params):
                p.copy_(new_p)

            return params

        gm = make_fx(fused_adam, tracing_mode='fake')(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )
        ensure_ops_have_val = [aten._fused_adam.default, operator.getitem]
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in ensure_ops_have_val:
                self.assertIn('val', n.meta)

    def test_alias(self):
        def f(x):
            return torch.ops.aten.alias(x)

        r = str(make_fx(f, tracing_mode="fake")(torch.randn(2)).code).strip()
        # NB: this should not have a detach call
        self.assertExpectedInline(r, """\
def forward(self, x_1):
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    return alias""")

    def test_meta(self):
        def f(x):
            a = x.cos()
            b = torch.var_mean(a, dim=0)
            c = b * 2
            return c

        out = make_fx(f, tracing_mode="fake")(torch.randn(5, 5))
        for n in out.graph.nodes:
            if n.op == 'output':
                continue
            self.assertTrue('val' in n.meta)

    def test_fake_tensor_mode(self):
        def f(a):
            d = a.cos()
            return d

        from torch._guards import detect_fake_mode

        existing_fake_mode = FakeTensorMode()
        with existing_fake_mode:
            out = make_fx(f, tracing_mode="real")(torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]))

        fake_mode = detect_fake_mode([node.meta.get('val', None) for node in out.graph.nodes])
        self.assertEqual(fake_mode, existing_fake_mode)

def _get_node(fx_g, cond):
    for n in fx_g.graph.nodes:
        if cond(n):
            return n
    raise AssertionError

def _get_free_symbols(shape_env):
    vars = tuple(shape_env.var_to_val.keys())
    return len([var for var in vars if var not in shape_env.replacements])

def _trace(f, *args):
    inps = [torch.randn(arg) for arg in args]
    return make_fx(f, tracing_mode="symbolic")(*inps)

# TODO: Need to test the guards themselves specifically as well
class TestSymbolicTracing(TestCase):
    def _test_dynamic(self, fn, trace_inputs, test_inputs, assert_eq=True):
        """
        Tests fn traced with trace_inputs against test_inputs
        Also returns shape env
        """
        trace_inputs = [torch.randn(shape) for shape in trace_inputs]
        traced_f = make_fx(fn, tracing_mode="symbolic")(*trace_inputs)
        for input in test_inputs:
            input = [torch.randn(shape) for shape in input]
            rx, ry = traced_f(*input), fn(*input)
            if assert_eq:
                self.assertEqual(rx, ry)
        return traced_f


    def test_debug_interpreter(self):
        import torch.library
        from torch.library import Library

        foo = Library("foo", "DEF")  # noqa: TOR901
        foo.define("foo(Tensor self) -> Tensor")

        # Operator where meta and cpu disagree on strides
        @torch.library.impl(foo, "foo", "CPU")
        def foo_cpu(x):
            return x.clone().T

        @torch.library.impl(foo, "foo", "Meta")
        def foo_meta(x):
            return x.clone()

        def f(x):
            return torch.ops.foo.foo.default(x)

        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2, 2))
        from torch._functorch.compilers import DebugInterpreter

        interp = DebugInterpreter(gm)

        # input mismatch is caught (indicates guard problem)
        self.assertRaisesRegex(
            AssertionError, r"3 != 1",
            lambda: interp.run(torch.randn(3, 3).T),
        )

        # Catch the incorrect meta
        self.assertRaisesRegex(
            AssertionError, r"\(3, 1\) != \(1, 3\)",
            lambda: interp.run(torch.randn(3, 3))
        )

    def test_int_input(self):
        def f(x, y):
            return x.view(y)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 4), 12).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    view = torch.ops.aten.view.default(x_1, [y_1]);  x_1 = y_1 = None
    return view""")

    def test_resize_from_zero(self):
        def f(x, y):
            x.resize_(y.size(0))

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(0), torch.empty(2)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    resize_ = torch.ops.aten.resize_.default(x_1, [sym_size_int]);  x_1 = sym_size_int = resize_ = None
    return None""")

    def test_broadcast_shapes(self):
        def f(x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0])

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 1), torch.empty(5)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    return (sym_size_int, sym_size_int_1)""")

    def test_deduped_shape(self):
        def f(s0, s1, x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
            x = fake_mode.from_tensor(x)
            y = fake_mode.from_tensor(y)
            r = str(make_fx(f, tracing_mode="real")(x.shape[0], y.shape[0], x, y).code).strip()
            self.assertExpectedInline(r, """\
def forward(self, s0_1, s1_1, x_1, y_1):
    empty = torch.ops.aten.empty.memory_format([s0_1], device = device(type='cpu'), pin_memory = False)
    return ((s0_1, s1_1), empty)""")

    def test_non_deduped_shape(self):
        def f(x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
            x = fake_mode.from_tensor(x)
            y = fake_mode.from_tensor(y)
            r = str(make_fx(f, tracing_mode="real")(x, y).code).strip()
            self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    empty = torch.ops.aten.empty.memory_format([sym_size_int], device = device(type='cpu'), pin_memory = False)
    return ((sym_size_int, sym_size_int_1), empty)""")

    def test_unary(self):
        def f(x):
            assert x.shape[0] < 20
            return x.cos()
        test_inputs = []
        test_inputs.append([(2, 5)])
        test_inputs.append([(6, 8)])
        gm = self._test_dynamic(f, [(3, 4)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(4, 5)))
        self.assertEqual(repr(bind_symbols(gm, torch.randn(4, 5))), "{s75: 4, s96: 5}")
        self.assertFalse(eval_guards(gm, torch.randn(25, 5)))
        self.assertExpectedInline(show_guards(gm), """L['x'].size()[0] <= 19""")

    def test_repeat_interleave(self):
        def f(src_tokens, beam_size_src):
            return src_tokens.repeat_interleave(beam_size_src.size(0), 0)

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens, torch.randn(5))
        self.assertEqual(len(gm.shape_env.guards), 0)

    def test_non_symint_size_spec(self):
        # this isn't really a proxy tensor test, but it's the most convenient
        # way to get a fake tensor with symbolic sizes
        def f(x):
            torch._C._non_sym_sizes(x)
            return x + 1

        x = torch.randn(2, 3)
        make_fx(f, tracing_mode="symbolic")(x)

    # https://github.com/pytorch/pytorch/issues/108195
    def test_symbolic_repeat_interleave(self):
        def f(y, x):
            return y.repeat_interleave(x, dim=1)

        y = torch.tensor([[1, 2], [3, 4]])
        x = torch.tensor([2, 3])
        r = str(make_fx(f, tracing_mode="symbolic")(y, x).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, y_1, x_1):
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1);  x_1 = None
    index_select = torch.ops.aten.index_select.default(y_1, 1, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select""")

    def test_mod_gcd_unbacked(self):
        def f(_a, _b, _stride):
            a = _a.item()
            b = _b.item()
            stride = _stride.item()
            ta = torch.randn(a * stride)
            tb = torch.randn(b * stride)
            r = torch.cat([ta, tb])
            return r.view(a + b, stride)

        _a = torch.tensor(30)
        _b = torch.tensor(20)
        _stride = torch.tensor(10)
        r = str(make_fx(f, tracing_mode="symbolic")(_a, _b, _stride).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, _a_1, _b_1, _stride_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(_a_1);  _a_1 = None
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(_b_1);  _b_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(_stride_1);  _stride_1 = None
    mul = _local_scalar_dense * _local_scalar_dense_2
    randn = torch.ops.aten.randn.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    mul_1 = _local_scalar_dense_1 * _local_scalar_dense_2
    randn_1 = torch.ops.aten.randn.default([mul_1], device = device(type='cpu'), pin_memory = False);  mul_1 = None
    cat = torch.ops.aten.cat.default([randn, randn_1]);  randn = randn_1 = None
    add = _local_scalar_dense + _local_scalar_dense_1;  _local_scalar_dense = _local_scalar_dense_1 = None
    view = torch.ops.aten.view.default(cat, [add, _local_scalar_dense_2]);  cat = add = _local_scalar_dense_2 = None
    return view""")

    def test_cumsum_unbacked(self):
        def f(x):
            y = x.item()
            z = torch.randn((3, y, 3))
            return z.cumsum(0)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([5])).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    randn = torch.ops.aten.randn.default([3, _local_scalar_dense, 3], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    cumsum = torch.ops.aten.cumsum.default(randn, 0);  randn = None
    return cumsum"""  # noqa: B950
        )


    def test_repeat_interleave_unbacked_output_size(self):
        def f(x, y):
            s = x.sum().item()
            return y.repeat_interleave(x, dim=0, output_size=s)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([2, 3]), torch.randn(2)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1, y_1):
    sum_1 = torch.ops.aten.sum.default(x_1)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(sum_1);  sum_1 = None
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1, output_size = _local_scalar_dense);  x_1 = _local_scalar_dense = None
    index_select = torch.ops.aten.index_select.default(y_1, 0, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select"""  # noqa: B950
        )

    def test_arange_unbacked_output_size(self):
        def f(x):
            return torch.arange(0, x)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    arange = torch.ops.aten.arange.start(0, _local_scalar_dense, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return arange"""  # noqa: B950
        )

    def test_adv_index_batch(self):
        def f(src_tokens):
            bsz, src_len = src_tokens.size()[:2]
            start_step = src_tokens.shape[1]
            beam_size = 1
            generate_size = 64
            max_len = src_len + generate_size
            tokens = torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(0)
            tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
            return tokens

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens)
        # Guards to rule out batch_size == sys.maxsize (wobbling between 2 and
        # 1 ok)
        self.assertEqual(len(gm.shape_env.guards), 0)

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_cpu_scalar_cuda(self):
        # Extracted from wave2vec2
        def f(a, b):
            return (a * b) @ b

        r = str(
            make_fx(f, tracing_mode="symbolic")(
                torch.tensor(1.0), torch.randn(2, 2, device='cuda')
            ).code
        ).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1, b_1):
    mul = torch.ops.aten.mul.Tensor(a_1, b_1);  a_1 = None
    mm = torch.ops.aten.mm.default(mul, b_1);  mul = b_1 = None
    return mm""")

    def test_binary_broadcast(self):
        def f(a, b):
            c = a * b
            return c

        test_inputs = []
        test_inputs.append([(1, 5), (3, 1)])
        test_inputs.append([(1, 4), (4, 1)])
        shape_env = self._test_dynamic(f, [(1, 2), (3, 1)], test_inputs).shape_env
        assert len(shape_env.guards) == 0

    def test_multiply_shape(self):
        def f(a):
            return torch.empty(a.shape[0] * 2)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    mul = sym_size_int * 2;  sym_size_int = None
    empty = torch.ops.aten.empty.memory_format([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    return empty""")

    def test_item(self):
        def f(a):
            r = a.item()
            return r * a

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(1)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1)
    mul = torch.ops.aten.mul.Tensor(a_1, _local_scalar_dense);  a_1 = _local_scalar_dense = None
    return mul""")

    def test_tensor_symfloat(self):
        def f(a):
            r = torch.tensor(a.size(0) ** 2.0)
            assert r.dtype is torch.float
            return r

        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2))
        r = str(gm.code).strip()
        # NB: this specializes, which is fine, the point is to make sure the
        # dtype inference is correct
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    return lift_fresh_copy""")
        self.assertEqual(gm._tensor_constant0, torch.tensor(4.0))

    def test_item_to_constructor(self):
        def f(a):
            r = a.item()
            return torch.empty(r)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randint(5, (1,))).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return empty"""  # noqa: B950
        )


    def test_setitem_symint(self):
        # from moco
        # https://github.com/pytorch/pytorch/issues/101939
        def f(x):
            x[0] = x.size(0)
            return x

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(10)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    scalar_tensor = torch.ops.aten.scalar_tensor.default(sym_size_int, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  sym_size_int = None
    select = torch.ops.aten.select.int(x_1, 0, 0)
    copy_ = torch.ops.aten.copy_.default(select, scalar_tensor);  select = scalar_tensor = copy_ = None
    return x_1"""  # noqa: B950
        )

    def test_dynamic_pointwise_scalar(self):
        def f(gravity, mask):
            g
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_proxy_tensor.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_proxy_tensor.py_docs.md_docs.md`
- **Keyword Index**: `test_proxy_tensor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

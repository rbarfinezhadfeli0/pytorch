# Documentation: test_autograd.py

## File Metadata
- **Path**: `test/test_autograd.py`
- **Size**: 561004 bytes
- **Lines**: 15333
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: autograd"]
# ruff: noqa: F841

import collections
import contextlib
import functools
import gc
import io
import math
import operator
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import weakref
from collections import OrderedDict
from copy import deepcopy
from functools import partial, reduce
from itertools import product
from operator import mul
from typing import TYPE_CHECKING

import torch
import torch.autograd._functions
import torch.autograd.forward_ad as fwAD
from torch import inf, nan, nn
from torch.autograd import (
    _calculate_shape,
    detect_anomaly,
    Function,
    kineto_available,
    Variable,
)
from torch.autograd.function import InplaceFunction, once_differentiable
from torch.autograd.graph import GradientEdge
from torch.autograd.profiler import emit_itt, emit_nvtx, profile, record_function
from torch.autograd.profiler_util import (
    _format_time,
    EventList,
    FunctionEvent,
    FunctionEventAvg,
)
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    expectedFailureMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_utils import (
    disable_gc,
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    scoped_load_inline,
    set_warn_always_context,
    skipCUDANonDefaultStreamIf,
    skipIfMPS,
    skipIfNoLapack,
    skipIfTorchDynamo,
    skipIfWindows,
    skipIfXpu,
    slowTest,
    TestCase,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    checkpoint,
    checkpoint_sequential,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch.utils.flop_counter import FlopCounterMode


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


def graph_desc(fn):
    if fn is None:
        return "None"
    result = type(fn).__name__ + "("
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ", "
    if next_functions:
        result = result[:-2]
    return result + ")"


class TestAutograd(TestCase):
    def tearDown(self):
        torch.autograd._force_original_view_tracking(False)
        super(TestCase, self).tearDown()

    def test_copy_slices_graph_task_updates(self):
        def f1(x, y):
            out = x.clone().view(-1)
            out += y
            return out

        def f2(x, y):
            out = x.clone().view(-1)
            b = out * 2
            out += y
            return out + b

        x = torch.rand(2, requires_grad=True)
        y = torch.rand(2, requires_grad=True)

        y_safe = torch._C._functions.DelayedError("Boom!", 1)(y)

        for f in [f1, f2]:
            # Ensure that the error Node works
            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                out.sum().backward()

            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                torch.autograd.grad(out.sum(), y)

            # Ensure that if we don't ask for y, it doesn't crash
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), x)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), y_safe)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), (x, y_safe))

        # Ensure that we don't run extra view Node
        def f3(x, y):
            out = x.clone().view(-1)

            def hook(*args):
                # This should never be called!
                self.assertTrue(False)

            out.register_hook(hook)

            b = out + y
            out += y
            return out + b, b

        out, b = f3(x, y_safe)
        torch.autograd.grad(out.sum(), (b, y_safe))

    def test_grad_mode_class_decoration(self):
        # Decorating class is deprecated and should not be used
        with self.assertWarnsRegex(FutureWarning, "Decorating classes is deprecated"):

            @torch.no_grad()
            class Foo:
                def __init__(self) -> None:
                    assert not torch.is_grad_enabled()

                def foo(self):
                    # Not applied to methods
                    assert torch.is_grad_enabled()

            # Show that we can actually construct the class
            foo = Foo()
            foo.foo()

        # Decorating functions or methods is fine though
        with warnings.catch_warnings(record=True) as w:

            @torch.no_grad()
            def foo():
                assert not torch.is_grad_enabled()

            foo()

            class Foo2:
                @torch.no_grad()
                def __init__(self) -> None:
                    assert not torch.is_grad_enabled()

                @torch.no_grad()
                def foo(self):
                    assert not torch.is_grad_enabled()

            foo2 = Foo2()
            foo2.foo()

        self.assertEqual(len(w), 0)

    def test_tensor_grad_warnings(self):
        dummy = torch.empty(1)

        with warnings.catch_warnings(record=True) as w:
            # Accessing .grad on leaf
            dummy.requires_grad_()
            foo = dummy.grad
            self.assertEqual(len(w), 0)

            # Accessing .grad on non-leaf
            dummy = dummy.clone()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

            # Accessing .grad on non-leaf that retains gradients
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)

        return x, y

    def test_function(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_tensors
                # NOTE: self is the test case here
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (
                    grad_output + grad_output * var2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * var1,
                )

        x, y = self._function_test(MyFunction)

        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertExpected(x_grad_desc, "x_grad_desc")
        self.assertExpected(y_grad_desc, "y_grad_desc")

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_once_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                self.assertFalse(torch.is_grad_enabled())
                t1, t2 = ctx.saved_tensors
                return (
                    grad_output + grad_output * t2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * t1,
                )

        x, y = self._function_test(MyFunction)
        self.assertEqual(
            graph_desc(x.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )
        self.assertEqual(
            graph_desc(y.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_function_returns_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

    def test_function_returns_undefined_tensor(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return None

        # Test that undefined tensors returned from custom backward function
        # are propagated as undefined and not tensor full of zeroes
        x = torch.ones(1, requires_grad=True)

        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x**2).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)

        self.assertIsNone(
            torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0]
        )

    def test_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(grad, torch.zeros(1))
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_dont_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertIsNone(grad)
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_set_materialize_non_diff_grads(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out0 = x.clone()
                out1 = x.clone()
                ctx.mark_non_differentiable(out1)
                ctx._materialize_non_diff_grads = False
                return out0, out1

            @staticmethod
            def backward(ctx, g0, g1):
                self.assertIsNone(g1)
                return g0

        a = torch.tensor(1.0, requires_grad=True)
        out = Func.apply(a)[0]
        out.backward()

    def test_legacy_function_deprecation_exception(self):
        # Trigger exception
        class MyFunction(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        # Check exception occurs
        with self.assertRaisesRegex(
            RuntimeError,
            "Legacy autograd function with non-static forward method is deprecated",
        ):
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):
        @staticmethod
        def forward(ctx, input):
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            raise Exception("Simulate error on backward pass")  # noqa: TRY002

    def test_custom_function_exception(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        tmp = (t1 + t2) * (t1 + t2)
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        with self.assertRaisesRegex(Exception, "Simulate error on backward pass"):
            t3.sum().backward()

    def test_custom_function_non_tensor_inputs_outputs(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale

                # Save scale
                ctx.scale = scale
                ctx.save_for_backward(t1, t2, t3)
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                # Verify grads
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])

                scale = ctx.scale
                var1, var2, var3 = ctx.saved_tensors
                return (
                    grads[1] * scale + grads[4] * var2 * scale + grads[6],
                    grads[1] * var3 * scale + grads[4] * var1 * scale,
                    None,
                    grads[1] * var2 * scale + grads[4] * scale,
                )

        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # Test gradcheck
        def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

        gradcheck(foo, (t1, t2, t3))

    def test_custom_function_no_tensors(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                return (args[0], args[1], None, args[2])

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_invalid_gradients(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return torch.randn(10, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, "expected shape"):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()

    def test_unrelated_inputs(self):
        # test to ensure grad(grad)check runs successfully even if there is an
        # unrelated (but differentiable) inputs

        def my_function(x, y):
            return x * x

        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)

        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

    def test_not_implemented_grad(self):
        a = torch.rand(2, requires_grad=True)
        # if grad for nextafter ends up being implemented, this should be changed
        y = torch.nextafter(a, a).sum()
        with self.assertRaisesRegex(
            NotImplementedError, "the derivative for .* is not implemented"
        ):
            y.backward()

    def test_not_implemented_fwad(self):
        x = torch.randn(3)
        v = torch.rand(3)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, v)

            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                # if forward AD ends up being implemented for torch.igamma, choose a different op
                torch.igamma(dual_x, dual_x)

    def test_saved_tensor_hooks_extra_exit_during_bw_no_crash(self):
        # This usage of saved tensor is not supported, but should not crash
        def unpack(x):
            ctx_1.__exit__()
            return x

        ctx_1 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, unpack)
        ctx_2 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x)

        for _ in range(10):
            with ctx_2:
                ctx_1.__enter__()
                x = torch.randn(3, 3, requires_grad=True)
                x.sin().sum().backward()

        # Clean up
        for _ in range(10):
            ctx_1.__exit__()

        # Validate there are no more hooks on the stack
        a = torch.tensor(1.0, requires_grad=True)
        y = a.exp()
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)

    def test_saved_tensor_hooks_extra_enter_during_bw_no_leak(self):
        # This usage of saved tensor is not supported, but should not leak
        def scope():
            def unpack(x):
                weak_ctx_1().__enter__()
                return x

            ctx_1 = torch.autograd.graph.saved_tensors_hooks(lambda x: x, unpack)
            weak_ctx_1 = weakref.ref(ctx_1)

            x = torch.randn(3, 3, requires_grad=True)
            with ctx_1:
                x.sin().sum().backward()
            return weakref.ref(unpack)

        with disable_gc():
            unpack_hook_ref = scope()
            self.assertIsNone(unpack_hook_ref())

    def test_will_engine_execute_node(self):
        counter = [0]

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, gO):
                return gO * 2

        def get_grad_fn(t):
            if t.requires_grad and t.grad_fn is None:
                return t.clone().grad_fn.next_functions[0][0]
            else:
                return t.grad_fn

        a = torch.randn(2, 3, 4, requires_grad=True)
        a2 = torch.randn(2, 3, 4, requires_grad=True)
        b = a * a2
        b2 = b.cos()
        c = MyFunction.apply(b)

        should_execute = list(map(get_grad_fn, (a, b, c)))
        should_not_execute = list(map(get_grad_fn, (a2, b2)))

        def fn(x):
            counter[0] += 1

            for g in should_execute:
                self.assertTrue(torch._C._will_engine_execute_node(g))

            for g in should_not_execute:
                self.assertFalse(torch._C._will_engine_execute_node(g))

        h1 = b.register_hook(fn)
        h2 = c.register_hook(fn)

        # .backward(inputs=) is OK
        out = c.sum()
        torch.autograd.backward(out, inputs=(a, b), retain_graph=True)
        self.assertEqual(counter[0], 2)

        # .backward() is OK
        should_execute = list(map(get_grad_fn, (a, a2, b, c)))
        should_not_execute = list(map(get_grad_fn, (b2,)))
        torch.autograd.backward(out, retain_graph=True)

        # .grad is NOT OK when leaf is passed (this is the current state, subject to change)
        with self.assertRaisesRegex(
            RuntimeError, "are currently running autograd.grad()"
        ):
            torch.autograd.grad(out, (a,))

        # .grad is OK when non-leaf is passed
        a = torch.randn(1, 2, 3, requires_grad=True) * 2
        b = a * 2

        def fn(x):
            # Check a non-leaf
            counter[0] += 1
            self.assertTrue(torch._C._will_engine_execute_node(b.grad_fn))

        h3 = b.register_hook(fn)
        counter[0] = 0
        torch.autograd.grad(b.sum(), (a,))
        self.assertEqual(counter[0], 1)

        # Verify other errors are raised
        with self.assertRaisesRegex(RuntimeError, "during the backward pass"):
            torch._C._will_engine_execute_node(out.grad_fn)

        with self.assertRaisesRegex(RuntimeError, "expects an grad_fn"):
            torch._C._will_engine_execute_node(out)

        # Ensure we don't leak memory
        h1.remove()
        h2.remove()
        h3.remove()

    def test_custom_function_vmap_defaults(self):
        class MySquare(Function):
            @staticmethod
            def forward(x):
                return x**2

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        self.assertFalse(MySquare.generate_vmap_rule)
        self.assertTrue(hasattr(MySquare, "vmap"))

    def test_custom_function_setup_context_simple(self):
        class MySquare(Function):
            @staticmethod
            def forward(x):
                return x**2

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        x = torch.randn([], requires_grad=True)
        y = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_output(self):
        # Multiple outputs with some non-Tensor outputs.
        class MySquare(Function):
            @staticmethod
            def forward(x):
                two_x = x.item() * 2
                return x**2, two_x

            @staticmethod
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                _, two_x = output
                ctx.two_x = two_x

            @staticmethod
            @once_differentiable
            def backward(ctx, gO, _):
                return gO * ctx.two_x

        x = torch.randn([], requires_grad=True)
        y, _ = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_input(self):
        class MyReshape(Function):
            @staticmethod
            def forward(x, shape, scale_forward, scale_backward):
                return x.reshape(shape) * scale_forward

            @staticmethod
            def setup_context(ctx, inputs, output):
                x, shape, scale_forward, scale_backward = inputs
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape

            @staticmethod
            def backward(ctx, gO):
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        class MyReshapeRef(Function):
            @staticmethod
            def forward(ctx, x, shape, scale_forward, scale_backward):
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape
                return x.reshape(shape) * scale_forward

            @staticmethod
            def backward(ctx, gO):
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        def test(x, shape, scale_forward, scale_backward):
            y = MyReshape.apply(x, shape, scale_forward, scale_backward).sum()
            (gx,) = torch.autograd.grad(y, x)

            y_expected = MyReshapeRef.apply(
                x, shape, scale_forward, scale_backward
            ).sum()
            (gx_expected,) = torch.autograd.grad(y_expected, x)

            self.assertEqual(y_expected, y)
            self.assertEqual(gx_expected, gx)

        test(torch.randn(24, requires_grad=True), (3, 8), 7, 11)
        test(torch.randn(2, 3, 4, requires_grad=True), (6, 4), -1, 2)

    def test_multiple_insert_removal_caching(self):
        torch._C._set_cached_tensors_enabled(True)
        try:
            x = torch.rand([4])

            torch._C._add_cached_tensor(x)
            self.assertTrue(torch._C._is_cached_tensor(x))

            torch._C._add_cached_tensor(x)
            torch._C._remove_cached_tensor(x)

            self.assertFalse(torch._C._is_cached_tensor(x))
        finally:
            torch._C._set_cached_tensors_enabled(False)

    def test_accumulate_grad(self):
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return x_grad, x_grad_clone

        # Accumulate in-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

        # Accumulate out-of-place when create_graph is True
        x_grad, x_grad_clone = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)

    def test_accumulate_grad_tensor_reference(self):
        def _test_grad_tensor(
            params_grad_tensor,
            backward_grad_tensor,
            should_preserve_reference,
            create_graph,
        ):
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(
                id(grad_saved) == id(params.grad), should_preserve_reference
            )

        for create_graph in (False, True):
            # Accumulate dense gradient to sparse gradient will change the `params.grad` reference
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.tensor([1.5, 1.5]),
                False,  # never accumulates in-place
                create_graph,
            )

            # Accumulate dense gradient to dense gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph,
            )

            # Accumulate sparse gradient to sparse gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                not create_graph,
                create_graph,
            )

    def test_accumulate_grad_with_zero_numel_grad(self):
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        assert c.shape == (4, 0)
        c.sum().backward()

        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad, x_grad + x_hv)
        self.assertEqual(y.grad, y_grad + y_hv)

        # Avoid leaking memory
        x.grad = None
        y.grad = None

    def test_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum],
            grad_outputs=[torch.ones(2, 2)],
            inputs=[x],
            create_graph=True,
        )
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # Avoid leaking memory
        x.grad = None
        y.grad = None

        # Test that grad_outputs and outputs have the same shape
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(
                outputs=[grad_sum],
                grad_outputs=[grad_out],
                inputs=[x],
                create_graph=True,
            )
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(
                str(error),
                "Mismatch in shape: grad_output[0] has a shape of "
                + str(grad_out.shape)
                + " and output[0] has a shape of "
                + str(grad_sum.shape)
                + ".",
            )

    def test_grad_to_node(self):
        def check_matches(out, inp):
            ref = torch.autograd.grad(out.sum(), inp)

            edge = torch.autograd.graph.get_gradient_edge(inp)
            new = torch.autograd.grad(out.sum(), edge)
            self.assertEqual(ref, new)

        # We need to ensure that our main types of Node work (regular cpp Nodes,
        # AccumulateGrad Nodes and custom Function)
        x = torch.rand(2, requires_grad=True)
        out = x.clone()
        check_matches(out, x)

        x = x.clone()
        out = x.clone()
        check_matches(out, x)

        x = torch.autograd._functions.Resize.apply(x, (2,))
        out = x.clone()
        check_matches(out, x)

        x = torch.var_mean(x)[1]
        out = x.clone()
        check_matches(out, x)

    def test_grad_to_node_set(self):
        x = torch.rand(2, requires_grad=True)
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        out = x.clone()

        with torch.no_grad():
            x.set_(torch.rand_like(x))

        with self.assertRaisesRegex(RuntimeError, "to not have been used in the graph"):
            torch.autograd.grad(out.sum(), x)

        # Works
        torch.autograd.grad(out.sum(), x_edge)

    def test_grad_to_node_inplace(self):
        x = torch.rand(2, requires_grad=True).clone()
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        x *= 2

        g_old, g_new = torch.autograd.grad(x.sum(), (x_edge, x))
        self.assertEqual(g_old, 2 * torch.ones_like(x))
        self.assertEqual(g_new, torch.ones_like(x))

    def test_grad_to_node_multi(self):
        x = torch.rand(2, requires_grad=True).clone()
        y = torch.rand(2, requires_grad=True).clone()

        out = x + y

        ref = torch.autograd.grad(out.sum(), (x, y))

        inp_edges = (
            GradientEdge(x.grad_fn, x.output_nr),
            GradientEdge(y.grad_fn, y.output_nr),
        )
        new = torch.autograd.grad(out.sum(), inp_edges)

        self.assertEqual(ref, new)

    def test_grad_to_node_materialize(self):
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        out = x.clone()

        # Works
        torch.autograd.grad(
            out.sum(), (edge_x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(
            out.sum(), (x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(out.sum(), (x, edge_y), allow_unused=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "materialize_grads cannot be used when the given input is a GradientEdge",
        ):
            torch.autograd.grad(
                out.sum(), (x, edge_y), allow_unused=True, materialize_grads=True
            )

    def test_backward_to_node(self):
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        out = x.clone()

        # All should work in this case
        torch.autograd.backward(out.sum(), inputs=(edge_x, y))
        torch.autograd.backward(out.sum(), inputs=(x, y))
        torch.autograd.backward(out.sum(), inputs=(x, edge_y))
        torch.autograd.backward(out.sum(), inputs=(edge_x, edge_y))

    def test_grad_fn_input_metadata(self):
        x = torch.rand(2, requires_grad=True, dtype=torch.float32)
        y = torch.rand(2, requires_grad=True, dtype=torch.float32)
        z = x * y
        z_metadata = z.grad_fn._input_metadata[0]
        self.assertEqual(z_metadata.shape, (2,))
        self.assertEqual(z_metadata.dtype, torch.float32)

        # Multiple outputs
        b = torch.rand(3, 3, requires_grad=True)
        var, _ = torch.var_mean(b, dim=0)

        metadata_0 = var.grad_fn._input_metadata[0]
        metadata_1 = var.grad_fn._input_metadata[1]
        self.assertEqual(metadata_0.shape, (3,))
        self.assertEqual(metadata_1.shape, (3,))

        # Preserves symints
        nt = torch.nested.nested_tensor(
            [torch.randn(3, 2), torch.randn(2, 2)],
            layout=torch.jagged,
            requires_grad=True,
        )

        nt_metadata = nt.clone().grad_fn._input_metadata[0]

        self.assertIsInstance(nt_metadata.shape[1], torch.SymInt)
        self.assertEqual(nt_metadata.shape, nt.shape)
        self.assertTrue(nt_metadata.is_nested_tensor)
        self.assertFalse(nt_metadata.is_cpp_nested_tensor)
        self.assertEqual(nt_metadata.dtype, nt.dtype)

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        x = torch.randn(3, 3, requires_grad=True)
        x = Test.apply(x)
        metadata = x.grad_fn._input_metadata[0]
        self.assertEqual(metadata.shape, (3, 3))

    def test_gradient_edge_output(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)

        def fn(x, reduce=True):
            tmp = x.sin().cos()
            if reduce:
                tmp = tmp.sum()
            out = tmp.exp().clone().sin().sum()
            tmp_edge = torch.autograd.graph.get_gradient_edge(tmp)
            return out, tmp_edge

        # Compute fn backward in two steps
        out, tmp_edge = fn(x)
        (tmp_grad,) = torch.autograd.grad(out, (tmp_edge,))

        (x_grad,) = torch.autograd.grad(tmp_edge, (x,), grad_outputs=(tmp_grad,))

        # Compare with as if we did it in one go.
        out, _ = fn(x)
        (x_grad_ref,) = torch.autograd.grad(out, (x,))
        self.assertEqual(x_grad, x_grad_ref)

        # Incorrect case: grad_outputs not passed/implicitly None and output is
        # not a scalar
        out, tmp_edge = fn(x, reduce=False)
        with self.assertRaisesRegex(
            RuntimeError, "grad can be implicitly created only for scalar output"
        ):
            torch.autograd.grad(tmp_edge, (x,))

        # grad_outputs is None, and output is a scalar is fine
        out, tmp_edge = fn(x, reduce=True)
        torch.autograd.grad(tmp_edge, (x,))

        # Incorrect case: grad_outputs wrong size
        out, tmp_edge = fn(x)
        with self.assertRaisesRegex(RuntimeError, "Mismatch in shape"):
            torch.autograd.grad(
                tmp_edge, (x,), grad_outputs=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )

        # Incorrect case: wrong dtype
        out, tmp_edge = fn(x)
        (tmp_grad,) = torch.autograd.grad(out, (tmp_edge,))
        with self.assertRaisesRegex(RuntimeError, "required to have the same dtype"):
            torch.autograd.grad(
                tmp_edge,
                (x,),
                grad_outputs=torch.rand_like(tmp_grad, dtype=torch.complex64),
            )

        # Run with .backward() and compare with .grad()
        out, tmp_edge = fn(x)
        torch.autograd.backward(tmp_edge, retain_graph=True)
        (x_grad_ref,) = torch.autograd.grad(tmp_edge, (x,), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # Pass a tuple of GradientEdges
        x.grad = None
        torch.autograd.backward((tmp_edge,), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # Mixing GradientEdge and Tensors
        out1, tmp_edge1 = fn(x)
        out2, tmp_edge2 = fn(x)
        (x_grad_ref,) = torch.autograd.grad((tmp_edge1, out2), (x,), retain_graph=True)
        x.grad = None
        torch.autograd.backward((tmp_edge1, out2), retain_graph=True)
        self.assertEqual(x.grad, x_grad_ref)

        # .backward(): wrong shape
        out, tmp_edge = fn(x)
        with self.assertRaisesRegex(RuntimeError, "Mismatch in shape"):
            torch.autograd.backward(
                tmp_edge, inputs=(x,), grad_tensors=torch.tensor([1.0, 2.0, 3.0, 4.0])
            )

    def test_gradient_edge_graph_ownership(self):
        # Ensure we own the graph properly
        class Clone(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gX):
                return gX.clone()

        inp = torch.rand(1, requires_grad=True).clone()

        # C++ Node
        out = inp.clone()
        edge = torch.autograd.graph.get_gradient_edge(out)
        torch.autograd.backward(edge)
        del out
        torch.autograd.backward(edge)

        # python Node
        out = Clone.apply(inp)
        edge = torch.autograd.graph.get_gradient_edge(out)
        torch.autograd.backward(edge)
        del out
        torch.autograd.backward(edge)

    def test_grad_nonleaf(self):
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            return x**2 + y * x + y**2

        for _ in range(5):
            (grad_x,) = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True
            )

            grad_x_expected = 2 * x + y
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x, grad_x_expected)

            x = x + 0.05 * grad_x

        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
        self.assertGreater(val_final, val_init)

        x.backward(grad_output)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(x_init.grad)

    def test_grad_nonleaf_many_outputs(self):
        # This checks an edge case for function callbacks
        # We want to capture two grads of a function, but can only
        # register a single callback.
        x = torch.randn(4, 2, requires_grad=True)
        a, b = x.chunk(2)

        def hook(*grads):
            hook_called[0] = True

        hook_called = [False]
        x.register_hook(hook)

        go = torch.randn(2, 2)
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True
        )

        self.assertEqual(grad_a, go)
        self.assertEqual(grad_b, go * 2)
        self.assertFalse(hook_called[0])
        self.assertIsNone(x.grad)

    def test_grad_nonleaf_register_hook(self):
        # This checks an edge case for register_hook.
        # We want to capture grad of a nonleaf tensor,
        # but avoid segfault during backward of other nonleaf tensors
        x = torch.randn(5, requires_grad=True)
        x_list = x.unbind()

        x0 = x_list[0]
        hook_results = [None]

        def hook(grad):
            hook_results[0] = grad

        x0.register_hook(hook)

        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.0))
        expected_grad = torch.tensor([1.0, 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)

        for i in range(1, 5, 1):
            x_list[i].backward()
            self.assertEqual(hook_results[0], None)
            expected_grad[i] = 1.0
            self.assertEqual(x.grad, expected_grad)
            self.assertIsNone(x_list[i].grad)

    def test_grad_materialize_grads(self):
        x = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        y = x * a
        dydx = torch.autograd.grad(y, x, create_graph=True)
        d2ydx2_none = torch.autograd.grad(dydx, x, create_graph=True, allow_unused=True)
        d2ydx2 = torch.autograd.grad(
            dydx, x, create_graph=True, allow_unused=True, materialize_grads=True
        )
        # `allow_unused` set to True implicitly
        d3ydx3 = torch.autograd.grad(d2ydx2, x, materialize_grads=True)
        self.assertIsNone(d2ydx2_none[0])
        self.assertEqual(d2ydx2[0].item(), 0)
        self.assertEqual(d3ydx3[0].item(), 0)
        with self.assertRaisesRegex(
            ValueError, "Expected allow_unused to be True or not passed when"
        ):
            torch.autograd.grad(y, x, allow_unused=False, materialize_grads=True)

    def test_post_accumulate_grad_hook_on_non_leaf(self):
        def hook(tensor):
            tensor.sub_(1.0)

        leaf = torch.rand(3, requires_grad=True)
        non_leaf = 2.0 * leaf

        with self.assertRaisesRegex(
            RuntimeError,
            "post accumulate grad hooks cannot be registered on non-leaf tensors",
        ):
            non_leaf.register_post_accumulate_grad_hook(hook)

    def test_post_accumulate_grad_hook_multiple_hooks(self):
        def hook1(tensor):
            tensor.sub_(tensor.grad)

        def hook2(tensor):
            tensor.mul_(4.0)

        tensor = torch.rand(3, requires_grad=True)
        tensor_ref = tensor.detach().clone()
        tensor.register_post_accumulate_grad_hook(hook1)
        tensor.register_post_accumulate_grad_hook(hook2)
        sum = tensor.sum()
        sum.backward()
        # both hooks should be called, in order
        self.assertEqual(4.0 * (tensor_ref - 1.0), tensor)

    def test_post_accumulate_grad_hook_multiple_tensors(self):
        def hook(tensor):
            tensor.sub_(tensor.grad)

        tensor1 = torch.rand(3, requires_grad=True)
        tensor1_ref = tensor1.detach().clone()
        tensor2 = torch.rand(5, requires_grad=True)
        tensor2_ref = tensor2.detach().clone()
        tensor1.register_post_accumulate_grad_hook(hook)
        tensor2.register_post_accumulate_grad_hook(hook)
        tensor1.sum().backward()
        tensor2.sum().backward()
        # both tensors should have been modified
        self.assertEqual(tensor1_ref - 1.0, tensor1)
        self.assertEqual(tensor2_ref - 1.0, tensor2)

    def test_post_accumulate_grad_hook_returns_not_None(self):
        def bad_hook(tensor):
            return tensor.grad

        tensor = torch.rand(2, 3, requires_grad=True)
        tensor.register_post_accumulate_grad_hook(bad_hook)
        # should error!
        with self.assertRaisesRegex(RuntimeError, "hooks should return None."):
            tensor.sum().backward()

    def test_post_accumulate_grad_hook_e2e(self):
        def setup_optim_in_bwd(model):
            optims = {}
            handles = []

            def optim_step_hook(param):
                optims[param].step()
                optims[param].zero_grad()

            for p in model.parameters():
                optims[p] = torch.optim.Adam([p])
                handles.append(p.register_post_accumulate_grad_hook(optim_step_hook))

            return handles

        model = torch.nn.Linear(3, 2)
        input = torch.rand(2, 3)
        handles = setup_optim_in_bwd(model)

        # make a copy for reference
        model_copy = deepcopy(model)
        optim_copy = torch.optim.Adam(model_copy.parameters())

        iters = 5

        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()

            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()

        params_copy = []  # freeze a copy of the params to compare later
        for p_reference, p in zip(model_copy.parameters(), model.parameters()):
            self.assertEqual(p_reference, p)
            params_copy.append(p_reference.detach().clone())

        # After removing the handle, the model should no longer update.
        for h in handles:
            h.remove()

        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()

            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()

        for p_static, p_reference, p in zip(
            params_copy, model_copy.parameters(), model.parameters()
        ):
            self.assertEqual(p_static, p)
            self.assertNotEqual(p_reference, p)

    def test_post_accumulate_grad_hook_gets_cleaned_up(self):
        def fun_stuff_with_hook():
            thing_to_put_in_hook = torch.rand(3)

            def hook(tensor):
                tensor.sub_(tensor.grad)
                tensor.add_(thing_to_put_in_hook)

            tensor = torch.rand(3, requires_grad=True)
            tensor.register_post_accumulate_grad_hook(hook)
            tensor.sum().backward()
            ref = weakref.ref(thing_to_put_in_hook)
            gc.collect()
            return tensor, ref

        with disable_gc():
            tensor, ref = fun_stuff_with_hook()
            self.assertIsNotNone(
                ref()
            )  # thing_to_put_in_hook should be kept alive by tensor

            del tensor
            gc.collect()
            self.assertIsNone(ref())  # thing_to_put_in_hook should be cleaned

    def test_post_accumulate_grad_hook_ordering(self):
        tensor = torch.rand(3, requires_grad=True)

        def pre_hook(grad):
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            tensor.grad.add_(0.5)

        def acc_grad_node_post_hook(grad_in, grad_out):
            tensor.grad = grad_out[0].mul(10)

        acc_grad = tensor.view_as(tensor).grad_fn.next_functions[0][0]
        tensor.register_hook(pre_hook)
        acc_grad.register_prehook(acc_grad_node_pre_hook)
        tensor.register_post_accumulate_grad_hook(post_acc_grad_hook)
        acc_grad.register_hook(acc_grad_node_post_hook)
        tensor.sum().backward()

        # the hooks should run in the order of:
        #   1. tensor prehook
        #   2. acc_grad prehook
        #   3. tensor post acc_grad hook
        #   4. acc_grad posthook
        # so that would be ((1 - 2) / 5 + 0.5) * 10 = 3
        self.assertEqual(torch.tensor([3.0, 3.0, 3.0]), tensor.grad)

    def test_hook_with_no_name(self):
        # Create a hook that do not have a __name__ attribute
        class MyHookClass:
            def __call__(self, grad):
                return grad.clone()

        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # Should run fine

    def test_prehook_ordering(self):
        # Hooks registered to tensor are ordered before those
        # that are registered to grad_fn
        log = []

        def hook1(g):
            log.append(1)
            return g * 3

        def hook2(gs):
            log.append(2)
            return tuple(g * 2 for g in gs)

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()

        b.grad_fn.register_prehook(hook2)
        b.register_hook(hook1)
        b.grad_fn.register_prehook(hook2)

        acc = b.grad_fn.next_functions[0][0]
        a.register_hook(hook1)
        acc.register_prehook(hook2)
        a.register_hook(hook1)

        b.sum().backward(retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1, 2])

        # grad also runs hooks on accumulate grad nodes, even though
        # the accumulate grad nodes are not actually executed
        log = []
        torch.autograd.grad(b.sum(), inputs=(a,), retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1])

        log = []
        b.sum().backward(inputs=(b,))
        self.assertEqual(log, [1, 2, 2])
        # retains_grad hooks would not observe modifications by all pre hooks
        # because they are executed after
        self.assertEqual(b.grad.item(), 3)

    def test_retains_grad_can_always_observe_tensor_prehook(self):
        def tensor_prehook(g):
            return g * 2

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.register_hook(tensor_prehook)
        b.retain_grad()
        b.register_hook(tensor_prehook)

        b.clone().backward()
        self.assertEqual(b.grad.item(), 4)

        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.retain_grad()
        b.register_hook(tensor_prehook)

        b.clone().backward()
        self.assertEqual(b.grad.item(), 2)

    def test_accumulate_grad_posthooks_can_observe_tensor_prehook(self):
        # Post hooks on accumulate should be able to observe changes to
        # grad made by tensor prehooks
        a = torch.tensor(1.0, requires_grad=True)

        def tensor_prehook(g):
            return g * 2

        def posthook(gO, gI):
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gO), 0)

        def prehook(gI):
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gI), 1)

        b = a.clone()
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        a.register_hook(tensor_prehook)

        b.backward()

    def test_accumulate_grad_posthooks_should_not_execute(self):
        def tensor_prehook(g):
            raise RuntimeError

        def posthook(gO, gI):
            raise RuntimeError

        a = torch.tensor(1.0, requires_grad=True)
        a.register_hook(tensor_prehook)
        b = torch.tensor(1.0, requires_grad=True)
        c = a.clone()
        acc = c.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)

        out = a + b + c
        out.sum().backward(inputs=[b])

    def test_hook_edge_case_when_called_with_grad(self):
        # grad executes the tensor hooks of the next node but not
        # grad_fn pre hooks or the post hooks
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        c = b * 2

        tensor_hook_count = [0]
        prehook_count = [0]
        posthook_count = [0]

        def reset_counts():
            nonlocal tensor_hook_count, prehook_count, posthook_count
            tensor_hook_count = [0]
            prehook_count = [0]
            posthook_count = [0]

        def tensor_prehook(g):
            tensor_hook_count[0] += 1

        def prehook(g):
            prehook_count[0] += 1

        def posthook(gI, gO):
            posthook_count[0] += 1

        a.register_hook(tensor_prehook)
        b.register_hook(tensor_prehook)
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        b.grad_fn.register_hook(posthook)
        b.grad_fn.register_prehook(prehook)

        torch.autograd.grad(c, inputs=(b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 1)
        self.assertEqual(posthook_count[0], 0)
        self.assertEqual(prehook_count[0], 0)
        reset_counts()

        torch.autograd.grad(c, inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 1)
        self.assertEqual(prehook_count[0], 1)
        reset_counts()

        c.backward(retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)
        reset_counts()

        c.backward(inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)

    def test_sharded_grad(self):
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]
        intermediates = [l * i + l * l for i, l in enumerate(leaves)]
        loss = sum(v * i for i, v in enumerate(intermediates)).sum()

        # define a helper for dividing intermediates into groups
        def group(l, group_size):
            return (l[i : i + group_size] for i in range(0, len(l), group_size))

        # Compute the d loss / d intermediates in chunks of shard_size
        shard_size = 2
        d_intermediates = [
            d_i
            for intermediates_batch in group(intermediates, shard_size)
            for d_i in torch.autograd.grad(loss, intermediates_batch)
        ]
        # Compute rest of backward pass
        torch.autograd.backward(intermediates, d_intermediates)

        for i, l in enumerate(leaves):
            self.assertEqual(l.grad, i * i * (1 + l))

    def test_backward_badcalls(self):
        x = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            x.backward()

    def test_grad_badcalls(self):
        x = torch.ones(1)
        y = x**2
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(x, y)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(y, x)

        x = torch.ones(1, requires_grad=True)
        y = x**2
        torch.autograd.grad(y, x)  # this should succeed now

    def test_grad_empty_inputs(self):
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "grad requires non-empty inputs."):
            torch.autograd.grad(2 * x, [], grad_outputs=torch.tensor([1.0]))

    def test_grad_fn_badcalls(self):
        error_regex = "expected .* arguments, got .* instead"
        x = torch.ones(1, requires_grad=True)
        y = x**2
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())  # too many
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()  # too few

        y.grad_fn(x.detach())  # this should succeed

    def test_grad_unreachable(self):
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # Make sure x and y have grad accumulators allocated
        z = x * 2
        w = y * 2

        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # This is slightly different than the case above, because z doesn't even
        # have a grad accumulator allocated.
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

        # allow_unused=False, but grads contains None inside, should throw
        with self.assertRaisesRegex(RuntimeError, "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

    def test_grad_unreachable_discovery(self):
        # Test that certain nodes are not erroneously executed when an input
        # is unreachable. See #39784
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                self.fail("This node should not be executed!")

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        (gY,) = torch.autograd.grad(x, (y,), allow_unused=True)
        self.assertIsNone(gY)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y,))  # allow_unused is implicitly True!
        self.assertIsNone(y.grad)

    def test_grad_batched_grad(self):
        x = torch.randn(2, 2, requires_grad=True)

        out = x.clone()  # Size([2, 2])
        batched_grad = (
            torch.arange(3).expand(2, 2, 3).transpose(0, 2)
        )  # Size([3, 2, 2])
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # Detect shape mismatch
        grad_out = torch.ones(2, 2)
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

        # Scalar outputs
        out = x.sum()  # Size([])
        batched_grad = torch.arange(3)  # Size([3])
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # We consider scalar and sized-1 to be a mismatch. This is consistent with current non-batched behavior.
        grad_out = torch.ones(2).unsqueeze(1)
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

    def test_hooks(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

        counter = [0]

        def bw_hook(inc, grad):
            self.assertIsInstance(grad, torch.Tensor)
            counter[0] += inc

        z = x**2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)

        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def _get_mul2(self, use_custom_function):
        if use_custom_function:

            class Mul2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x * 2

                @staticmethod
                def backward(ctx, gO):
                    return gO * 2

            return Mul2.apply
        else:
            return lambda x: x * 2

    def test_grad_fn_prehooks(self):
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)

            a = torch.tensor([1.0], requires_grad=True)
            b = mul2(a)

            post_counter = [0]
            pre_counter = [0]

            def posthook(grad_input, grad_output):
                self.assertEqual(pre_counter[0], 3)
                self.assertTrue(torch.allclose(grad_output[0], torch.ones(1) * 8))
                self.assertTrue(torch.allclose(grad_input[0], torch.ones(1) * 16))
                post_counter[0] += 1
                return grad_input

            def prehook(grad_output):
                pre_counter[0] += 1
                return (grad_output[0] * 2,)

            # register posthook x 2
            b.grad_fn.register_hook(posthook)
            b.grad_fn.register_hook(posthook)
            # register prehook x 3
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: None)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: x)
            b.grad_fn.register_prehook(lambda x: None)

            b.sum().backward()

            self.assertEqual(post_counter[0], 2)
            self.assertEqual(pre_counter[0], 3)

            # Return None
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)

            def prehook(grad_output):
                pre_counter[0] += 1
                return None

            b.grad_fn.register_prehook(prehook)
            b.sum().backward()
            self.assertEqual(pre_counter[0], 4)
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))

    def test_grad_fn_prehooks_multiple_outputs(self):
        # Compute gradients without hooks
        b = torch.rand(3, 3, requires_grad=True)
        var, mean = torch.var_mean(b, dim=0)
        (var + mean).sum().backward()

        # Compute gradients with hooks
        a = b.detach().requires_grad_()
        counter = [0]

        def prehook(grad_output):
            gvar, gmean = grad_output
            counter[0] += 1
            return (gvar * 2, gmean * 2)

        var, mean = torch.var_mean(a, dim=0)
        mean.grad_fn.register_prehook(prehook)
        (var + mean).sum().backward()

        self.assertEqual(counter[0], 1)
        # Compare
        self.assertTrue(torch.allclose(a.grad, b.grad * 2))

        # Test with custom Function
        class DoubleMul2(Function):
            @staticmethod
            def forward(ctx, x, a, y):
                ctx.a = a
                return a * x * 2, a, a * y * 2

            @staticmethod
            def backward(ctx, g1, _a, g2):
                return ctx.a * g1 * 2, None, ctx.a * g2 * 2

        counter = [0]

        def prehook(grad_output):
            g1, ga, g2 = grad_output
            self.assertIsNone(ga)
            counter[0] += 1
            return (g1 * 2, None, g2 * 2)

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        k = 3
        c, _, d = DoubleMul2.apply(a, k, b)
        c.grad_fn.register_prehook(prehook)
        (c + d).sum().backward()

        self.assertEqual(counter[0], 1)
        self.assertTrue(torch.allclose(a.grad, torch.ones(1) * 4 * k))
        self.assertTrue(torch.allclose(b.grad, torch.ones(1) * 4 * k))

    def test_grad_fn_prehooks_remove_hooks(self):
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)

            # Simply remove hooks

            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook(grad_output):
                counter[0] += 1
                return None

            handle = b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            handle.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

            # Remove hooks during backward
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook1(grad_output):
                handle2.remove()
                # Remove hook that is already removed is OK
                handle3.remove()
                return None

            def prehook2(grad_output):
                counter[0] += 1
                return None

            # Hooks that registered first run first
            b.grad_fn.register_prehook(prehook1)
            handle2 = b.grad_fn.register_prehook(prehook2)
            handle3 = b.grad_fn.register_prehook(prehook2)
            handle3.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

    def test_node_post_hook_registered_during_unpack_hook(self):
        """
        Test that post hooks registered during one of the node's
        unpack hooks are properly restricted and will run properly.
        """
        test_case = self

        class RegisterPostNodeHook(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self) -> None:
                def pack_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    return tensor

                def unpack_tensor(tensor: torch.Tensor) -> torch.Tensor:
                    node = torch._C._current_autograd_node()

                    def hook(outputs, inputs):
                        # Assert that inputs passed in are None
                        test_case.assertTrue(all(i is None for i in inputs))
                        halved_outputs = tuple(
                            o / 2.0 if o is not None else None for o in outputs
                        )
                        return halved_outputs

                    node.register_hook(hook)
                    return tensor

                super().__init__(pack_tensor, unpack_tensor)

        a = torch.rand(3, 3, requires_grad=True)

        def model():
            var, mean = torch.var_mean(a, dim=0)
            loss = (var + mean).sum()
            loss.backward()

        model()
        ref_grad = a.grad.clone()

        with RegisterPostNodeHook():
            model()

        # Verify that the post hook got called and the grad propagation worked
        self.assertEqual(ref_grad / 2.0 + ref_grad, a.grad)

    def test_hooks_cpp(self):
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()

        self.assertEqual(counter[0], 1, msg="bw_hook not called")
        self.assertEqual(
            x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0
        )

    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                assert ctx.needs_input_grad[0]
                assert not ctx.needs_input_grad[1]
                return x, y

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                return grad_x, None

        was_called = [False]

        def hook(grad):
            self.assertIsNotNone(grad)
            was_called[0] = True

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        rx, ry = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()

        # It should be possible to call retain_grad() multiple times
        h1.retain_grad()
        h1.retain_grad()

        # Gradient should be accumulated
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    # NB: See test/cpp/api/autograd.cpp for more tests on the interaction between
    #     retains_grad and hooks in cpp
    def test_retain_grad_inplace(self):
        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        # Inplace multiple times is OK
        a.mul_(2)
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # When in-place over view is done, the retains_grad hooks should be
        # moved from base's original grad_fn to the copyslices node.
        x = torch.tensor([1.0], requires_grad=True).clone()
        x.retain_grad()
        x_view = x[:]
        x_view *= 2
        x *= 2
        x.sum().backward()
        # The grad is 1, not 4, because we are computing grad wrt the latest
        # version of x.
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # If the base did not originally require grad, there should be no hook
        # to move. Make sure this case runs without error.
        x = torch.zeros(4)
        y = x.view(2, 2)
        y.add_(torch.randn(2, 2, requires_grad=True))

    def test_retains_grad_inplace_multiple_outputs(self):
        class DoubleMul(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, g1, g2):
                return g1 * 2 + g2 * 3

        var_mean = partial(torch.var_mean, dim=0)

        for fn in (DoubleMul.apply, var_mean):
            b = torch.rand(3, 3, requires_grad=True)
            var, mean = fn(b)
            var.retain_grad()
            mean.retain_grad()
            # node has two retains_grad hooks
            var.mul_(2)
            # the retain_grad hook multi-output node refers should now be a nullptr
            (var + mean).sum().backward()
            gvar = var.grad
            gmean = mean.grad

            a = b.detach().requires_grad_(True)
            var, mean = fn(a)
            var.mul_(2)
            out = (var + mean).sum()
            gvar_expected, gmean_expected = torch.autograd.grad(out, inputs=(var, mean))
            self.assertTrue(torch.allclose(gvar, gvar_expected))
            self.assertTrue(torch.allclose(gmean, gmean_expected))

    def test_retain_grad_inplace_over_view(self):
        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.retain_grad()
        view2.retain_grad()
        view.mul_(2)
        (view + view2).sum().backward()

        # The old grad_fn, slice, wouldn't be part of the graph during backward
        # so if the retains grad were not properly updated to the new grad_fn,
        # the grad would still be None
        self.assertEqual(view.grad, view2.grad)
        self.assertEqual(view.grad, torch.tensor([1.0]))

    def test_tensor_hooks_inplace(self):
        # Check that the second hook gets registered to the new version of tensor
        count1 = [0]
        count2 = [0]

        def fn1(grad):
            count1[0] += 1
            # x2 from mul, x2 from fn2
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        def fn2(grad):
            count2[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))
            return grad * 2

        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn1)
        b.mul_(2)
        b.register_hook(fn2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(count2[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

        count3 = [0]

        def fn3(grad):
            count3[0] += 1
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn3)
        # Inplace multiple times is OK
        b.mul_(2)
        b.mul_(2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

    def test_tensor_hooks_inplace_multiple_outputs(self):
        class DoubleMul(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 3

            @staticmethod
            def backward(ctx, g1, g2):
                return g1 * 2 + g2 * 3

        var_mean = partial(torch.var_mean, dim=0)

        for fn in (DoubleMul.apply, var_mean):
            counts = [0, 0, 0]

            def fn0(grad):
                counts[0] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 2)

            def fn1(grad):
                counts[1] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 3)

            def fn2(grad):
                counts[2] += 1
                self.assertEqual(grad, torch.ones_like(out1))

            b = torch.rand(3, 3, requires_grad=True)
            out1, out2 = fn(b)
            h1 = out1.register_hook(fn0)
            h2 = out2.register_hook(fn1)
            # node refers to two hook dicts
            # out1 no longer no longer points to its old hook dict
            out1.mul_(2)
            # fn2 is registered to out1's new hook dict
            h3 = out1.register_hook(fn2)
            (out1 + out2 * 3).sum().backward()
            self.assertEqual(counts, [1, 1, 1])

            # Avoid leaking memory
            h1.remove()
            h2.remove()
            h3.remove()

    def test_tensor_hooks_inplace_over_view(self):
        # There might be a better UX here, but this is the way it is now
        count = [0]

        def fn0(grad):
            self.fail()

        def fn1(grad):
            self.fail()

        def fn2(grad):
            count[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))

        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.register_hook(fn0)
        view2.register_hook(fn1)
        view.mul_(2)
        # We need to explicitly trigger an update to view to update its grad_fn
        view2.grad_fn
        view2.register_hook(fn2)
        (view + view2).sum().backward()
        # The hooks originally registered to view are not fired, one must explicitly
        # trigger an update to the view's grad_fn, and then register a new hook
        self.assertEqual(count[0], 1)

    def test_retain_grad_cycle(self):
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            y = x * 2
            y.retain_grad()

            return y / 2, torch._C._WeakTensorRef(y)

        z, ref = run_test()
        self.assertTrue(ref.expired())
        z.sum().backward()

    def test_backward(self):
        v = torch.randn(5, 5, requires_grad=True)
        x = torch.randn(5, 5, requires_grad=True)
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        z = torch.randn(5, 5, requires_grad=True)
        grad_output = torch.randn(5, 5)

        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        a = x + (y * z) + 4 * z**2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_to_sparse_backward(self):
        to_attr_names = (
            "to_dense",
            "to_sparse",
            "to_sparse_csr",
            "to_sparse_csc",
            "to_sparse_bsr",
            "to_sparse_bsc",
        )
        to_params = ((), (), (), (), (2,), (2,))
        to_attr_names_params = dict(zip(to_attr_names, to_params))

        def check_inversion_possible(
            t, layout1, layout1_params, layout2, layout2_params
        ):
            l = (layout1, layout2)
            p = (layout1_params, layout2_params)
            for l1, l2, p1, p2 in ((*l, *p), (*l[::-1], *p[::-1])):
                try:
                    to_l1 = getattr(t, l1)(*p1)
                    to_l2 = getattr(to_l1, l2)(*p2)
                except RuntimeError:
                    return False

            return True

        self_strided = torch.rand(4, 4, dtype=torch.double) + 1
        grad_strided = torch.rand(4, 4, dtype=torch.double) + 1

        for from_to_attr in to_attr_names:
            from_params = to_attr_names_params[from_to_attr]
            self_from = getattr(self_strided, from_to_attr)(
                *from_params
            ).requires_grad_(True)

            for to_to_attr in to_attr_names[1:]:
                to_params = to_attr_names_params[to_to_attr]

                if check_inversion_possible(
                    self_strided, from_to_attr, from_params, to_to_attr, to_params
                ):
                    self_to = getattr(self_from, to_to_attr)(*to_params)
                    grad_to = getattr(grad_strided, to_to_attr)(*to_params)

                    # No gradcheck support for BSR/BSC, so the grads are checked explicitly
                    grad_res = torch.autograd.grad(self_to, self_from, grad_to)[0]

                    self.assertEqual(grad_res.layout, self_from.layout)
                    self.assertEqual(grad_res.to_dense(), grad_strided)

    def test_sparse_mm_backward(self):
        size = (3, 3)

        mm_test_cases = product(*(([False, True],) * 4))

        for a_req_grad, a_is_sparse, b_req_grad, b_is_sparse in mm_test_cases:
            # We should only be testing cases with sparse inputs, and at least one
            # input needs to require grad so we can call a backward pass
            if not ((a_is_sparse or b_is_sparse) and (a_req_grad or b_req_grad)):
                continue
            a = torch.randn(size)
            if a_is_sparse:
                # detaching as `a` needs to be a leaf
                a = a.to_sparse().detach()
            b = torch.randn(size)
            if b_is_sparse:
                # detaching as `b` needs to be a leaf
                b = b.to_sparse().detach()

            a = a.requires_grad_(a_req_grad)
            b = b.requires_grad_(b_req_grad)

            r = a.mm(b)
            s = r.sum().backward()
            a_grad = None if a.grad is None else a.grad.detach().clone()
            b_grad = None if b.grad is None else b.grad.detach().clone()

            # Redo with only dense tensors
            a = (
                (a.to_dense() if a.is_sparse else a)
                .clone()
                .detach()
                .requires_grad_(a_req_grad)
            )
            b = (
                (b.to_dense() if b.is_sparse else b)
                .clone()
                .detach()
                .requires_grad_(b_req_grad)
            )

            r = a.mm(b)
            r.sum().backward()

            self.assertEqual(a_grad, a.grad)
            self.assertEqual(b_grad, b.grad)

    def test_multi_backward(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q = torch.randn(5, 5, requires_grad=True)

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        torch.autograd.backward([z, c], [grad_z, grad_c])

        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)

        z = x + y
        q = y * 2

        # NB: we currently raise an exception if any arguments to backwards
        # have requires_grad=False and don't have a grad_fn. We may want to
        # relax that check to a warning.
        def call_backwards():
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])

        self.assertRaises(RuntimeError, call_backwards)

    def test_backward_with_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        def fn():
            return x**2 + y * x + y**2

        gradient = torch.ones(2, 2)
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        @torch.no_grad()
        def reset_grad():
            x.grad.zero_()
            y.grad.zero_()

        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, y_grad_expected)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[x])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[y])
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=y)
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        self.assertRaisesRegex(
            RuntimeError,
            "cannot be empty",
            lambda: torch.autograd.backward(fn(), gradient, inputs=[]),
        )

    def test_backward_with_scalar_input(self):
        x = torch.randn([], dtype=torch.double, requires_grad=True)
        out = x**2
        out.backward(inputs=x)
        self.assertEqual(x.grad, 2 * x)

    def test_backward_with_nonleaf_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        out = x_nonleaf**2 + y * x_nonleaf + y**2

        out.backward(
            torch.ones(2, 2, dtype=torch.double),
            create_graph=True,
            inputs=[x, y, x_nonleaf],
        )
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y
        x_non_leaf_expected = 2 * x_nonleaf + y

        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(x_nonleaf.grad, x_non_leaf_expected)

        # backward doesn't have an allow_unused flag, so the behavior of backward
        # when variable is not part of the graph is as if allow_used were true
        # x.grad will simply be None.
        out.backward(
            torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z]
        )
        self.assertIsNone(z.grad)

        # Avoid leaking memory
        x.grad = None
        y.grad = None
        x_nonleaf.grad = None

    def test_dependent_backward(self):
        x = torch.randn(10, requires_grad=True)
        y = x**2
        z = y**3

        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])

        xd = x
        self.assertEqual(x.grad, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

    def test_save_output_nr(self):
        x = torch.randn(10, requires_grad=True)

        class MultiOutputFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:5], x[5:]

            @staticmethod
            def backward(ctx, *grad):
                return torch.cat(grad)

        a, b = MultiOutputFn.apply(x)
        self.assertEqual(b.output_nr, 1)

        class TestFn(Function):
            @staticmethod
            def forward(ctx, b):
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                (b,) = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)

        TestFn.apply(b).sum().backward()

    def test_first_grad_fn_access_in_no_grad_mode(self):
        a = torch.tensor([1 + 1j], requires_grad=True).clone()
        v = a.real
        a.add_(1)
        with torch.autograd.grad_mode.no_grad():
            v.grad_fn

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph(self):
        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build a "chain" computation graph
            for _ in range(depth):
                y = y + y * 0.000001

            # graph deletion occurs when the above locals go out of scope.
            # In this case `del y` will trigger it but it's easier to leave
            # it to Python to delete the locals.

        # Should not stack overflow
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_complicated(self):
        def scope():
            depth = 100000
            randchoice = torch.randint(2, [depth, 2])
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # Hold the two previous values
            prev_values = [None, None]

            # Build a "chain with skip connections" graph
            for _ in range(depth):
                prev_tensors = [
                    tensor for tensor in prev_values[:-1] if tensor is not None
                ]
                prev_values.append(y)
                prev_values.pop(0)

                # Definitely pick one tensor to add
                y += y * 0.000001

                # Possibly add other tensors
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_pyfunction(self):
        class MyOp(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build deeply nested computation graph
            for _ in range(depth):
                y = MyOp.apply(y, y)

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_no_unnecessary_save(self):
        # If we kept x in the derivative Function of x * 2 we would
        # get an error in the backward that would complain that we've
        # modified x, which was needed for gradient computation.
        # Since we should elide unnecessary saves, this test should pass.
        mu = torch.ones(1, requires_grad=True)
        x = torch.empty(1)
        loss = 0
        for i in range(3):
            x.detach_()
            x.copy_(mu + i)
            ft = torch.tensor([float(i)])
            multiplied = x * ft
            s = multiplied.sum()
            loss += s
        loss.backward()

    def test_no_grad(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        with torch.no_grad():
            w = x + y

        def adder(x, y):
            return x + y

        adders = [torch.no_grad()(adder), torch.no_grad(adder)]

        for adder in adders:
            z = adder(x, y)

            self.assertFalse(w.requires_grad)
            self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
            self.assertIsNone(w.grad_fn)
            self.assertFalse(z.requires_grad)
            self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))
            self.assertIsNone(z.grad_fn)

        # test nested decorator and with-statement on no_grad
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            w = adder(x, y)
            self.assertFalse(torch.is_grad_enabled())

    def test_enable_grad_decorator_no_paren(self):
        x = torch.ones(1, requires_grad=True)

        @torch.enable_grad
        def doubler(x):
            return x * 2

        with torch.no_grad():
            z = doubler(x)
        self.assertTrue(z.requires_grad)

    def test_set_grad_generator_functions(self):
        @torch.no_grad()
        def gen_no_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i

        with torch.enable_grad():
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i

        with torch.no_grad():
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        # enable_grad_decorator_recursive and no_grad_decorator_recursive call each other
        # recursively, to ensure that the decorators preserve the caller's setting
        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        # enable_grad_context_manager_recursive and no_grad_context_manager_recursive call
        # each other recursively, to ensure that the decorators preserve the caller's setting
        def enable_grad_context_manager_recursive(depth):
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_coroutines(self):
        @torch.no_grad()
        def coro_no_grad(n=10):
            self.assertFalse(torch.is_grad_enabled())
            for i in range(n):
                self.assertFalse(torch.is_grad_enabled())
                r = yield i
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertFalse(torch.is_grad_enabled())

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            self.assertTrue(torch.is_grad_enabled())
            for i in range(n):
                self.assertTrue(torch.is_grad_enabled())
                r = yield i
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertTrue(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            coro, r = coro_no_grad(), None
            try:
                while True:
                    self.assertTrue(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertTrue(torch.is_grad_enabled())

            except StopIteration:
                pass

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            coro, r = coro_enable_grad(), None
            try:
                while True:
                    self.assertFalse(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertFalse(torch.is_grad_enabled())

            except StopIteration:
                pass

    def test_set_grad_coroutines_benign_exceptions(self):
        class RecoverableException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True

        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        class UnrecoverableException(Exception):
            pass

        class SecondaryException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException from None

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException from None

        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

    def test_set_grad_coroutines_exit(self):
        @torch.no_grad()
        def coro_no_grad(state):
            for i in range(10):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertFalse(torch.is_grad_enabled())
                    state.add("GeneratorExit")
                    raise

        @torch.enable_gr

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 197 class(es): TestAutograd, Foo, Foo2, MyFunction, MyFunction, MyFunction, MyFunction, MyFunction, MyFunction, Func, MyFunction, SimulateBackwardError, MyFunction, MyFunction, MyFunction, MyFunction, MySquare, MySquare, MySquare, MyReshape

### Functions
This file defines 1267 function(s): graph_desc, tearDown, test_copy_slices_graph_task_updates, f1, f2, f3, hook, test_grad_mode_class_decoration, __init__, foo, foo, __init__, foo, test_tensor_grad_warnings, _function_test, test_function, forward, backward, test_once_differentiable, forward, backward, test_function_returns_input, forward, backward, test_function_returns_undefined_tensor, forward, backward, test_materialize_grads, forward, backward


## Key Components

The file contains 44599 words across 15333 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 561004 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

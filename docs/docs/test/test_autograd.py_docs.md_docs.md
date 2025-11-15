# Documentation: `docs/test/test_autograd.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_autograd.py_docs.md`
- **Size**: 361,889 bytes (353.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_autograd.py`

## File Metadata

- **Path**: `test/test_autograd.py`
- **Size**: 561,004 bytes (547.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
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

        @torch.enable_grad()
        def coro_enable_grad(state):
            for i in range(10):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertTrue(torch.is_grad_enabled())
                    state.add("GeneratorExit")
                    raise

        state = set()
        with torch.enable_grad():
            coro = coro_no_grad(state)
            for _ in range(5):
                next(coro)

            coro.close()
        self.assertTrue("GeneratorExit" in state)

        state = set()
        with torch.no_grad():
            coro = coro_enable_grad(state)
            for _ in range(5):
                next(coro)

            coro.close()
        self.assertTrue("GeneratorExit" in state)

    def test_no_grad_python_function(self):
        """Python Functions should respect grad mode."""
        x = torch.ones(5, 5, requires_grad=True)

        class MyOp(Function):
            @staticmethod
            def forward(self, x):
                return x + 1

            @staticmethod
            def backward(self, dy):
                return dy

        with torch.no_grad():
            y = MyOp.apply(x)
        self.assertFalse(y.requires_grad)

    def test_indexing(self):
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)

            indexed_var.sum().backward()
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)

        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ((slice(None), [2, 3])))
        check_index(x, y, (([2, 3], slice(None))))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0]))
        check_index(x, y, ([0],))

        x = torch.arange(1.0, 49).view(4, 3, 4)
        y = Variable(x, requires_grad=True)

        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0],))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))

        # advanced indexing, with a tensor wrapped in a variable
        z = torch.LongTensor([0, 1])
        zv = Variable(z, requires_grad=False)
        seq = (z, Ellipsis)
        seqv = (zv, Ellipsis)

        if y.grad is not None:
            with torch.no_grad():
                y.grad.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self):
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad, expected_grad)

        # with advanced indexing
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = ([1, 1, 3, 2, 1, 2], [0])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = ([[1, 2], [0, 0]], [[0, 1], [1, 1]])
        y[idx].sum().backward()
        expected_grad = torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1.0, 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        idx = ([1, 1, 1], slice(None), slice(None))
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        # Example from https://github.com/pytorch/pytorch/issues/24853.
        # if `index(tensor, indices)` saves `tensor` for backwards, then it will
        # trigger a version check on `tensor` during the backward pass, which
        # will cause the following code to error because `tensor` gets modified
        # by the indexing line.
        a = torch.tensor([1.0, 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        tensor[a != 0] = tensor[a != 0]
        tensor.backward(torch.zeros_like(tensor))

    def test_volatile_deprecated(self):
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(v.volatile)
        self.assertIn("volatile", str(w[0].message))

    def test_saved_variables_deprecated(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_variables
                return (grad_output, grad_output)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            MyFunction.apply(x, y).sum().backward()

            has_deprecated = (
                "deprecated" in str(warn) and "saved_variables" in str(warn)
                for warn in warns
            )
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            self.assertTrue(has_deprecated)

    def test_requires_grad(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)

        def error():
            raise RuntimeError

        # Make sure backward isn't called on these
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks["test"] = error
        x._backward_hooks["test"] = error
        y._backward_hooks["test"] = error
        b.backward(torch.ones(5, 5))

    def test_requires_grad_(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        z = x * y
        self.assertRaises(RuntimeError, lambda: z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)

        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    def test_requires_grad_inplace(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

        # non-leaf
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

    def test_no_requires_grad_inplace(self):
        # basic case, should be able to modify inplace while requires_grad is False
        a = torch.randn(2, 3)
        a.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # same but with a view
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # should fail if requires_grad = True when we modify inplace
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_attribute_deletion(self):
        x = torch.randn((5, 5), requires_grad=True)
        del x.grad
        self.assertIsNone(x.grad)
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    def test_duplicate_backward_root(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])

        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    def test_backward_no_grad(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_backward_twice_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        self.assertRaisesRegex(
            RuntimeError,
            "Specify retain_graph=True",
            lambda: c.backward(torch.tensor([1, 1, 1], dtype=torch.double)),
        )

    def test_backward_twice_retained_graph_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = b + 1
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_retained_graph_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_create_graph_warns(self):
        with set_warn_always_context(True):
            b = torch.randn(3, requires_grad=True, dtype=torch.double)
            c = b * b
            with warnings.catch_warnings(record=True) as ws:
                c.backward(torch.ones_like(c), create_graph=True)
            b.grad = None
            self.assertTrue(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

            # Should not warn for grad
            with warnings.catch_warnings(record=True) as ws:
                torch.autograd.grad(c, b, torch.ones_like(c), create_graph=True)
            self.assertFalse(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

    def test_next_functions(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        self.assertIsNotNone(a.grad_fn)
        next_functions = a.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[0][1], 0)
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[1][1], 0)

        b = a + 5
        next_functions = b.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIs(next_functions[0][0], a.grad_fn)
        self.assertIs(next_functions[1][0], None)

    def test_inplace(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_graph=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_graph=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_graph=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    def test_mark_non_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 0).to(torch.double)

        x = torch.randn(5, 5, requires_grad=True)
        mask = MyFunction.apply(x)
        self.assertFalse(mask.requires_grad)
        y = x.masked_fill(mask, 0)
        y.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_mark_non_differentiable_mixed(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                a = input + 1
                b = input + 2
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                self.assertTrue((grad_a == 0).all())
                self.assertTrue((grad_b == 1).all())
                return grad_b

        x = torch.randn(5, 5, requires_grad=True)
        a, b = MyFunction.apply(x)
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        # This used to segfault because MyFunction would send back null
        # gradients to MulBackward, which is implemented in C++. C++
        # implemented functions expect incoming grad_outputs to be non-null.
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input.clone()
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return None

        x = torch.randn(5, 5, requires_grad=True)
        r = MyFunction.apply(x * x)
        (r * x).sum().backward()

    def test_return_duplicate(self):
        class DoubleDuplicate(Function):
            @staticmethod
            def forward(ctx, x):
                output = x * 2
                return output, output

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def fn(x):
            a, b = DoubleDuplicate.apply(x)
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(fn, [x])
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        class DoubleInplace(Function):
            @staticmethod
            def forward(ctx, x):
                x.mul_(2)
                ctx.mark_dirty(x)
                return x, x

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def inplace_fn(x):
            a, b = DoubleInplace.apply(x.clone())
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(inplace_fn, [x])
        gradgradcheck(inplace_fn, [x])

        # Can't modify leaf variables in-place
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x))
        # Functions which modify views in-place must return only one output
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x.clone()[0]))

    def _test_setitem(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones_like(value))

        # case when x broadcasts to as y[1]
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        y[1] = x
        y.backward(torch.randn(2, 3, 4))
        self.assertEqual(x.size(), x.grad.size())

    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), ([0, 4, 2]))
        self._test_setitem((5, 5), ([0, 4], [2, 2]))
        self._test_setitem((5, 5, 5), (slice(None), slice(None), [1, 3]))
        self._test_setitem((5, 5, 5), (slice(None), [1, 3], slice(None)))
        self._test_setitem((5, 5, 5), ([1, 3], slice(None), slice(None)))
        self._test_setitem((5, 5, 5), (slice(None), [2, 4], [1, 3]))
        self._test_setitem((5, 5, 5), ([1, 3], [2, 4], slice(None)))
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), ([0, 1], [1, 0]))
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor(
            (5,), Variable(torch.LongTensor([3]), requires_grad=False).sum()
        )
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), (slice(None), slice(None), [1, 3]))
        self._test_setitem_tensor((5, 5, 5), (slice(None), [1, 3], slice(None)))
        self._test_setitem_tensor((5, 5, 5), ([1, 3], slice(None), slice(None)))
        self._test_setitem_tensor((5, 5, 5), (slice(None), [2, 4], [1, 3]))
        self._test_setitem_tensor((5, 5, 5), ([1, 3], [2, 4], slice(None)))
        self._test_setitem_tensor(
            (5, 5, 5),
            (
                Variable(torch.LongTensor([1, 3]), requires_grad=False),
                [2, 4],
                slice(None),
            ),
        )

    def test_setitem_mask(self):
        mask = torch.BoolTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # both select and sum return Scalars in ATen; ensure they work together.
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            return x.select(0, 1).sum()

        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        value = torch.rand([])
        v_expanded = torch.tensor(value).expand(10)
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        (result,) = torch.autograd.grad(a.diagonal(), a, v_expanded)
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        v_expanded = torch.rand(10).expand(10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[0], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[3:5], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        self.assertEqual(result, expected)

    def test_unused_output(self):
        x = torch.randn(10, 10, requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad, expected_grad)

        with torch.no_grad():
            x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    # TODO: opinfo this or move to the sparse test suite
    def _test_sparse_gather(self, size_x, size_ind, dim):
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        out = torch.gather(x, dim, ind, sparse_grad=False)
        grad = torch.rand_like(out)
        out.backward(grad)
        grad_dense = x.grad.clone()
        x.grad = None
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        self.assertEqual(grad_dense, x.grad.to_dense())

    def test_sparse_gather_dim0(self):
        self._test_sparse_gather((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        self._test_sparse_gather((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        self._test_sparse_gather((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        self._test_sparse_gather((), (), 0)

    @skipIfTorchDynamo("grad_dtype not supported in compile")
    def test_grad_dtype(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        # Default to tensor's dtype
        self.assertEqual(leaf.grad_dtype, torch.float32)
        leaf.grad_dtype = torch.float16
        self.assertEqual(leaf.grad_dtype, torch.float16)
        leaf.grad_dtype = None  # Allow any dtype
        self.assertIsNone(leaf.grad_dtype)

        # get/set grad_dtype is only allowed on leaf tensors
        non_leaf = leaf * 2
        self.assertFalse(non_leaf.is_leaf)
        with self.assertRaisesRegex(
            RuntimeError, "grad_dtype can only be accessed on leaf tensors"
        ):
            _ = non_leaf.grad_dtype
        with self.assertRaisesRegex(
            RuntimeError, "grad_dtype can only be set on leaf tensors"
        ):
            non_leaf.grad_dtype = torch.float16

        # Manual setting
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        grad_match = torch.tensor([1.0, 1.0])
        x.grad = grad_match
        self.assertEqual(x.grad.dtype, torch.float32)

        x.grad = None
        x.grad_dtype = torch.float16
        grad_mismatch = torch.tensor([1.0, 1.0])
        with self.assertRaisesRegex(
            RuntimeError,
            "attempting to assign a gradient with dtype.*float.*to a tensor with grad_dtype.*Half",
        ):
            x.grad = grad_mismatch

        # When grad_dtype is None, any dtype is allowed
        x.grad = None
        x.grad_dtype = None
        grad_any = torch.tensor([1.0, 1.0], dtype=torch.float64)
        x.grad = grad_any
        self.assertEqual(x.grad.dtype, torch.float64)

        # Incoming gradient case
        class MismatchedGradientFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.to(torch.float64)

        d = torch.tensor([1.0, 2.0], requires_grad=True)
        output = MismatchedGradientFunction.apply(d)
        loss = output.sum()
        loss.backward()
        # Default behavior is to cast to tensor dtype
        self.assertEqual(d.grad.dtype, torch.float32)
        self.assertTrue(torch.allclose(d.grad, torch.tensor([1.0, 1.0])))

        e = torch.tensor([3.0, 4.0], requires_grad=True)
        e.grad_dtype = None
        output_e = MismatchedGradientFunction.apply(e)
        loss_e = output_e.sum()
        loss_e.backward()
        # No casting is done if set to None.
        self.assertTrue(
            torch.allclose(e.grad, torch.tensor([1.0, 1.0], dtype=torch.float64))
        )

        f = torch.tensor([5.0, 6.0], requires_grad=True)
        f.grad_dtype = torch.float16  # Expect float16 gradients
        output_f = MismatchedGradientFunction.apply(f)
        loss_f = output_f.sum()
        loss_f.backward()
        self.assertTrue(
            torch.allclose(f.grad, torch.tensor([1.0, 1.0], dtype=torch.float16))
        )

        # Setting grad_dtype when gradient already exists
        g = torch.tensor([1.0, 2.0], requires_grad=True)
        g.grad = torch.tensor([1.0, 1.0])
        g.grad_dtype = torch.float32
        self.assertEqual(g.grad_dtype, torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot set grad_dtype.*because there is already a gradient"
        ):
            g.grad_dtype = torch.float16
        g.grad_dtype = None
        self.assertIsNone(g.grad_dtype)
        g.grad = None
        g.grad_dtype = torch.float16
        self.assertEqual(g.grad_dtype, torch.float16)

        # Test the case where there is an existing accumulate grad
        h = torch.tensor([1.0, 2.0], requires_grad=True)
        _ = h.clone()
        h.grad_dtype = None
        output = MismatchedGradientFunction.apply(h)
        output.sum().backward()
        self.assertEqual(h.grad.dtype, torch.float64)

        # Mixed accumulation cases
        k = torch.tensor([1.0, 2.0], requires_grad=True)
        k.grad_dtype = None
        y = k * 2
        y.sum().backward()
        k.grad = k.grad.to(torch.bfloat16)
        y2 = k * 3
        # Doesn't type promote to float32, always coerce to current .grad's dtype.
        # This is because the accumulation is done in-place on the existing grad.
        self.assertEqual(k.grad.dtype, torch.bfloat16)

        l = torch.tensor([3.0, 4.0], requires_grad=True, dtype=torch.bfloat16)
        l.grad_dtype = None
        z = l * 2
        z.sum().backward()
        l.grad = l.grad.to(torch.float32)
        z2 = l * 3
        z2.sum().backward()
        self.assertEqual(l.grad.dtype, torch.float32)

    def test_gc_in_destructor(self):
        """
        Previously, if a Function destructor triggered a garbage collection,
        the Variable's tp_dealloc handler would get called twice leading to a
        segfault.
        """

        class CollectOnDelete(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

            def __del__(self):
                gc.collect()

        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    def test_naughty_autograd_function_attribute_access(self):
        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_x):
                return grad_x

        with self.assertWarnsRegex(DeprecationWarning, "should not be instantiated"):
            f = Id()

        # After raising warning, should still return an instance
        self.assertIsInstance(f, Id)
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "non-static forward method is deprecated"
        ):
            f(x)
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), "IdBackward")

        # THPFunction is the base class of both grad_fn and autograd functions,
        # which means that a lot of accessors on them may segfault. Test that we
        # properly error in this case.
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = {}
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_register_hook_dict' is invalid"
        ):
            f._register_hook_dict(t)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute 'register_hook' is invalid"
        ):
            f.register_hook(lambda x, y: None)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute 'next_functions' is invalid"
        ):
            f.next_functions
        with self.assertRaisesRegex(RuntimeError, "Attribute 'name' is invalid"):
            f.name()
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_sequence_nr' is invalid"
        ):
            f._sequence_nr()
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_set_sequence_nr' is invalid"
        ):
            f._set_sequence_nr(2)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_input_metadata' is invalid"
        ):
            f._input_metadata
        with self.assertRaisesRegex(
            RuntimeError, "underlying PyNode has already been deallocated"
        ):
            f.metadata

    @unittest.expectedFailure
    def test_naughty_anomaly_access(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                return g

        x = torch.zeros(1, requires_grad=True)
        y = MyFunction.apply(x)
        y.backward()
        y.grad_fn.metadata
        g = y.grad_fn
        del y
        g.metadata  # this currently fails, but shouldn't

    def test_naughty_autograd_function_stashing_ctx(self):
        saved_ctx = []

        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                saved_ctx.append(ctx)
                return ctx.saved_tensors

        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        # At this point in time, it complains that the graph has been freed
        # (which indeed true, although a somewhat indirect way of stating the
        # problem).
        self.assertRaises(RuntimeError, lambda: saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        # This test failed the equality check in PR #22983; it's an interesting
        # and different test case worth enshrining.  mult1 is not testing
        # anything that interesting, but mult2 is the interesting case.

        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_1,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_2,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        # This test failed complaining that buffers had already been freed
        # prior to #22983.  Also pretty interesting test case.
        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))

        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)  # should not error!

    def test_custom_autograd_ac_early_stop(self):
        refs = []

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.clone()
                ctx.save_for_backward(y)
                refs.append(weakref.ref(y))
                return y

            @staticmethod
            def backward(ctx, *args):
                _ = ctx.saved_tensors
                return None

        def fn(inp):
            return Test.apply(inp)

        inp = torch.randn(5, 5, requires_grad=True)

        def scope():
            # Early-stop is true by default in non-reentrant torch.utils.checkpoint
            out = torch.utils.checkpoint.checkpoint(fn, inp, use_reentrant=False)
            out.sum().backward()

        with disable_gc():
            scope()

            for ref in refs:
                self.assertIsNone(ref())

    def test_detach(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)

        # in-place detach on a view raises an exception
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, "view", lambda: view.detach_())

    def test_detach_base(self):
        "detaching base does not detach view"
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)

    def test_detach_then_inplace_raises_in_autograd(self):
        x = torch.randn([], requires_grad=True)
        orig_x = x.detach().clone()

        y = x**2  # saves x
        z = x.detach()
        z.zero_()
        with self.assertRaisesRegex(RuntimeError, "has been modified by an inplace"):
            y.backward()

    def _test_type_conversion_backward(self, t):
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)

        for t in [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.IntTensor,
            torch.ByteTensor,
        ]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                # TODO: t.dtype should work
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            _, y_type = y_c.type().rsplit(".", 1)
                            y_typestr = ("torch.cuda." if y_cuda else "torch.") + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(
                                y_c.data_ptr(),
                                y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr(),
                            )

        self._test_type_conversion_backward(lambda x: x)
        if torch.cuda.is_available():
            self._test_type_conversion_backward(lambda x: x.cuda())
            if torch.cuda.device_count() >= 2:
                # one of these has to be the non-default device
                self._test_type_conversion_backward(lambda x: x.cuda(0))
                self._test_type_conversion_backward(lambda x: x.cuda(1))

    def test_isolated_node(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        o = (b + a).sum()
        o.backward()

    def test_shape(self):
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        with torch.no_grad():
            x.numpy()

        x = torch.randn(2, 2)
        x.numpy()

        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        class Identity(Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Identity.apply(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):
            @staticmethod
            def forward(ctx, a, b):
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Inplace.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y

        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(
            RuntimeError, "modified by an inplace operation", lambda: z.backward()
        )

    def test_increment_version(self):
        a = torch.rand(5, requires_grad=True)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)

        a = torch.zeros(5, dtype=torch.int)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)

        with torch.inference_mode():
            a = torch.rand(5, requires_grad=True)
            # does not error
            torch.autograd.graph.increment_version(a)

        # does not error
        torch.autograd.graph.increment_version(a)

    def test_no_grad_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(self, x):
                return x

            @staticmethod
            def backward(self, grad_output):
                return grad_output

        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)

        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        # This tests checks backward engine for a very subtle bug that appeared
        # in one of the initial versions of autograd. Gradients tensors were
        # simply stored in lists while the function waited for all its gradients
        # to be computed. However, sometimes an output was used multiple times,
        # so the gradients needed to be summed. Engine used to keep a need_copy
        # set of tensors that will need a clone upon next addition and removed
        # them from the set as soon as the clone was performed. However, this
        # could lead to incorrect results if the same gradient tensor was
        # buffered in three places in the graph:
        # 1. When accumulating gradients in one of these places it was cloned
        #    and removed from need_copy set.
        # 2. When accumulating in second place, it wasn't in the need_copy set,
        #    so the gradients were simply accumulated in-place (which already
        #    modified the grad in 3rd place)
        # 3. When accumulating in the third place, it wasn't in the need_copy set
        #    as well, so the incoming gradient was summed in-place, yielding
        #    incorrect results in all functions, except the first one.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # Simulate that we're in the middle of the graph
        a = x + 2
        b = y + 2
        c = x + 2
        # This op will just return grad_output two times in backward
        add1 = a + b
        add2 = add1 + c
        # Simulate a long branch, so grad_output will get buffered.
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # expected gradients are:
        # for x: 34 (16 from final a, 16 from final c, 2 from add2)
        # for y: 17 (16 from final b, 1 from add2)
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        test_case = self

        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_dep_nograd(self):
        class F1(Function):
            @staticmethod
            def forward(ctx, input):
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return input, out

            @staticmethod
            def backward(ctx, grad_output, ignored):
                return grad_output

        class F2(Function):
            @staticmethod
            def forward(ctx, input, ignored):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        x = torch.randn(5, requires_grad=True)
        a, b = F1.apply(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        x = torch.tensor([1.0], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_set_grad_enabled_wraps(self):
        for decorator in [True, False]:
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())

                if decorator:
                    # This should not mutate the global grad mode!
                    @torch.set_grad_enabled(False)
                    def inner_func(x):
                        return x.sin()

                else:

                    def inner_func(x):
                        return x.sin()

                    # This is non-idiomatic usage!
                    # More idiomatic usage: torch.set_grad_enabled(False)(inner_func)
                    obj = torch.set_grad_enabled(False)
                    self.assertTrue(not torch.is_grad_enabled())

                    # this will consume the set_grad_enabled global mutation!
                    inner_func = obj(inner_func)
                    self.assertTrue(torch.is_grad_enabled())

                self.assertTrue(torch.is_grad_enabled())

                x = torch.zeros(1, requires_grad=True)
                self.assertTrue(not inner_func(x).requires_grad)

    def test_simple_reentrant(self):
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishes on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # Parent graph.
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

        # Reentrant child graph.
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        f = TestAutograd.SimulateBackwardError.apply(e)
        reentrant_root = f.sum()

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will throw an error.
                reentrant_root.backward()
                return grad

        d = ReentrantFunc.apply(c)
        with self.assertRaisesRegex(Exception, "Simulate error"):
            d.sum().backward()

    def test_var_mean_differentiable(self):
        dim = [2, 4]
        keepdim = False
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        input2 = deepcopy(input1)
        var1, mean1 = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        var2 = input2.var(dim=dim, keepdim=keepdim)
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)

        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        self.assertEqual(r1, r2, rtol=0.01, atol=0.0)

        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        self.assertEqual(input1.grad, input2.grad, rtol=0.01, atol=0.0)

    @skipIfNoLapack
    def test_lobpcg(self):
        def func(k, A, largest=True, B=None):
            X_shape = list(A.shape)
            X_shape[-1] = k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            if A.dim() > 2:
                X = X.expand(X_shape)

            D, U = torch.lobpcg(A=A, k=k, B=B, X=X, largest=largest)

            # LOBPCG uses a random initial eigenspace approximation
            # if parameter `X` is not provided.
            # This may cause a non-deterministic behavior
            # when it comes to the sign of an eigenvector
            # (note if v is an eigenvector, so is -v),
            # hence we eliminate this non-determinism
            # by making sure that each column of U
            # gets multiplied by the sign of its max (in absolute value) element.
            # Also, gradcheck changes the content of the input by +/- eps (default to 1e-06)
            # to compute the numerical gradient which can also cause the signs to flip.
            _, idx = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return D, U

        # TODO: review if this can be ported to OpInfos or moved to test_linalg.py
        def run_symeig_test(k, sizes, largest=True):
            A = torch.rand(*sizes).double()
            A = (A @ A.mT) / 10
            A.requires_grad_(True)

            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)

            # Custom gradient vectors for better stability due to some
            # non-determinism in the lobpcg's forward.
            # Note it is not required if symeig is in forward instead (tested).
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(
                lambda A: func(k, A, largest),
                A,
                [D_grad, U_grad],
                atol=1e-4,
                check_batched_grad=False,
            )

            # check whether A.grad is symmetric
            A = A.detach().requires_grad_(True)
            D, U = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.mT)

        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)

    def test_variable_traverse(self):
        def get_out_and_unrefed_cycle():
            inp = torch.randn(10, requires_grad=True)
            tmp = inp.view(10, 1)
            out = tmp.view(10)

            # Create a reference cycle that contains an
            # intermediary Variable in the graph
            my_list = []
            my_list.append(tmp)
            my_list.append(my_list)

            return out

        out = get_out_and_unrefed_cycle()
        gc.collect()
        # This will segfault if things have been erroneously released
        out.backward(torch.randn(out.size()))

    # TODO: review porting these to OpInfo tests
    def test_pow_zero_tensor_gradient(self):
        def run_test(input_size, exponent):
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.abs().sum(), 0)

        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_node_ordering_when_none_returned(self):
        class Matmul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w):
                # x: [M, N]
                # w: [N, K]
                ctx.save_for_backward(x, w)
                return x @ w

            @staticmethod
            def backward(ctx, g_out):
                # g_out: [M, K]
                x, w = ctx.saved_tensors
                g_x = g_out @ w.T
                g_w = x.T @ g_out
                w.main_grad = g_w.float()
                return g_x, None

        executed = []

        class HookFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                executed.append("A")
                return g

        def hook(*args, **kwargs):
            executed.append("B")

        x = torch.randn((3, 3), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x = HookFunction.apply(x)
        w = torch.randn((3, 3), dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w.register_hook(hook)
        o = Matmul.apply(x, w)
        o.sum().backward()

        self.assertEqual(executed, ["B", "A"])

    def test_current_graph_task_id(self):
        id = [-1]

        def hook(_):
            id[0] = torch._C._current_graph_task_id()

        t = torch.tensor(1.0, requires_grad=True).clone()
        t.register_hook(hook)

        t.backward(retain_graph=True)
        base = id[0]
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 1)
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 2)

        self.assertEqual(torch._C._current_graph_task_id(), -1)

    def test_current_graph_task_execution_order(self):
        predicted = [None]
        all_hooks = []

        def hook(_):
            predicted[0] = torch._C._current_graph_task_execution_order()

        def names(nodes):
            return ", ".join([node.name().split(" ")[-1] for node in nodes]) + "\n"

        def grad_fns(*tensors):
            # or grad accumulator
            out = []
            for t in tensors:
                if t.requires_grad and t.grad_fn is None:
                    out.append(t.clone().grad_fn.next_functions[0][0])
                else:
                    out.append(t.grad_fn)
            return out

        actual = []

        def register_logging_hooks(*tensors):
            # register hooks that log the order in which they are called
            def get_hook(i):
                def hook(t_):
                    actual.append(tensors[i])

                return hook

            for i, t in enumerate(tensors):
                all_hooks.append(t.register_hook(get_hook(i)))

        # Basic example: single path
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        all_hooks.append(t.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            t.backward()
        self.assertExpectedInline(
            names(predicted[0]),
            """\
ExpBackward0, SinBackward0, CloneBackward0, torch::autograd::AccumulateGrad
""",
        )

        # We don't exactly follow sequence_nr order
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = b.sin()
        d = a.cos()
        out = c * d
        register_logging_hooks(a, b, c, d, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Accumulate grad node has more than one input
        a = torch.tensor(1.0, requires_grad=True)
        b = a.sin()
        c = a.cos()
        out = b * c
        register_logging_hooks(a, b, c, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Multiple roots are also OK
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        out2 = b.cos()
        out3 = b.cos()
        register_logging_hooks(a, b, out, out2, out3)
        all_hooks.append(out3.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out, out3, out2), inputs=(a,))
        self.assertExpectedInline(
            names(predicted[0]),
            """\
CosBackward0, CosBackward0, SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where next node is nullptr
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where two `inputs` on the same path
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a, b))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Case where `inputs` specifies a subgraph
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        c = a * b
        out = c.sin()
        register_logging_hooks(a, b, c, out)
        all_hooks.append(out.register_hook(hook))
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a,))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: Uncomment after update to hooks behavior
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # Errors when not called in a backward
        with self.assertRaisesRegex(
            RuntimeError, "should only be called during the backward pass"
        ):
            torch._C._current_graph_task_execution_order()

        # Errors when context manager not enabled
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        all_hooks.append(t.register_hook(hook))
        with self.assertRaisesRegex(
            RuntimeError,
            "expects the current backward to be executed with multithreading disabled",
        ):
            t.backward()

        # Avoid leaking memory
        for h in all_hooks:
            h.remove()

    @skipIfWindows(msg="node name demangling inconsistent on windows")
    def test_backward_hook_relative_ordering(self):
        order = []

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        x = torch.randn(10, 10, requires_grad=True)
        module = MyModule()
        module.register_full_backward_hook(
            lambda _1, _2, _3: order.append(
                "module_full_backward_hook_BackwardHookFunctionBackward0"
            )
        )

        def make_pre_hook(id):
            return lambda _: order.append(f"pre_hook_{id}")

        def make_post_hook(id):
            return lambda _1, _2: order.append(f"post_hook_{id}")

        count = 0

        def register_hooks_on_all_nodes(nodes):
            nonlocal count
            for node, _ in nodes:
                count += 1
                id = f"{node.name()}_{count}"
                node.register_prehook(make_pre_hook(id))
                node.register_hook(make_post_hook(id))
                register_hooks_on_all_nodes(node.next_functions)

        loss = module(x).sum()
        register_hooks_on_all_nodes(((loss.grad_fn, None),))

        def make_tensor_pre_hook(id):
            return lambda _: order.append(f"tensor_pre_hook_{id}")

        def make_post_acc_grad_hook(id):
            return lambda _: order.append(f"post_acc_grad_hook_{id}")

        x.register_hook(make_tensor_pre_hook("x"))
        module.linear.weight.register_hook(make_tensor_pre_hook("weight"))
        module.linear.bias.register_hook(make_tensor_pre_hook("bias"))

        x.register_post_accumulate_grad_hook(make_post_acc_grad_hook("x"))
        module.linear.weight.register_post_accumulate_grad_hook(
            make_post_acc_grad_hook("weight")
        )
        module.linear.bias.register_post_accumulate_grad_hook(
            make_post_acc_grad_hook("bias")
        )

        loss.backward()

        expected_order = [
            "pre_hook_SumBackward0_1",
            "post_hook_SumBackward0_1",
            "pre_hook_BackwardHookFunctionBackward_2",
            "post_hook_BackwardHookFunctionBackward_2",
            "pre_hook_AddmmBackward0_3",
            "post_hook_AddmmBackward0_3",
            "tensor_pre_hook_bias",
            "pre_hook_torch::autograd::AccumulateGrad_4",
            "post_acc_grad_hook_bias",
            "post_hook_torch::autograd::AccumulateGrad_4",
            "pre_hook_TBackward0_7",
            "post_hook_TBackward0_7",
            "tensor_pre_hook_weight",
            "pre_hook_torch::autograd::AccumulateGrad_8",
            "post_acc_grad_hook_weight",
            "post_hook_torch::autograd::AccumulateGrad_8",
            "pre_hook_BackwardHookFunctionBackward_5",
            "module_full_backward_hook_BackwardHookFunctionBackward0",
            "post_hook_BackwardHookFunctionBackward_5",
            "tensor_pre_hook_x",
            "pre_hook_torch::autograd::AccumulateGrad_6",
            "post_acc_grad_hook_x",
            "post_hook_torch::autograd::AccumulateGrad_6",
        ]

        self.assertEqual(len(expected_order), len(order))
        for expected, actual in zip(expected_order, order):
            self.assertEqual(expected, actual)

    def test_view_replay_enabled(self):
        def f(x):
            out = x.clone().view(-1)
            # mutate the view, triggering autograd view-replay logic
            out.add_(1)
            return out

        x = torch.ones(2, 2, requires_grad=True)

        # Test as a context manager
        with torch.autograd._force_original_view_tracking(False):
            out = f(x)
            self.assertTrue("AsStridedBackward" in str(out.grad_fn))
            self.assertFalse(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        with torch.autograd._force_original_view_tracking(True):
            out = f(x)
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        with torch.autograd._force_original_view_tracking(False):
            torch.autograd._force_original_view_tracking(True)
            out = f(x)
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        # Test as a function
        torch.autograd._force_original_view_tracking(False)
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        torch.autograd._force_original_view_tracking(True)
        out = f(x)
        self.assertTrue("ViewBackward" in str(out.grad_fn))
        self.assertTrue(torch.autograd.is_view_replay_enabled())

    def test_unsafe_set_version_counter(self):
        x = torch.ones(2, requires_grad=True).clone()
        x.add_(1)
        x.add_(2)
        self.assertEqual(2, x._version)
        with torch.autograd._unsafe_preserve_version_counter(x):
            x.mul_(2)
            x.mul_(3)
        # version counter doesn't change inside of the context manager
        self.assertEqual(2, x._version)

        torch._C._autograd._unsafe_set_version_counter((x,), (0,))
        self.assertEqual(0, x._version)
        with self.assertRaisesRegex(RuntimeError, "Cannot set"):
            torch._C._autograd._unsafe_set_version_counter((x,), (-1,))

        y = torch.ones(2, requires_grad=True).clone()
        with torch.autograd._unsafe_preserve_version_counter((x, y)):
            x.mul_(2)
            y.mul_(3)
        # version counter doesn't change inside of the context manager
        self.assertEqual(0, x._version)
        self.assertEqual(0, y._version)

    def test_current_node(self):
        pr = []

        class MyMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                node = torch._C._current_autograd_node()
                # Don't use node.name() here as it is not consistent on windows
                node_name = node.__class__.__name__ if node else "None"
                pr.append(f"Running {func} from within {node_name}")
                return func(*args, **(kwargs or {}))

        with MyMode():
            pr.append("FW")
            a = torch.rand(10, requires_grad=True)
            b = a.mul(2).div(3).sum()
            pr.append("BW")
            b.backward()
            pr.append("Done")

        self.assertExpectedInline(
            "\n".join(pr),
            """\
FW
Running aten.rand.default from within None
Running aten.mul.Tensor from within None
Running aten.div.Tensor from within None
Running aten.sum.default from within None
BW
Running aten.ones_like.default from within None
Running aten.expand.default from within SumBackward0
Running aten.div.Tensor from within DivBackward0
Running aten.mul.Tensor from within MulBackward0
Running aten.detach.default from within AccumulateGrad
Done""",
        )

    def test_profiler(self):
        x = torch.randn(10, 10)

        with profile(use_kineto=kineto_available()) as p:
            self.assertTrue(torch.autograd._profiler_enabled())
            y = x * 2 + 4

        self.assertFalse(torch.autograd._profiler_enabled())

        names = ["aten::mul", "aten::add"]
        found_indices = set()
        for evt in p.function_events:
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        self.assertEqual(len(found_indices), len(names))

    def test_profiler_seq_nr(self):
        with profile(use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = z.sum(dim=None)
            s.backward()
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        # expecting aten::add, aten::sum to have the sequence numbers,
        # expecting the corresponding backward nodes to have the same numbers
        # as the forward ops
        autograd_ops = {
            ("aten::add", "Add"): [],
            ("aten::sum", "Sum"): [],
        }
        accumulate_ops = []
        found_empty = False
        for e in p.function_events:
            for (fwd_name, bwd_name), ops in autograd_ops.items():
                if e.name == fwd_name or (bwd_name in e.name and "Backward" in e.name):
                    ops.append(e)

            if "AccumulateGrad" in e.name:
                accumulate_ops.append(e)

            # check that nested ops (e.g. empty) don't have
            # sequence number
            if e.name == "aten::empty":
                self.assertEqual(e.sequence_nr, -1)
                found_empty = True

        for idx, ((fwd_name, bwd_name), ops) in enumerate(autograd_ops.items()):
            self.assertEqual(len(ops), 3)
            self.assertEqual(ops[0].name, fwd_name)
            self.assertEqual(
                ops[1].name,
                f"autograd::engine::evaluate_function: {bwd_name}Backward{idx}",
            )
            self.assertEqual(ops[2].name, f"{bwd_name}Backward{idx}")
            self.assertGreaterEqual(ops[0].sequence_nr, 0)
            self.assertEqual(ops[1].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[2].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[0].fwd_thread, 0)
            self.assertEqual(ops[1].fwd_thread, ops[0].thread)
            self.assertEqual(ops[2].fwd_thread, ops[0].thread)
        self.assertTrue(found_empty)

    def test_profiler_unboxed_only(self):
        x = torch.rand(3, 4)

        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            x.resize_([3, 2])

    def test_profiler_propagation(self):
        def foo(x):
            with record_function("in_foo") as rf:
                return x * 2

        x = torch.rand(3, 4)
        traced_foo = torch.jit.trace(foo, x)

        def bar(x):
            with record_function("in_bar") as rf:
                # we expect that profiler will be able
                # propagate across fork
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                # note: continuation (and rf's end) can
                # be executed in a different thread
                with record_function("in_bar_after_wait") as rf2:
                    y = y * 2
                return y

        traced_bar = torch.jit.trace(bar, x)

        with profile(use_kineto=kineto_available()) as p:
            traced_bar(x)

        found_foo = False
        found_bar = False
        found_bar_after_wait = False
        for info in p.function_events:
            if info.name == "in_foo":
                self.assertFalse(found_foo)
                found_foo = True
            elif info.name == "in_bar":
                self.assertFalse(found_bar)
                found_bar = True
            elif info.name == "in_bar_after_wait":
                self.assertFalse(found_bar_after_wait)
                found_bar_after_wait = True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    def test_record_function_callbacks(self):
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            with record_function("foo"):
                y = x * 2 + 4

        function_events = p.function_events
        foo_event = next(event for event in function_events if "foo" in event.name)
        self.assertEqual(foo_event.count, 1)

    def test_record_function_legacy(self):
        # Test the new _record_function ops work
        # Note: Remove once record_function uses these directly
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            handle = torch.ops.profiler._record_function_enter("bar", None)
            try:
                y = x * 2 + 4
            finally:
                torch.ops.profiler._record_function_exit(handle)

        function_events = p.function_events
        foo_event = next(event for event in function_events if "bar" in event.name)
        self.assertEqual(foo_event.count, 1)

    def test_profiler_aggregation_fake(self):
        events = EventList()
        id = [0]

        def get_id():
            id[0] = id[0] + 1
            return id[0]

        # [[thread_id, [(start, end, id), ....]], ...]
        # Using list instead of a dict so order is guaranteed for any Python
        # version
        threads = [
            [1, [(0, 1, get_id()), (1, 2, get_id())]],
            [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]],
        ]
        for thread, ranges in threads:
            for range in ranges:
                assert len(range) == 3
                events.append(
                    FunctionEvent(
                        id=range[2],
                        node_id=0,
                        name="",
                        thread=thread,
                        start_us=range[0],
                        end_us=range[1],
                    )
                )

        events._populate_cpu_children()

        # Note that [1, 3] pushes out [0, 2] first. Then we record [1, 2]
        # as a child of [1, 3]
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            return [child.id for child in event.cpu_children]

        assert [get_children_ids(event) for event in events] == res

    def test_profiler_aggregation_table(self):
        """
        Test if the profiling result is aggregated for `str(prof)`

        See: https://github.com/pytorch/pytorch/issues/37500
        """

        x = torch.randn(1024)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum("i->", x)

        prof_str = str(prof)
        prof_table = prof.table()

        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        avg = FunctionEventAvg()
        avg.add(
            FunctionEvent(id=0, node_id=0, name="foo", thread=0, start_us=10, end_us=15)
        )
        avg.add(
            FunctionEvent(id=1, node_id=0, name="foo", thread=0, start_us=20, end_us=30)
        )
        avg.add(avg)
        self.assertEqual(avg.key, "foo")

        # aggregate stats
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.device_time_total, 0)

        # average stats
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.device_time_total, 0)

    def test_profiler_shapes(self):
        print()
        layer1 = torch.nn.Linear(20, 30)
        layer2 = torch.nn.Linear(30, 40)
        input = torch.randn(128, 20)
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            layer2(layer1(input))

        print(prof.function_events)

        linear_expected_shapes = [
            [[128, 20], [30, 20], [30]],
            [[128, 30], [40, 30], [40]],
        ]

        found_indices = set()
        for event in prof.function_events:
            if event.name == "aten::linear":
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    def test_profiler_aggregation_lstm(self):
        print()
        rnn = torch.nn.LSTM(10, 20, 2)
        total_time_s = 0
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            for _ in range(20):
                input = torch.randn(5, 3, 10)
                h = torch.randn(2, 3, 20)
                c = torch.randn(2, 3, 20)
                start = time.time()
                rnn(input, (h, c))
                end = time.time()
                total_time_s += end - start

        print(prof.table(sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10
            )
        )
        print(
            prof.table(
                sort_by="self_cpu_time_total",
                row_limit=10,
                max_src_column_width=300,
                header="TEST",
                top_level_events_only=True,
            )
        )
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=True
            )
        )

        total_time_us = (
            total_time_s * 1000.0 * 1000.0
        )  # make it us which is profiler default
        print("Total time based on python measurements: ", _format_time(total_time_us))
        print(
            f"CPU time measurement python side overhead: {(total_time_us / prof.self_cpu_time_total - 1.0) * 100.0:.2f}%"
        )

        if sys.platform != "win32":
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

    def test_record_function(self):
        x = torch.randn(10, 10)

        def forward(x):
            with record_function("outer"):
                y = x * 2 + 4
                with record_function("inner"):
                    y = y - 1
            y = y / 1

        forward(x)

        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        events = p.function_events
        important_events = [
            "outer",
            "aten::mul",
            "aten::add",
            "inner",
            "aten::sub",
            "aten::div",
        ]
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        # We can also use record_function to decorate arbitrary function
        @record_function("my_func")
        def f(x, y):
            return x + y

        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        self.assertTrue("my_func" in str(p))

    def test_record_function_multithreaded(self):
        rf = record_function("outer")
        rf.__enter__()
        with record_function("inner"):
            # test that exiting the record function after starting another one
            # doesn't throw.
            rf.__exit__(None, None, None)

        with record_function("inner"):
            rf.__enter__()
        # test that exiting the record function after ending another one
        # doesn't throw.
        rf.__exit__(None, None, None)

    def test_dir(self):
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn("shape", keys)

        # real and imag are only implemented for complex tensors.
        y = torch.randn(10, 10, dtype=torch.cfloat)
        imag_key = "imag"
        self.assertRaises(RuntimeError, lambda: hasattr(x, imag_key))
        self.assertTrue(hasattr(y, imag_key))
        keys.remove(imag_key)

        for key in keys:
            self.assertTrue(hasattr(x, key))

    def test_inplace_on_view_saved_output(self):
        # Test an in-place operation on a view in which the in-place op saves
        # its output. Previously, this created a reference cycle.
        dealloc = [0]

        class IncrementOnDelete:
            def __del__(self):
                dealloc[0] += 1

        def test():
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)

        test()
        self.assertEqual(dealloc[0], 1)

    def test_inplace_on_view_leaf_errors(self):
        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "a view of a leaf Variable that "
            "requires grad is being used in "
            "an in-place operation.",
        ):
            y.add_(1)

    def test_inplace_on_view_backward(self):
        # Issue #10532: Make sure that this does not raise RuntimeError.
        net = nn.Sequential(nn.InstanceNorm2d(2), nn.ReLU(True))

        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        (g,) = torch.autograd.grad(
            net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape), create_graph=True
        )
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # https://discuss.pytorch.org/t/freeing-buffer-strange-behavior/31955/8
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0.0, 0.0, True)
        prob_interpolated = torch.sigmoid(tmp2)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=inputs,
            grad_outputs=torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward0")

    def test_inplace_on_view_weak_grad_fn(self):
        # Issue 23502: Test that b's grad_fn is preserved.
        a = torch.arange(10.0, requires_grad=True)

        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        c = b.clone()
        del b
        gc.collect()

        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

    def test_out_variant_raises_when_inputs_require_grad(self):
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)

        # out=... functions don't support automatic differentiation currently
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

        # the inputs can require grad if we're in no_grad() mode
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)

        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        # we should throw an exception if the output requires grad
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

    def test_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()  # Should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 0th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn("No forward pass information", str(w[0].message))

        inp = torch.rand(size, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 1th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out = MyFunc.apply(inp, inp, False)
                    out.backward()
            self.assertIn("MyFunc.apply", str(w[0].message))

    def test_calculate_shape_util(self):
        out = torch.randn(10, 5, requires_grad=True)
        grad = torch.randn(5, 10, requires_grad=True)
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        assert out_shape == torch.Size([10, 5])
        assert grad_shape == torch.Size([5, 10])

        out = torch.nested.as_nested_tensor(
            [
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
            ]
        )
        grad = torch.nested.as_nested_tensor(
            [
                torch.randn(5, 10, requires_grad=True),
                torch.randn(5, 10, requires_grad=True),
            ]
        )
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        assert torch.equal(out_shape, torch.tensor([[10, 5], [10, 5], [10, 5]]))
        assert torch.equal(grad_shape, torch.tensor([[5, 10], [5, 10]]))

    def test_nested_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                (inp,) = ctx.saved_tensors
                fail_0th = ctx.fail_0th
                g = gO.clone().expand(size)
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                return gI, None

        class MyFunc2(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1 * 2.0 + inp2

            @staticmethod
            def backward(ctx, gO):
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                # generate a nan
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return g1, g2, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        gsum.backward()  # should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 0th output.",
            ):
                with detect_anomaly():
                    gsum.backward()
        self.assertIn("No forward pass information", str(w[1].message))

        inp = torch.rand(size, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 1th output.",
            ):
                with detect_anomaly():
                    out = MyFunc.apply(inp, False)
                    (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
                    gsum = ginp.sum()
                    gsum.backward()
        self.assertIn("MyFunc2.apply", str(w[1].message))
        self.assertIn("MyFunc.apply", str(w[2].message))

    def test_anomaly_grad_warnings(self):
        # PyTorch won't throw warnings if there is an error
        # but we'd want to at least see them in stderr

        class StdErrDiverter:
            def __enter__(self):
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig

        # if the warnings don't throw, they will be handled as regular warnings
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1**2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 2)
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", str(w[1].message))

        # if the warning throws, it will be printed to sys.stderr
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    warnings.simplefilter("error")
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1**2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 1)
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", s.captured)

    def test_anomaly_assign_parent_cleanup(self):
        # Test that python objects created are properly cleaned up when assign_parent is called

        def get_ref():
            # we use torch.exp here but any function that will construct a new node in its
            # backward call in grad mode will work
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()

            # ExpBackward calls mul, creating the MulBackward node when create_graph=True.
            # In anomaly mode, a PyObject referencing MulBackward's "parent" ExpBackward is added to
            # MulBackward's anomaly metadata dict, creating the following reference chain:
            #
            # grad -> MulBackward -> PyObject -> ExpBackward
            #
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            # We add a weak reference to a new Foo object, which we insert into ExpBackward's metadata dict
            #
            # (PyObject) -> ExpBackward -> dict -> *Foo*
            #            t ----^        WeakRef ---^
            #
            # We want to test that when grad goes out of scope at the end of this function that PyObject is destroyed
            # We can test this by seeing whether Foo is not kept alive once t is destroyed
            class Foo:
                pass

            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return t, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_nested_anomaly_printstack_cleanup(self):
        # Test if metadata dict PyObject is properly destroyed
        def get_ref():
            # This is similar to the construction in test_anomaly_assign_parent_cleanup:
            #
            # MyFuncBackward2 -> PyObject -> MyFuncBackward -> dict -> Foo
            #                               out ---^         WeakRef ---^
            #
            # We want to check that Foo is still properly destroyed even when MyFunc2Backward's
            # AnomalyMetadata calls printstack, which does some python object manipulation.
            #
            # You might be wondering why we still have to test_anomaly_assign_parent_cleanup,
            # since if PyObject is not destroyed here, wouldn't this test would detect that also?
            # The answer is that custom function's PyObject (THPFunction) actually only hold
            # a weak reference to the c++ node!
            class MyFunc(Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO + float("NaN")

            inp = torch.rand(1, requires_grad=True)
            out = MyFunc.apply(inp)
            (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)

            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Function 'MyFunc2Backward' returned nan values in its 0th output.",
                ):
                    with detect_anomaly():
                        ginp.backward()

            class Foo:
                pass

            my_obj = Foo()
            meta_dict = out.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return out, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_anomaly_mode_no_check_nan(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                return torch.tensor(float("nan")).expand(10, 10)

        def run_fn(a):
            out = MyFunc.apply(a)
            return out.sum()

        with warnings.catch_warnings(record=True) as w:
            with torch.autograd.detect_anomaly(check_nan=False):
                inp = torch.rand(10, 10, requires_grad=True)
                out = run_fn(inp)
                out.backward(retain_graph=True)

                with torch.autograd.detect_anomaly(check_nan=True):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Function 'MyFuncBackward' returned nan values in its 0th output.",
                    ):
                        out.backward(retain_graph=True)

                out.backward()

    def test_no_grad_copy(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)

        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # non-contiguous grad should be copied
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        # test case that should trigger no copy for one of a,b
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

    def test_no_grad_copy_sparse(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # Create a sparse tensor with non-contiguous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for one of a,b
        emb_matrix = MyFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for _ in range(10):
            loss.backward(retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        a.grad = b.grad = None
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # Verify we cloned both grads.
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for _ in range(10):
            loss.backward(retain_graph=True)

    def test_gradcheck_single_input(self):
        def check(fast_mode):
            def f(inp):
                return inp.mul(5)

            gradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )
            gradgradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )

        check(fast_mode=True)
        check(fast_mode=False)

    @parametrize(
        "layout",
        (
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ),
    )
    def test_gradcheck_input(self, layout):
        if layout in {torch.sparse_bsr, torch.sparse_bsc}:
            blocksize = (2, 2)
            size = (4, 8)
        else:
            blocksize = None
            size = (2, 2)

        def check(fast_mode, masked):
            def fn(sparse):
                return torch.sum(sparse)

            gradcheck(
                fn,
                torch.rand(size, dtype=torch.double)
                .to_sparse(layout=layout, blocksize=blocksize)
                .requires_grad_(),
                masked=masked,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        for fast_mode, masked in product(*[(True, False)] * 2):
            check(fast_mode=fast_mode, masked=masked)

    def test_gradcheck_nondeterministic(self):
        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        def check(fast_mode):
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-6),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradgradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-12),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            gradcheck(
                lambda x: NonDetFunc.apply(x, 1e-6),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            gradgradcheck(
                lambda x: NonDetFunc.apply(x, 1e-12),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_validates_inputs(self):
        def check(fast_mode):
            x = torch.rand(10, requires_grad=True).to_sparse()
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    check_batched_grad=False,
                    atol=1e-1,
                    fast_mode=fast_mode,
                    masked=True,
                )
            )
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=False,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(masked_grad=False),
                    (x,),
                    masked=False,
                    atol=1e-1,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when none of the inputs require grad (always raises even if raise_exception=False)
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(
                ValueError, "at least one input tensor to require gradient"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

            # (warning) when inputs are not double precision
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning, "Input #0 requires gradient and is not a double precision"
            ):
                self.assertTrue(
                    gradcheck(lambda x: x, (x,), atol=1e-1, fast_mode=fast_mode)
                )

            # when layout is not mkldnn(aka has strides) and input has a dimension with stride 0. (always raises
            # even if raise_exception=False)
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))
            with self.assertRaisesRegex(
                RuntimeError, "The 0th input has a dimension with stride 0"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_validates_input_mkldnn(self):
        # when mkldnn inputs, forward mode testing is not allowed
        # Update tolerances below to make sure the gradient match even in single precision floats
        # Use the warning assert to hide the float32 warning
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not support for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=False,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not support for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=True,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_test_outputs(self):
        def check(fast_mode):
            # when sparse outputs (always raise even if raise_exception=False)
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                ValueError, "Sparse output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x,
                    (x,),
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

            # when mkldnn outputs (always raise even if raise_exception=False)
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(
                ValueError, "MKLDNN output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x.to_mkldnn(),
                    (root,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_no_differentiable_outputs(self):
        def check(fast_mode):
            # When none of the outputs are differentiable, but numerical gradient is not zero
            x = torch.ones((1,), requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "Numerical gradient for function expected to be zero"
            ):
                gradcheck(lambda x: torch.tensor([x]), x)
            self.assertFalse(
                gradcheck(
                    lambda x: torch.tensor([x]),
                    x,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # succeed when no outputs at all
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        def check(fast_mode):
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            # runtime error while compute batched grad (print big error)
            with self.assertRaisesRegex(
                RuntimeError,
                "gradcheck or gradgradcheck failed while testing batched gradient",
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_backward_mul_by_grad_output(self):
        # when grad_input is sparse and has incorrect sparse_dim/dense_dim
        def check(fast_mode):
            def fn(x):
                def hook(grad):
                    if grad is not None:
                        return grad.to_dense().to_sparse(1)
                    return grad

                y = x.clone()
                y.register_hook(hook)
                return y.to_dense()

            x = torch.ones((2, 2), dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                RuntimeError, "grad is sparse tensor, but has incorrect sparse_dim"
            ):
                gradcheck(
                    fn,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    fn,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when backward not multiplied by grad_output (non-sparse case)
            def fn2(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "backward not multiplied by grad_output"
            ):
                gradcheck(fn2, (x,), atol=1e-1, fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(
                    fn2, (x,), atol=1e-1, raise_exception=False, fast_mode=fast_mode
                )
            )

            # when backward not multiplied by grad_output (sparse case)
            def fn3(x):
                y = x.clone().to_dense()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                RuntimeError, "backward not multiplied by grad_output"
            ):
                gradcheck(
                    fn3,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            self.assertFalse(
                gradcheck(
                    fn3,
                    (x,),
                    atol=1e-1,
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # when layout of grad_input is not the same as input
            class Test(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, x):
                    return x.to_sparse()

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "grad is incorrect layout"):
                gradcheck(
                    Test.apply, (x,), check_batched_grad=False, fast_mode=fast_mode
                )
            self.assertFalse(
                gradcheck(
                    Test.apply,
                    (x,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_undefined_grad(self):
        def check(fast_mode):
            # when encounter runtime error while running backward
            def fn(x):
                def hook(x):
                    if x is None:
                        raise RuntimeError("x is undefined")

                y = x.clone()
                y.register_hook(hook)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning,
                "Backwards compatibility: New undefined gradient support checking feature",
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected backward function to handle undefined output grads",
                ):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                self.assertFalse(
                    gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
                )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_jacobian_mismatch(self):
        def check(fast_mode):
            def fn(x):  # R -> R, C -> C
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn, (x_c,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False)
            )

            def fn2(x):  # R -> C
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn2, (x,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn2, (x,), raise_exception=False, fast_mode=False)
            )

            def fn3(x):  # C -> R
                y = torch.real(x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn3, (x_c,), fast_mode=False)
            self.assertFalse(
                gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False)
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_dense_and_sparse_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x * y.coalesce().to_dense()

            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            b = torch.rand(2, 2, dtype=torch.double).to_sparse().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn,
                    (a, b),
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_multiple_mkldnn_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x + y.to_dense()

            a = torch.rand(10, requires_grad=True)
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn, (a, b), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

            def fn2(x, y):
                return x.to_dense() + y.to_dense()

            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(
                gradcheck(
                    fn, (a, c), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        def check(fast_mode):
            def fn(x):
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x

            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same shape when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            def fn2(x):
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x

            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same dtype when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_complex_non_complex_outputs(self):
        def fn(x, y):
            z = torch.complex(x, y)
            return z, x + 1

        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        self.assertTrue(gradcheck(fn, (a, b)))

        def fn2(z):
            return z, torch.real(z)

        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        self.assertTrue(gradcheck(fn2, (c)))

    def test_gradcheck_get_numerical_jacobian(self):
        # get_numerical_jacobian is deprecated and no longer used internally by gradcheck
        from torch.autograd.gradcheck import get_numerical_jacobian

        def fn(inputs):
            # get_numerical_jacobian requires fn to take inputs as a tuple
            # and returns the jacobian wrt the first output
            x = inputs[0]
            y = inputs[1]
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))

        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6, grad_out=2.0)

    def test_gradcheck_get_analytical_jacobian(self):
        from torch.autograd.gradcheck import get_analytical_jacobian

        def fn(x, y):
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        outputs = fn(a, b)
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a, b), outputs[0])
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        self.assertTrue(reentrant)

        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        outputs = NonDetFunc.apply(a, 1e-6)
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a,), outputs)
        self.assertFalse(reentrant)

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobians, _, _, _ = get_analytical_jacobian((a,), outputs, grad_out=2.0)

    def test_gradcheck_custom_error(self):
        from torch.autograd.gradcheck import GradcheckError

        def check(fast_mode):
            def fn(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(
                GradcheckError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            def fn2(x):
                raise RuntimeError("Not a GradcheckError!")

            # Checks that when raise_exception=False, non-GradcheckErrors are not caught by gradcheck
            with self.assertRaisesRegex(RuntimeError, "Not a GradcheckError!"):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_forward_ad(self):
        def fn(x, y):
            return x + y, y

        def bad_fn(x, y):
            # Hacky way to check if we're currently inside a forward ad level
            is_running_forward_ad = fwAD._current_level >= 0

            if is_running_forward_ad:
                y_p, y_d = fwAD.unpack_dual(y)
                y = fwAD.make_dual(y_p, y_d * 1.1)

            return x + y, y

        err_msg = "Jacobian computed with forward mode mismatch for output 0 with respect to input 1"

        for fast_mode in [True, False]:
            # Test for all inputs and outputs being real
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            def basic_mul(x):
                return torch.view_as_real(torch.resolve_conj(x * 1j))

            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)

            # Test for one input and one output being complex
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # Test for all inputs and outputs being complex
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

    def test_gradcheck_forward_ad_runs_with_no_requires_grad(self):
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        # This test checks that when the inputs are passed to the function they should not have
        # requires_grad=True even though they may have requires_grad=True when passed
        # to gradcheck
        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                if fwAD._current_level >= 0:
                    self.assertFalse(x.requires_grad)
                    self.assertFalse(y.requires_grad)
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                return x_t, y_t

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=False)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

    def test_gradcheck_forward_ad_respects_requires_grad(self):
        # Currently requires_grad is used as a easy way for gradcheck to know
        # which inputs of the function are meant to be differentiable
        jvp_count = [0]

        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                jvp_count[0] += 1
                return x_t, y_t

        # NB: In slow gradcheck we need to loop through numel times so use numel = 1 to ensure
        #     that fast and slow have the same counts
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=True)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )
        self.assertEqual(jvp_count[0], 2)  # (2) once per input
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )
        self.assertEqual(
            jvp_count[0], 6
        )  # (+4): (once with normal ZT (+1), once with efficient ZT (+1)) for each input (x2)
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
        self.assertEqual(
            jvp_count[0], 12
        )  # (+6): (compute batch of 2 with vmap (+1), with a loop (+2)) for each input (x2)
        jvp_count = [0]

        # Repeat the previous test except we mark one input with requires_grad=False
        # NB: _test_undefined_forward_mode is only (+1), when function has single differentiable input, not (+2)!
        #     Otherwise, other counts are halved.
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=False)
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
        self.assertEqual(jvp_count[0], 5)  # 1 + 1 + 3

    def test_gradcheck_check_forward_or_backward_only(self):
        """Depending on settings for check_forward_ad and check_backward_ad, the
        correct codepaths should be reached (or not reached)
        """
        fwd_fail_err_msg = "FAIL FWD"
        bwd_fail_err_msg = "FAIL BWD"

        class UserFn(Function):
            @staticmethod
            def forward(ctx, foo, fwd_bad, bwd_bad):
                ctx.fwd_bad = fwd_bad
                ctx.bwd_bad = bwd_bad
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)
                else:
                    return 2 * gO, None, None

            @staticmethod
            def jvp(ctx, gI, _1, _2):
                if ctx.fwd_bad:
                    raise RuntimeError(fwd_fail_err_msg)
                else:
                    return 2 * gI

        for fast_mode in (True, False):
            for check_forward_ad in (True, False):
                for check_backward_ad in (True, False):
                    for fwd_bad in (True, False):
                        for bwd_bad in (True, False):
                            fwd_should_fail = fwd_bad and check_forward_ad
                            bwd_should_fail = bwd_bad and check_backward_ad

                            def run():
                                gradcheck(
                                    UserFn.apply,
                                    (x, fwd_bad, bwd_bad),
                                    check_forward_ad=check_forward_ad,
                                    check_backward_ad=check_backward_ad,
                                    check_undefined_grad=check_backward_ad,
                                    check_batched_grad=check_backward_ad,
                                    fast_mode=fast_mode,
                                )

                            x = torch.rand(2, dtype=torch.double, requires_grad=True)

                            if not check_forward_ad and not check_backward_ad:
                                with self.assertRaisesRegex(
                                    AssertionError, "Expected at least one of"
                                ):
                                    run()
                                continue

                            if not fwd_should_fail and not bwd_should_fail:
                                run()
                            else:
                                # If both fail, backward AD failure "hides" forward AD failure
                                if fwd_should_fail:
                                    fail_msg = fwd_fail_err_msg
                                if bwd_should_fail:
                                    fail_msg = bwd_fail_err_msg
                                with self.assertRaisesRegex(RuntimeError, fail_msg):
                                    run()

    def test_gradcheck_forward_ad_batched_grad(self):
        x = torch.rand(2, dtype=torch.double, requires_grad=True)

        # multiple inputs and outputs with non-tensors inputs
        def fn1(a: torch.Tensor, b: int):
            return a.clone(), a + 1

        gradcheck(
            fn1,
            (x, 1),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_undefined_grad=False,
            check_batched_forward_grad=True,
        )

        # unrelated inputs: tangent for c is None
        def fn2(a: torch.Tensor, c: torch.Tensor):
            return a.clone()

        gradcheck(
            fn2,
            (x, x.clone()),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_undefined_grad=False,
            check_batched_forward_grad=True,
        )

        class Fn(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                return gO * 2

            @staticmethod
            def jvp(ctx, gI):
                torch.randn_like(gI)
                return gI * 2

        msg = "vmap: We do not yet support calling random operations inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            gradcheck(
                Fn.apply, (x,), check_forward_ad=True, check_batched_forward_grad=True
            )

    def test_version_counter(self):
        x = torch.randn(1, 2)

        # In-place op bumps version
        x_saved_version = x._version
        x.add_(1).add_(1)
        self.assertTrue(x._version > x_saved_version)

        # Differentiable view shares version counter
        xz = x[:]
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

        # `x.data = y` preserves version counter of `x`
        x_saved_version = x._version
        x.data = torch.randn(2, 3)
        self.assertTrue(x._version == x_saved_version)
        x.add_(1)
        self.assertTrue(x._version > x_saved_version)
        # Make sure `x` is still using the same version counter it shares with `xz`
        self.assertTrue(x._version == xz._version)

        # In-place op on `xz` also updates version of `x`,
        # because they share the version counter
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while sparse tensor has impl
        # of type `SparseTensorImpl`.
        x = torch.randn(1, 2)
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        with self.assertRaisesRegex(RuntimeError, "incompatible tensor type"):
            x.data = x_s

    def test_set_data_preserve_pyobj(self):
        a = torch.randn(1, 2)
        b = torch.randn(1, 2)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    def test_set_data_self_requires_grad(self):
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0)
        c = torch.tensor(3, dtype=torch.int64)
        a.data = b
        with self.assertRaisesRegex(
            RuntimeError, "must be floating point or complex dtype"
        ):
            a.data = c

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        code = """import torch
from torch.autograd import Function
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad

# Run on cuda if it is available to ensure that the worker thread
# is properly initialized by the time we exit.
device = "cuda" if torch.cuda.is_available() else "cpu"

for shape in [(1,), ()]:
    v = torch.ones(shape, requires_grad=True, device=device)
    MyFunction.apply(v).backward()
"""
        s = TestCase.runWithPytorchAPIUsageStderr(code)
        # The autograd engine creates worker threads only when GPU devices are present.
        # So make sure that we do shutdown threads when we're testing cuda and make sure
        # that there is no thread to shutdown when we're not using cuda.
        if TEST_CUDA or torch.backends.mps.is_available() or torch.xpu.is_available():
            self.assertRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")
        else:
            self.assertNotRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")

    @unittest.skipIf(
        IS_MACOS,
        "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941",
    )
    def test_deep_reentrant(self):
        class DeepReentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x

        # Test stack overflow escape mechanism
        v = torch.tensor(2000.0, requires_grad=True)
        # This will cause stack overflow if reentrant calls are handled
        # in the same thread recursively
        DeepReentrant.apply(v).sum().backward()

        # Test stack overflow escape mechanism multiple times
        # to ensure reusing workers in the pool works fine
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()

    def test_reentrant_priority(self):
        order = []

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        class Reentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")

    @slowTest
    def test_checkpointing(self):
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        feat_combined = []
        for _ in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r, use_reentrant=True)
            feat_combined.append(feat_r)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

    def _test_checkpointing_non_reentrant_autocast(self, device_type):
        for enabled in [True, False]:

            def foo(x, y, z):
                # torch.mm is on autocast's list of ops that should run in
                # the autocast precision
                x = torch.mm(x, y)
                y = torch.mm(x, z)
                z = torch.mm(z, z)
                expected_dtype = torch.float32 if not enabled else torch.bfloat16
                self.assertEqual(expected_dtype, z.dtype)
                return z

            x = torch.randn(3, 3, requires_grad=True)
            y = torch.randn(3, 3, requires_grad=True)
            z = torch.randn(3, 3, requires_grad=True)
            if device_type == "cuda":
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()

            with torch.autocast(
                enabled=enabled, device_type=device_type, dtype=torch.bfloat16
            ):
                loss = checkpoint(foo, x, y, z, use_reentrant=False)
                loss = loss.sum()

            # Without saving + recasting the autocast type, would raise error in autograd
            # about mismatched dtypes.
            loss.backward()  # triggers recomputation to check it runs in bfloat

    def test_checkpointing_non_reentrant_autocast_cpu(self):
        """
        Test that autocast args such as the dtype are preserved during non-reentrant
        checkpoint recomputation on CPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cpu")

    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "Test requires CUDA bf16 support",
    )
    def test_checkpointing_non_reentrant_autocast_gpu(self):
        """
        Test that autocast args/kwargs such as the dtype are preserved during
        non-reentrant checkpoint recomputation on GPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cuda")

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    @slowTest
    def test_checkpointing_without_reentrant_memory_savings(self):
        class MyModel(nn.Module):
            def __init__(self, n, use_checkpoint, use_reentrant):
                super().__init__()
                self.n = n
                self.use_checkpoint = use_checkpoint
                self.use_reentrant = use_reentrant
                self.layers = nn.ModuleList()
                for _ in range(self.n):
                    layer = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    self.layers.append(layer)
                # pre-allocate the grad so that increased memory usage is mainly
                # due to activations.
                for layer in self.layers:
                    for lin in layer:
                        lin.weight.grad = torch.ones_like(lin.weight)
                        lin.bias.grad = torch.ones_like(lin.bias)

            def forward(self, x):
                for i in range(self.n):
                    if not self.use_checkpoint:
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(
                            self.layers[i], x, use_reentrant=self.use_reentrant
                        )

                return x

        model_no_checkpoint = MyModel(
            8, use_checkpoint=False, use_reentrant=False
        ).cuda()
        model_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=True
        ).cuda()
        model_no_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=False
        ).cuda()

        x = torch.randn(100, 256, requires_grad=True, device="cuda")

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        loss = model_no_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        self.assertTrue(mem_reentrant_checkpoint < mem_no_checkpoint)
        self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)

    def test_checkpointing_without_reentrant_custom_function_works(self):
        msg = "Unpack is being triggered for a tensor that was already unpacked once"

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y, z):
                w = x * y * z
                out = w + w
                ctx.save_for_backward(x, y, z, w, out)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                x, y, z, w, out = ctx.saved_tensors
                # Accessing the saved Tensors a second time will raise because
                # recomputed tensors get cleared as soon as they are unpacked.
                # A recomputation is only triggered if your backward has a new
                # graph-task id.
                with self.assertRaisesRegex(RuntimeError, msg):
                    x_2, y_2, z_2, w_2, out_2 = ctx.saved_tensors
                return x, y, z

        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = torch.tensor(3.0, requires_grad=True)

        def foo(x, y, z):
            x = x * y * z
            y = y * y * z
            z = z * z
            out = MyFunc.apply(x, y, z)
            return out

        out = checkpoint(foo, x, y, z, use_reentrant=False)
        out.sum().backward()

    def test_checkpointing_without_reentrant_with_context_fn(self):
        class VerboseTorchDispatchMode(TorchDispatchMode):
            def __init__(self) -> None:
                self.operators = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                self.operators.append(func.__name__)
                return func(*args, **kwargs)

        x = torch.tensor(1.0, requires_grad=True)
        verbose_mode = VerboseTorchDispatchMode()

        def context_fn():
            return verbose_mode, contextlib.nullcontext()

        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        self.assertEqual(verbose_mode.operators, ["exp.default"])

        verbose_mode.operators = []

        def context_fn():
            return contextlib.nullcontext(), verbose_mode

        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        out.backward()
        self.assertEqual(verbose_mode.operators, ["exp.default", "detach.default"])

        with self.assertRaisesRegex(
            Exception, "only supported when use_reentrant=False"
        ):
            out = checkpoint(
                lambda x: x.sin(), x, use_reentrant=True, context_fn=context_fn
            )

    def test_checkpoint_warns_if_use_reentrant_not_passed_explcitly(self):
        a = torch.randn(1, requires_grad=True)

        # Passing explicitly should not warn
        self.assertNotWarn(lambda: checkpoint(lambda x: x, a, use_reentrant=False))

        # Not passing explicitly warns
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint(lambda x: x, a)

    def test_checkpoint_sequential_warns_if_use_reentrant_not_passed_explcitly(self):
        a = torch.randn(3, requires_grad=True)
        modules_list = [
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
        ]

        # Passing explicitly should not warn
        self.assertNotWarn(
            lambda: checkpoint_sequential(modules_list, 3, a, use_reentrant=False)
        )

        # Not passing explicitly warns
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint_sequential(modules_list, 3, a)

    @skipIfTorchDynamo("GraphExecGroup does not support compile")
    def test_checkpoint_graph_execution_group(self):
        def run(use_graph_execution_group):
            counter = [0]

            def fn(x):
                counter[0] += 1
                y = x.sin().cos()
                z = y.sin().cos()
                return y, z

            x = torch.randn(3, 3, requires_grad=True)

            y, z = checkpoint(fn, x, use_reentrant=False)

            group = torch.utils.checkpoint.GraphExecGroup()

            ctx = contextlib.nullcontext()
            if use_graph_execution_group:
                ctx = group

            with ctx:
                (grad_y,) = torch.autograd.grad(
                    z, inputs=(y,), grad_outputs=(torch.ones(3, 3),)
                )

                (grad_x,) = torch.autograd.grad(
                    y,
                    inputs=(x,),
                    grad_outputs=(grad_y,),
                )

            if use_graph_execution_group:
                self.assertEqual(counter[0], 2)
            else:
                self.assertEqual(counter[0], 3)

        run(use_graph_execution_group=True)
        run(use_graph_execution_group=False)

        # Test the not actually disjoint case (using retain_graph=True since
        # otherwise autograd itself will catch this)
        def fn(x):
            return x.sin().cos()

        x = torch.randn(3, 3, requires_grad=True)
        out = checkpoint(fn, x, use_reentrant=False)
        with torch.utils.checkpoint.GraphExecGroup():
            # Under this context, we will enforce that two backward are disjoint
            # even if retain_graph=True.
            out.sum().backward(retain_graph=True)
            with self.assertRaisesRegex(
                RuntimeError, "Performing two backward calls that overlap"
            ):
                out.sum().backward()

    def test_checkpoint_detects_non_determinism(self):
        def save_3_tensors(x):
            out = x.sin().exp()
            out = out.sin()
            return out

        def save_2_tensors(x):
            return x.sin().exp()

        def save_2_tensors_alt(x):
            return x.sin() * torch.tensor([1.0, 2.0])

        def get_non_det_fn(orig_fn, recompute_fn):
            counter = [0]

            def fn(x):
                if counter[0] == 0:
                    counter[0] += 1
                    return orig_fn(x)
                else:
                    return recompute_fn(x)

            return fn

        a = torch.randn(1, requires_grad=True)

        # Save fewer tensors during recompute
        fn = get_non_det_fn(orig_fn=save_3_tensors, recompute_fn=save_2_tensors)
        with self.assertRaisesRegex(
            RuntimeError, "A different number of tensors was saved"
        ):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()

        # Save more tensors during recompute
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            with self.assertRaisesRegex(
                RuntimeError, "trying to save more tensors during recomputation"
            ):
                out = checkpoint(fn, a, use_reentrant=False)
                out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        # If early stopping is enabled, we would not raise (the results would be correct anyway)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()

        # Save the same number of tensors but the shape is different
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, "tensors have different metadata"):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()

        # Get the debug message if debug=True
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError,
            "You are seeing this error because you passed `debug=True` to checkpoint",
        ):
            out = checkpoint(fn, a, use_reentrant=False, debug=True)
            out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError,
            "You are seeing this error because you passed `debug=True` to checkpoint",
        ):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
                out = checkpoint(fn, a, use_reentrant=False, debug=False)
                out.backward()

        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)

        with self.assertRaisesRegex(
            RuntimeError, "Recomputed values for the following tensors have different"
        ):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(False):
                out = checkpoint(fn, a, use_reentrant=False, debug=True)
                out.backward()

    def test_access_saved_tensor_twice_without_recomputation_works(self):
        count = [0]

        def foo(a):
            count[0] += 1
            b = a * a
            c = a * b
            d = torch.exp(a)
            return d

        a = torch.randn(5, requires_grad=True)
        d = checkpoint(foo, a, use_reentrant=False)
        self.assertEqual(count[0], 1)
        # Recomputed variables only persist within a particular backward call.
        # If _saved_result is accessed outside of a backward, it will trigger
        # a recompute. And afterwards, those recomputed results are immediately
        # cleared.
        d.grad_fn._saved_result
        self.assertEqual(count[0], 2)
        # Second access will trigger another recompute
        d.grad_fn._saved_result
        self.assertEqual(count[0], 3)
        # Backward clears the saved variable
        d.sum().backward()
        self.assertEqual(count[0], 4)
        # Now it raises an error
        with self.assertRaisesRegex(
            RuntimeError,
            "or directly access saved tensors after they have already been freed",
        ):
            d.grad_fn._saved_result

    @slowTest
    @parametrize("input_requires_grad", [True, False])
    def test_checkpointing_without_reentrant(self, input_requires_grad):
        """
        Basic test for checkpoint without reentrant autograd.
        """
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        # Module holder for testing activation checkpointing with no_reentrant
        # supports kwargs.
        class MyModule(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.module = mod

            def forward(self, data):
                return self.module(data)

        module = MyModule(mod=module)

        # Run model with and without checkpointing and verify gradients are
        # equivalent, regardless of if inputs require grads or not.
        module_copy = deepcopy(module)

        feat_combined = []
        feat_combined_no_checkpoint = []
        for _ in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = input_requires_grad
            data_r_copy = data_r.clone()
            feat_r = checkpoint(module, data=data_r, use_reentrant=False)
            feat_combined.append(feat_r)
            feat_r_no_checkpoint = module_copy(data_r)
            feat_combined_no_checkpoint.append(feat_r_no_checkpoint)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()
        mean_combined_no_checkpoint = torch.stack(feat_combined_no_checkpoint).mean()
        mean_combined_no_checkpoint.backward()

        for checkpoint_param, param in zip(
            module.parameters(), module_copy.parameters()
        ):
            self.assertEqual(checkpoint_param.grad, param.grad)

    def test_checkpoint_valid_reset_on_error(self):
        a = torch.randn(2, 2, requires_grad=True)

        with self.assertRaisesRegex(
            Exception, "torch.utils.checkpoint is incompatible"
        ):
            b = checkpoint(torch.exp, a, use_reentrant=True).sum()
            torch.autograd.grad(b, (a,))

        c = checkpoint(torch.exp, a, use_reentrant=True).sum()
        c.backward()

    @parametrize("use_reentrant", [True, False])
    def test_checkpointing_without_reentrant_detached_tensor(self, use_reentrant):
        class NoGradModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.lin2 = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                with torch.no_grad():
                    return self.lin2(self.linear(x))

        module = NoGradModule()

        err_ctx = (
            self.assertRaisesRegex(
                RuntimeError, "none of output has requires_grad=True"
            )
            if use_reentrant
            else contextlib.nullcontext()
        )

        a = torch.randn(2, 2, requires_grad=True)
        for _ in range(3):
            with err_ctx:
                # out does not require grad
                out = checkpoint(module, a, use_reentrant=use_reentrant)
                # Make loss require grad, otherwise we would run into
                # "element 0 of tensors does not require grad and does not have a grad_fn"
                out += a
                out.sum().backward()

    def test_checkpointing_without_reentrant_saved_object_identity(self):
        x_backward = None

        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(y)
                return x

            @staticmethod
            def backward(ctx, x):
                nonlocal x_backward
                (x_backward,) = ctx.saved_tensors
                return x, None

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=False)

        Test.apply(a, b).backward()
        self.assertIs(b, x_backward)

        x_backward = None
        checkpoint(Test.apply, a, b, use_reentrant=False).backward()
        self.assertIs(b, x_backward)

    def test_checkpointing_without_reentrant_correct_grad(self):
        """
        Verifies that correct gradients are calculated for checkpoint
        without reentrant autograd, for both backward() and autograd.grad().
        """
        a = torch.randn(2, 2, requires_grad=True)

        b = torch.exp(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None
        c = checkpoint(torch.exp, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        a.grad = None
        d = checkpoint(torch.exp, a, use_reentrant=False).sum()
        (d_grad,) = torch.autograd.grad(d, (a,))

        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    @skipIfXpu(msg="torch._C._scatter Not implemented on XPU, issue #143239")
    def test_checkpointing_without_reentrant_dataparallel(self):
        """
        Verifies gradient correctness when checkpoint without reentrant autograd
        is used in conjunction with DataParallel.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                return self.linear(inp)

        a = torch.randn(2, 2, requires_grad=True)
        if torch.cuda.is_available():
            a = a.cuda()

        model = LinearModule()
        if torch.cuda.is_available():
            model = model.cuda()

        b = deepcopy(model)(a).sum()
        b.backward()
        b_grad = a.grad

        a.grad = None

        module = torch.nn.DataParallel(deepcopy(model))
        c = checkpoint(module, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad

        self.assertEqual(b_grad, c_grad)

    def test_checkpointing_without_reentrant_parameter_used_in_an_out(self):
        """
        Ensures that gradient hooks are only called once per tensor.
        """
        w = torch.randn(10, 10, requires_grad=True)
        count = 0

        def hook(grad):
            nonlocal count
            count += 1

        w.register_hook(hook)
        x = torch.rand(10, 10, requires_grad=True)
        h = w * x  # Using w outside the checkpoint
        out = checkpoint(
            lambda x: w * x, h, use_reentrant=False
        )  # Using w inside the checkpoint

        out.sum().backward()
        # should only call hook once
        self.assertEqual(count, 1)

    def test_checkpointing_without_reentrant_arbitrary_input_output(self):
        """
        Ensures checkpointing without reentrant autograd works with functions
        with arbitrary input/output structures.
        """

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Linear(5, 5, bias=False)

            def forward(self, dict_input):
                tensor = dict_input["tensor"]
                return {"result": self.layer(tensor)}

        model_no_checkpoint = MyModel()
        model_checkpoint_without_reentrant = deepcopy(model_no_checkpoint)

        inp = {"tensor": torch.randn(5, 5)}

        out_no_checkpoint = model_no_checkpoint(inp)["result"].sum()

        out_checkpoint = checkpoint(
            model_checkpoint_without_reentrant, inp, use_reentrant=False
        )["result"].sum()

        self.assertEqual(out_checkpoint, out_no_checkpoint)

        out_no_checkpoint.backward()
        out_checkpoint.backward()

        for param, checkpoint_param in zip(
            model_no_checkpoint.parameters(),
            model_checkpoint_without_reentrant.parameters(),
        ):
            self.assertEqual(param.grad, checkpoint_param.grad)

    def test_callback_adds_callback(self):
        called = [0]

        def callback_final():
            called[0] += 1

        def callback_adds_callback():
            called[0] += 1
            Variable._execution_engine.queue_callback(callback_final)

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                Variable._execution_engine.queue_callback(callback_adds_callback)
                return grad

        a = torch.rand((3, 3), requires_grad=True)
        b = MyFunc.apply(a)
        b.sum().backward()

        self.assertEqual(called[0], 2)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_callback_propagates_errors_from_device_thread(self):
        def callback():
            raise RuntimeError("blah")

        def hook_with_callback(*args):
            torch.autograd.Variable._execution_engine.queue_callback(callback)

        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        t.register_hook(hook_with_callback)
        output = t**2
        loss = output.sum()

        with self.assertRaisesRegex(RuntimeError, "blah"):
            loss.backward()

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        counter = {}
        counter["inner"] = 0
        counter["outer"] = 0

        def inc_inner_counter():
            counter["inner"] += 1

        def inc_outer_counter():
            counter["outer"] += 1

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 1 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_inner_counter)

                return input

        class MyReentrantFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 0 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                # Reentrant backward call.
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = (MyFunc.apply(tmp_inp)).sum()
                tmp_out.backward()
                return input

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        torch.autograd.backward([t3])

        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(ret["outer"], 1)
        self.assertEqual(ret["inner"], 0)

    def test_reentrant_with_callbacks_depth_1(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(ret["outer"], 0)
        self.assertEqual(ret["inner"], 1)

    def test_reentrant_with_callbacks_both_depths(self):
        # Verify callback is called twice.
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(ret["outer"], 1)
        self.assertEqual(ret["inner"], 1)

    def test_reentrant_with_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def add_gradient_penalty_to_grad(grad):
            handle.remove()
            old_param_grad = grad
            param.grad = None
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                out = ((g * 2) + new_param).sum()
                out.backward()
            res = g.grad + grad
            param.grad = old_param_grad
            return res

        handle = param.register_hook(add_gradient_penalty_to_grad)
        # Forward pass
        tmp = param * param
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()

    def test_reentrant_with_non_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def manual_increase_gradient(grad):
            handle.remove()
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                out = ((g * 2) + 5).sum()
                out.backward()
            res = g.grad + grad
            return res

        # Forward pass
        tmp = param * param
        handle = tmp.register_hook(manual_increase_gradient)
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()
        self.assertEqual(param.grad, 6 * param)

    def test_grad_fn_attr_bindings(self):
        # Check that the getter of each type returns what we want
        # See `gen_autograd_functions.py` for how the getters are generated
        #
        # This test is only meant to check if the codegen'd bindings work
        # Please help update this test if you update the names of any the fields we check!
        #
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        out1 = torch.stack([a, b], dim=0)
        out2 = (a * 2) * b
        # TODO: I don't think we have a backward saving a list of tensors
        #       at the moment. It used to be stack, but for no reason...
        #       see discussion in #84993
        # self.assertEqual(out.grad_fn._saved_tensors, (a, b))              # TewnsorList -> Tuple[Tensor]
        self.assertEqual(out2.grad_fn._saved_self, a * 2)
        self.assertIsInstance(out2.grad_fn._saved_self, torch.Tensor)
        self.assertIsInstance(
            out2.grad_fn._raw_saved_self, torch._C._autograd.SavedTensor
        )
        self.assertEqual(out1.grad_fn._saved_dim, 0)  # int64_t -> int
        self.assertIsInstance(out1.grad_fn._saved_dim, int)

        out2.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)

        out2.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out2.grad_fn._saved_self
        # TODO: interestingly, this only happens if indexing into a list grad_fn._raw_saved_tensors[0],
        #       not when using a saved tensor, see discussion in #84993
        # with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
        #     out2.grad_fn._raw_saved_self
        self.assertEqual(out1.grad_fn._saved_dim, 0)

        a = torch.ones(2, 2, requires_grad=True)
        indices = torch.tensor([0, 1])
        out = a[:, indices]
        self.assertEqual(
            out.grad_fn._saved_indices, (None, indices)
        )  # c10::List<std::optional<Tensor>> -> Tuple[Tensor?]
        self.assertIsInstance(out.grad_fn._saved_indices[1], torch.Tensor)
        self.assertIsInstance(
            out.grad_fn._raw_saved_indices[1], torch._C._autograd.SavedTensor
        )
        self.assertEqual(
            out.grad_fn._saved_self_sym_sizes, a.shape
        )  # SymIntArrayRef -> Tuple[SymInt]
        self.assertIsInstance(out.grad_fn._saved_self_sym_sizes[0], int)

        out.grad_fn._raw_saved_indices[1].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            out.grad_fn._raw_saved_indices[0].register_hooks(lambda x: x, lambda x: x)

        out = a.mean()
        self.assertEqual(
            out.grad_fn._saved_self_sym_sizes, a.shape
        )  # IntArrayRef -> Tuple[int]

        a = torch.ones(2, 2, requires_grad=True)
        out = a * a
        out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after it has been freed"):
            out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, 4, mode="linear")
        self.assertEqual(
            out.grad_fn._saved_output_size, (4,)
        )  # std::optional<IntArrayRef> -> int[]?
        self.assertIsInstance(out.grad_fn._saved_output_size[0], int)
        self.assertEqual(out.grad_fn._saved_align_corners, False)  # bool -> bool
        self.assertIsInstance(out.grad_fn._saved_align_corners, bool)
        if hasattr(out.grad_fn, "_saved_scale_factors"):
            self.assertIsNone(
                out.grad_fn._saved_scale_factors
            )  # std::optional<ArrayRef<double>> -> float[]?
        else:
            self.assertIsNone(
                out.grad_fn._saved_scales
            )  # std::optional<ArrayRef<double>> -> float[]?

        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = nn.Conv2d(1, 1, 3)(a)
        self.assertEqual(
            out.grad_fn._saved_bias_sym_sizes_opt, (1,)
        )  # std::optional<SymIntArrayRef> -> SymInt[]?
        out = nn.Conv2d(1, 1, 3, bias=False)(a)
        # TODO: This is BAD! we converted a std::nullopt into a (0,)
        self.assertEqual(out.grad_fn._saved_bias_sym_sizes_opt, (0,))

        a = torch.ones(1, 3, 3, requires_grad=True)
        out = torch.addbmm(a.squeeze(0), a, a)
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_0, 1)  # int64_t
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_1, 3)  # int64_t

        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = torch.nn.functional.unfold(a, 3)
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_2, 3)  # SymInt
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_1, 3)  # SymInt

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, scale_factor=0.5, mode="linear")
        self.assertEqual(out.grad_fn._saved_scales, 0.5)

        a = torch.ones(2, 2, requires_grad=True)
        out = torch.pdist(a, p=1)
        self.assertEqual(out.grad_fn._saved_p, 1.0)  # double -> float
        self.assertIsInstance(out.grad_fn._saved_p, float)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.logit(a, 1.0)
        self.assertEqual(out.grad_fn._saved_eps, 1.0)  # c10:optional<double> -> float?
        self.assertIsInstance(out.grad_fn._saved_eps, float)
        out = torch.logit(a)
        self.assertIsNone(out.grad_fn._saved_eps)

        if torch._C.has_lapack:
            a = torch.ones(1, 1, requires_grad=True)
            q, r = torch.linalg.qr(a, mode="reduced")
            self.assertEqual(q.grad_fn._saved_mode, "reduced")  # std::string -> str

        a = torch.tensor([1.0], requires_grad=True)
        out = torch.div(a, 2.0, rounding_mode="trunc")
        self.assertEqual(
            out.grad_fn._saved_rounding_mode, "trunc"
        )  # std::optional<std::string> -> str?
        out = torch.div(a, 2.0, rounding_mode=None)
        self.assertIsNone(
            out.grad_fn._saved_rounding_mode
        )  # std::optional<std::string> -> str?

        x = torch.zeros(5, requires_grad=True)
        out = torch.threshold(x, threshold=(1 + 0j), value=(1 + 0j))
        self.assertIsInstance(
            out.grad_fn._saved_threshold, complex
        )  # Scalar(complex double) -> complex
        cfloat = torch.tensor(1 + 0j, dtype=torch.complex64)
        out = torch.threshold(x, threshold=cfloat, value=(1 + 0j))
        self.assertIsInstance(
            out.grad_fn._saved_threshold, complex
        )  # Scalar(complex float) -> complex
        out = torch.threshold(x, threshold=1.0, value=1.0)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, float
        )  # Scalar(floating point) -> float
        out = torch.threshold(x, threshold=1, value=1)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, int
        )  # Scalar(integral) -> int
        out = torch.threshold(x, threshold=False, value=False)
        self.assertIsInstance(
            out.grad_fn._saved_threshold, bool
        )  # Scalar(bool) -> bool

        a = torch.ones(2, 2, requires_grad=True)
        out = a.as_strided((3,), (1,), 1)
        self.assertEqual(
            out.grad_fn._saved_storage_offset, 1
        )  # c10:optional<int64_t> -> int?
        self.assertIsInstance(out.grad_fn._saved_storage_offset, int)
        out = a.as_strided((3,), (1,))
        self.assertIsNone(out.grad_fn._saved_storage_offset)

        a = torch.ones(2, requires_grad=True)
        out = torch.tanh(a)
        self.assertEqual(out, out.grad_fn._saved_result)  # saved variable when output

        a = torch.randn(3, 5, requires_grad=True)
        b = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        out = loss(a, b)
        self.assertIsNone(out.grad_fn._saved_weight)
        loss = nn.NLLLoss(weight=torch.ones((5,)))
        out = loss(a, b)
        self.assertEqual(
            out.grad_fn._saved_weight, torch.ones((5,))
        )  # c10:optional<Tensor> -> Tensor?

        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out.grad_fn._saved_weight

        num_tensors = 3
        input_tensors = [
            torch.ones(2, 2, requires_grad=True) for _ in range(num_tensors)
        ]
        scalars = [
            0.0 for _ in range(num_tensors)
        ]  # ArrayRef<Scalar> -> Tuple[Scalar, ...]
        results = torch._foreach_maximum(input_tensors, scalars)
        for t in results:
            self.assertEqual(t.grad_fn._saved_scalars, scalars)

    def test_get_data_and_hooks_from_raw_saved_variable(self):
        def pack_hook(t):
            return t

        def unpack_hook(t):
            return t

        a = torch.tensor(2.0, requires_grad=True)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            b = a**2

        c = b.exp()
        d = c**2

        pow_sv = b.grad_fn._raw_saved_self
        exp_sv = c.grad_fn._raw_saved_result
        pow2_sv = d.grad_fn._raw_saved_self

        # Returns the packed object as-is
        self.assertTrue(pow_sv.data is a)
        self.assertTrue(pow_sv.unpack_hook is unpack_hook)
        # Returns the detached data when the output/leaf is saved
        self.assertFalse(exp_sv.data is c)
        self.assertIsNone(exp_sv.unpack_hook)
        # Returns the un-detached data when input is saved
        self.assertTrue(pow2_sv.data is c)
        self.assertIsNone(pow2_sv.unpack_hook)

    def test_cant_create_saved_tensors(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to create a SavedTensor object from Python is forbidden",
        ):
            torch.autograd.SavedTensor()

    def test_custom_function_saved_tensors(self):
        def getFn(save=True):
            class MyFn(Function):
                @staticmethod
                def forward(ctx, x):
                    if save:
                        ctx.save_for_backward(x, None)
                    return x

                @staticmethod
                def backward(ctx, g):
                    return g

            return MyFn

        a = torch.randn(5, requires_grad=True)

        y = getFn(True).apply(a)

        self.assertEqual((a, None), y.grad_fn.saved_tensors)
        saved = y.grad_fn._raw_saved_tensors
        self.assertIsInstance(saved[0], torch._C._autograd.SavedTensor)
        # We can't tell the underlying tensor is None without unpacking it
        self.assertIsInstance(saved[1], torch._C._autograd.SavedTensor)

        # We catch that error when the user calls register_hooks on it
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            saved[1].register_hooks(lambda x: x, lambda x: x)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(lambda x: x)
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(1, 1)
        saved[0].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "already been set"):
            saved[0].register_hooks(lambda x: x, lambda x: x)
        y.sum().backward()

        # Using a reference to the SavedTensor object after the
        # saved variables have been released can lead to undefined behavior
        del saved
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn._raw_saved_tensors
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn.saved_tensors

        y = getFn(False).apply(a)
        self.assertEqual(y.grad_fn.saved_tensors, ())
        self.assertEqual(y.grad_fn._raw_saved_tensors, ())

    def test_autograd_node_isinstance(self):
        # Node is a "virtual" base class of codegen'd nodes. This means that
        # isinstance and issubclass are overridden, but mro is unchanged
        Node = torch.autograd.graph.Node

        a = torch.rand(3, 3, requires_grad=True)
        b = a.exp()

        # Some nodes have codegened registrations to the torch._C._function module
        self.assertIsInstance(b.grad_fn, Node)
        self.assertTrue(issubclass(type(b.grad_fn), Node))
        self.assertTrue(Node not in type(b.grad_fn).mro())

        # Other nodes have manual registrations to the torch._C._function module
        self.assertNotIsInstance(torch._C._functions.AccumulateGrad, Node)
        self.assertTrue(issubclass(torch._C._functions.AccumulateGrad, Node))
        self.assertIsInstance(b.grad_fn.next_functions[0][0], Node)
        self.assertTrue(issubclass(torch._C._functions.DelayedError, Node))

        # Special cases
        self.assertNotIsInstance(None, Node)
        self.assertNotIsInstance(1, Node)
        self.assertNotIsInstance(Node, Node)
        self.assertTrue(issubclass(Node, Node))

        # Custom function case
        self.assertTrue(issubclass(torch.autograd.function.BackwardCFunction, Node))

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                self.assertIsInstance(ctx, Node)
                return x

            @staticmethod
            def backward(ctx, x):
                self.assertIsInstance(ctx, Node)
                return x

        out = Func.apply(a)
        self.assertIsInstance(out.grad_fn, Node)
        self.assertTrue(issubclass(type(out.grad_fn), Node))
        self.assertTrue(Node not in type(out.grad_fn).mro())
        out.sum().backward()

    def test_autograd_views_codegen(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks the behavior of two codegen functions (view_as and unbind)
        # with respect to view tracking and inplace operation on the output.

        def run_test(grad_mode, requires_grad, is_view, should_raise_tuple):
            def maybe_check_raise(fn, should_raise):
                self.assertTrue(should_raise is None or isinstance(should_raise, str))
                if should_raise is not None:
                    with self.assertRaisesRegex(RuntimeError, should_raise):
                        fn()
                else:
                    fn()

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.view_as(inp)
            # Are they differentiable views?
            self.assertTrue(out._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out.add_(1), should_raise_tuple[0])

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.unbind()
            # Are they differentiable views?
            self.assertTrue(out[0]._is_view() == is_view)
            self.assertTrue(out[1]._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out[0].add_(1), should_raise_tuple[1])
            maybe_check_raise(lambda: out[1].add_(1), should_raise_tuple[2])

        # should_raise contains None if it should not raise
        # should_raise contains a string of the error if it should raise
        # The 3 elements are for view_as, first output of unbind and second output of unbind
        run_test(
            grad_mode=True,
            requires_grad=False,
            is_view=True,
            should_raise_tuple=(None, None, None),
        )
        inp_change_err = (
            "Output {} of UnbindBackward0 is a view and is being modified inplace."
        )
        run_test(
            grad_mode=True,
            requires_grad=True,
            is_view=True,
            should_raise_tuple=(
                None,
                inp_change_err.format("0"),
                inp_change_err.format("1"),
            ),
        )
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        run_test(
            grad_mode=False,
            requires_grad=True,
            is_view=True,
            should_raise_tuple=(leaf_grad_err, leaf_grad_err, leaf_grad_err),
        )
        run_test(
            grad_mode=False,
            requires_grad=False,
            is_view=True,
            should_raise_tuple=(None, None, None),
        )

    def test_inplace_not_requires_grad(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        # Original Tensor does not require grad
        a = torch.rand(1, 2)

        # Tensor being written does require grad
        b = torch.rand(1, requires_grad=True)

        # Take an invalid view on 'a' that should raise an error (warns during deprecation)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            view_a += b

        # Extra test for copy_ that is a manual implementation and could be easily
        # forgotten when the codegen is updated (warns during deprecation)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            view_a.copy_(b)

        # Functions that should throw must properly throw
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]
        with self.assertRaisesRegex(
            RuntimeError,
            "This view is the output of a function that returns multiple views.",
        ):
            view_a.copy_(b)

        # Sanity check that views that should work still work
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    def _do_test_autograd_simple_views_python(self, dtype):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks the autograd.Function behavior when we return one or multiple outputs
        # while one of these is an input, a view of an input or of a temporary tensor.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]
        # This indicator is used to check if the argument `ga` contains non-zero values
        ga_nz = [False]

        class IdOneOutput(Function):
            @staticmethod
            def forward(ctx, a, make_view, pure_view):
                ctx._is_pure_view = pure_view
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a

            @staticmethod
            def backward(ctx, ga):
                bw_called[0] += 1
                return ga, None, None

        class IdTwoOutput(Function):
            @staticmethod
            def forward(ctx, a, b, make_view, pure_view):
                ctx._is_pure_view = pure_view
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                if ga.eq(0).all():
                    ga_nz[0] = False
                else:
                    ga_nz[0] = True
                return ga + gab, gab, None, None

        class ViewOfTemp(Function):
            @staticmethod
            def forward(ctx, a, make_view, pure_view):
                ctx._is_pure_view = pure_view
                ctx.save_for_backward(a)
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                b = a.clone()
                return b.select(0, 0)

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, 0).copy_(grad)
                return res, None, None

        fn_id_to_inplace_on_view_err_msg = {
            "one_output": (
                "Output 0 of IdOneOutputBackward is a view and is being "
                "modified inplace. This view was created inside a custom Function"
            ),
            "two_output": (
                "Output 0 of IdTwoOutputBackward is a view and is being modified inplace."
                " This view is the output of a function that returns multiple views.",
                "Pure view custom Function can only have one input Tensor and one output Tensor."
                " Open an issue if you need to support more.",
            ),
            "view_of_temp": (
                "Output 0 of ViewOfTempBackward is a view and is being "
                "modified inplace. This view was created inside a custom Function",
                "a view of a leaf Variable that requires grad is being used in an in-place operation",
            ),
        }

        for fn_id in ["one_output", "two_output", "view_of_temp"]:
            for inplace in [True, False]:
                for make_view in [True, False]:
                    for pure_view in [True, False]:
                        # Used for special casing the tests below
                        output_is_a_view = make_view or fn_id == "view_of_temp"

                        def fn(a, b):
                            # never modify a, b inplace for gracheck
                            a = a.clone()
                            b = b.clone()
                            if fn_id == "two_output":
                                tmp1, tmp2 = IdTwoOutput.apply(
                                    a, b, make_view, pure_view
                                )
                                if inplace:
                                    tmp1 += 3
                                    tmp2 += 3
                                else:
                                    tmp1 = tmp1 + 3
                                    tmp2 = tmp2 + 3
                                tmp = tmp1 * tmp2
                            else:
                                if fn_id == "one_output":
                                    tmp = IdOneOutput.apply(a, make_view, pure_view)
                                else:
                                    tmp = ViewOfTemp.apply(a + b, make_view, pure_view)
                                if inplace:
                                    tmp += 3
                                else:
                                    tmp = tmp + 3

                            return tmp.sum()

                        a = torch.ones(2, dtype=dtype, requires_grad=True)
                        b = torch.ones(2, dtype=dtype, requires_grad=True)

                        err_msg = fn_id_to_inplace_on_view_err_msg[fn_id][
                            int(pure_view)
                        ]

                        will_raise_error = (
                            (pure_view and fn_id == "two_output")
                            or (pure_view and fn_id == "view_of_temp" and inplace)
                            or (not pure_view and inplace and output_is_a_view)
                        )

                        if will_raise_error:
                            with self.assertRaisesRegex(RuntimeError, err_msg):
                                gradcheck(fn, (a, b), check_batched_grad=False)
                        else:
                            gradcheck(fn, (a, b), check_batched_grad=False)

                        # Was the custom backward called properly
                        bw_called[0] = 0
                        ga_nz[0] = True  # For the case where the backward is called

                        expected_called = 1
                        expected_ga_nz = True

                        if will_raise_error:
                            expected_called = 0
                            with self.assertRaisesRegex(RuntimeError, err_msg):
                                fn(a, b)
                        else:
                            fn(a, b).abs().backward()

                        if (
                            fn_id == "one_output"
                            and inplace
                            and output_is_a_view
                            and pure_view
                        ):
                            # We expect the op to have been replayed and we leveraged the pure view
                            # to re-create the graph, so the original backward was not called
                            expected_called = 0

                        self.assertTrue(bw_called[0] == expected_called)
                        self.assertTrue(ga_nz[0] == expected_ga_nz)

    def test_autograd_simple_views_python(self):
        self._do_test_autograd_simple_views_python(torch.double)
        self._do_test_autograd_simple_views_python(torch.cdouble)

    def test_autograd_inplace_views_creation_meta(self):
        # Tests creation_meta properly handled for inplace views

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, x):
                return x

        view_custom = Func.apply

        def run_test(
            fn, fn_type, grad_mode_view, grad_mode_iview, requires_grad, error1, error2
        ):
            # This test checks the behavior of inplace-view functions when
            # the views are created in grad mode or not
            base = torch.rand(2, 3, requires_grad=requires_grad).clone()
            # 1. Create a view with `grad_mode=grad_mode_view`
            with torch.set_grad_enabled(grad_mode_view):
                if fn_type == "multi_view":
                    inp = base.unbind()[0]
                elif fn_type == "custom":
                    inp = view_custom(base)
                else:
                    inp = base.view_as(base)

            # 2. Perform inplace view with `grad_mode=grad_mode_iview`
            with torch.set_grad_enabled(grad_mode_iview):
                if error1 is not None:
                    with self.assertRaisesRegex(RuntimeError, error1):
                        fn(inp)
                    return
                else:
                    # If error is None, check that runs without error
                    fn(inp)
            # 3. Do inplace on the (new) view
            if error2 is not None:
                with self.assertRaisesRegex(RuntimeError, error2):
                    inp.add_(1)
            else:
                # If error is None, check that runs without error
                inp.add_(1)

        no_grad_err = "A view was created in no_grad mode"
        multi_view_err = "function that returns multiple views"
        custom_err = "view was created inside a custom Function"

        def run_tests(fn):
            for fn_type in ("normal", "multi_view", "custom"):
                for grad_mode_view in (True, False):
                    for grad_mode_iview in (True, False):
                        for requires_grad in (True, False):
                            error1 = None  # expected error when we do inplace_view on original view
                            error2 = None  # expected error when we do inplace on the resulting view

                            if requires_grad:
                                if not grad_mode_view and grad_mode_iview:
                                    error1 = no_grad_err
                                if not grad_mode_view and not grad_mode_iview:
                                    error2 = no_grad_err

                                if fn_type == "multi_view":
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = multi_view_err
                                    if grad_mode_view and not grad_mode_iview:
                                        error2 = multi_view_err

                                if fn_type == "custom":
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = custom_err
                                    if grad_mode_view and not grad_mode_iview:
                                        error2 = custom_err

                            run_test(
                                fn,
                                fn_type,
                                grad_mode_view,
                                grad_mode_iview,
                                requires_grad,
                                error1,
                                error2,
                            )

        # This list was created by logging gen_inplace_or_view_type.py
        #   detach_ is excluded for this test because it cannot be applied to
        #   views and thus does not return a view
        run_tests(lambda v: v.as_strided_((1, 0), (2, 2)))
        run_tests(lambda v: v.transpose_(0, 0))
        run_tests(lambda v: v.t_())
        run_tests(lambda v: v.squeeze_(0))
        run_tests(lambda v: v.unsqueeze_(0))
        run_tests(lambda v: v.swapdims_(0, 0))
        run_tests(lambda v: v.swapaxes_(0, 0))

    def test_autograd_print_tensor(self):
        a = torch.ones(1, requires_grad=True)
        a_clone = a.clone()
        self.assertEqual(repr(a), "tensor([1.], requires_grad=True)")
        self.assertEqual(repr(a_clone), "tensor([1.], grad_fn=<CloneBackward0>)")

        with torch.no_grad():
            b = a[:]
            b *= 2

        # Special handling for printing view created in no-grad and modified
        # in-placed in no-grad.
        self.assertEqual(repr(b), "tensor([2.], grad_fn=<Invalid>)")

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                return x

        c = Func.apply(a)
        self.assertEqual(repr(c), "tensor([2.], grad_fn=<FuncBackward>)")

    def test_autograd_inplace_view_of_view(self):
        x = torch.zeros(2)
        with torch.no_grad():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the no_grad block"
        ):
            z /= 2

        x = torch.zeros(2)
        with torch.inference_mode():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the inference_mode"
        ):
            z /= 2

    # TODO This is not the correct behavior -
    # See https://github.com/pytorch/pytorch/issues/49825#issuecomment-794466627
    def test_autograd_inplace_views_cross_dtype(self):
        # This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b = b.transpose(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        non_inplace_grad = a_orig.grad

        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b.transpose_(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        inplace_grad = a_orig.grad

        # TODO: this is a bug!
        # once this is fixed, it should have the transpose removed:
        # self.assertEqual(non_inplace_grad, inplace_grad)
        self.assertEqual(non_inplace_grad.T, inplace_grad)

    def test_autograd_multiple_views_python(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks that multiples views in the forward are properly traced and how they
        # behave with respect to inplace operations.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]

        class ComplexView(Function):
            @staticmethod
            def forward(ctx, a, idx):
                res = a.narrow(0, idx, 1)
                res = a.select(0, idx)
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, ctx.idx).copy_(grad)
                return res, None

        a = torch.ones(2, requires_grad=True)
        idx = 1

        bw_called[0] = 0
        out = ComplexView.apply(a.clone(), idx)
        out.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        out = ComplexView.apply(a.clone(), idx)
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of ComplexViewBackward is a view and is being modified inplace",
        ):
            out += 1

    def test_autograd_python_custom_function_inplace(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks custom autograd.Function that perform inplace operations

        bw_called = [0]

        # I) Single output
        class MyAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        # No extra inplace
        c = MyAdder.apply(a.clone(), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c = MyAdder.apply(a.clone(), b)
        c += 2
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        bw_called[0] = 0
        c = MyAdder.apply(a.clone().view_as(a), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # Should not give non-inputs to mark_dirty
        class MyAdderBad(Function):
            @staticmethod
            def forward(ctx, a, b):
                c = 3 * a
                c.add_(b)
                ctx.mark_dirty(c)
                return c

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                grad = 3 * grad
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            MyAdderBad.apply(a.clone(), b)
        self.assertEqual(len(w), 1)

        # II) Multiple outputs
        class MyBadAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + gab

        # No extra inplace
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        c += 2
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        inplace_on_view_err = (
            "your Function modifies inplace an input that is a view of another Tensor"
        )
        with self.assertRaisesRegex(RuntimeError, inplace_on_view_err):
            c, d = MyBadAdder.apply(a.clone().view_as(a), b)

        # III) Inplace + other op
        class MyOutPlaceAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a.clone(), a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + 2 * gab

        # We don't reuse the input
        def fn(a, b):
            orig_a = a.clone().view_as(a)
            c, d = MyOutPlaceAdder.apply(orig_a, b)
            return (c * d).sum()

        bad_mark_dirty_err = "Some elements marked as dirty during the forward method were not returned as output."
        with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
            fn(a, b)

    def test_custom_function_mark_dirty_not_differentiable(self):
        def get_custom_fn(jvp_err):
            class InplaceMul(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    result = x.mul_(2)
                    ctx.mark_dirty(result)
                    return result

                @staticmethod
                def backward(ctx, grad_output):
                    pass

                @staticmethod
                def jvp(ctx, x_t):
                    if jvp_err:
                        return x_t
                    else:
                        return x_t.mul_(2)

            return InplaceMul

        for requires_grad, jvp_err in product([True, False], repeat=2):
            InplaceMul = get_custom_fn(jvp_err)
            # Make sure that tensor is always returned as-is if marked dirty
            z = torch.tensor(1.0, requires_grad=requires_grad)
            x = z.clone()
            y = InplaceMul.apply(x)
            self.assertTrue(x is y)
            self.assertEqual(x, z * 2)

            # jvp must properly modify the input grad if mark_dirty is set
            with fwAD.dual_level():
                x_tangent = torch.ones_like(x)
                x_dual = fwAD.make_dual(x, x_tangent)

                if jvp_err:
                    bad_mark_dirty_err = (
                        "jvp function must modify the corresponding gradient inplace"
                    )
                    with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
                        InplaceMul.apply(x_dual)
                else:
                    out_dual = InplaceMul.apply(x_dual)
                    _, out_tangent = fwAD.unpack_dual(out_dual)
                    self.assertTrue(out_dual is x_dual)
                    self.assertTrue(out_tangent is x_tangent)

    def test_custom_function_mark_output_view_of_intermediate(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                out = inp.clone().view_as(inp)
                ctx.mark_dirty(out)
                return out

            @staticmethod
            def backward(ctx, gO):
                pass

        a = torch.tensor([1.0], requires_grad=True)
        a_clone = a.clone()

        with self.assertRaisesRegex(
            RuntimeError, "received a tensor that was not an input."
        ):
            Func.apply(a_clone)

    def test_custom_function_inplace_on_non_default_view(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                inp.add_(1)
                ctx.mark_dirty(inp)
                return inp

            @staticmethod
            def backward(ctx, gO):
                pass

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        a_clone = a.clone()
        b, c = a.split_with_sizes([1, 1], dim=0)

        with self.assertRaisesRegex(
            RuntimeError, "output of a function that returns multiple view"
        ):
            Func.apply(b)

    def test_custom_function_inplace_on_view_of_leaf(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                inp.add_(1)
                ctx.mark_dirty(inp)
                return inp

            @staticmethod
            def backward(ctx, gO):
                pass

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = a.view_as(a)

        with self.assertRaisesRegex(
            RuntimeError, "a view of a leaf Variable that requires grad"
        ):
            Func.apply(b)

    def test_named_tensor_for_complex_views(self):
        names = ["batch", "height", "width", "complex"]
        z = torch.ones((2, 1, 2, 2), requires_grad=True)
        z_named = z.refine_names(*names)
        z_complex = torch.view_as_complex(z_named.rename(None)).refine_names(
            *names[:-1]
        )
        z_complex.sum().abs().backward()
        expected = torch.ones_like(z_complex).rename(None)
        abs_1_1j = abs(1 + 1j)
        expected.fill_(complex(abs_1_1j / 2, abs_1_1j / 2))
        self.assertEqual(z.grad, torch.view_as_real(expected))

    def test_custom_function_saving_mutated_view_no_leak(self):
        class Test(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad):
                pass

        def scope():
            x = torch.tensor(1.0, requires_grad=True).clone()
            x = x.view_as(x)
            y = Test.apply(x)
            return weakref.ref(x)

        ref = scope()
        self.assertIsNone(ref())

    def test_custom_function_return_view_in_nograd(self):
        class Alias(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:]

            @staticmethod
            def backward(ctx, gx):
                return gx

        inp = torch.rand(2, requires_grad=True)

        with torch.no_grad():
            output = Alias.apply(inp)

        with torch.no_grad():
            expected_output = inp[:]

        # Calling the custom function should operate as if we called an equivalent op
        self.assertEqual(output.requires_grad, expected_output.requires_grad)

        # Check that in-place modification on view throws
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_custom_function_preserve_torch_function_when_return_as_is(self):
        class Custom(torch.Tensor):
            def __init__(self, data):
                super().__init__()
                self._data = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                args = tuple(a._data if isinstance(a, cls) else a for a in args)
                out = func(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    out = cls(out)
                return out

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx):
                pass

        x = Custom(torch.randn(2, 3))
        y = Fn.apply(x)
        self.assertTrue(isinstance(y, Custom))

    def test_grad_mode_restored_reentrant(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, go):
                original = torch._C.is_grad_enabled()
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    foo = torch.rand(go.size(), requires_grad=True)
                    (grad,) = torch.autograd.grad(foo**3, foo, grad_outputs=go)
                    self.assertTrue(torch._C.is_grad_enabled())
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad

        inp = torch.rand(3, requires_grad=True)

        # Case where original==False
        MyFunction.apply(inp).sum().backward()
        # Case where original==True
        MyFunction.apply(inp).sum().backward(create_graph=True)

    def test_power_function(self):
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(a**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

        s = 0
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(s**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

    def test_custom_function_error(self):
        class BadFw(Function):
            @staticmethod
            def backward(ctx, foo):
                return foo

        class BadBw(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        class BadBw2(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

            @staticmethod
            def backward(ctx, foo):
                return foo

            @staticmethod
            def vjp(ctx, foo):
                return foo

        class BadJvp(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        inp = torch.rand(1, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "must implement the forward"):
            BadFw.apply(inp)

        with self.assertRaisesRegex(RuntimeError, "must implement either the backward"):
            BadBw.apply(inp).sum().backward()

        with self.assertRaisesRegex(
            RuntimeError, "Implementing both 'backward' and 'vjp'"
        ):
            BadBw2.apply(inp).sum().backward()

        with self.assertRaisesRegex(RuntimeError, "must implement the jvp function"):
            with fwAD.dual_level():
                d = fwAD.make_dual(inp, torch.rand_like(inp))
                res = BadJvp.apply(d)

    def test_custom_function_forward_mode_view_checks(self):
        flag_to_error = {
            "ok": None,
            "not_a_view": "jvp is not returning a view",
            "not_a_view_of_inp": "jvp is not returning a view of the given",
            "not_a_view_of_inp_base": "jvp is not returning a view of the same base",
        }

        class ViewFn(Function):
            @staticmethod
            def forward(ctx, foo, flag):
                ctx.flag = flag
                ctx.size = foo.size()
                return foo.narrow(0, 0, 2)

            @staticmethod
            def vjp(ctx, gO):
                gI = gO.new_zeros(ctx.size)
                gI.narrow(0, 0, 2).copy_(gO)
                return gI, None

            @staticmethod
            def jvp(ctx, gI, _):
                res = gI.narrow(0, 0, 2)
                if ctx.flag != "ok":
                    # Break the view in the gradients!
                    res = res.clone()
                if ctx.flag in ["not_a_view_of_inp", "not_a_view_of_inp_base"]:
                    # Result should be a view, just of the wrong thing
                    res = res.view_as(res)
                return res

        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        for flag, msg in flag_to_error.items():

            def test_fn(inp):
                if flag == "not_a_view_of_inp_base":
                    inp = inp.view_as(inp)
                return ViewFn.apply(inp, flag)

            if msg is None:
                gradcheck(test_fn, inp, check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, msg):
                    gradcheck(test_fn, inp, check_forward_ad=True)

    def test_custom_function_forward_mode_inplace_checks(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, foo, flag):
                ctx.mark_dirty(foo)
                ctx.flag = flag
                foo.mul_(2)
                return foo

            @staticmethod
            def vjp(ctx, gO):
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                if ctx.flag:
                    # Don't do the change inplace
                    return 2 * gI
                else:
                    gI.mul_(2)
                    return gI

        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        def test_fn(inp, flag):
            inp = inp.clone()
            return InplaceFn.apply(inp, flag)

        gradcheck(test_fn, (inp, False), check_forward_ad=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "inplace custom Function is not modifying the forward mode gradients inplace",
        ):
            gradcheck(test_fn, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_wrong_formula(self):
        class UserFn(Function):
            @staticmethod
            def forward(ctx, foo, should_fail):
                ctx.should_fail = should_fail
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                if ctx.should_fail:
                    # Wrong gradient formula
                    return 3 * gI
                else:
                    return 2 * gI

        inp = torch.rand(10, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (inp, False), check_forward_ad=True)

        with self.assertRaisesRegex(
            RuntimeError, "Jacobian computed with forward mode mismatch for output 0"
        ):
            gradcheck(UserFn.apply, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_non_tensor_before_tensor_args(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, nt, x, nt2, y):
                return x * 2 + y * 3

            @staticmethod
            def jvp(ctx, nt, x_t, nt2, y_t):
                self.assertIsNone(nt)
                self.assertIsNone(nt2)
                return x_t * 2 + y_t * 3

        x = torch.tensor(1.0, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        y = torch.tensor(1.0, dtype=torch.double)

        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, t)
            MyFn.apply(1, dual_x, 1, y)

        gradcheck(
            MyFn.apply,
            (1, x.requires_grad_(True), 1, y.requires_grad_(True)),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
        )

    def test_custom_function_forward_mode_forward_is_no_op(self):
        error_regex = (
            "A custom Function's forward is returning a view \\(or an input as-is\\)"
        )

        return_lambdas = {
            # If we return an input as-is in forward, that is treated
            # as if self.view_as(self) is performed. If jvp returns x.view_as(x),
            # this is OK.
            "view_as": lambda x: x.view_as(x),
            # Expect this to raise an error
            "self": lambda x: x,
            # Expect this to raise the same error
            "mul_by_2": lambda x: x * 2,
        }

        for k, fn in return_lambdas.items():

            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    return x + y, x

                @staticmethod
                def vjp(ctx, gO1, gO2):
                    return gO1 + gO2, gO1

                @staticmethod
                def jvp(ctx, x_t, y_t):
                    return x_t + y_t, fn(x_t)

            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            t = torch.tensor(1.0, dtype=torch.double)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)

            c = torch.tensor(1.0, dtype=torch.double)
            t2 = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)

            with fwAD.dual_level():
                a_dual = fwAD.make_dual(a, t)
                c_dual = fwAD.make_dual(c, t2)

                if k == "view_as":
                    _, out2 = MyFn.apply(a_dual, b)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t)

                    _, out2 = MyFn.apply(c_dual, d)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t2)
                else:
                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(a_dual, b)

                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(c_dual, d)

            if k == "view_as":
                gradcheck(MyFn.apply, (a, c), check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, error_regex):
                    gradcheck(MyFn.apply, (a, c), check_forward_ad=True)

    def test_custom_function_save_for_forward(self):
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
                ctx.save_for_backward(x, y)
                ctx.save_for_forward(x, y)
                ctx.z = z
                ctx.prod = x * y
                return z * ctx.prod

            @staticmethod
            def jvp(ctx, x_t, y_t, _):
                x_p, y_p = ctx.saved_tensors
                z = ctx.z
                return z * (y_p * x_t + x_p * y_t)

            @staticmethod
            def vjp(ctx, grad_out):
                x, y = ctx.saved_tensors
                z = ctx.z
                return z * grad_out * y, z * grad_out * x, None

        a = torch.tensor(1.0, requires_grad=True, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        b = torch.tensor(2.0, requires_grad=True, dtype=torch.double)
        c = 4

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual, b, c)
            out.backward()

        gradcheck(Func.apply, (a, b, c), check_forward_ad=True)

        # When saved for backward, but not saved for forward
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def jvp(ctx, x_t):
                self.assertEqual(len(ctx.saved_tensors), 0)
                return x_t

            @staticmethod
            def vjp(ctx, grad_out):
                (x,) = ctx.saved_tensors
                self.assertEqual(len(ctx.saved_tensors), 1)
                return grad_out

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual)
            out.backward()

        gradcheck(Func.apply, (a,), check_forward_ad=True)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_autograd_function.py")
    def test_custom_function_forward_mode_non_differentiable(self):
        # returns differentiable type, marked non-differentiable
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                return x_tangent, None

        x = torch.tensor(2.0)
        x_tangent = torch.tensor(1.0)
        y = torch.tensor(3.0)

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            _, out2_dual = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        y = torch.tensor(3)

        # returns non-differentiable type, NOT marked non-differentiable
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                self.assertIsNone(y_tangent)
                return x_tangent, None

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            _, out2_dual = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        class FuncWrong(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                return x_tangent, x_tangent.clone()

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            with self.assertRaisesRegex(
                RuntimeError, "You should return None at that position instead"
            ):
                FuncWrong.apply(x_dual, y)

        # returns non-tensor
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone(), object(), x.clone()

            @staticmethod
            def jvp(ctx, x_tangent):
                return x_tangent, None, x_tangent

        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            out_dual, _, out2_dual = Func.apply(x_dual)
            self.assertEqual(fwAD.unpack_dual(out_dual).tangent, x_tangent)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, x_tangent)

    def test_custom_function_local_inplace(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        base = torch.rand(10, requires_grad=True)

        foo = MyFn.apply(base, False)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

        foo = MyFn.apply(base, True)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

    def test_integer_outputs(self):
        inp = torch.rand(4, requires_grad=True)

        out = inp.argmax()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argmin()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argsort()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.rand((), requires_grad=True)

        out = torch.searchsorted(inp, val)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        bins = torch.linspace(0, 1.0, steps=100, requires_grad=True)
        vals = torch.rand(5, 5, requires_grad=True)
        out = torch.bucketize(vals, bins)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.empty(5).requires_grad_()
        out = val.count_nonzero()
        self.assertFalse(out.requires_grad)

        def assert_only_first_requires_grad(res):
            if not isinstance(res, tuple):
                res = (res,)
            self.assertTrue(res[0].requires_grad)
            for out in res[1:]:
                if out is not None:
                    self.assertFalse(out.requires_grad)

        for sort in [True, False]:
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    res = torch.unique(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        dim=0,
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(
                        inp, return_inverse=return_inverse, return_counts=return_counts
                    )
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(
                        inp,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                        dim=0,
                    )
                    assert_only_first_requires_grad(res)

                    # Here we test the internal functions to make sure all of them are
                    # covered on top of the public API
                    res = torch._unique(inp, sorted=sort, return_inverse=return_inverse)
                    assert_only_first_requires_grad(res)

                    # This looks public but is actually manually deleted from the
                    # torch namespace in torch/functional.py
                    res = torch._VF.unique_dim(
                        inp,
                        dim=0,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

                    # We don't test `unique_dim_consecutive` here.
                    # It looks public but the python binding is actually manually disabled in
                    # tools/autograd/gen_python_functions.py

                    res = torch._unique2(
                        inp,
                        sorted=sort,
                        return_inverse=return_inverse,
                        return_counts=return_counts,
                    )
                    assert_only_first_requires_grad(res)

    def test_custom_function_cycle(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x, metadata):
                x = x.clone()
                ctx.meta = metadata
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                self.assertEqual(x, 3.14)
                self.assertEqual(ctx.meta["foo"], 3.14)
                return gO * x, None

        def get_refs(with_backward):
            a = torch.tensor(3.14, requires_grad=True)

            metadata = {}
            out = MyFn.apply(a, metadata)

            metadata["foo"] = out

            if with_backward:
                out.sum().backward()
                self.assertEqual(a.grad, a)

            return torch._C._WeakTensorRef(out)

        with disable_gc():
            ref = get_refs(False)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

        # The backward clears the saved_variables but not the __dict__
        with disable_gc():
            ref = get_refs(True)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

    def test_create_graph_and_full_backward_hook_cycle(self):
        # If BackwardHook saves grad_output, it can create a cycle when we perform backward
        # with create_graph=True
        #
        #   grad_output -> grad_output.grad_fn -> graph -> hook -> grad_output
        #
        class TestCls:
            # Dummy class for the purpose of creating a weakref
            pass

        def get_ref(input_requires_grad, nb_hooks):
            t = torch.randn(10, requires_grad=input_requires_grad)
            a = torch.tensor(1.0, requires_grad=True)

            class Test(nn.Module):
                def forward(self, x):
                    return x**2 * a**2

            mod = Test()

            for _ in range(nb_hooks):
                mod.register_full_backward_hook(lambda a, b, c: None)

            tmp = mod(t)

            # Save dummy object to graph and get a weak ref to it
            test = TestCls()
            ref = weakref.ref(test)
            tmp.grad_fn.metadata["a"] = test

            with set_warn_always_context(True):
                with warnings.catch_warnings(record=True) as w:
                    tmp.exp().sum().backward(create_graph=True)
                    self.assertTrue(w)
                    found = 0
                    for warning in w:
                        if "Using backward() with create_graph=True" in str(
                            warning.message
                        ):
                            found += 1
                    self.assertEqual(found, 1)

            # Remove the backward + create_graph=True cycle
            a.grad = None
            t.grad = None

            return ref

        for nb_hooks in (1, 2, 3):
            for input_requires_grad in (True, False):
                ref_ = get_ref(
                    input_requires_grad=input_requires_grad,
                    nb_hooks=nb_hooks,
                )
                gc.collect()
                self.assertIsNone(ref_())

    @parametrize("use_custom_function", [True, False])
    @parametrize("use_tensor_hook", [True, False])
    def test_hook_closure_cycle(self, use_custom_function, use_tensor_hook):
        # This creates a cycle between the hook and grad_fn_b
        # hook -> closure -> grad_fn_b (python) -> grad_fn (cpp) -> hook (cpp)
        # -> dict -> hook
        #
        # This test is testing that the grad_fn_b (python) only traverses the
        # dict if it is the only one holding a reference to the grad_fn_b (cpp)
        # shared_ptr
        #
        # See: https://github.com/pytorch/pytorch/issues/102174
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad

        class Test:
            pass

        count = [0]

        def scope():
            a = torch.tensor(1.0, requires_grad=True)
            if use_custom_function:
                b = Function.apply(a)
            else:
                b = a.clone()
            grad_fn_b = b.grad_fn
            obj = Test()

            def hook(*args):
                # Make sure this hook's closure holds onto grad_fn_b
                # This forms a cycle between the hook and grad_fn_b
                # We also hold onto a sentinel object 'obj' to track
                # whether this cycle is still alive. See 'ref' below.
                grad_fn_b
                obj
                count[0] += 1

            if use_tensor_hook:
                b.register_hook(hook)
            else:
                b.grad_fn.register_hook(hook)
            c = b.clone()
            ref = weakref.ref(obj)
            return c, ref

        with disable_gc():
            out, ref = scope()
            out.backward(retain_graph=True)

            gc.collect()

            # Make sure gc does not clear the cycle noted above.
            # e.g. the hook is alive and gets fired even after gc runs
            out.backward(retain_graph=True)
            self.assertEqual(count[0], 2)

            # ref is still alive because the use_count of the cpp grad_fn
            # shared_ptr > 1 since (1) the python grad_fn is alive, and (2) the
            # rest of the graph holds onto the shared_ptr
            self.assertIsNotNone(ref())

            # Then delete the rest of the graph and check that ref is dead
            del out
            gc.collect()
            self.assertIsNone(ref())

    def test_full_backward_hook_double_backward(self):
        x = torch.rand(1, requires_grad=True)
        y = torch.rand_like(x)

        func = torch.nn.MSELoss()
        counter = [0]

        def hook(module, grad_input, grad_output):
            counter[0] += 1

        func.register_full_backward_hook(hook)

        f = func(x, y)

        (gradx_f,) = torch.autograd.grad(f, x, create_graph=True)
        self.assertEqual(counter[0], 1)
        _ = torch.autograd.grad(gradx_f, x)
        # We should not error, and counter should not be incremented
        self.assertEqual(counter[0], 1)

    def test_input_buffer_accum(self):
        leaf = torch.rand(2, 2, requires_grad=True)

        # An op that returns sparse gradients
        ind = torch.tensor([[0, 0]], dtype=torch.long)
        out2 = leaf.gather(0, ind, sparse_grad=True)

        # An op that returns the gradients as-is
        out1 = leaf.clone()

        grad_out1_original = torch.rand_like(out1)
        grad_out1 = grad_out1_original.clone()
        grad_out2 = torch.rand_like(out2)

        torch.autograd.backward((out1, out2), (grad_out1, grad_out2))

        # Given gradients should not be modified inplace
        self.assertEqual(grad_out1, grad_out1_original)

    def test_no_unnecessary_unwrapping(self):
        a = torch.randn(5, requires_grad=True)
        a_orig = a.detach().clone()
        b = a * a
        c = a * b
        d = torch.exp(a)

        # a is leaf
        self.assertIs(b.grad_fn._saved_self, a)
        self.assertIs(b.grad_fn._saved_other, a)
        self.assertIs(c.grad_fn._saved_self, a)

        # b is not an output
        self.assertIs(c.grad_fn._saved_other, b)

        # d is an output
        self.assertEqual(d.grad_fn._saved_result, d)
        self.assertIsNot(d.grad_fn._saved_result, d)

        c.sum().backward()

        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            c.grad_fn._saved_self

        # a is left untouched
        self.assertEqual(a, a_orig)

    def test_saved_variable_version_counter(self):
        a = torch.rand(2, requires_grad=True)

        b = torch.exp(a)

        b_unpacked = b.grad_fn._saved_result
        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

        with torch.no_grad():
            b += 1

        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

    def test_saved_variable_packing_unpacking_saved_original_with_hooks(self):
        # Tests that packing/unpacking a SavedVariable works correctly with user-defined hooks
        # The saved_original / did_not_save_original distinction corresponds to the `save_original`
        # attribute of `SavedVariable`.

        def test(get_input, is_leaf):
            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x / 2)
            self.assertEqual(a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(2 * a, a.grad)

            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x)
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(3 * a, a.grad)

            # double backward
            a = get_input()
            grad_fn = a.grad_fn
            y = a**3
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                g.sum().backward()
            else:
                g.sum().backward()
                self.assertEqual(6 * a, a.grad)

            a = get_input()
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: 1)
            with self.assertRaisesRegex(
                TypeError, "Output of saved tensor unpack_hook expected to be a Tensor"
            ):
                print(y.grad_fn._saved_self)

            a = get_input()
            y = a * a
            with self.assertRaisesRegex(
                TypeError, "missing 1 required positional argument"
            ):
                y.grad_fn._raw_saved_self.register_hooks(lambda x, b: x, lambda x: x)

            a = get_input()
            y = a * a
            with self.assertRaisesRegex(
                TypeError, "missing 1 required positional argument"
            ):
                y.grad_fn._raw_saved_self.register_hooks(
                    lambda x, b: (x, b), lambda x: x
                )

            def inplace_double(x):
                x *= 2
                return x

            a = get_input()
            t = a * a

            with self.assertRaisesRegex(
                RuntimeError,
                "A saved tensor pack hook is modifying its input in place.",
            ):
                t.grad_fn._raw_saved_self.register_hooks(
                    inplace_double, lambda x: x / 2
                )

        # leaf
        test(lambda: torch.randn(5, requires_grad=True), True)

        # not leaf, not output
        test(lambda: (1 + torch.randn(5, requires_grad=True)), False)

    def test_saved_variable_saved_original_inplace_detach(self):
        # Detaching a tensor that is saved input raises
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.sin()
        a.detach_()
        with self.assertRaisesRegex(
            RuntimeError, "Trying to use a saved tensor that has been detached"
        ):
            b.backward()



... [truncated: 160 more lines]
```

*Note: Content truncated due to size*

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
- **Context Manager**: Implements context manager protocol
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
- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_autograd.py_docs.md
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

- **File Documentation**: `test_autograd.py_docs.md_docs.md`
- **Keyword Index**: `test_autograd.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: test_eager_transforms.py

## File Metadata
- **Path**: `test/functorch/test_eager_transforms.py`
- **Size**: 181764 bytes
- **Lines**: 5422
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: functorch"]
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import os
import subprocess
import sys
import unittest
import warnings
from functools import partial, wraps

# NB: numpy is a testing dependency!
import numpy as np
from common_utils import expectedFailureIf

import functorch
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F
from functorch import (
    combine_state_for_ensemble,
    grad,
    grad_and_value,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    make_functional,
    make_functional_with_buffers,
    make_fx,
    vjp,
    vmap,
)
from functorch.experimental import functionalize, replace_all_batch_norm_modules_
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dynamo import allow_in_graph
from torch._functorch.eager_transforms import _slice_argnums
from torch._functorch.make_functional import (
    functional_init,
    functional_init_with_buffers,
)
from torch._functorch.utils import enable_single_level_autograd_function
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.func import functional_call, linearize, stack_module_state
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    SM70OrLater,
    TEST_CUDA,
    tf32_on_and_off,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import get_all_fp_dtypes
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_CUDA_MEM_LEAK_CHECK,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
)
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


USE_TORCHVISION = False
try:
    import torchvision  # noqa: F401

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )

# TestCase for _slice_argnums, an important helper function


class VmapTearDownMixin:
    def tearDown(self):
        # Ensure that in the case of a test failure, the next test won't fail
        # because of a previous call to _vmap_increment_nesting that wasn't undone
        # i.e. test_vmap_free_tensor fails when PYTORCH_TEST_WITH_DYNAMO=1
        # and the call to increment nesting is not undone
        if not TEST_WITH_TORCHDYNAMO:
            return

        warn = False
        while ci := torch._C._functorch.peek_interpreter_stack():
            if ci.key() == torch._C._functorch.TransformType.Vmap:
                warn = True
                torch._C._functorch._vmap_decrement_nesting()
            else:
                break

        if warn:
            msg = (
                "Interpreter stack is not empty. Test should have called "
                "'torch._C._functorch._vmap_decrement_nesting()'"
            )
            warnings.warn(msg)


@markDynamoStrictTest
class TestSliceArgnums(TestCase):
    def test_invalid_argnum_type(self):
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _slice_argnums(args, 0.0)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _slice_argnums(args, [0])
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _slice_argnums(args, (0.0,))

        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _slice_argnums(args, ((0, 1), 2))

    def test_out_of_bounds_argnum_values(self):
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, 1)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, -2)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, (-2,))

    def test_not_enough_argnums(self):
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            _slice_argnums(args, ())

    def test_duplicate_argnums(self):
        x = torch.randn(3)
        args = (x, x)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _slice_argnums(args, (0, 0))
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _slice_argnums(args, (0, -2))

    def test_flat_args_with_positive_int_argnum(self):
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        res = _slice_argnums(args, 0)
        self.assertEqual(res, (0.1,))

        res = _slice_argnums(args, 4)
        self.assertEqual(res, (4.1,))

    def test_flat_args_with_negative_int_argnum(self):
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        res = _slice_argnums(args, -1)
        self.assertEqual(res, (4.1,))

        res = _slice_argnums(args, -5)
        self.assertEqual(res, (0.1,))

    def test_flat_args_with_tuple_argnum(self):
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        res = _slice_argnums(args, (0, 1, 2, 3, 4))
        self.assertEqual(res, args)

        res = _slice_argnums(args, (0, -3))
        self.assertEqual(res, (0.1, 2.1))

    def test_pytree_args(self):
        args = ((0.1, 1.1), 2.0, [3.1])

        res = _slice_argnums(args, 0)
        self.assertEqual(res, args[0:1])

        res = _slice_argnums(args, (0,))
        self.assertEqual(res, args[0:1])

        res = _slice_argnums(args, -1)
        self.assertEqual(res, args[-1:])

        res = _slice_argnums(args, (0, -2))
        self.assertEqual(res, args[0:2])

    def test_argnums_reorders(self):
        args = ((0.1, 1.1, 2.1), 3.1, 4.1)

        res = _slice_argnums(args, (1, 0))
        self.assertEqual(res, (args[1], args[0]))


def _get_weights_and_functional_call(net, mechanism):
    if mechanism == "make_functional":
        return make_functional(net)
    else:
        assert mechanism == "functional_call"
        # this makes it so the function from make_functional and this call have the same signature

        def net_func(weights, data):
            return functional_call(net, weights, (data,))

        return net_func, dict(net.named_parameters())


def _get_weights_and_functional_call_with_buffers(net, mechanism):
    if mechanism == "make_functional":
        return make_functional_with_buffers(net)
    else:
        assert mechanism == "functional_call"

        # this makes it so the function from make_functional and this call have the same signature
        def net_func(weights, buffers, data):
            return functional_call(net, (weights, buffers), (data,))

        return net_func, dict(net.named_parameters()), dict(net.named_buffers())


@markDynamoStrictTest
class TestGradTransform(TestCase):
    def test_primitive(self, device):
        x = torch.randn([], device=device)
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self, device):
        x = torch.randn(2, 3, 4, device=device)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    def test_fn_with_kwargs(self, device):
        def foo(x, y):
            return (x * y).sum()

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected = grad(foo)(x, y)
        result = grad(foo)(x, y=y)
        self.assertEqual(result, expected)

    def test_composite_complicated(self, device):
        x = torch.randn(3, device=device)
        y = torch.randn(3, 5, device=device)

        def foo(x, y):
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_composite_two_ops(self, device):
        N, C = 2, 5
        y = torch.randn(N, C, device=device)
        targets = torch.randint(0, C, (N,), device=device)

        def foo(y, targets):
            return F.cross_entropy(y, targets)

        result = grad(foo)(y, targets)

        y.requires_grad_()
        (expected,) = torch.autograd.grad(foo(y, targets), y)

        self.assertEqual(result, expected)

    def _test_attributes(self, get_attr_lambda, device):
        x = torch.randn(2, 3, 5, dtype=torch.double, device=device)
        expected = get_attr_lambda(x)

        def foo(x):
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    def test_shape(self, device):
        self._test_attributes(lambda x: x.shape, device)

    def test_dtype(self, device):
        self._test_attributes(lambda x: x.dtype, device)

    def test_is_cuda(self, device):
        self._test_attributes(lambda x: x.is_cuda, device)

    def test_numel(self, device):
        self._test_attributes(lambda x: x.numel(), device)

    def test_layout_sparse(self, device):
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], device=device)
        values = torch.tensor([3.0, 4.0, 5.0], device=device)
        sparse_x = torch.sparse_coo_tensor(indices, values, (2, 3), device=device)

        # Verify the input is sparse
        self.assertEqual(sparse_x.layout, torch.sparse_coo)

        def foo(x):
            # assert GradTrackingTensor still reports sparse layout
            self.assertEqual(x.layout, torch.sparse_coo)
            return x.coalesce()._values().sum()

        result = grad(foo)(sparse_x)

        # The gradient should also be sparse
        self.assertEqual(result.layout, torch.sparse_coo)

    def test_inplace(self, device):
        x = torch.randn([], device=device)

        def foo(x):
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_inplace_on_view(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y0.sin_()
            return y.sum()

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_view_base(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y.sin_()
            return y0

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_captures(self, device):
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        captured = torch.randn(3, device=device)

        def foo(x):
            captured.copy_(x)
            return (x * captured).sum()

        with self.assertRaisesRegex(RuntimeError, "mutate a captured Tensor"):
            grad(foo)(x)

    def test_nesting_simple(self, device):
        x = torch.randn([], device=device)
        result = grad(grad(torch.sin))(x)
        self.assertEqual(result, -torch.sin(x))

    @skipIfTorchDynamo("Ref: https://github.com/pytorch/pytorch/issues/103613")
    def test_escaped_wrappers_are_marked_as_dead(self, device):
        x = torch.randn([], device=device)
        escaped = []

        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        grad(foo)(x)
        self.assertEqual(torch._C._functorch.dlevel(escaped[0]), -1)

    @skipIfTorchDynamo("Ref: https://github.com/pytorch/pytorch/issues/103613")
    def test_escaped_wrappers_are_ignored(self, device):
        x = torch.randn([], device=device)
        escaped = []

        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        grad(foo)(x)

        something = escaped[0].sum()
        self.assertEqual(torch._C._functorch.dlevel(something), 0)
        self.assertEqual(something, x.sin().sum())

    def test_manual_seed_inside_grad(self, device):
        x = torch.randn([], device=device)

        def f(x):
            torch.manual_seed(0)
            return x * torch.randn_like(x)

        with freeze_rng_state():
            result = grad(f)(x)
            x.requires_grad_()
            (expected,) = torch.autograd.grad(f(x), x)
            self.assertEqual(result, expected)

    def test_vjp(self, device):
        x = torch.randn([], device=device)
        out, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(out, x.sin())

        v = torch.randn([], device=device)
        (result,) = vjp_fn(v)
        self.assertEqual(result, v * x.cos())

    def test_vjp_two_outputs(self, device):
        def f(x):
            return x, x

        result, vjp_fn = vjp(f, torch.tensor(1.0))
        vjp_fn(result)

    def test_conj_bit(self):
        x = torch.tensor(1 + 1j)

        def foo(x):
            assert not x.is_conj()
            y = x.conj()
            assert y.is_conj()
            return y.abs()

        res = grad(foo)(x)
        with torch.no_grad():
            self.assertEqual(res, torch.ones_like(res) * torch.sgn(x))

    def test_composed_with_autograd(self, device):
        x = torch.randn([], requires_grad=True, device=device)

        y = grad(torch.sin)(x)
        (result,) = torch.autograd.grad(y, x)
        self.assertEqual(result, -x.sin())

    def test_grad_of_vjp_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            out, vjp_fn = vjp(torch.sin, x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_vjp_of_grad_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            out, vjp_fn = vjp(grad(torch.sin), x)
            return vjp_fn(y)[0]

        result = foo(x, y)
        expected = -y * x.sin()
        self.assertEqual(result, expected)

    def test_grad_of_vjp_of_grad_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            df, vjp_fn = vjp(grad(lambda x: -torch.cos(x)), x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_views(self, device):
        x = torch.randn([], requires_grad=True, device=device)
        y = torch.randn([], requires_grad=True, device=device)

        def silly_sin(x):
            x = x.view([])
            x = x.sin()
            return x

        def foo(x, y):
            z1 = grad(silly_sin)(x)
            z2 = torch.cos(y)
            return z1 + z2

        result = foo(x, y)
        grads = torch.autograd.grad(result, [x, y])
        self.assertEqual(grads[0], -x.sin())
        self.assertEqual(grads[1], -y.sin())

    def test_view_inplace_simple(self, device):
        def foo(x):
            x = x.clone()
            x.view([]).sin_()
            return x

        x = torch.randn([], requires_grad=True, device=device)
        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_invalid_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "but only"):
            grad(torch.mul, argnums=-3)(x, y)
        with self.assertRaisesRegex(RuntimeError, "but only"):
            grad(torch.mul, argnums=2)(x, y)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            grad(torch.mul, argnums=[0])(x, y)
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            grad(torch.mul, argnums=("0",))(x, y)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            grad(torch.mul, argnums=(0, 0))(x, y)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            grad(torch.mul, argnums=(0, -2))(x, y)

    def test_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=0)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=1)(x, y)
        self.assertEqual(gy, x)

        (gx,) = grad(torch.mul, argnums=(0,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(0, 1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_out_of_order_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gy, gx = grad(torch.mul, argnums=(1, 0))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_negative_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=-2)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=-1)(x, y)
        self.assertEqual(gy, x)

        (gx,) = grad(torch.mul, argnums=(-2,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(-2, -1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_grad_pytree_inputs(self, device):
        x = torch.randn([], device=device)

        def f(a, b):
            x, y = a
            return 1 * x + 2 * y + 3 * b["foo"]

        args = ((x, x), {"foo": x})

        gx, gy = grad(f)(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))

        ((gx, gy),) = grad(f, argnums=(0,))(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))

        (gx, gy), gz = grad(f, argnums=(0, 1))(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))
        self.assertEqual(gz["foo"], torch.tensor(3.0, device=device))

    def test_grad_aux_tensor(self, device):
        x = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            r"grad_and_value\(f\)\(\*args\): output of function f should be a tuple",
        ):
            grad(lambda t: [t, t], has_aux=True)(x)

        with self.assertRaisesRegex(
            RuntimeError,
            r"grad_and_value\(f\)\(\*args\): output of function f should be a tuple",
        ):
            grad(lambda t: (t, t + 2, t + 3), has_aux=True)(x)

        def f(t):
            y = t.sin()
            return y.sum(), t.cos()

        out, aux = grad(f, has_aux=True)(x)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.cos())

    def test_grad_aux_pytree(self, device):
        def f(x):
            y = x.sin()
            return y.sum(), {"a": x.cos(), "b": [x.tan()]}

        x = torch.randn(3, device=device)

        out, aux = grad(f, has_aux=True)(x)
        _, expected_aux = f(x)
        self.assertEqual(aux, expected_aux)
        self.assertEqual(out, x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = grad(lambda x: (x.sum(), aux), has_aux=True)(x)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = grad(lambda x: (x.sum(), [x, aux]), has_aux=True)(x)

    def test_zero_grad(self, device):
        def f(x):
            return (x["a"] ** 2.0).sum()

        inps = {
            "a": torch.randn(10, device=device) + 3,
            "b": torch.randn(10, device=device),
        }
        grads = grad(f)(inps)
        self.assertNotEqual(grads["a"].sum(), 0.0)
        self.assertEqual(grads["b"].sum(), 0.0)

    def test_unrelated_grad(self, device):
        x = torch.tensor(1.0, device=device)
        y = torch.tensor(2.0, device=device)

        def unrelated(x):
            return y

        result = grad(unrelated)(x)
        self.assertEqual(result, torch.zeros_like(x))

    def test_unrelated_vjp(self, device):
        x = torch.tensor(1.0, device=device)
        y = torch.tensor(2.0, device=device)
        v = torch.tensor(1.0, device=device)

        def unrelated(x):
            return y

        out, vjp_fn = vjp(unrelated, x)
        result = vjp_fn(v)
        expected = (torch.zeros_like(x),)
        self.assertEqual(result, expected)

    def test_unrelated_vjp_multiple_inputs_outputs(self, device):
        w = torch.tensor(3.0, device=device)
        x = torch.tensor(4.0, device=device)
        y = torch.tensor(2.0, device=device)
        v = torch.tensor(1.0, device=device)

        def unrelated(w, x):
            return y, y, x

        out, vjp_fn = vjp(unrelated, w, x)
        result = vjp_fn((v, v, v))
        expected = (torch.zeros_like(x), torch.ones_like(x))
        self.assertEqual(result, expected)

    # TODO: https://github.com/pytorch/functorch/issues/12
    @onlyCPU
    def test_unrelated_hessian(self, device):
        N = 5
        M = 3
        W = torch.randn(N, M, device=device)

        def f(x):
            return W @ x

        x = torch.randn(M)
        result = jacrev(jacrev(f))(x)
        expected = torch.zeros(N, M, M, device=device)
        self.assertEqual(result, expected)

    def test_vjp_pytree_input(self, device):
        def f(x):
            return x[0] * x[1][0]

        x = torch.randn([], device=device)
        v = torch.randn([], device=device)
        out, vjp_fn = vjp(f, (x, (x, x)))
        self.assertEqual(out, x * x)
        result = vjp_fn(v)
        self.assertEqual(result, ((x * v, (x * v, 0.0)),))

    def test_vjp_pytree_output(self, device):
        def f(x):
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        (result,) = vjp_fn((v1, (v2, v3)))
        self.assertEqual(result, v1 + v2 + v3)

    def test_vjp_outputs_can_any_pytree(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"vjp\(f, \*primals\): Expected f to be a function that has non-empty output",
            ):
                _, vjp_fn = vjp(lambda _: output, x)
                vjp_fn(t)

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"vjp\(f, \*primals\): expected f\(\*primals\) to return only tensors",
            ):
                _, vjp_fn = vjp(lambda _: output, x)
                vjp_fn(t)

        # Check list output
        output, vjp_fn = vjp(lambda x: [x, x.sum()], x)
        (vjp_out,) = vjp_fn([t, t.sum()])
        assert isinstance(output, list) and len(output) == 2
        assert isinstance(vjp_out, torch.Tensor)

        # Check dict output
        output, vjp_fn = vjp(lambda x: {"x": x, "xsum": x.sum()}, x)
        (vjp_out,) = vjp_fn({"x": t, "xsum": t.sum()})
        assert isinstance(output, dict) and len(output) == 2 and "xsum" in output
        assert isinstance(vjp_out, torch.Tensor)

        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        output, vjp_fn = vjp(composite_output, x)
        (vjp_out,) = vjp_fn(
            [
                (t.sum(), {"a": t, "out": [t, t.sum()]}),
            ]
        )
        assert isinstance(output, list)
        assert isinstance(output[0], tuple) and isinstance(output[0][1], dict)
        assert isinstance(vjp_out, torch.Tensor)

    def test_vjp_pytree_error(self, device):
        def f(x):
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        with self.assertRaisesRegex(RuntimeError, "Expected pytree structure"):
            (result,) = vjp_fn(((v1, (v2, v3)),))

    def test_vjp_aux_tensor(self, device):
        x = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError, r"vjp\(f, \*primals\): output of function f should be a tuple"
        ):
            vjp(lambda t: [t, t], x, has_aux=True)

        with self.assertRaisesRegex(
            RuntimeError, r"vjp\(f, \*primals\): output of function f should be a tuple"
        ):
            vjp(lambda t: (t, t + 2, t + 3), x, has_aux=True)

        def f(t):
            y = t.sin()
            return y, t.cos()

        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.sin())

        v = torch.randn(3, device=device)
        (grad_x,) = vjp_fn(v)
        self.assertEqual(grad_x, v * x.cos())

    def test_vjp_aux_pytree(self, device):
        def f(x):
            y = x.sin()
            return y, {"a": x.cos(), "b": [x.tan()]}

        x = torch.randn(3, device=device)

        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        expected_out, expected_aux = f(x)
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)

        v = torch.randn(3, device=device)
        (grad_x,) = vjp_fn(v)
        self.assertEqual(grad_x, v * x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = vjp(lambda x: (x, aux), x, has_aux=True)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = vjp(lambda x: (x, [x, aux]), x, has_aux=True)

    def test_functional_init(self, device):
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, fn, _ = functional_init(MLPClassifier, (B,), device=device)(32, 2)
        inputs = torch.randn(B, 7, 2, device=device)
        vmap(fn)(weights, (inputs,))

    def test_functional_init_with_buffers(self, device):
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.bn = nn.BatchNorm1d(self.hidden_dim, affine=True)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, buffers, fn, _, _ = functional_init_with_buffers(
            MLPClassifier, [B], device=device
        )(32, 2)
        inputs = torch.randn(B, 7, 2, device=device)
        vmap(fn)(weights, buffers, (inputs,))

    def test_advanced_indexing(self, device):
        def f(value):
            log_prob = torch.ones((), device=device)
            val = torch.zeros(()) > 0
            log_prob[val] = 0
            return value

        result = grad(f)(torch.randn((), device=device))
        self.assertEqual(result, torch.ones_like(result))

        def f2(value):
            value = value.clone()
            value[value > 0] = 0
            return value.sum()

        x = torch.randn(100, device=device)
        result = grad(f2)(x)
        self.assertEqual(result, (x <= 0).type_as(x))

    def test_tensor_ctor_inside_grad(self, device):
        def foo(x):
            return x * torch.tensor(2.0, device=device)

        x = torch.tensor(3.14, device=device)
        functorch.grad(foo)(x)

    @parametrize(
        "op_list_data",
        [
            subtest(
                (
                    [
                        vmap,
                    ],
                    [(4, 2), (64, 3, 32, 32)],
                ),
                name="vmap",
            ),
            subtest(([vmap, vmap], [(4, 3, 2), (64, 3, 32, 32)]), name="vmap_vmap"),
            subtest(
                (
                    [
                        grad,
                    ],
                    [(0,), [], (4, 2), (64, 3, 32, 32)],
                ),
                name="grad",
            ),
            subtest(
                (
                    [grad, grad],
                    [
                        [],
                    ],
                ),
                name="grad_grad",
            ),
            subtest(([vmap, grad], [(4, 2)]), name="vmap_grad"),
        ],
    )
    def test_tensor_print(self, device, op_list_data):
        op_list, shapes = op_list_data

        for dt in get_all_fp_dtypes():
            data = [torch.randn(s, dtype=dt, device=device) for s in shapes]

            for x in data:
                buf = None

                def foo(t):
                    nonlocal buf
                    buf = repr(t)
                    return t.mean()

                fn = foo
                bdim = 0
                for op in reversed(op_list):
                    if op == vmap:
                        fn = op(fn, in_dims=bdim)
                        bdim += 1
                    else:
                        fn = op(fn)

                expected = f"{repr(x)}"
                level = 0
                for op in op_list:
                    level += 1  # noqa: SIM113
                    if op == grad:
                        expected = f"GradTrackingTensor(lvl={level}, value={expected})"
                    elif op == vmap:
                        bdim -= 1
                        expected = (
                            f"BatchedTensor(lvl={level}, bdim={bdim}, value={expected})"
                        )

                fn(x)
                buf = buf.replace("\n", "").replace("  ", "")
                expected = expected.replace("\n", "").replace("  ", "")
                self.assertEqual(expected, buf)

    def test_print_captured_tensor_inside_transform(self, device):
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        out = None

        def f(y):
            nonlocal out
            out = repr(x)
            return y

        vjp(f, torch.randn(4, device=device))
        self.assertEqual(out, repr(x))

    def test_no_grad_outside(self, device):
        x = torch.randn([], device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(torch.sin)(x)
        self.assertEqual(y, x.cos())
        self.assertFalse(y.requires_grad)

    def test_no_grad_inside(self, device):
        def f(x):
            with torch.no_grad():
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device)
        y = grad(f)(x)
        self.assertEqual(y, 2 * x)
        y = grad(grad(f))(x)
        self.assertEqual(y, 2)

        x = torch.randn([], device=device, requires_grad=True)
        y = grad(f)(x)
        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 2)

    def test_no_grad_mixed(self, device):
        def f(x):
            with torch.no_grad():
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(f)(x)

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)

    def test_no_grad_nested_simple(self, device):
        def h(x):
            with torch.no_grad():
                shift = grad(lambda x: 0.25 * x**4)(x)
            return x**3 - shift

        x = torch.tensor(1.5, device=device, requires_grad=True)
        y = grad(h)(x)
        self.assertEqual(y, 3 * x**2)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 6 * x)

    def test_no_grad_nested_complicated(self, device):
        def f(x):
            with torch.no_grad():
                shift = x**3
            return x**3 - shift

        def g(x):
            r1 = grad(f)(x)
            with torch.no_grad():
                shift = grad(f)(x)
            return r1 - shift

        x = torch.randn([], requires_grad=True, device=device)
        y = grad(g)(x)
        # The only differential part of g is x ** 3
        self.assertEqual(y, 6 * x)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 6)

    def test_no_grad_value(self, device):
        def h(x):
            with torch.no_grad():
                gvalue, value = grad_and_value(lambda x: x**3)(x)
            return x**3 - value

        x = torch.tensor(1.6, device=device, requires_grad=True)
        y = grad(h)(x)
        self.assertEqual(y, 3 * x**2)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 6 * x)

    def test_no_grad_outside_vjp(self, device):
        def h(x):
            return x**2

        x = torch.tensor(2.0, requires_grad=True, device=device)
        with torch.no_grad():
            out, vjp_fn = vjp(h, x)
            (y,) = vjp_fn(torch.tensor(1.0, device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(out.requires_grad)

    def test_no_grad_outside_vjp_fn(self, device):
        def h(x):
            return x**2

        x = torch.tensor(3.14, requires_grad=True, device=device)
        out, vjp_fn = vjp(h, x)
        with torch.no_grad():
            (y,) = vjp_fn(torch.tensor(1.0, device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)
        self.assertTrue(out.requires_grad)

        (z,) = torch.autograd.grad(out, x)
        self.assertEqual(z, 2 * x)

    def test_no_grad_outside_vjp_only(self, device):
        def h(x):
            return x**2

        x = torch.tensor(3.14, requires_grad=True, device=device)
        with torch.no_grad():
            out, vjp_fn = vjp(h, x)
        (y,) = vjp_fn(torch.tensor(1.0, device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(out.requires_grad)

        # This one is a little weird...
        self.assertTrue(y.requires_grad)

        (z,) = torch.autograd.grad(y, x)
        self.assertEqual(z, 2)


@markDynamoStrictTest
class TestAutogradFunction(TestCase):
    def test_set_materialize_grads(self, device):
        class A(torch.autograd.Function):
            @staticmethod
            def forward(x, y):
                return x, y

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.set_materialize_grads(False)

            @staticmethod
            def backward(ctx, gx, gy):
                self.assertIsNotNone(gx)
                self.assertIsNone(gy)
                return gx, gy

        def f(y, x):
            x, y = A.apply(x, y)
            return x**2

        x = torch.tensor(2.0, device=device)
        y = torch.tensor(3.0, device=device)
        # grad differentiates w.r.t. arg 0 by default
        grad(f)(y, x)
        grad(grad(f))(y, x)

    @parametrize("inner_requires_grad", [True, False])
    @parametrize("save_for", ["jvp", "vjp"])
    @parametrize("save_tensors", ["input", "output", "neither"])
    @parametrize("mark_dirty", [True, False])
    def test_function_returns_input(
        self, device, inner_requires_grad, save_for, save_tensors, mark_dirty
    ):
        class A(torch.autograd.Function):
            @staticmethod
            def forward(x):
                return x

            @staticmethod
            def setup_context(ctx, inputs, output):
                if save_for == "jvp":
                    save_fn = ctx.save_for_forward
                else:
                    save_fn = ctx.save_for_backward

                if mark_dirty:
                    ctx.mark_dirty(inputs[0])

                if save_tensors == "input":
                    save_fn(inputs[0])
                elif save_tensors == "output":
                    save_fn(output)
                elif save_tensors == "neither":
                    pass

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

            @staticmethod
            def jvp(ctx, x_t):
                # NB: the logic to check ctx.save_for_forward happens
                #     before we reach this!
                if mark_dirty:
                    ret = x_t.add_(0)
                else:
                    ret = x_t.view_as(x_t)
                return ret

        def fn(x):
            return A.apply(x.clone())

        err_msg = "A input that has been returned as-is"

        a = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad)
        a_t = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad)
        if save_tensors in ("input", "output") and not mark_dirty:
            with self.assertRaisesRegex(RuntimeError, err_msg):
                grad(fn)(a)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                jvp(fn, (a,), (a_t,))
        else:
            grad(fn)(a)
            jvp(fn, (a,), (a_t,))

        a = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad).clone()
        a_t = torch.tensor(
            2.0, device=device, requires_grad=inner_requires_grad
        ).clone()

        if save_tensors in ("input", "output") and not mark_dirty:
            with self.assertRaisesRegex(RuntimeError, err_msg):
                A.apply(a)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                with fwAD.dual_level():
                    A.apply(fwAD.make_dual(a, a_t))
        else:
            b = A.apply(a)
            if mark_dirty:
                self.assertTrue(a is b)
            if not (
                mark_dirty and save_for == "vjp" and save_tensors in ("input", "output")
            ):
                # TODO(soulitzer): https://github.com/pytorch/pytorch/issues/97827
                with fwAD.dual_level():
                    a_dual = fwAD.make_dual(a, a_t)
                    b_dual = A.apply(a_dual)
                if mark_dirty:
                    self.assertTrue(a_dual is b_dual)

    def test_needs_input_grads(self, device):
        class A(torch.autograd.Function):
            @staticmethod
            def forward(x, y):
                return x * y

            @staticmethod
            def setup_context(ctx, inputs, output):
                return

            @staticmethod
            def backward(ctx, grad_output):
                self.assertTrue(ctx.needs_input_grad[0])
                self.assertFalse(ctx.needs_input_grad[1])
                return None, None

        x = torch.tensor(2.0, device=device)
        y = torch.tensor(3.0, device=device)
        # grad differentiates w.r.t. arg 0 by default
        grad(A.apply)(x, y)
        grad(grad(A.apply))(x, y)

    def _get_NumpyCubeNotComposable(self):
        class NumpyCubeNotComposable(torch.autograd.Function):
            @staticmethod
            def forward(input):
                input_np = input.cpu().numpy()
                return torch.tensor(input_np**3, device=input.device), input_np

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.input_np = output[1]
                ctx.device = inputs[0].device

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad_output, grad_saved):
                result_np = 3 * (ctx.input_np**2)
                return torch.tensor(result_np, device=ctx.device)

        return NumpyCubeNotComposable

    def test_once_differentiable_autograd_vjp(self, device):
        NumpyCubeNotComposable = self._get_NumpyCubeNotComposable()

        def f(x):
            y, _ = NumpyCubeNotComposable.apply(x)
            return y

        # regular autograd x vjp
        x = torch.randn([], requires_grad=True, device=device)
        grad_y = torch.randn_like(x, requires_grad=True)
        _, vjp_fn = vjp(f, x)
        (gx,) = vjp_fn(grad_y)

        with self.assertRaisesRegex(RuntimeError, "marked with @once_differentiable"):
            gx.backward()

    # TODO: support torch.autograd.function.once_differentiable
    # (or, if impossible, figure out how to raise a nice error)
    # https://github.com/pytorch/pytorch/issues/90224
    @unittest.expectedFailure
    def test_once_differentiable_grad_vjp(self, device):
        # grad x vjp
        x = torch.randn([], device=device)
        grad_y = torch.randn_like(x)

        def h(x, grad_y):
            _, vjp_fn = vjp(f, x)  # noqa: F821
            (gx,) = vjp_fn(grad_y)
            return gx

        grad(h, argnums=(0, 1))(x, grad_y)

    def test_grad_fn_name(self, device):
        names = []

        class FooBar(torch.autograd.Function):
            @staticmethod
            def forward(x):
                return x.clone()

            @staticmethod
            def setup_context(ctx, inputs, output):
                return

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        def f(x):
            y = FooBar.apply(x)
            names.append(type(y.grad_fn).__name__)
            return y

        x = torch.tensor(1.0)
        grad(f)(x)
        self.assertEqual(names, ["FooBarGeneratedBackward"])


@markDynamoStrictTest
class TestAutogradFunctionVmapAPI(TestCase):
    def test_no_vmap_staticmethod_and_no_generate_vmap_rule(self, device):
        class NumpyCube(torch.autograd.Function):
            @staticmethod
            def forward(input):
                input_np = to_numpy(input)  # noqa: F821
                dinput = torch.tensor(3 * input_np**2, device=input.device)
                return torch.tensor(input_np**3, device=input.device), dinput

            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.save_for_backward(inputs, output[1])

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                raise RuntimeError("foobar")

        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "does not have vmap support"):
            vmap(NumpyCube.apply)(x)

    def test_has_vmap_staticmethod_and_has_generate_vmap_rule(self, device):
        class NumpyCube(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(input):
                input_np = to_numpy(input)  # noqa: F821
                dinput = torch.tensor(3 * input_np**2, device=input.device)
                return torch.tensor(input_np**3, device=input.device), dinput

            @staticmethod
            def setup_context(ctx, outputs, input):
                ctx.save_for_backward(input, outputs[1])

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                raise RuntimeError("foobar")

            @staticmethod
            def vmap(infos, in_dims, x):
                raise RuntimeError("foobar")

        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "generate_vmap_rule=True and"):
            vmap(NumpyCube.apply)(x)

    def test_info_object(self, device):
        batch_size = 10

        class Id(torch.autograd.Function):
            @staticmethod
            def forward(input):
                pass

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                self.assertEqual(info.batch_size, batch_size)
                self.assertEqual(info.randomness, randomness)
                return input, in_dims[0]

        x = torch.randn(batch_size, 3, device=device)

        for randomness in ("error", "different", "same"):
            vmap(Id.apply, randomness=randomness)(x)

    def test_in_dims_single_input(self, device):
        class Id(torch.autograd.Function):
            @staticmethod
            def forward(input):
                pass

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                self.assertEqual(in_dims, (1,))
                return input, in_dims[0]

        B = 10
        x = torch.randn(3, B, device=device)
        vmap(Id.apply, in_dims=1)(x)
        vmap(Id.apply, in_dims=(1,))(x)

    def test_in_dims_multiple_inputs(self, device):
        class Id(torch.autograd.Function):
            @staticmethod
            def forward(x, y):
                pass

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass

            @staticmethod
            def vmap(info, in_dims, x, y):
                self.assertEqual(in_dims, (0, [0, 0]))
                self.assertTrue(isinstance(in_dims, tuple))
                self.assertTrue(isinstance(in_dims[1], list))
                return (x, y), in_dims

        x = torch.randn(2, device=device)
        vmap(Id.apply)(x, [x, x])

    def test_skips_empty_layer(self, device):
        class Id(torch.autograd.Function):
            @staticmethod
            def forward(input):
                return input

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                raise RuntimeError("expected to not be called")

        def f(x):
            y = torch.tensor(1.0)
            y = Id.apply(y)
            return x * 1

        x = torch.randn(2, 3)
        vmap(f)(x)

    def test_none_returns(self, device):
        class Zeros(torch.autograd.Function):
            @staticmethod
            def forward(input):
                return torch.zeros(input.shape, device=input.device)

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                assert in_dims == (0,)
                return torch.zeros(input.shape[1:], device=input.device), None

        B = 2
        x = torch.randn(B, 3)
        y = vmap(Zeros.apply)(x)
        self.assertEqual(y, torch.zeros_like(x))

        class TwoZeros(torch.autograd.Function):
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r, r

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                assert in_dims == (0,)
                r = torch.zeros(input.shape[1:], device=input.device)
                return (r, r), None

        B = 2
        x = torch.randn(B, 3)
        result = vmap(TwoZeros.apply)(x)

        self.assertTrue(isinstance(result, tuple))
        y, z = result
        self.assertEqual(y, torch.zeros_like(x))
        self.assertEqual(z, torch.zeros_like(x))

    def test_should_have_two_returns(self, device):
        class Zeros(torch.autograd.Function):
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return r

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "to have two returns"):
            vmap(Zeros.apply)(x)

        class TwoZeros(torch.autograd.Function):
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r, r

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return r, r, 0, 0

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "to have two returns"):
            vmap(Zeros.apply)(x)

    def test_incompatible_out_dims_error_msg(self, device):
        class Zeros(torch.autograd.Function):
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return r, (None,)

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            vmap(Zeros.apply)(x)

        class Zeros(torch.autograd.Function):
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return [r]

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return [r], (None,)

        B = 2
        x = torch.randn(B, 3)
        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            vmap(Zeros.apply)(x)

    def test_kwarg_only_tensors(self, device):
        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            class MyClass(torch.autograd.Function):
                @staticmethod
                def forward(x, *, y):
                    return x + y

                @staticmethod
                def setup_context(ctx, inputs, output):
                    pass

                @staticmethod
                def vmap(info, in_dims, x, *, y):
                    assert in_dims == (0,)
                    return x + y, 0

            x = torch.randn(3)
            y = torch.randn(3)

            vmap(MyClass.apply)(x, y=y)


@markDynamoStrictTest
class TestVmapOfGrad(TestCase):
    def test_per_sample_grads_inplace_view(self, device):
        def compute_loss(weight, x, t):
            x = x.mm(weight)
            y = x.squeeze_(0)
            return (y - t).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 1, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_new_zeros_materializes_tensor(self, device):
        N = 3
        C = 5

        def foo(y, x):
            result = x.new_zeros((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_new_empty_materializes_tensor(self, device):
        N = 3
        C = 5

        def foo(y, x):
            result = x.new_empty((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_per_sample_grads_simple(self, device):
        def compute_loss(weight, x, t):
            y = x @ weight
            return ((y - t) ** 2).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def _compare_expected_and_result(self, expected, result, mechanism):
        if mechanism == "make_functional":
            expected = zip(*expected)
            expected = tuple(torch.stack(shards) for shards in expected)
            for r, e in zip(result, expected):
                self.assertEqual(r, e, atol=0, rtol=1.5e-3)
        else:
            assert mechanism == "functional_call"
            expected = {
                k: tuple(d[k] for d in expected) for k, v in expected[0].items()
            }
            expected = {k: torch.stack(shards) for k, shards in expected.items()}
            for key in result:
                self.assertEqual(result[key], expected[key], atol=0, rtol=1.5e-3)

    @tf32_on_and_off(0.005)
    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_per_sample_grads_embeddingnet(self, device, mechanism):
        class SampleNet(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 16)
                self.fc1 = nn.Linear(16, 16)
                self.fc2 = nn.Linear(16, 2)

            def forward(self, x):
                x = self.emb(x)
                x = torch.transpose(x, -1, -2)
                x = torch.mean(x, -1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

            def name(self):
                return "SampleNet"

        # Create our inputs...
        vocab_size = 1000
        batch_shape = [64]
        words_per_sentence = 5
        data = torch.randint(
            0, vocab_size, (*batch_shape, words_per_sentence), device=device
        )
        targets = torch.randint(0, 1, (*batch_shape,), device=device)

        # Construct our module
        net = SampleNet(vocab_size).to(device=device)
        criterion = nn.CrossEntropyLoss()

        net_func, weights = _get_weights_and_functional_call(net, mechanism)

        def compute_loss(weights, data, target):
            output = net_func(weights, data)
            result = criterion(output, target)
            return result

        expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
        result = vmap(partial(grad(compute_loss), weights))(data, targets)
        self._compare_expected_and_result(expected, result, mechanism)

    def test_log_softmax(self, device):
        x = torch.randn(3, 5, device=device)
        v = torch.randn(5, device=device)

        def foo(x, v):
            _, vjp_fn = vjp(partial(torch.log_softmax, dim=-1), x)
            return vjp_fn(v)[0]

        result = vmap(foo, (0, None))(x, v)

        v = v.expand_as(x)
        x.requires_grad_()
        output = torch.log_softmax(x, dim=-1)
        output.backward(v)
        self.assertEqual(result, x.grad)


jacrev_and_jacfwd = parametrize(
    "jacapi", [subtest(jacrev, name="jacrev"), subtest(jacfwd, name="jacfwd")]
)

FIXME_jacrev_only = parametrize("jacapi", [subtest(jacrev, name="jacrev")])


@markDynamoStrictTest
class TestJac(VmapTearDownMixin, TestCase):
    @jacrev_and_jacfwd
    def test_simple(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = jacapi(torch.sin)(x)
        expected = torch.diagflat(x.cos())
        assert torch.allclose(y, expected)

    @jacrev_and_jacfwd
    def test_simple_not_flat(self, device, jacapi):
        x = torch.randn(2, 3, device=device)
        y = jacapi(torch.sin)(x)
        expected = torch.diagflat(x.view(-1).cos())
        expected = expected.view(2, 3, 2, 3)
        assert torch.allclose(y, expected)

    @jacrev_and_jacfwd
    def test_take(self, device, jacapi):
        x = torch.rand(5)

        def func(x):
            y = torch.ones(3, dtype=torch.long)
            z = torch.take(x, y)
            return z

        self.assertEqual(jacrev(func)(x), torch.autograd.functional.jacobian(func, x))

    @jacrev_and_jacfwd
    def test_diff_numel(self, device, jacapi):
        x = torch.randn(2, 4, device=device)

        # Tensor[2, 4] -> Tensor[3, 1]
        def f(x):
            return x[0, 1:].unsqueeze(-1)

        y = jacapi(f)(x)
        self.assertEqual(y.shape, (3, 1, 2, 4))

        expected = x.new_zeros(3, 1, 2, 4)
        expected[0, 0, 0, 1] = 1
        expected[1, 0, 0, 2] = 1
        expected[2, 0, 0, 3] = 1
        self.assertEqual(y, expected)

    @jacrev_and_jacfwd
    def test_vmap_on_jac_simple(self, device, jacapi):
        x = torch.randn(2, 3, device=device)
        y = vmap(jacapi(torch.sin))(x)
        expected = torch.stack([torch.diagflat(x[i].cos()) for i in range(2)])
        assert torch.allclose(y, expected)

    @jacrev_and_jacfwd
    def test_nested_jac_simple(self, device, jacapi):
        def foo(x):
            return x.sin().sum()

        x = torch.randn(3, device=device)
        y = jacapi(jacapi(foo))(x)
        expected = torch.diagflat(-x.sin())
        assert torch.allclose(y, expected)

    @jacrev_and_jacfwd
    def test_multiple_args(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=1)(x, y)
        expected = torch.diagflat(x)
        assert torch.allclose(z, expected)

    @jacrev_and_jacfwd
    def test_multiple_outputs_multiple_argnums(self, device, jacapi):
        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f, argnums=(0, 1))(x, y)
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        expected_out0_y = torch.diagflat(torch.full_like(y, 3))
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))
        expected_out1_y = torch.diagflat(torch.full_like(y, 5))

        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertEqual(len(z[0]), 2)
        self.assertTrue(isinstance(z[0], tuple))
        self.assertEqual(z[0][0], expected_out0_x)
        self.assertEqual(z[0][1], expected_out0_y)
        self.assertEqual(z[1][0], expected_out1_x)
        self.assertEqual(z[1][1], expected_out1_y)

    @jacrev_and_jacfwd
    def test_multiple_outputs_single_argnums(self, device, jacapi):
        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))

        z = jacapi(f, argnums=0)(x, y)
        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertEqual(z, (expected_out0_x, expected_out1_x))

        z = jacapi(f, argnums=(0,))(x, y)
        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertTrue(isinstance(z[0], tuple))
        self.assertEqual(z, ((expected_out0_x,), (expected_out1_x,)))

    @jacrev_and_jacfwd
    def test_multiple_outputs_pytree(self, device, jacapi):
        def f(x, y):
            return {"left": 2 * x + 3 * y, "right": 4 * x + 5 * y}

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f, argnums=(0, 1))(x, y)
        expected_left_x = torch.diagflat(torch.full_like(x, 2))
        expected_left_y = torch.diagflat(torch.full_like(y, 3))
        expected_right_x = torch.diagflat(torch.full_like(x, 4))
        expected_right_y = torch.diagflat(torch.full_like(y, 5))
        expected = {
            "left": (expected_left_x, expected_left_y),
            "right": (expected_right_x, expected_right_y),
        }
        self.assertTrue(isinstance(z, dict))
        self.assertTrue(isinstance(z["left"], tuple))
        self.assertTrue(isinstance(z["right"], tuple))
        self.assertEqual(z, expected)

    @jacrev_and_jacfwd
    def test_multiple_inputs_pytree(self, device, jacapi):
        def f(a, b, c):
            a0, a1 = a
            return a0 + a1 * 2 + b * 3 + c * 4

        x = torch.randn([], device=device)
        args = ((x, x), x, x)

        result = jacapi(f, argnums=(0, 1, 2))(*args)
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
            torch.tensor(3.0, device=device),
            torch.tensor(4.0, device=device),
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0,))(*args)
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
        )
        self.assertEqual(result, expected)

        result = jacapi(f)(*args)
        expected = (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device))
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_dimensionality(self, device, jacapi):
        def f(x):
            return x

        x = torch.randn([], device=device)
        result = jacapi(f)(x)
        self.assertEqual(result.dim(), 0)
        self.assertEqual(result, torch.ones_like(x))

        x = torch.randn([1], device=device)
        result = jacapi(f)(x)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result, x.new_ones(1, 1))

    @jacrev_and_jacfwd
    def test_aux_tensor(self, device, jacapi):
        def f(x):
            y = x.clone()
            return y, y.cos()

        x = torch.randn(3, device=device)
        result, aux = jacapi(f, has_aux=True)(x)

        self.assertEqual(result, torch.eye(3, 3, device=device))
        self.assertEqual(aux, x.cos())

    @jacrev_and_jacfwd
    def test_aux_pytree(self, device, jacapi):
        def f(x):
            y = x.clone()
            return y, {"a": y.cos(), "b": [y.tan()]}

        x = torch.randn(3, device=device)

        result, aux = jacapi(f, has_aux=True)(x)
        self.assertEqual(result, torch.eye(3, 3, device=device))
        _, expected_aux = f(x)
        self.assertEqual(aux, expected_aux)

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jacapi(lambda x: (x, aux), has_aux=True)(x)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jacapi(lambda x: (x, [x, aux]), has_aux=True)(x)

    @jacrev_and_jacfwd
    def test_outputs_can_any_pytree(self, device, jacapi):
        x = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"(vjp|jvp).+: Expected f to be a function that has non-empty output",
            ):
                jacapi(lambda _: output)(x)

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"(vjp|jvp).+: expected f\(\*primals\) to return only tensors",
            ):
                jacapi(lambda _: output)(x)

        # Check list output
        out = jacapi(lambda x: [x, x.sum()])(x)
        assert isinstance(out, list) and len(out) == 2

        # Check dict output
        out = jacapi(lambda x: {"x": x, "xsum": x.sum()})(x)
        assert isinstance(out, dict) and len(out) == 2 and "xsum" in out

        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jacapi(composite_output)(x)
        assert isinstance(out, list)
        assert isinstance(out[0], tuple) and isinstance(out[0][1], dict)

    @jacrev_and_jacfwd
    def test_multiple_inputs_outputs_pytree(self, device, jacapi):
        def f(a, b, c):
            a0, a1 = a
            return a0 + a1 * 2, {"foo": b * 3 + c * 4}

        x = torch.randn([], device=device)
        zero = torch.zeros([], device=device)
        args = ((x, x), x, x)

        result = jacapi(f)(*args)
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
            {"foo": (zero, zero)},
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0,))(*args)
        expected = (
            ((torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),),
            {"foo": ((zero, zero),)},
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0, 1))(*args)
        expected = (
            (
                (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
                zero,
            ),
            {"foo": ((zero, zero), torch.tensor(3.0, device=device))},
        )
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_multiple_inputs_outputs_pytree_multidim(self, device, jacapi):
        def f(dct):
            a = dct["a"]
            b = dct["b"]
            return {"c": a.sin(), "d": b.cos()}

        x = torch.randn(3, device=device)
        args = ({"a": x, "b": x},)

        result = jacapi(f)(*args)
        expected = {
            "c": {"a": x.cos().diagflat(), "b": x.new_zeros(3, 3)},
            "d": {"a": x.new_zeros(3, 3), "b": -x.sin().diagflat()},
        }
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_unrelated_input(self, device, jacapi):
        def f(x, y):
            return x

        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)

        result = jacapi(f, argnums=(0, 1))(x, y)
        expected0 = torch.eye(6, 6, device=device).view(2, 3, 2, 3)
        expected1 = y.new_zeros(2, 3, 2, 3)
        expected = (expected0, expected1)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_unrelated_output(self, device, jacapi):
        y = torch.randn(2, 3, device=device)

        def f(x):
            return y

        x = torch.randn(2, 3, device=device)

        result = jacapi(f)(x)
        expected = x.new_zeros(2, 3, 2, 3)
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_empty_output(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)

        def f(x, y):
            return ()

        with self.assertRaisesRegex(RuntimeError, "xpected"):
            jacapi(f)(x, y)

    @jacrev_and_jacfwd
    def test_argnums_tuple(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=(0, 1))(x, y)
        expected0 = torch.diagflat(y)
        expected1 = torch.diagflat(x)
        assert len(z) == 2
        assert torch.allclose(z[0], expected0)
        assert torch.allclose(z[1], expected1)

    @jacrev_and_jacfwd
    def test_argnums_effect_on_return(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=(0,))(x, y)
        expected0 = torch.diagflat(y)
        assert isinstance(z, tuple)
        assert len(z) == 1
        assert torch.allclose(z[0], expected0)

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=0)(x, y)
        expected0 = torch.diagflat(y)
        assert isinstance(z, torch.Tensor)
        assert torch.allclose(z, expected0)

    @jacrev_and_jacfwd
    def test_argnums_defaults_to_zero(self, device, jacapi):
        def f(x, y):
            return x * 2 + y * 3

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f)(x, y)
        expected = torch.diagflat(torch.full_like(x, 2))
        self.assertEqual(z, expected)

    @jacrev_and_jacfwd
    def test_empty_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            jacapi(torch.sin, argnums=())(x)

    @jacrev_and_jacfwd
    def test_out_of_bounds_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=2)(x)

    @jacrev_and_jacfwd
    def test_negative_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=-2)(x)

    @jacrev_and_jacfwd
    def test_repeated_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            jacapi(torch.sin, argnums=(0, 0))(x)

    @jacrev_and_jacfwd
    def test_float_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be int or Tuple"):
            jacapi(torch.sin, argnums=0.0)(x)
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            jacapi(torch.multiply, argnums=(1, 0.0))(x, x)

    def test_hessian_simple(self, device):
        def f(x):
            return x.sin()

        x = torch.randn(3, device=device)
        hessian(f)(x)

    def _test_against_reference(self, f, inputs, jacapi):
        def foo(inputs):
            return f(*inputs)

        expected = torch.autograd.functional.jacobian(f, inputs)
        result = jacapi(foo)(inputs)
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_against_reference_simple(self, device, jacapi):
        def f(x):
            return 3 * x**2

        x = torch.randn(2, 3, 5, device=device)
        self._test_against_reference(f, (x,), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_multi_input(self, device, jacapi):
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_multi_input_multi_output(self, device, jacapi):
        def f(x, y):
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        x = torch.randn(5, 3, device=device)
        y = torch.randn(3, 5, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_unrelated_outputs(self, device, jacapi):
        def f(x, y):
            return x, y, x, y

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_zero_dim(self, device, jacapi):
        # zero-dim output
        def f(x, y):
            return x.sum(), y.sum(), x * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

        # zero-dim input
        def g(x):
            return torch.stack([x, x, x])

        x = torch.randn([], device=device)
        self._test_against_reference(g, (x,), jacapi)

        # Mixed zero-dim input / zero-dim output
        def h(x, y):
            return y.sum(), x * y

        x = torch.randn([], device=device)
        y = torch.randn(1, device=device)
        self._test_against_reference(h, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_correctness_different_devices(self, device, jacapi):
        def f(x, y):
            return x * y, (x * y).to(device=device)

        x = torch.randn(3)
        y = torch.randn(3)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_default_arg(self, device, jacapi):
        def f(x, y, z=3.0):
            return x * y * z

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_inplace(self, device, jacapi):
        def f(x, y):
            y.copy_(x)
            return y

        out = jacapi(f, argnums=0)  # x is differentiable
        x, y = torch.randn(2, device=device), torch.randn(2, device=device)
        self.assertEqual(out(x, y), torch.eye(y.shape[0]))

        # testing tuple of argnums with the example that raised this issue originally
        def g(x, y, z):
            x[:2] = y
            return torch.vstack([(x**2).sum(), (z**3).sum()])

        out = jacapi(g, argnums=(1, 2))
        x, y, z = (
            torch.randn(3, device=device),
            torch.randn(2, device=device),
            torch.randn(2, device=device),
        )

        expected_out = (
            torch.zeros(2, 1, 2, device=device),
            torch.zeros(2, 1, 2, device=device),
        )
        expected_out[0][0][0] = 2 * y  # top left corner
        expected_out[1][1][0] = 3 * (z**2)  # bottom right corner

        out_val = out(x, y, z)
        self.assertEqual(out_val, expected_out)

    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev(self, device, _preallocate_and_copy):
        x = torch.randn(10, 2, device=device)
        y = torch.randn(1, 2, device=device)

        def f(x, y):
            return (x.sin(), x + y), (x + 2, x.sum())

        for chunk_size in (1, 2, 3, 4, 7, 10, 1000):
            expected = jacrev(f, argnums=(0, 1))(x, y)
            actual = jacrev(
                f,
                argnums=(0, 1),
                chunk_size=chunk_size,
                _preallocate_and_copy=_preallocate_and_copy,
            )(x, y)
            self.assertEqual(actual, expected)

        err_msg = "jacrev: `chunk_size` should be greater than 0."
        with self.assertRaisesRegex(ValueError, err_msg):
            jacrev(f, argnums=(0,), chunk_size=0)(x, y)

        with self.assertRaisesRegex(ValueError, err_msg):
            jacrev(f, argnums=(0,), chunk_size=-2)(x, y)

    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev_composition(self, device, _preallocate_and_copy):
        x = torch.randn(10, 2, device=device)
        chunk_size = 3

        def f(x):
            return (x.sin(), x), (x + 2, x.sum())

        expected = vmap(jacrev(jacrev(f)))(x)
        actual = vmap(
            jacrev(
                jacrev(
                    f,
                    chunk_size=chunk_size,
                    _preallocate_and_copy=_preallocate_and_copy,
                ),
                chunk_size=chunk_size,
            )
        )(x)
        self.assertEqual(actual, expected)

    # https://github.com/pytorch/pytorch/issues/127036
    @xfailIfTorchDynamo
    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev_chunksize_one(self, device, _preallocate_and_copy):
        # With chunk_size=1, we shouldn't `vmap` and hence not be limited
        # by it's constraints.
        x = torch.randn(3, 3, device=device)

        # Function with Dynamic Op in Backward.
        # This should cause jacrev/vmap(vjp) to fail.
        class IdentityWithDynamicBackwardOp(torch.autograd.Function):
            @staticmethod
            def forward(input):
                return input

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output):
                # dynamic op in backward pass.
                grad_output.nonzero()
                return grad_output

        def f(x):
            return IdentityWithDynamicBackwardOp.apply(x)

        # With `chunk_size=1`, we don't use vmap. So the following should work.
        jacfn = jacrev(f, chunk_size=1, _preallocate_and_copy=_preallocate_and_copy)
        actual = jacfn(x)
        expected = torch.autograd.functional.jacobian(f, x, vectorize=False)
        self.assertEqual(actual, expected)

        # Should fail with `chunk_size=2`.
        msg = (
            r"vmap: We do not support batching operators that can output dynamic shape."
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            jacrev(f, chunk_size=2, _preallocate_and_copy=_preallocate_and_copy)(x)

    def test_complex_error(self, device):
        # Verify complex input raises error
        # C -> C
        def fn(x):
            return x.conj()

        x = torch.randn(1, device=device, dtype=torch.cfloat)

        with self.assertRaisesRegex(RuntimeError, "jacrev: Expected all inputs"):
            jacrev(fn)(x)

        with self.assertRaisesRegex(RuntimeError, "jacfwd: Expected all inputs"):
            jacfwd(fn)(x)

        # Verify complex output raises error
        # R -> C
        def fn(x):
            return torch.conj(x * 0.5j)

        x = torch.randn(1, device=device, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, "jacrev: Expected all outputs"):
            jacrev(fn)(x)

        with self.assertRaisesRegex(RuntimeError, "jacfwd: Expected all outputs"):
            jacfwd(fn)(x)

    @jacrev_and_jacfwd
    def test_jac_with_non_tensor_args(self, device, jacapi):
        def f(t, int_x):
            return t + int_x

        t = torch.randn(3, 3, device=device)

        actual = jacapi(f)(t, 3)
        expected = torch.autograd.functional.jacobian(partial(f, int_x=3), t)
        self.assertEqual(actual, expected)


@markDynamoStrictTest
class TestHessian(TestCase):
    def _test_against_reference(self, f, inputs):
        def foo(inputs):
            return f(*inputs)

        expected = torch.autograd.functional.hessian(f, inputs)
        result = hessian(foo)(inputs)
        self.assertEqual(result, expected)

    def test_hessian_vectorize_correctness_simple(self, device):
        def f(x):
            return (3 * x**2).sum()

        x = torch.randn(2, 3, 5, device=device)
        self._test_against_reference(f, (x,))

    def test_hessian_vectorize_correctness_multi_input(self, device):
        def f(x, y, z):
            return ((x.relu() * x) @ y.sin() @ z).sum()

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        z = torch.randn(5, 5, device=device)
        self._test_against_reference(f, (x, y, z))

    def test_hessian_vectorize_correctness_unrelated_outputs(self, device):
        # output unrelated to one input
        def f(x, y):
            return (x**2).sum()

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y))

        # output unrelated to all inputs
        def f(x, y):
            return torch.ones([])

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y))

    def test_jacfwd_different_levels(self, device):
        # Test case from:
        # https://github.com/pytorch/functorch/issues/597
        b = 8
        n = 100
        d = 2
        x1 = torch.randn(b, n, d, device=device)
        x2 = x1
        A = 0.1 * torch.randn(b, d, d, device=device)

        def loss(A, x1, x2):
            x2_hat = (A @ (x1.T)).T
            res = x2 - x2_hat
            res_sqr = res**2
            return res_sqr.sum()

        hess1 = vmap(jacrev(jacrev(loss)))(A, x1, x2)
        hess2 = vmap(hessian(loss))(A, x1, x2)
        self.assertEqual(hess2, hess1)


@markDynamoStrictTest
class TestJvp(TestCase):
    def test_inplace_on_captures(self, device):
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        captured = torch.randn(3, device=device)

        def foo(x):
            captured.copy_(x)
            return (x * captured).sum()

        with self.assertRaisesRegex(RuntimeError, "mutate a captured Tensor"):
            grad(foo)(x)

    def test_simple(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)
        result = jvp(torch.sin, (x,), (t,))
        expected = (x.sin(), x.cos() * t)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_multiple_inputs(self, device):
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        def f(x, y):
            return x * y

        result = jvp(f, (x, y), (tx, ty))
        expected = (x * y, y * tx + x * ty)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_pytree_inputs(self, device):
        def f(x, y, z):
            a, b = x
            return a + 2 * b + 3 * y + 4 * z

        one = torch.tensor(1.0, device=device)
        primal_outs, tangent_outs = jvp(
            f, ((one, one), one, one), ((one, one), one, one)
        )
        self.assertEqual(primal_outs, one * 10)
        self.assertEqual(tangent_outs, one * 10)

    def test_pytree_inputs_error_cases(self, device):
        def f(x):
            return x

        one = torch.tensor(1.0, device=device)

        with self.assertRaisesRegex(RuntimeError, "Expected primals to be a tuple"):
            jvp(f, one, one)
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(f, ((one, one), one), (one, one))
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(f, ((one, one), 1), ((one, one), one))
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(f, ((one, one), 1), ((1, one), one))
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            jvp(f, ((),), ((),))

    def test_unrelated_input(self, device):
        def f(x, y):
            return x

        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        result = jvp(f, (x, y), (tx, ty))
        expected = (x, tx)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_unrelated_output(self, device):
        y = torch.randn(2, 3, device=device)

        def f(x):
            return y

        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        result = jvp(f, (x,), (tx,))
        expected = (y, torch.zeros_like(y))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_strict_mode(self, device):
        y = torch.randn(2, 3, device=device)

        def f(x):
            return x, y

        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        with self.assertRaisesRegex(RuntimeError, "strict"):
            jvp(f, (x,), (tx,), strict=True)

    def test_multiple_outputs(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        def f(x):
            return torch.sin(x), torch.cos(x)

        result = jvp(f, (x,), (t,))
        expected = (f(x), (x.cos() * t, -x.sin() * t))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_multiple_inputs_outputs(self, device):
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        result = jvp(f, (x, y), (tx, ty))
        expected = (f(x, y), f(tx, ty))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_jvp_new_tensor(self):
        def f(x):
            y = x.new_tensor(0.5)
            return x + y

        x = torch.rand(10, 10)
        tangents = torch.zeros_like(x)
        actual = jvp(f, (x,), (tangents,))
        expected = (f(x), torch.zeros_like(x))
        self.assertEqual(actual, expected)

    def test_primals_tangents_length_mismatch(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        msg = "same python structure"
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x,), (t, t))
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x, x), (t, t, t))

    def test_nonempty_primals_and_tangents(self, device):
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            jvp(torch.sin, (), ())

    def test_inputs_are_tuples_of_tensors(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        with self.assertRaisesRegex(RuntimeError, "be a tuple"):
            jvp(torch.sin, x, (t,))
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(torch.sin, (x,), t)
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(torch.sin, (x,), [t])
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(torch.sin, (1.0,), (t,))
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(torch.sin, (x,), (1.0,))

    def test_outputs_can_any_pytree(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"jvp\(f, primals, tangents\): Expected f to be a function that has non-empty output",
            ):
                jvp(lambda _: output, (x,), (t,))

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"jvp\(f, primals, tangents\): expected f\(\*primals\) to return only tensors",
            ):
                jvp(lambda _: output, (x,), (t,))

        # Check list output
        out = jvp(lambda x: [x, x.sum()], (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], list) and len(out[i]) == 2

        # Check dict output
        out = jvp(lambda x: {"x": x, "xsum": x.sum()}, (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], dict) and len(out[i]) == 2 and "xsum" in out[i]

        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jvp(composite_output, (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], list)
            assert isinstance(out[i][0], tuple) and isinstance(out[i][0][1], dict)

    def test_aux_tensor(self, device):
        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            r"jvp\(f, primals, tangents\): output of function f should be a tuple",
        ):
            jvp(lambda t: [t, t], (x,), (t,), has_aux=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"jvp\(f, primals, tangents\): output of function f should be a tuple",
        ):
            jvp(lambda t: (t, t + 2, t + 3), (x,), (t,), has_aux=True)

        def f(z):
            y = z.sin()
            return y, z.cos()

        out, jvp_out, aux = jvp(f, (x,), (t,), has_aux=True)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.sin())
        self.assertEqual(jvp_out, t * x.cos())

    def test_aux_pytree(self, device):
        def f(x):
            y = x.sin()
            return y, {"a": x.cos(), "b": [x.tan()]}

        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        out, jvp_out, aux = jvp(f, (x,), (t,), has_aux=True)
        expected_out, expected_aux = f(x)
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)
        self.assertEqual(jvp_out, t * x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jvp(lambda x: (x, aux), (x,), (t,), has_aux=True)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = jvp(lambda x: (x, [x, aux]), (x,), (t,), has_aux=True)

    def test_autograd_function_disables_fwd_grad(self, device):
        # Sanity check. We don't really assume this anywhere so
        # it's fine if this breaks one day.
        class MySquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertFalse(enabled)
                return x * x

            @staticmethod
            def backward(ctx, gx):
                return gx

        x = torch.randn(3, requires_grad=True)
        MySquare.apply(x)

    def test_disable_fwd_grad_outside(self, device):
        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        with fwAD._set_fwd_grad_enabled(False):
            _, y = jvp(torch.sin, (x,), (t,))
        self.assertEqual(y, x.cos())

    def test_disable_fwd_grad_inside(self, device):
        def f(x):
            with fwAD._set_fwd_grad_enabled(False):
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        _, y = jvp(f, (x,), (t,))
        self.assertEqual(y, 2 * x)
        _, y = jvp(lambda x: jvp(f, (x,), (t,))[1], (x,), (t,))
        self.assertEqual(y, 2)

    def test_disable_fwd_grad_mixed(self, device):
        def f(x):
            with fwAD._set_fwd_grad_enabled(False):
                shift = x**2
            return x**2 - shift

        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        with fwAD._set_fwd_grad_enabled(True):
            _, y = jvp(f, (x,), (t,))

        self.assertEqual(y, 2 * x)

    def test_jvp_inside_autograd_function(self, device):
        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                t = torch.ones_like(x)
                _, neg_sin_x = jvp(torch.cos, (x,), (t,))
                ctx.save_for_backward(x)
                return -neg_sin_x

            @staticmethod
            def backward(ctx, gx):
                (x,) = ctx.saved_tensors
                t = torch.ones_like(x)
                _, cos_x = jvp(torch.sin, (x,), (t,))
                return gx * cos_x

        x = torch.randn([], device=device, requires_grad=True)
        y = MySin.apply(x)
        self.assertEqual(y, x.sin())

        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    def test_zerotensor_vmapjvp_interaction(self, device):
        dummy = torch.ones(4, 1)
        x = torch.randn(4, 2)
        x_tangent = torch.randn(2)

        def push_jvp(dummy, x):
            result = jvp(torch.cov, (x,), (x_tangent,))
            return result

        # Should not error
        vmap(vmap(push_jvp, (0, None)))(dummy, x)


@markDynamoStrictTest
class TestLinearize(TestCase):
    @dtypes(torch.float)
    def test_linearize_basic(self, device, dtype):
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(x):
            return x.cos()

        actual_output, jvp_fn = linearize(fn, x_p)
        actual_jvp = jvp_fn(x_t)
        expected_output, expected_jvp = jvp(fn, (x_p,), (x_t,))
        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_jvp, expected_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_return(self, device, dtype):
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(x):
            return (x.cos(), x.sum())

        actual_output, jvp_fn = linearize(fn, x_p)
        actual_jvp = jvp_fn(x_t)
        expected_output, expected_jvp = jvp(fn, (x_p,), (x_t,))
        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_jvp, expected_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_composition_vmap(self, device, dtype):
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 3, 1), device=device, dtype=dtype)

        def fn(x):
            return (x.cos(), x.sum())

        _, jvp_fn = linearize(fn, x_p)
        actual_batched_jvp = vmap(jvp_fn)(x_t)

        def jvp_fn(x_t):
            return jvp(fn, (x_p,), (x_t,))[1]

        expected_batched_jvp = vmap(jvp_fn)(x_t)

        self.assertEqual(actual_batched_jvp, expected_batched_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_composition_grad(self, device, dtype):
        x_p = make_tensor((3,), device=device, dtype=dtype)
        x_t = make_tensor((3,), device=device, dtype=dtype)

        def fn(x):
            z = torch.ones(3, device=device, dtype=dtype)
            return grad(lambda x: z @ x)(x)

        _, jvp_fn = linearize(fn, x_p)
        actual_batched_jvp = jvp_fn(x_t)

        def jvp_fn(x_t):
            return jvp(fn, (x_p,), (x_t,))[1]

        expected_batched_jvp = jvp_fn(x_t)

        self.assertEqual(actual_batched_jvp, expected_batched_jvp)

    @dtypes(torch.float)
    @unittest.skipIf(
        TEST_CUDA_MEM_LEAK_CHECK,
        "Leaking memory, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_linearize_nested_input_nested_output(self, device, dtype):
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)
        y_p = make_tensor((3, 1), device=device, dtype=dtype)
        y_t = make_tensor((3, 1), device=device, dtype=dtype)
        z_p = make_tensor((3, 1), device=device, dtype=dtype)
        z_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(arg):
            x = arg["x"]
            y = arg["yz"][0]
            z = arg["yz"][1]

            return {"a": x.sum(), "b": {"c": y + z, "d": (x * z, y.exp())}}

        inp_p = {"x": x_p, "yz": (y_p, z_p)}
        inp_t = {"x": x_t, "yz": (y_t, z_t)}
        actual_output, jvp_fn = linearize(fn, inp_p)
        actual_jvp = jvp_fn(inp_t)

        expected_output, expected_jvp = jvp(fn, (inp_p,), (inp_t,))

        self.assertEqual(actual_output, expected_output)
        self.assertEqual(actual_jvp, expected_jvp)

    @onlyCUDA
    def test_linearize_errors(self):
        dtype = torch.float
        device = torch.device("cpu")
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        def fn(x):
            return x.sin()

        _, jvp_fn = linearize(fn, x_p)

        with self.assertRaisesRegex(
            RuntimeError, "to have the same argspec as the primals"
        ):
            jvp_fn((x_t, x_t))

        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the shape"
        ):
            jvp_fn(x_t.unsqueeze(0))

        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the dtype"
        ):
            jvp_fn(x_t.to(torch.double))

        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the device"
        ):
            jvp_fn(x_t.to(torch.device("cuda")))


# The tests here follow the cases in [Forward Grad View/inplace]
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/autograd_meta.cpp#L18-L43
@markDynamoStrictTest
class TestVmapJvpInplaceView(TestCase):
    # Case 1 in [Forward Grad View/inplace]
    def test_all_dual_no_view(self, device):
        B = 2

        def push_jvp(f):
            def inner(x, xt, y, yt):
                return jvp(f, (x, y), (xt, yt))

            return inner

        def f(x, y):
            x.copy_(y)
            return x

        x = torch.randn(3, B, device=device)
        xt = torch.randn(3, B, device=device)
        y = torch.randn(3, B, device=device)
        yt = torch.randn(3, B, device=device)
        out, out_tangent = vmap(push_jvp(f), in_dims=1)(x, xt, y, yt)
        self.assertEqual(out, x.movedim(1, 0))
        self.assertEqual(out_tangent, yt.movedim(1, 0))

        x = torch.randn(3, B, device=device)
        xt = torch.randn(3, B, device=device)
        y = torch.randn(3, 3, device=device)[:, 1]
        yt = torch.randn(6, device=device)[::2]
        out, out_tangent = vmap(push_jvp(f), in_dims=(1, 1, None, None))(x, xt, y, yt)
        self.assertEqual(out, x.movedim(1, 0))
        self.assertEqual(out_tangent, yt.expand(B, 3))

    # Case 2 in [Forward Grad View/inplace]
    def test_all_dual_base_view_inplace(self, device):
        B = 2

        def push_jvp(f):
            def inner(x, xt, y, yt):
                return jvp(f, (x, y), (xt, yt))

            return inner

        # with view, propagate from view to base

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 63 class(es): VmapTearDownMixin, TestSliceArgnums, TestGradTransform, MLPClassifier, MLPClassifier, TestAutogradFunction, A, A, A, NumpyCubeNotComposable, FooBar, TestAutogradFunctionVmapAPI, NumpyCube, NumpyCube, Id, Id, Id, Id, Zeros, TwoZeros

### Functions
This file defines 622 function(s): tearDown, test_invalid_argnum_type, test_out_of_bounds_argnum_values, test_not_enough_argnums, test_duplicate_argnums, test_flat_args_with_positive_int_argnum, test_flat_args_with_negative_int_argnum, test_flat_args_with_tuple_argnum, test_pytree_args, test_argnums_reorders, _get_weights_and_functional_call, net_func, _get_weights_and_functional_call_with_buffers, net_func, test_primitive, test_composite_simple, test_fn_with_kwargs, foo, test_composite_complicated, foo, test_composite_two_ops, foo, _test_attributes, foo, test_shape, test_dtype, test_is_cuda, test_numel, test_layout_sparse, foo


## Key Components

The file contains 14915 words across 5422 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 181764 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

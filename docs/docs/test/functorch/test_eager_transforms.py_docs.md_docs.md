# Documentation: `docs/test/functorch/test_eager_transforms.py_docs.md`

## File Metadata

- **Path**: `docs/test/functorch/test_eager_transforms.py_docs.md`
- **Size**: 54,821 bytes (53.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/functorch/test_eager_transforms.py`

## File Metadata

- **Path**: `test/functorch/test_eager_transforms.py`
- **Size**: 181,764 bytes (177.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
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
        with 
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
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/functorch/test_eager_transforms.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/functorch`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_aot_joint_with_descriptors.py_kw.md_docs.md`](./test_aot_joint_with_descriptors.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`functorch_additional_op_db.py_kw.md_docs.md`](./functorch_additional_op_db.py_kw.md_docs.md)
- [`test_ac_knapsack.py_docs.md_docs.md`](./test_ac_knapsack.py_docs.md_docs.md)
- [`common_utils.py_kw.md_docs.md`](./common_utils.py_kw.md_docs.md)
- [`test_logging.py_kw.md_docs.md`](./test_logging.py_kw.md_docs.md)
- [`test_rearrange.py_kw.md_docs.md`](./test_rearrange.py_kw.md_docs.md)
- [`test_dims.py_kw.md_docs.md`](./test_dims.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_eager_transforms.py_docs.md_docs.md`
- **Keyword Index**: `test_eager_transforms.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

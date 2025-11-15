# Documentation: test_vmap.py

## File Metadata
- **Path**: `test/functorch/test_vmap.py`
- **Size**: 237111 bytes
- **Lines**: 6432
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

import contextlib
import functools
import itertools
import os
import random
import types
import unittest
import warnings
from collections import namedtuple, OrderedDict
from unittest.case import skipIf

from common_utils import (
    check_vmap_fallback,
    compute_quantities_for_vmap_test,
    decorate,
    DisableVmapFallback,
    generate_vmap_inputs,
    get_fallback_and_vmap_exhaustive,
    is_batch_norm_training,
    is_valid_inplace_sample_input,
    opsToleranceOverride,
    skip,
    skipOps,
    tol1,
    xfail,
    xfailIf,
)
from functorch_additional_op_db import additional_op_db

import functorch
import torch
import torch.nn.functional as F
from functorch import grad, grad_and_value, jacfwd, jvp, vjp, vmap
from functorch.experimental import chunk_vmap
from torch import Tensor
from torch._C._functorch import reshape_dim_into, reshape_dim_outof
from torch._functorch.make_functional import functional_init_with_buffers
from torch._functorch.vmap import restore_vmap
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.autograd_function_db import autograd_function_db
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    tf32_on_and_off,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    OpDTypes,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    markDynamoStrictTest,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    unMarkDynamoStrictTest,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.utils import _pytree as pytree


def get_platform_specific_sdpa():
    ret = [SDPBackend.MATH]
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    return ret


PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()

FALLBACK_REGEX = "There is a performance drop"


class EnableVmapFallbackWarnings:
    def __enter__(self):
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)


@markDynamoStrictTest
class TestVmapAPI(TestCase):
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(ValueError, "got type <class 'float'>"):
            vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'>"):
            vmap(multiple_outputs)(torch.ones(3))

    def test_different_map_dim_size_raises(self):
        x = torch.randn(2)
        y = torch.randn(3)
        expected_msg = (
            "Expected all tensors to have the same size in the mapped dimension"
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(torch.mul)(x, y)
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
                {"x": x, "y": y}
            )

    def test_func_with_no_inputs(self):
        expected_msg = "got no inputs"

        def foo():
            return torch.randn(3)

        def bar(x):
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(foo)()

        with self.assertRaisesRegex(ValueError, expected_msg):
            vmap(bar)()

    def test_func_with_no_tensors(self):
        def foo(x):
            return torch.randn(3)

        with self.assertRaisesRegex(ValueError, "at least one Tensor"):
            vmap(foo, (None,))(1)

    def test_constant_function(self):
        output = vmap(lambda x: torch.tensor(3.14))(torch.ones(3))
        self.assertEqual(output, torch.tensor([3.14, 3.14, 3.14]))

    def test_single_input(self):
        x = torch.randn(2, 3)

        def square(x):
            return x * x

        output = vmap(square)(x)
        self.assertEqual(output, x * x)

    def test_multiple_inputs(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul)(x, y)
        self.assertEqual(output, x * y)

    def test_multiple_outputs(self):
        def foo(x):
            return x * x, x * x * x

        x = torch.randn(3)
        outputs = vmap(foo)(x)
        self.assertEqual(outputs[0], x * x)
        self.assertEqual(outputs[1], x * x * x)

    def test_multiple_outputs2(self):
        # This is the same thing as
        # def returns_tuple_of_tensors(x):
        #     return x, x
        def returns_tuple_of_tensors(x):
            return (x, x)

        def returns_list_of_two_tensors(x):
            return [x, x]

        def returns_list_of_one_tensor(x):
            return [x]

        x = torch.randn(3)

        # should not throw
        vmap(returns_tuple_of_tensors)(x)
        vmap(returns_list_of_two_tensors)(x)
        vmap(returns_list_of_one_tensor)(x)

    def test_nested_with_same_map_dim(self):
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

    def test_nested_with_diag_embed(self):
        # diag_embed requires special testing because it is registered with conditional functionalization.
        x = torch.randn(3, 3, 5)
        output = vmap(vmap(torch.diag_embed))(x)
        self.assertEqual(output, torch.diag_embed(x))

    def test_nested_with_different_map_dim(self):
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        output = vmap(lambda x: vmap(lambda y: x * y)(y))(x)
        self.assertEqual(output.shape, (2, 5, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        z = torch.randn(7, 3)
        output = vmap(lambda x: vmap(lambda y: vmap(lambda z: x * y * z)(z))(y))(x)
        self.assertEqual(output.shape, (2, 5, 7, 3))
        self.assertEqual(output, x.view(2, 1, 1, 3) * y.view(5, 1, 3) * z)

    def test_noop_in_inner_vmap(self):
        x = torch.randn(3)
        y = torch.randn(5)
        output = vmap(lambda x: vmap(lambda y: x)(y))(x)
        self.assertEqual(output, x.view(3, 1).expand(3, 5))

    def test_checkpoint(self):
        A = torch.randn((3, 8, 8), dtype=torch.float64, requires_grad=True)

        def get_grad(checkpoint):
            A.grad = None

            def get_loss(A):
                ortho_A, _ = torch.func.vmap(torch.linalg.qr)(A)
                return torch.sum(ortho_A)

            if checkpoint:
                loss = torch.utils.checkpoint.checkpoint(
                    get_loss, A, use_reentrant=False
                )
            else:
                loss = get_loss(A)
            loss.backward()
            return A.grad

        expected = get_grad(checkpoint=False)
        result = get_grad(checkpoint=True)
        self.assertEqual(result, expected)

    def test_unsupported_op_err_msg(self):
        # Unsupported view op
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        # TODO: find a view op
        # with self.assertRaisesRegex(RuntimeError, msg):
        #     vmap(torch.ravel)(tensor)

        def out_op(x, y):
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.equal)(tensor, tensor)

    def test_nonzero_out_dims(self):
        # Basic test
        tensor = torch.randn(2, 3)
        result = vmap(lambda x: x, out_dims=1)(tensor)
        self.assertEqual(result, tensor.permute(1, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # Test that the batch dimension gets permuted to dim 2
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 0, 3))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # negative out_dim
        tensor = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x: x, out_dims=-1)(tensor)
        self.assertEqual(result, tensor.permute(1, 2, 3, 0))
        self.assertEqual(result.data_ptr(), tensor.data_ptr())

        # check that out_dims works on ALL outputs
        tensor = torch.randn(2, 3, 5, 7)
        other = torch.randn(2, 3, 5, 7)
        result = vmap(lambda x, y: (x, y), out_dims=2)(tensor, other)
        self.assertEqual(
            result, (tensor.permute(1, 2, 0, 3), other.permute(1, 2, 0, 3))
        )

        # use out_dims with the maximum vmap-able tensor dims (64 dims)
        ndims = 64
        shape = [2] + [1] * (ndims - 1)
        expected_shape = [1, 1, 2] + [1] * (ndims - 3)
        tensor = torch.randn(shape)
        result = vmap(lambda x: x, out_dims=2)(tensor)
        self.assertEqual(result.shape, expected_shape)

        # test something that is not the identity function
        def foo(x, y):
            return x, x * y, x * y * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=1)(x, y)
        self.assertEqual(
            result,
            (
                x.permute(1, 0, 2),
                (x * y).permute(1, 0, 2),
                (x * y * y).permute(1, 0, 2),
            ),
        )

    def test_multiple_out_dims(self):
        def foo(x):
            return x, x

        def bar(x, y):
            return x, x, x, x * y

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        result = vmap(foo, out_dims=(0, 1))(x)
        self.assertEqual(result, (x, x.permute(1, 0, 2)))

        result = vmap(bar, out_dims=(-1, 0, 1, 2))(x, y)
        expected = (
            x.permute(1, 2, 0),
            x,
            x.permute(1, 0, 2),
            (x * y).permute(1, 2, 0),
        )
        self.assertEqual(result, expected)

    def test_nested_out_dims(self):
        y = torch.randn(2, 3, 5, 7)

        # Inner vmap has non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y))(y)
        self.assertEqual(result.shape, (2, 5, 3, 7))
        self.assertEqual(result, y.permute(0, 2, 1, 3))

        # all vmaps have non-zero out_dim
        result = vmap(lambda y: vmap(lambda x: x, out_dims=1)(y), out_dims=1)(y)
        self.assertEqual(result.shape, (5, 2, 3, 7))
        self.assertEqual(result, y.permute(2, 0, 1, 3))

        # throwing in some negative out_dims
        result = vmap(lambda y: vmap(lambda x: x, out_dims=-1)(y), out_dims=-1)(y)
        self.assertEqual(result.shape, (5, 7, 3, 2))
        self.assertEqual(result, y.permute(2, 3, 1, 0))

        # testing fn that isn't the identity
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        result = vmap(lambda y: vmap(lambda x: x * y, out_dims=1)(x), out_dims=-1)(y)
        self.assertEqual(result.shape, (3, 2, 5))
        self.assertEqual(result, (y.view(5, 1, 3) * x).permute(2, 1, 0))

    def test_out_dims_edge_case(self):
        def foo(x):
            return x

        # Test that we accept out_dims=(1,) for a function with one output.
        tensor = torch.randn(2, 3)
        expected = vmap(foo, out_dims=1)(tensor)
        result = vmap(foo, out_dims=(1,))(tensor)
        self.assertEqual(result, expected)

    def test_out_dims_none_tuple(self):
        def foo(x):
            return x, "hello world"

        tensor = torch.randn(2, 3)
        result = vmap(foo, out_dims=(0, None))(tensor)
        self.assertEqual(result[1], "hello world")
        self.assertEqual(result[0], tensor)

        def foo(x):
            x.add_(1)
            return None, "hello world"

        result = vmap(foo, out_dims=(None, None))(tensor)
        self.assertEqual(result, (None, "hello world"))

    def test_out_dims_none(self):
        def foo(x):
            return x

        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(
            ValueError, "can not return a BatchedTensor when out_dim is None"
        ):
            vmap(foo, out_dims=None)(tensor)

        def foo(x):
            x.add_(1)
            return "hello world"

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, "hello world")

    def test_out_dims_normal_tensor(self):
        def foo(x):
            return torch.arange(3)

        tensor = torch.randn(2, 3)
        result = vmap(foo)(tensor)
        self.assertEqual(result.shape, [2, 3])

        result = vmap(foo, out_dims=None)(tensor)
        self.assertEqual(result, torch.arange(3))

    def test_pytree_returns(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y), [y, (y, y)]

        y0, (y1, y2), (y3, (y4, y5)) = vmap(f)(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y0, y1)
        self.assertEqual(y2, y1)
        self.assertEqual(y2, y3)
        self.assertEqual(y4, y3)
        self.assertEqual(y5, y4)

    def test_pytree_odict_returns(self):
        x = torch.randn(2, 3)

        def f(t):
            y = t.sin()
            return OrderedDict([("sin", y), ("cos", t.cos())])

        out = vmap(f)(x)
        assert isinstance(out, OrderedDict)
        expected = f(x)
        self.assertEqual(out["sin"], expected["sin"])
        self.assertEqual(out["cos"], expected["cos"])

    def test_pytree_returns_outdims(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=(0, (0, 1)))(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, x.sin())
        self.assertEqual(y2, x.sin().t())

    def test_pytree_returns_broadcast_simple(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=1)(x)
        self.assertEqual(y0, x.sin().t())
        self.assertEqual(y1, y0)
        self.assertEqual(y2, y0)

    def test_pytree_returns_broadcast_nested(self):
        x = torch.randn(2, 3)

        def f(x):
            y = x.sin()
            return y, (y, y)

        y0, (y1, y2) = vmap(f, out_dims=(0, 1))(x)
        self.assertEqual(y0, x.sin())
        self.assertEqual(y1, y0.t())
        self.assertEqual(y2, y0.t())

    def test_out_dims_must_be_int_or_collection_of_int_err_msg(self):
        msg = "must be an int, None or a python collection of ints"
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        msg = "not compatible"
        x = torch.randn(2, 3, 5)

        # Too many out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(0, 0))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0, 0, 0))(x)

        # Too few out_dims
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x), out_dims=(0,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: (x, x, x), out_dims=(0, 0))(x)

    def test_out_dim_out_of_bounds_err_msg(self):
        # TODO(rzou): This error message isn't that great. It comes straight
        # from maybe_wrap_dim. Consider doing a try-catch-(add some context) to
        # the error message in the future in C++
        msg = "Dimension out of range"
        x = torch.randn(2, 3, 5)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=3)(x)
        with self.assertRaisesRegex(IndexError, msg):
            vmap(lambda x: x, out_dims=-4)(x)

    def test_non_zero_in_dims(self):
        tensor = torch.randn(2, 3, 5)

        # Implicit out_dims = 0; vmap will move the batch dim to the front.
        output = vmap(lambda x: x, (1,))(tensor)
        self.assertEqual(output, tensor.permute(1, 0, 2))
        self.assertEqual(output.data_ptr(), tensor.data_ptr())

        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        output = vmap(torch.mul, (0, 1))(x, y)
        self.assertEqual(output, x * y.t())
        output = vmap(torch.mul, (1, 0))(x, y)
        self.assertEqual(output, x.t() * y)

    def test_none_in_dims(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # None in_dim for a Tensor means we don't map over it
        output = vmap(torch.mul, (0, None))(x, y)
        self.assertEqual(output.shape, (2, 2, 3))
        self.assertEqual(output, x.view(2, 1, 3) * y)

        # None in_dim for non-tensor arguments
        output = vmap(torch.mul, (0, None))(x, 2)
        self.assertEqual(output, x * 2)

    def test_nested_non_default_in_dims(self):
        x = torch.rand(5, 2, 3)
        y = torch.rand(3, 5, 2)
        result = vmap(vmap(vmap(torch.mul), (1, 0)), (1, 2))(x, y)
        self.assertEqual(result, x.permute(1, 2, 0) * y.permute(2, 0, 1))

    def test_nested_negative_in_dims(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        output = vmap(torch.mul, (-1, -1))(x, y)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output, (x * y).permute(1, 0))

    def test_non_default_in_dims_out_dims(self):
        x = torch.randn(2, 3, 5)

        # Same in_dim as out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x)
        self.assertEqual(result.data_ptr(), x.data_ptr())

        # Different in_dim from out_dim, vmap over identity
        result = vmap(lambda x: x, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, x.transpose(1, 2))
        self.assertEqual(result.data_ptr(), x.data_ptr())

        def foo(x):
            return x * 2

        # Same in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=1, out_dims=1)(x)
        self.assertEqual(result, x * 2)

        # Different in_dim as out_dim, vmap over operation
        result = vmap(foo, in_dims=2, out_dims=1)(x)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertEqual(result, (x * 2).transpose(1, 2))

        # Basic nested test.
        result = vmap(vmap(foo, 1, 1), 1, 1)(x)
        self.assertEqual(result, x * 2)

    def test_item_throws(self):
        def f(x):
            return x.item()

        with self.assertRaisesRegex(RuntimeError, r"item\(\) on a Tensor"):
            vmap(f)(torch.randn(3))

    def test_data_dependent_control_flow_throws(self):
        def f(x):
            if x:
                return x
            return 0

        with self.assertRaisesRegex(RuntimeError, r"data-dependent control flow"):
            vmap(f)(torch.randn(3))

    def test_accepts_nested_inputs(self):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        # Single layer of nesting
        out = vmap(lambda z: z[0] + z[1])((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))((x, y))
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))((x, y))
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z[0] + z[1])([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=(0,))([x, y])
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, y])
        self.assertEqual(out, x + y)

        out = vmap(lambda z: z["x"] + z["y"])({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=(0,))({"x": x, "y": y})
        self.assertEqual(out, x + y)
        out = vmap(lambda z: z["x"] + z["y"], in_dims=({"x": 0, "y": 0},))(
            {"x": x, "y": y}
        )
        self.assertEqual(out, x + y)

        # Multiple layers of nesting
        out_fn = vmap(lambda z: z["x"][0] + z["x"][1][0] + z["y"][0] + z["y"][1])
        out = out_fn({"x": [x, (x,)], "y": [y, y]})
        self.assertEqual(out, x + x + y + y)

    def test_in_dims_wrong_type_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"expected `in_dims` to be int or a \(potentially nested\) tuple"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, [0, 0])(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, set({0}))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, "lol")(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=[0, 0])([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_not_enough_in_dims_err_msg(self):
        x = torch.randn(3)
        y = torch.randn(3)
        msg = r"in_dims is not compatible with the structure of `inputs`"

        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0,))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.mul, (0, 0, 0))(x, y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0],))([x, y])
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=((0, 0),))([x, y])
        # The following should not throw
        vmap(torch.mul, (0, 0))(x, y)

    def test_integer_in_dim_but_not_tensor_input_err_msg(self):
        # noqa: F841

        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)

        # the following are errors in jax (and will always be errors)
        msg = "Got in_dim=0 for an input but the input is of type"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum)(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(torch.sum, (0, 0))(x, 0)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([0, 0],))([x, 1])
        # The following should not throw
        vmap(torch.sum, (0, None))(x, 0)

    def test_in_dim_not_in_tensor_err_msg(self):
        def foo(x):
            return x * x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        msg = r"Got in_dim=-?\w for some input, but that input is a Tensor of dimensionality \w"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo)(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(0,))(torch.randn([]))
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(-3,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            torch.vmap(op)(x, y)
            # The single warning here is the "vmap is experimental"
            # warning, not a warning from the vmap fallback path.
            self.assertEqual(len(wa), 1)

    @skipIfTorchDynamo("Flaky test")
    @unittest.expectedFailure
    def test_fallback_warns_when_warnings_are_enabled(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                torch.vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        return
        # with warnings.catch_warnings(record=True) as wa:
        #     with EnableVmapFallbackWarnings():
        #         result = vmap(*vmap_args)(*inputs)
        #     self.assertEqual(len(wa), 2)
        #     self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        op = torch._test_functorch_fallback
        x = torch.randn(11)
        y = torch.randn(11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        B0, B1 = 0, 3
        x = torch.randn(B0, 11)
        y = torch.randn(11)

        msg = "The fallback path does not support vmap over dims of size 0"

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

        x = torch.randn(B0, B1, 11)
        y = torch.randn(B1, 11)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (0, None))(x, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, (None, 0))(y, x)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(x, x)

    def test_fallback_warning(self):
        # We use a dummy function _test_functorch_fallback
        # defined in prim_native_functions.cpp for this
        op = torch._test_functorch_fallback

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # nested vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # big batch size (total 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    # TODO: No clue what is wrong here.
    @unittest.skip
    def test_fallback_masked_fill(self):
        # NB: One day we will implement a batching rule for masked_fill
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        def run_test(batch_size):
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 13)

            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 1, 13))
            self.assertEqual(result, expected)

        run_test(batch_size=5)
        run_test(batch_size=1237)

    def test_fallback_multiple_returns(self):
        # NB: One day we will implement a batching rule for torch.var_mean
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        B0, B1, B2 = 2, 3, 1237
        tensor = torch.randn(B0, 10)

        self._assert_uses_vmap_fallback((torch.var_mean,), (tensor,))

        # fallback correctness on torch.var_mean
        result = vmap(torch.var_mean)(tensor)
        expected = torch.var_mean(tensor, dim=1)
        self.assertEqual(result, expected)

        # nested vmap
        tensor = torch.randn(B0, B1, 10)
        result = vmap(vmap(torch.var_mean))(tensor)
        expected = torch.var_mean(tensor, dim=2)
        self.assertEqual(result, expected)

        # big batch size, nested vmap
        tensor = torch.randn(B0, B1, B2, 10)
        result = vmap(vmap(vmap(torch.var_mean)))(tensor)
        expected = torch.var_mean(tensor, dim=3)
        self.assertEqual(result, expected)

    def test_inplace_fallback_unary(self):
        # Test the in-place fallback on an in-place method that takes no
        # additional Tensor arguments. This is the simplest case of the fallback.
        # NB: One day we will implement a batching rule for acos_.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.acos_
        B0, B1, B2 = 2, 3, 10000

        x = torch.randn(B0, 5)
        self._assert_uses_vmap_fallback((op,), (x,))

        # Single vmap
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op)(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Single vmap + different out_dim produces a view(!)
        x_orig = torch.rand(B0, 5)
        x = x_orig.clone()
        result = vmap(op, out_dims=(1,))(x)
        self.assertTrue(result._base is x)
        self.assertEqual(result, x_orig.t().acos())

        # Nested vmap
        x_orig = torch.randn(B0, B1, 5)
        x = x_orig.clone()
        result = vmap(vmap(op))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

        # Nested vmap, large batch size
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        result = vmap(vmap(vmap(op)))(x)
        self.assertTrue(result is x)
        self.assertEqual(result, x_orig.acos())

    def test_inplace_fallback_nary_same_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)
        self._assert_uses_vmap_fallback((op,), (x, y))

        # Single vmap
        B0 = 5
        x_orig = torch.randn(7, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, 7, 11)
        vmap(op, (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim(0, 2)))

        # Nested vmap
        B0, B1 = 5, 7
        x_orig = torch.randn(B1, 11, B0)
        x = x_orig.clone()
        y = torch.randn(B0, B1, 11)
        vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.movedim([0, 1], [2, 0])))

        # big batch size (total 10000)
        B0, B1, B2 = 100, 10, 10
        x_orig = torch.randn(B0, B1, B2, 5)
        x = x_orig.clone()
        y = torch.randn(B0, B1, B2)
        vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    # ("Fallback isInplaceVmapCompatible check is broken")
    @unittest.expectedFailure
    def test_inplace_fallback_nary_different_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2
        B0, B1 = 2, 3

        x = torch.rand(B0, 7)
        y = torch.rand(7)
        self._assert_uses_vmap_fallback((op, (0, None)), (x, y))

        # op(left, right): All of the levels in right are found in left
        x_orig = torch.rand(B0, 7)
        x = x_orig.clone()
        y = torch.rand(7)
        vmap(op, in_dims=(0, None))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y))

        x_orig = torch.rand(B0, B1, 7)
        x = x_orig.clone()
        y = torch.rand(B0, 7)
        vmap(vmap(op, in_dims=(0, None)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, 1, 7)))

        # op(left, right): Some of the levels in right are not found in left
        msg = r"vmap: aten::atan2_\(self, \*extra_args\) is not possible"
        x = torch.rand(7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(B0, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 0))(x, y)

        x = torch.rand(B1, 7)
        y = torch.rand(7, B0)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(0, None)), in_dims=(None, 1))(x, y)

        x = torch.rand(B0, 7)
        y = torch.rand(B0, B1, 7)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(vmap(op, in_dims=(None, 0)))(x, y)

    def test_backward_unsupported_interaction(self):
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(5)
        grad = torch.randn_like(x)
        err_msg = r"backward\(\) called inside a functorch transform"

        def backward_on_vmapped_tensor(x):
            x.sum().backward()

        # FIXME
        return self.skipTest(
            "error: element 0 of tensors does not require grad and does not have a grad_fn"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            x.sum().backward()
            return y

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

    @unittest.expectedFailure
    def test_grad_unsupported_interaction(self):
        input_tensor = torch.randn(3, requires_grad=True)
        err_msg = "autograd.grad.* called inside torch.vmap"

        captured = torch.randn(3, requires_grad=True)

        def output_to_grad_is_vmapped(input_tensor):
            output = (captured * input_tensor).sum()
            return torch.autograd.grad([output], [captured])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(output_to_grad_is_vmapped)(input_tensor)

        output = (input_tensor**2).sum()

        def input_to_grad_is_vmapped(input_tensor):
            return torch.autograd.grad([output], [input_tensor])[0]

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(input_to_grad_is_vmapped)(input_tensor)

    def test_batched_gradient_basic(self):
        N = 3
        x = torch.randn(N, requires_grad=True)
        y = torch.randn(N)

        def vjp_mul(v):
            return torch.autograd.grad([x * y], [x], grad_outputs=[v])[0]

        batched_v = torch.eye(N)
        jacobian = vmap(vjp_mul)(batched_v)
        self.assertEqual(jacobian, torch.diagflat(y))

    def test_functools_partial(self):
        x = torch.randn(3)
        y = torch.randn(2, 3)
        result = vmap(functools.partial(torch.mul, x))(y)
        self.assertEqual(result, x * y)

    def test_nn_module(self):
        tensor = torch.randn(2, 3)
        model = torch.nn.Linear(3, 3, bias=False)
        result = vmap(model)(tensor)
        self.assertEqual(result, model(tensor))

    def test_fallback_with_undefined_grad(self):
        B0 = 7
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        weight = torch.randn(3, 3, 1, 1)
        v = torch.randn(B0, 2, 3, 4, 5)

        def get_vjp(v):
            result = torch.nn.functional.conv2d(x, weight)
            (grad_x,) = torch.autograd.grad(result, x, v)
            return grad_x

        # Runs vmap(get_vjp)(v), which should not error out.
        # The backward formula for convolution returns an undefined
        # Tensor for grad_bias because the original bias does not exist.
        #
        # In the future we'll probably add a batching rule for convolution
        # backward. When this happens, we should modify this test to use a
        # different op (and/or create and use a dummy operator) to avoid bitrot.
        self._assert_uses_vmap_fallback([get_vjp], [v])

    def test_reshape_dim_into(self):
        x = torch.randn(2, 3, 5, 7)

        y = reshape_dim_into(0, 0, x)
        self.assertEqual(y, x.reshape(6, 5, 7))

        y = reshape_dim_into(0, 1, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        y = reshape_dim_into(0, 2, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        y = reshape_dim_into(1, 2, x)
        self.assertEqual(y, x.movedim(1, 2).reshape(2, 5, 3 * 7))

        y = reshape_dim_into(0, -2, x)
        self.assertEqual(y, x.movedim(0, 1).reshape(3, 2 * 5, 7))

        y = reshape_dim_into(0, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

        y = reshape_dim_into(-4, -1, x)
        self.assertEqual(y, x.movedim(0, 2).reshape(3, 5, 2 * 7))

    def test_reshape_dim_outof(self):
        x = torch.randn(12, 12, 12).permute(2, 1, 0)

        y = reshape_dim_outof(0, 2, x)
        self.assertEqual(y, x.reshape(2, 6, 12, 12))

        y = reshape_dim_outof(1, 4, x)
        self.assertEqual(y, x.reshape(12, 4, 3, 12))

        y = reshape_dim_outof(2, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y, x.reshape(12, 12, 6, 2))

        # Case: `0` sized dim.
        x = torch.randn(12, 12, 0)
        y = reshape_dim_outof(-1, 6, x)
        self.assertEqual(y.shape, torch.Size((12, 12, 6, 0)))

    def test_batch_rule_does_not_need_to_handle_no_batched_input(self):
        def f(x, y):
            res = torch.dot(y, torch.ones(2))
            return x + res

        x = torch.randn(7, 5)
        y = torch.randn(3, 2)
        out = vmap(vmap(f, in_dims=(0, None)), in_dims=(None, 0))(x, y)
        expected = torch.mv(y, torch.ones(2)).view(3, 1, 1) + x
        self.assertEqual(out, expected)

    def test_decomposition_under_python_dispatcher(self):
        # This test will raise an error if the vmap fallback gets invoked.
        # Here we test that decomps registered to FuncTorchBatchedDecomposition
        # are respected by the Python Dispatcher.
        t = torch.ones(3, 3) * 5
        with DisableVmapFallback():
            with torch._dispatch.python.enable_python_dispatcher():
                o = torch.vmap(torch.square)(t)
        self.assertEqual(o, torch.square(t))

    def _test_vmap_autocast(self, device):
        if torch.device(device).type == "cpu":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        a_float32 = torch.rand(4, 2, 3, device=device)
        b_float32 = torch.rand(4, 3, 2, device=device)
        c_float32 = torch.rand(4, 2, 2, device=device)
        d_float32 = torch.rand(4, 3, 2, device=device)

        # Case 1, autocast inside vmapped function
        def func1(x, y, z, w):
            with torch.autocast(dtype=amp_dtype, device_type=device):
                e_float16 = torch.matmul(x, y)
                assert e_float16.dtype == amp_dtype, e_float16.dtype
                f_float16 = torch.matmul(z, e_float16)
                assert f_float16.dtype == amp_dtype, f_float16.dtype
            return torch.matmul(w, f_float16.float())

        expected = func1(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func1)(a_float32, b_float32, c_float32, d_float32)
        assert expected.allclose(out)

        # Case 2, autocast decorator inside vmapped function
        @torch.autocast(dtype=amp_dtype, device_type=device)
        def func2(x, y, z, w):
            e_float16 = torch.matmul(x, y)
            assert e_float16.dtype == amp_dtype, e_float16.dtype
            f_float16 = torch.matmul(z, e_float16)
            assert f_float16.dtype == amp_dtype, f_float16.dtype
            return torch.matmul(w, f_float16)

        expected = func2(a_float32, b_float32, c_float32, d_float32)
        out = vmap(func2)(a_float32, b_float32, c_float32, d_float32)
        assert expected.allclose(out)

        # Case 3, autocast is outside vmapped function
        def func3(x, y, z, w):
            e_float16 = torch.matmul(x, y)
            assert e_float16.dtype == amp_dtype, e_float16.dtype
            f_float16 = torch.matmul(z, e_float16)
            assert f_float16.dtype == amp_dtype, f_float16.dtype
            return torch.matmul(w, f_float16)

        with torch.autocast(dtype=amp_dtype, device_type=device):
            expected = func3(a_float32, b_float32, c_float32, d_float32)
            out = vmap(func3)(a_float32, b_float32, c_float32, d_float32)

        assert expected.allclose(out)

    @unittest.skip("Somehow, vmap and autocast do not work on CPU")
    def test_vmap_autocast_cpu(self):
        self._test_vmap_autocast("cpu")

    @skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_vmap_autocast_cuda(self):
        self._test_vmap_autocast("cuda")

    def test_restore_vmap_pytree_input_output(self):
        def f(x, y):
            output0 = x[0] + x[1]
            output1 = y
            return {"a": output0, "b": output1}

        B = 2
        x0 = torch.randn(B, 3)
        x1 = torch.randn(B)
        y = torch.randn(4, B)

        out, out_dims = restore_vmap(f, ((0, 0), 1), B, "error")((x0, x1), y)
        expected = vmap(f, in_dims=((0, 0), 1), out_dims={"a": 0, "b": 1})((x0, x1), y)
        self.assertEqual(out, expected)
        self.assertEqual(out_dims, {"a": 0, "b": 1})

    def test_restore_vmap_no_vmapped_inputs(self):
        def f(x, y, z):
            return x, y * z, z

        B = 2
        # Mix of tensor and non-tensor inputs
        x = torch.randn(3)
        y = torch.randn(4)
        z = 5
        out, out_dims = restore_vmap(f, (None, None, None), B, "error")(x, y, z)
        self.assertEqual(out, f(x, y, z))
        self.assertEqual(out_dims, (None, None, None))

    def test_restore_vmap_unexpanded_outputs(self):
        def f(x, y):
            # Mix of tensor and non-tensor outputs
            return 3 * y, y.sum(), None

        B = 2
        x = torch.randn(B, 3)
        y = torch.randn(4)
        out, out_dims = restore_vmap(f, (0, None), B, "error")(x, y)
        self.assertEqual(out, f(None, y))
        self.assertEqual(out_dims, (None, None, None))

    def test_data_attribute(self):
        def foo(x):
            y = x.data  # noqa: F841
            return x

        with self.assertRaisesRegex(
            RuntimeError, "accessing `data` under vmap transform"
        ):
            torch.func.vmap(foo)(torch.randn(3, 3))

        def foo(x):
            x.data = torch.ones(3, 3)
            return x

        with self.assertRaisesRegex(
            RuntimeError, "mutating directly with `.data` under vmap"
        ):
            torch.func.vmap(foo)(torch.randn(3, 3))


def slice_inputs(inputs, bdims, i):
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def reference_vmap(op, inputs, in_dims=0, out_dims=0, return_nt=False):
    if isinstance(in_dims, int):
        in_dims = (in_dims,) * len(inputs)
    bdim_sizes = [inp.size(dim) for inp, dim in zip(inputs, in_dims) if dim is not None]
    assert all(bdim_size == bdim_sizes[0] for bdim_size in bdim_sizes)
    bdim_size = bdim_sizes[0]
    results = tuple(op(*slice_inputs(inputs, in_dims, i)) for i in range(bdim_size))

    assert len(results) > 0
    op_has_single_return = not isinstance(results[0], tuple)
    if op_has_single_return:
        assert all(isinstance(result, torch.Tensor) for result in results)
        if isinstance(out_dims, int):
            out_dims = (out_dims,) * 1
        if return_nt:
            return torch.nested.nested_tensor(list(results))
        else:
            return torch.stack(results, dim=out_dims[0])

    assert all(isinstance(result, tuple) for result in results)
    num_returns = len(results[0])
    assert all(len(result) == num_returns for result in results)
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
    if return_nt:
        return tuple(
            torch.nested.nested_tensor(list(result_shards))
            for result_shards in zip(*results)
        )
    else:
        return tuple(
            torch.stack(result_shards, out_dim)
            for result_shards, out_dim in zip(zip(*results), out_dims)
        )


class TensorFactory:
    @staticmethod
    def rand(size, device="cpu", dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype)

    @staticmethod
    def randn(size, device="cpu", dtype=torch.float):
        return torch.randn(size, device=device, dtype=dtype)

    @staticmethod
    def randp1(size, device="cpu", dtype=torch.float):
        return torch.rand(size, device=device, dtype=dtype) + 1


# Tests vmap(op, in_dims, out_dims)(*inputs) by comparing the output to a
# (slow) sequential map+stack fallback.
#
# check_view: Test if the first returned output is a view of the first input
# check_propagates_grad: Test if the operation propagates gradients.


def _vmap_test(
    self,
    op,
    inputs,
    in_dims=0,
    out_dims=0,
    check_view=False,
    check_propagates_grad=True,
):
    result = vmap(op, in_dims, out_dims)(*inputs)
    are_nested = [t.is_nested for t in pytree.tree_leaves(result)]
    reference_result = reference_vmap(
        op, inputs, in_dims, out_dims, return_nt=any(are_nested)
    )
    self.assertEqual(result, reference_result)
    op_has_single_return = not isinstance(result, tuple)

    if check_view:
        result_as_tuple = (result,) if op_has_single_return else result
        for output in result_as_tuple:
            input0_base = inputs[0] if inputs[0]._base is None else inputs[0]._base
            self.assertTrue(
                output._base is input0_base,
                msg="result was not a view of the first input!",
            )

    if not check_propagates_grad:
        return
    # Assuming input[0] is a floating-point tensor. Check if the vmap
    # operation propagates the requires_grad flag to the zeroth output.
    # Some vmap operators are implemented in a way that assumes that
    # they are composite with respect to autograd. If the operator ever is
    # changed to not be composite with respect to autograd, then the
    # following check should fail.
    inputs_clone = list(inputs)
    inputs_clone[0] = inputs[0].clone().requires_grad_()
    result = vmap(op, in_dims, out_dims)(*inputs_clone)
    result_as_tuple = (result,) if op_has_single_return else result
    self.assertTrue(result[0].requires_grad)


def should_allow_vmap_fallback_usage(fn):
    return getattr(fn, "_allow_vmap_fallback_usage", False)


def allowVmapFallbackUsage(fn):
    fn._allow_vmap_fallback_usage = True
    return fn


# All tests of TestVmapBase check that the slow vmap fallback is never invoked.
# This is so that we can incrementally add batching rules for operators to
# replace the slow vmap fallback path for said operators. To skip this check,
# please use the allowVmapFallbackUsage decorator.
#
# NB: Don't add tests to TestVmapBase directly, unless you want them to run
# on every subclass of TestVmapBase. Add them to e.g. TestVmapOperators.
#
# NB: TestVmapBase is a nested class. This prevents test runners from picking
# it up and running it.


class Namespace:
    class TestVmapBase(TestCase):
        def __init__(self, method_name="runTest"):
            super().__init__(method_name)

            test_method = getattr(self, method_name, None)
            if test_method is None:
                return

            if not should_allow_vmap_fallback_usage(test_method):
                setattr(
                    self,
                    method_name,
                    self._wrap_method_with_vmap_fallback_check(test_method),
                )

        def _wrap_method_with_vmap_fallback_check(self, method):
            # msg = (
            #     'Expected the test to not invoke the vmap fallback path, i.e., '
            #     'all of the operators being tested in this test should have batching '
            #     'rules implemented. If you are intentionally testing something to '
            #     'do with the fallback path, use allowVmapFallbackUsage. Otherwise, '
            #     'please make sure that batching rules are implemented for the '
            #     'operator(s) being tested.'
            # )

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    # for captured_warning in wa:
                    #     self.assertNotRegex(str(captured_warning.message), FALLBACK_REGEX, msg)

            return types.MethodType(wrapper, self)

        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))

        @unittest.expectedFailure
        def test_vmap_fallback_check(self):
            @self._wrap_method_with_vmap_fallback_check
            def no_fallback(self):
                pass

            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean

            @self._wrap_method_with_vmap_fallback_check
            def uses_fallback(self):
                vmap(op_using_fallback)(torch.rand(3))

            no_fallback(self)

            with self.assertRaises(AssertionError):
                uses_fallback(self)


def _make_case(op, input_getter=TensorFactory.randn):
    return (op, input_getter)


@markDynamoStrictTest
class TestVmapOperators(Namespace.TestVmapBase):
    def _vmap_test(self, *args, **kwargs):
        return _vmap_test(self, *args, **kwargs)

    def _vmap_view_test(self, *args, **kwargs):
        self._vmap_test(*args, **kwargs, check_view=True)

    def _test_unary(self, op, getter, device, *args, **kwargs):
        test = functools.partial(self._vmap_test, *args, **kwargs)
        B0, B1 = 7, 11

        # Single vmap, various in_dims / out_dims
        test(op, [getter([B0, 3], device)])
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2)
        test(op, [getter([2, 5, B0, 3], device)], in_dims=2, out_dims=2)

        # Doubly nested vmap
        test(vmap(op), [getter([B0, B1], device)])
        test(vmap(op), [getter([B1, 2, 5, B0, 3], device)], in_dims=2)
        test(
            vmap(op, in_dims=2),
            [getter([2, 5, B0, B1, 3], device)],
            in_dims=2,
            out_dims=2,
        )

    @parametrize(
        "case",
        [
            (torch.abs, TensorFactory.randn),
            (torch.acos, TensorFactory.rand),
            (torch.asin, TensorFactory.rand),
            (torch.atan, TensorFactory.rand),
            (torch.ceil, TensorFactory.randn),
            (torch.cos, TensorFactory.rand),
            (torch.cosh, TensorFactory.rand),
            (torch.digamma, TensorFactory.rand),
            (torch.exp, TensorFactory.randn),
            (torch.expm1, TensorFactory.randn),
            (torch.floor, TensorFactory.randn),
            (torch.frac, TensorFactory.randn),
            (torch.lgamma, TensorFactory.rand),
            (torch.log, TensorFactory.randp1),
            (torch.log10, TensorFactory.randp1),
            (torch.log1p, TensorFactory.randp1),
            (torch.log2, TensorFactory.randp1),
            (torch.neg, TensorFactory.randn),
            (torch.reciprocal, TensorFactory.randp1),
            (torch.relu, TensorFactory.randn),
            (torch.round, TensorFactory.randn),
            (torch.rsqrt, TensorFactory.randp1),
            (torch.sigmoid, TensorFactory.randn),
            (torch.sign, TensorFactory.randn),
            (torch.sin, TensorFactory.rand),
            (torch.sinh, TensorFactory.rand),
            (torch.sqrt, TensorFactory.rand),
            (torch.tan, TensorFactory.rand),
            (torch.tanh, TensorFactory.rand),
            (torch.trunc, TensorFactory.randn),
        ],
        name_fn=lambda x: x[0].__name__,
    )
    def test_unary_pointwise(self, case):
        op, getter = case
        self._test_unary(op, getter, "cpu")

        # test in-place
        method = getattr(Tensor, f"{op.__name__ + '_'}")
        self._test_unary(method, getter, "cpu", check_propagates_grad=False)

    def test_clone(self):
        # Some basic tests
        self._test_unary(lambda x: x.clone(), TensorFactory.randn, "cpu")
        self._test_unary(
            lambda x: x.clone(memory_format=torch.preserve_format),
            TensorFactory.randn,
            "cpu",
        )
        self._test_unary(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            TensorFactory.randn,
            "cpu",
        )

        # Test that the per-examples are contiguous when using torch.contiguous_format
        def clone_contiguous(x):
            return x.clone(memory_format=torch.contiguous_format)

        B0, B1 = 3, 5
        x = torch.randn(2, B0, 7)
        y = vmap(clone_contiguous, in_dims=1, out_dims=1)(x)
        self.assertTrue(y.movedim(1, 0).is_contiguous())
        self.assertTrue(y[:, 0, :].is_contiguous())

        x = torch.randn(2, B0, 7, B1)
        y = vmap(vmap(clone_contiguous, in_dims=2), in_dims=1)(x)
        self.assertTrue(y.is_contiguous())
        self.assertTrue(y[0][0].is_contiguous())

        msg = r"only supported with memory_format torch.preserve_format or torch.contiguous_format"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last))(torch.randn(B0))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(lambda x: x.clone(memory_format=torch.channels_last_3d))(
                torch.randn(B0)
            )

    def test_weird_matmul_case(self):
        # Check that this doesn't crash.
        # https://github.com/pytorch/functorch/issues/417
        x = torch.randn(5, 2, 2, 2)
        y = torch.randn(5, 7, 2)

        vmap(vmap(torch.matmul, in_dims=(None, 0)))(x, y)

    @parametrize(
        "case",
        (
            (torch.clamp_min_, TensorFactory.randn),
            (torch.clamp_max_, TensorFactory.randn),
        ),
        name_fn=lambda x: x[0].__name__,
    )
    def test_clamp_inplace_variant(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0], device), getter([B0], device)),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([2, B0, 3], device), getter([2, B0, 3], device)),
            in_dims=(1, 1),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 2, 3], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 2, 3], device), getter([1, 1], device)),
            in_dims=(0, None),
            check_propagates_grad=False,
        )
        test(
            op,
            (getter([B0, 3], device), getter([B0, 3], device)),
            in_dims=(0, 0),
            check_propagates_grad=False,
        )

        # Nested vmap: op(Tensor, Tensor)
        test(
            vmap(op),
            (getter([B0, B1, 2, 3], device), getter([B0, B1, 1, 3], device)),
            check_propagates_grad=False,
        )

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(
            lambda t: op(t, number), getter, device, check_propagates_grad=False
        )

    @parametrize(
        "case",
        [
            subtest(_make_case(torch.clamp_min), name="clamp_min"),
            subtest(_make_case(torch.clamp_max), name="clamp_max"),
        ],
    )
    def test_clamp_variant(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(None, 0))

        # Nested vmap: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python number overload: op(Tensor, Number)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)

    def test_copy_(self):
        x = torch.randn(3)
        y = torch.randn(3)
        vmap(Tensor.copy_)(x, y)
        self.assertEqual(x, y)

        x = torch.randn(3)
        y = torch.randn(3, 2)
        vmap(Tensor.copy_, in_dims=(1, None))(y, x)
        self.assertEqual(y, x.expand(2, 3).t())

        x = torch.randn(3)
        y = torch.randn(2, 3)
        with self.assertRaisesRegex(RuntimeError, "inplace"):
            vmap(Tensor.copy_, in_dims=(None, 0))(x, y)

    def test_silu_backward(self):
        test = self._vmap_test
        device = "cpu"
        getter = TensorFactory.randp1
        B0 = 7
        op = torch.ops.aten.silu_backward

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([], device), getter([B0], device)), in_dims=(None, 0))
        test(op, (getter([2, B0], device), getter([2], device)), in_dims=(1, None))

    @parametrize(
        "case",
        [
            subtest(_make_case(torch.add), name="add"),
            subtest(_make_case(lambda x, y: x + y), name="add_dunder"),
            subtest(_make_case(torch.sub), name="sub"),
            subtest(_make_case(lambda x, y: x - y), name="sub_dunder"),
            subtest(_make_case(torch.mul), name="mul"),
            subtest(_make_case(lambda x, y: x * y), name="mul_dunder"),
            subtest(
                _make_case(torch.div, input_getter=TensorFactory.randp1), name="div"
            ),
            subtest(
                _make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
                name="div_dunder",
            ),
            subtest(
                _make_case(torch.pow, input_getter=TensorFactory.randp1), name="pow"
            ),
            subtest(
                _make_case(lambda x, y: x**y, input_getter=TensorFactory.randp1),
                name="pow_dunder",
            ),
        ],
    )
    def test_arithmetic(self, case):
        test = self._vmap_test

        def get_number(getter):
            return getter([]).item()

        op, getter = case
        device = "cpu"
        B0, B1 = 7, 11

        # Single vmap: op(Tensor, Tensor)
        test(op, (getter([B0, 3], device), getter([B0, 3], device)))
        test(op, (getter([B0], device), getter([B0, 2, 3], device)))
        test(op, (getter([B0], device), getter([2, B0, 3], device)), in_dims=(0, 1))
        test(
            op,
            (getter([B0], device), getter([2, B0, 3], device)),
            in_dims=(0, 1),
            out_dims=1,
        )
        test(op, (getter([B0], device), getter([2, 3], device)), in_dims=(0, None))
        test(op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None))

        # Nested vmap: op(Tensor, Tensor)
        test(vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device)))
        test(
            vmap(op, in_dims=(None, 0)),
            (getter([B0, 2, 3], device), getter([B1, 3], device)),
            in_dims=(0, None),
        )

        # Python number overload: op(Tensor, Number) (and vice-versa)
        number = get_number(getter)
        self._test_unary(lambda t: op(t, number), getter, device)
        number = get_number(getter)
        self._test_unary(lambda t: op(number, t), getter, device)

        # Type promotion: op(Logical Scalar Tensor, Logical Scalar Tensor)
        test(op, (getter([B0], device), getter([B0], device, dtype=torch.double)))
        test(op, (getter([B0], device, dtype=torch.double), getter([B0], device)))
        test(op, (getter([B0], device), getter([B0], device)))

        # Type promotion: op(Tensor, Logical Scalar Tensor) (and vice-versa)
        test(op, (getter([B0, 2], device), getter([B0], device, torch.double)))
        test(op, (getter([B0], device, torch.double), getter([B0, 2], device)))

        if not torch.cuda.is_available():
            return

        # TODO(rzou): fix the following
        # # Test cross-device scalars
        # number = get_number(getter)
        # self._test_unary(lambda t: op(t, number), getter, device='cuda')
        # self._test_unary(lambda t: op(number, t), getter, device='cuda')
        # self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    def test_as_strided(self):
        def _test(sizes, strides, offset, tensor, lambd):
            # bdim at dim 0 test
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

            # bdim at dim -1 test
            tensor = tensor.movedim(0, -1)
            result = vmap(lambda t: t.as_strided(sizes, strides, offset), -1)(tensor)
            expected = vmap(lambd, -1)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

        # single vmap test
        B0 = 5
        # Each Tensor has shape [B0, 2, 3]; the expressions below
        # are just to get tensors of different strides that have shape [B0, 2, 3]
        tensors = [
            # contiguous
            torch.randn(B0, 2, 3),
            # non-contiguous
            torch.randn(B0, 3, 2).transpose(1, 2),
            torch.randn(3, 2, B0).movedim(-1, 0).transpose(1, 2),
            # non-zero storage offset
            torch.randn(2, B0, 2, 3)[1],
            torch.randn(2, 2, B0, 3)[1].movedim(1, 0),
            # non-contiguous strides, zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0],
            torch.randn(2, 4, B0, 3, 7).movedim(2, 0)[:, :, 0, :, 0],
            # non-contiguous strides, non-zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1],
            torch.randn(2, 4, 3, 7, B0).movedim(-1, 0)[:, :, 2, :, 1],
        ]

        for x in tensors:
            S0, S1 = x.stride()[1:]
            offset = x.storage_offset()

            # Broadcast
            _test(
                [5, 5, 2, 3], [0, 0, S0, S1], offset, x, lambda x: x.expand(5, 5, 2, 3)
            )
            # transpose
            _test([3, 2], [S1, S0], offset, x, lambda x: x.transpose(0, 1))
            # select
            _test([2], [S0], offset + S1, x, lambda x: x[:, 1])
            # diagonal
            _test([2], [S0 + S1], offset, x, lambda x: x.diagonal())
            # strided slice
            _test([2], [S1 * 2], offset, x, lambda x: x[0, ::2])

        # Nested vmap test
        B1 = 7
        x = torch.randn(B1, B0, 2, 3)
        S0, S1 = x.stride()[2:]
        result = vmap(
            vmap(lambda t: t.as_strided([5, 5, 2, 3], [0, 0, S0, S1])), in_dims=1
        )(x)
        expected = vmap(vmap(lambda t: t.expand(5, 5, 2, 3)), in_dims=1)(x)
        self.assertTrue(result._base is expected._base)
        self.assertEqual(result, expected)

        # Check that mal-formatted size/strides doesn't crash
        with self.assertRaisesRegex(
            RuntimeError, "size and stride must have the same length"
        ):
            x = torch.randn(B0, 2, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([1, 1, 1], [1, 1]))(x)

        # All the Sanity check #1{a,b,c} cases check that
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # doesn't index memory that is out of bounds of xs[i]. This condition
        # is important to the correctness of the as_strided batching rule
        # (see NOTE: [When will the as_strided_batching_rule fail?])

        # Sanity check #1a: The maximum indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is less than or equal to the maximum indexable location of xs[i].
        msg = "This is not supported inside of vmap"
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3)
            vmap(lambda x: x.as_strided([3], [1], 1))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 3, 5)
            vmap(lambda x: x.as_strided([4, 4], [4, 1], 0))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, B1, 3, 5)
            vmap(vmap(lambda x: x.as_strided([4, 4], [4, 1], 0)))(x)

        # Sanity check #1b: The min indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is greater than or equal to the min indexable location of xs[i].
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda x: x.as_strided([3], [1], B0 * 3 - 1))(x)

        # Sanity check #1c:
        # xs[i] is a zero-dim tensor, but
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is not
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 0, 3)
            vmap(lambda x: x.as_strided([3], [1]))(x)

    def test_nll_loss(self):
        test = self._vmap_test
        op = F.nll_loss
        B = 3

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (B, 2))
        test(op, (y, t))
        test(functools.partial(op, reduction="sum"), (y, t))
        test(functools.partial(op, reduction="none"), (y, t))

        y = torch.randn(B, 2, 5)
        t = torch.randint(0, 5, (2,))
        test(op, (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="sum"), (y, t), in_dims=(0, None))
        test(functools.partial(op, reduction="none"), (y, t), in_dims=(0, None))

    def test_adaptive_avg_pool2d(self):
        test = self._vmap_test
        op = functools.partial(F.adaptive_avg_pool2d, output_size=(3, 3))

        x = torch.randn(3, 5, 7, 9, 11)
        test(op, (x,))
        test(op, (x,), in_dims=(1,))
        test(op, (x,), in_dims=(4,))

    def test_bmm(self):
        op = torch.bmm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 3, 3, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(2, 5, 3)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(2, 5, 3)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5, 3), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5, 3), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 3, 5), torch.rand(B0, 2, 5, 3)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 3, 5), torch.rand(B0, B1, 2, 5, 3)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 3, 5), torch.rand(B0, 2, 5, 3)),
            in_dims=(None, 0),
        )

    def test_cat(self):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            def op(*tensors):
                return torch.cat(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 2), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(B0, 0), torch.rand(B0, 0)))
        test(get_op(0), (torch.rand(2), torch.rand(B0, 0)), in_dims=(None, 0))
        test(
            get_op(1),
            (torch.rand(2, 5), torch.rand(B0, 0), torch.rand(2, 3)),
            in_dims=(None, 0, None),
        )
        test(get_op(1), (torch.rand(B0, 2, 3), torch.rand(B0, 0)))
        test(get_op(1), (torch.rand(B0, 2, 3, 4), torch.rand(0)), in_dims=(0, None))
        test(
            get_op(0),
            (torch.rand(0), torch.rand(B0, 2), torch.rand(B0, 0)),
            in_dims=(None, 0, 0),
        )
        test(get_op(0), (torch.rand(2), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17), torch.rand(3, 17, B0)), in_dims=(None, 2))
        test(get_op(-1), (torch.rand(17, 2), torch.rand(17, 3, B0)), in_dims=(None, 2))
        test(
            vmap(get_op(0), in_dims=(0, None)),
            (torch.rand(B1, 2), torch.rand(B0, 3)),
            in_dims=(None, 0),
        )
        test(
            vmap(get_op(0), in_dims=(0, 0)),
            (torch.rand(B1, 2), torch.rand(B0, B1, 3)),
            in_dims=(None, 0),
        )

    def test_unsafe_view(self):
        # Unsafe view isn't exposed, so we get at it via
        # vmap(grad(matmul))
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B = 2
        x = torch.randn(B, 2, 3, 3)
        y = torch.randn(B, 3, 3)

        def baz(x, y):
            return (x @ y).sum()

        test(functorch.grad(baz), (x, y))

    def test_conj(self):
        op = torch.conj

        def run_test(dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11
            test = self._vmap_test

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, B0, 3])], in_dims=2)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        # correctness tests
        run_test(torch.float)
        run_test(torch.cfloat)

        # check that torch.conj on a non-complex tensor returns the same tensor
        real_tensor = torch.randn(3)
        result = vmap(op)(real_tensor)
        self.assertEqual(result.data_ptr(), real_tensor.data_ptr())

    def test_contiguous(self):
        op = Tensor.contiguous

        self._test_unary(op, TensorFactory.randn, "cpu")

        # check that contiguous returns the original tensor if the per-examples
        # are already contiguous
        B0 = 3
        x = torch.randn(B0, 2, 5, 7)
        x = x.movedim(0, 2)
        result = vmap(Tensor.contiguous, in_dims=2, out_dims=2)(x)
        self.assertTrue(result is x)

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(op, memory_format=torch.channels_last_3d))(tensor)

    def test_stride(self):
        B0 = 3

        x = torch.randn(B0, 2, 5, 7)

        def foo(x):
            assert x.stride() == (7 * 5, 7, 1)
            return x

        vmap(foo)(x)

        x = torch.randn(2, B0, 5, 7).movedim(1, 0)

        def bar(x):
            assert x.stride() == (7 * 5 * B0, 7, 1)
            return x

        vmap(bar)(x)

    def test_chunk(self):
        test = self._vmap_view_test
        op = torch.chunk
        B0, B1, B2 = 7, 11, 13

        # tests for torch.split(self, split_size: int, dim)
        test(op, (torch.rand(B0, 2, 1024), 15, -1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 1024), 9, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 1023, B0, 5), 4, 0),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(lambda t: op(t, 4, 1), in_dims=2)),
            (torch.rand(B1, 2, B0, 64, B2),),
            in_dims=2,
        )

    def test_clamp(self):
        clamp_cases = (
            (lambda t: t.clamp(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp(max=0.5), TensorFactory.randn),
            (lambda t: t.clamp(min=-0.5, max=0.5), TensorFactory.randn),
            (lambda t: t.clamp_min(min=-0.5), TensorFactory.randn),
            (lambda t: t.clamp_max(max=0.5), TensorFactory.randn),
        )
        for op, getter in clamp_cases:
            self._test_unary(op, getter, "cpu")

    def test_comparison_ops(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)

        getter = TensorFactory.randn
        B0, B1 = 7, 11

        ops = (
            torch.eq,
            lambda x, y: x == y,
            torch.gt,
            lambda x, y: x > y,
            torch.ge,
            lambda x, y: x >= y,
            torch.le,
            lambda x, y: x <= y,
            torch.lt,
            lambda x, y: x < y,
            torch.ne,
            lambda x, y: x != y,
        )

        for op in ops:
            # Single vmap: op(Tensor, Tensor)
            test(op, (getter([B0, 3]), getter([B0, 3])))
            test(op, (getter([B0]), getter([B0, 2, 3])))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1))
            test(op, (getter([B0]), getter([2, B0, 3])), in_dims=(0, 1), out_dims=1)
            test(op, (getter([B0]), getter([2, 3])), in_dims=(0, None))
            test(op, (getter([2, 3]), getter([B0, 3])), in_dims=(0, None))

            # Nested vmap: op(Tensor, Tensor)
            test(vmap(op), (getter([B0, B1, 2, 3]), getter([B0, B1, 3])))
            test(
                vmap(op, in_dims=(None, 0)),
                (getter([B0, 2, 3]), getter([B1, 3])),
                in_dims=(0, None),
            )

            # test number as inputs
            number = getter([]).item()
            self._test_unary(
                lambda t: op(t, number), getter, "cpu", check_propagates_grad=False
            )

    def test_cross_batch_size_three(self):
        # Let's test corner case when batch_size is 3 and cross' dim argument is not specified
        # According to the cross API, dim will be assigned to the first dim with value 3
        # In this test we ensure that found dim is not batch dim.
        op = torch.cross
        test = self._vmap_test
        B0 = B1 = 3
        test(op, (torch.rand(B0, 2, 3), torch.rand(B0, 2, 3)))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B0, B1, 2, 3), torch.rand(B0, B1, 2, 3)),
            in_dims=(None, 1),
        )

    def test_diagonal(self):
        tensor = torch.randn(3, 5, 7, 11, 13)
        test = self._vmap_view_test
        op = torch.diagonal
        test(op, (tensor, 1, 0, 1), in_dims=(0, None, None, None))
        test(op, (tensor, 0, 2, -1), in_dims=(0, None, None, None))
        test(op, (tensor, 2, 1, 2), in_dims=(1, None, None, None))
        test(op, (tensor, 0, -2, -1), in_dims=(1, None, None, None), out_dims=1)
        test(vmap(lambda t: op(t, 0, 0, -1)), (tensor,), in_dims=1, out_dims=1)
        test(
            vmap(vmap(lambda t: op(t, 0, 0, 1), in_dims=1), in_dims=3),
            (tensor,),
            in_dims=1,
            out_dims=1,
        )

    def test_dot(self):
        op = torch.dot
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 5), torch.rand(B0, 5)))
        test(vmap(op), (torch.rand(B1, B0, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_expand_as(self):
        op = torch.Tensor.expand_as
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 1, 5), torch.rand(B0, 2, 3, 5)))
        test(op, (torch.rand(B0, 1, 5), torch.rand(2, 3, 5)), in_dims=(0, None))
        test(op, (torch.rand(1, 5), torch.rand(B0, 2, 3, 5)), in_dims=(None, 0))
        test(vmap(op), (torch.rand(B0, B1, 1, 5), torch.rand(B0, B1, 2, 3, 5)))
        test(
            vmap(op),
            (torch.rand(B0, B1, 1, 5), torch.rand(B1, B0, 2, 3, 5)),
            in_dims=(0, 1),
        )
        test(vmap(op), (torch.rand(B0, B1), torch.rand(B1, 2, 3, 5)), in_dims=(0, None))
        test(vmap(vmap(op)), (torch.rand(B0, B1, B2), torch.rand(B0, B1, B2, 2, 3, 5)))

    def test_fill_and_zero_inplace(self):
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11
        ops = (
            lambda t: t.fill_(0.1),
            lambda t: t.fill_(torch.tensor(0.2)),
            lambda t: t.zero_(),
        )

        for op in ops:
            # Single vmap, various in_dims / out_dims
            test(op, [TensorFactory.randn([B0, 3])])
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2)
            test(op, [TensorFactory.randn([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [TensorFactory.randn([B0, B1])])
            test(vmap(op), [TensorFactory.randn([B1, 2, 5, B0, 3])], in_dims=2)
            test(
                vmap(op, in_dims=2),
                [TensorFactory.randn([2, 5, B0, B1, 3])],
                in_dims=2,
                out_dims=2,
            )

        # test when value is a batched tensor for fill_ operator
        B0, B1 = 3, 5
        test(Tensor.fill_, [TensorFactory.randn([B0, B1]), TensorFactory.randn(B0)])

        with self.assertRaisesRegex(RuntimeError, ""):
            # Runtime Error is thrown when the tensor being written to isn't being vmapped over
            vmap(Tensor.fill_, (None, 0))(
                TensorFactory.randn([B0, B1]), TensorFactory.randn([B0])
            )

    def _test_complex_views(self, op, dtypes):
        test = self._vmap_view_test

        def run_test(op, dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3])])
            test(op, [get([3, B0])], in_dims=1)
            test(op, [get([2, 5, B0, 3])], in_dims=2)
            test(op, [get([2, 5, B0, 3])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1])])
            test(vmap(op), [get([B1, 2, 5, 3, B0])], in_dims=4)
            test(vmap(op, in_dims=2), [get([2, 5, B0, B1, 3])], in_dims=2, out_dims=2)

        for dtype in dtypes:
            run_test(op, dtype)

    def test_real(self):
        self._test_complex_views(torch.real, dtypes=[torch.cfloat, torch.cdouble])

    def test_imag(self):
        self._test_complex_views(torch.imag, dtypes=[torch.cfloat, torch.cdouble])

    def test_view_as_real(self):
        self._test_complex_views(
            torch.view_as_real, dtypes=[torch.cfloat, torch.cdouble]
        )

    def test_view_as_complex(self):
        def run_test(dtype):
            def get(shape):
                return torch.randn(shape, dtype=dtype)

            op = torch.view_as_complex
            test = self._vmap_view_test
            B0, B1 = 7, 11

            # Single vmap, various in_dims / out_dims
            test(op, [get([B0, 3, 2])])
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2)
            test(op, [get([2, 5, B0, 3, 2])], in_dims=2, out_dims=2)

            # Doubly nested vmap
            test(vmap(op), [get([B0, B1, 2])])
            test(vmap(op), [get([B1, 2, 5, B0, 3, 2])], in_dims=2)
            test(
                vmap(op, in_dims=2), [get([2, 5, B0, B1, 3, 2])], in_dims=2, out_dims=2
            )

            # Interesting case #1: Batch dim directly before dim of size 2
            test(op, [get([3, B0, 2])], in_dims=1)
            test(vmap(op, in_dims=1), [get([3, B1, B0, 2])], in_dims=2)

            # Interesting case #2: Batch dim at end of tensor, success cases
            # view_as_complex requires that the dim with size 2 have stride 1
            # in order for the view to function property
            test(op, [get([B0, 2]).transpose(0, 1)], in_dims=1)
            test(vmap(op, in_dims=1), [get([B0, B1, 2]).movedim(1, 2)])
            test(vmap(op, in_dims=2), [get([B0, 3, B1, 2]).movedim(2, 3)])

            # Interesting case #3: Batch dim at end of tensor, failure cases
            msg = "Tensor must have a last dimension with stride 1"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([2, B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op, in_dims=1), in_dims=1)(get([2, B0, B1]))

            # Invalid input: no dimension of size 2
            msg = "Input tensor must have one or more dimensions"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op)(get([B0]))
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(vmap(op))(get([B0, B1]))

            # Invalid input: Batch dim has size 2, but the logical last dim does
            # not have size 2
            msg = "Tensor must have a last dimension of size 2"
            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(op, in_dims=1)(get([3, 2]))

        for dtype in [torch.float, torch.double]:
            run_test(dtype)

    def test_is_complex(self):
        ctensor = torch.randn(3, dtype=torch.cfloat)
        tensor = torch.randn(3)

        def foo(x):
            if x.is_complex():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(ctensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(tensor), torch.tensor([0, 0, 0]))

    def test_is_floating_point(self):
        float_tensor = torch.tensor([1.0, 2.0, 3.0])
        long_tensor = torch.tensor([1, 2, 3])

        def foo(x):
            if x.is_floating_point():
                return torch.tensor(1)
            else:
                return torch.tensor(0)

        self.assertEqual(vmap(foo)(float_tensor), torch.tensor([1, 1, 1]))
        self.assertEqual(vmap(foo)(long_tensor), torch.tensor([0, 0, 0]))

    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    def test_is_contiguous(self):
        def foo(x):
            if x.is_contiguous():
                return torch.tensor(1.0)
            else:
                return torch.tensor(0.0)

        B0, B1 = 3, 5

        # Single batch dim
        contig = torch.randn(B0, 2, 7)
        self.assertEqual(vmap(foo)(contig), torch.ones(B0))

        noncontig = torch.randn(2, B0, 7)
        self.assertEqual(vmap(foo, in_dims=1)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, B0, 7).movedim(1, 0)
        self.assertEqual(vmap(foo)(noncontig), torch.zeros(B0))

        noncontig = torch.randn(2, 7, B0)
        self.assertEqual(vmap(foo, in_dims=2)(noncontig), torch.zeros(B0))

        # Multiple batch dims
        contig = torch.randn(B0, B1, 3)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3)
        self.assertEqual(vmap(vmap(foo), in_dims=1)(contig), torch.ones(B0, B1))

        contig = torch.randn(B1, B0, 3).movedim(0, 1)
        self.assertEqual(vmap(vmap(foo))(contig), torch.ones(B0, B1))

        noncontig = torch.randn(B0, 3, B1)
        self.assertEqual(vmap(vmap(foo, in_dims=1))(noncontig), torch.zeros(B0, B1))

        # is_contiguous on empty tensor is True
        def bar(x):
            assert x.is_contiguous()
            return x

        vmap(bar)(torch.randn(B0, 0, 3))
        vmap(bar, in_dims=1)(torch.randn(0, B0, 3))
        vmap(bar)(torch.randn(B0, 0, 3).transpose(-1, -2))

        # is_contiguous with other memory formats
        def baz(x, memory_format):
            x.is_contiguous(memory_format=memory_format)
            return x

        msg = "NYI: querying is_contiguous inside of vmap for memory_format"
        tensor = torch.randn(B0, 2, 7, 3)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last))(tensor)
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(functools.partial(baz, memory_format=torch.channels_last_3d))(tensor)

        for mf in (torch.channels_last, torch.channels_last_3d):

            @torch.compile(backend="eager", fullgraph=True)
            def f(x):
                if x.is_contiguous(memory_format=mf):
                    return x.sin()
                return x.cos()

            with self.assertRaisesRegex(RuntimeError, msg):
                vmap(f)(torch.randn(3, 3))

    def test_unsqueeze(self):
        op = torch.unsqueeze
        test = self._vmap_view_test
        B0, B1 = 7, 11

        # unsqueeze dim 0
        test(op, (torch.rand(B0, 2, 5), 0), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 0), in_dims=(1, None))

        # unsqueeze last dim (positive)
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 2), in_dims=(1, None))

        # unsqueeze last dim (negative)
        test(op, (torch.rand(B0, 2, 5), -1), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), -1), in_dims=(1, None))

        # nested vmaps
        def unsqueeze_0(x):
            return torch.unsqueeze(x, 0)

        def unsqueeze_last(x):
            return torch.unsqueeze(x, -1)

        # bdims in canonical order
        test(vmap(unsqueeze_0), (torch.rand(B0, B1, 2),))
        test(vmap(unsqueeze_last), (torch.rand(B0, B1, 2),))

        # wild bdims
        test(vmap(unsqueeze_0), (torch.rand(B1, 2, B0),), in_dims=2)
        test(vmap(unsqueeze_0, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)
        test(vmap(unsqueeze_last), (torch.rand(B1, 2, B0),), in_dims=2)
        test(vmap(unsqueeze_last, in_dims=1), (torch.rand(2, B1, B0),), in_dims=2)

    def test_movedim(self):
        op = torch.movedim
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        # movedim(tensor, int, int) variant
        test(op, (torch.rand(B0, 2, 5), 0, 1), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 0, 1), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), 0, 1),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), 0, 1),
            in_dims=(2, None, None),
        )

        # movedim(tensor, intlist, intlist) variant
        test(op, (torch.rand(B0, 2, 3, 5), [1, 0], [0, 2]), in_dims=(0, None, None))
        test(op, (torch.rand(2, 3, B0, 5), [1, 0], [0, 2]), in_dims=(1, None, None))
        test(
            vmap(op, in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )
        test(
            vmap(vmap(op, in_dims=(2, None, None)), in_dims=(0, None, None)),
            (torch.rand(B1, 2, B0, 5, B2), [0, 1], [1, 0]),
            in_dims=(2, None, None),
        )

    def test_mm(self):
        op = torch.mm
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = "Shape mismatch"
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5, 2)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5, 2)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5, 2)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5, 2)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5, 2)))
        test(
            vmap(op),
            (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5, 2)),
            in_dims=(1, 0),
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5, 2)),
            in_dims=(None, 0),
        )

    def test_mv(self):
        op = torch.mv
        test = self._vmap_test
        B0, B1 = 7, 11

        # shape mismatch
        msg = ""
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op)(torch.randn(B0, 2, 2, 2), torch.randn(B0, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(0, None))(torch.randn(B0, 2, 2), torch.randn(2, 2))
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(op, in_dims=(None, 0))(torch.randn(2, 2), torch.randn(B0, 2, 2))

        # left arg is vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(5)), in_dims=(0, None))
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, B0, 2, 5), torch.rand(5)),
            in_dims=(1, None),
        )

        # right arg is vmapped
        test(op, (torch.rand(2, 5), torch.rand(B0, 5)), in_dims=(None, 0))
        test(
            vmap(op, in_dims=(None, 0)),
            (torch.rand(2, 5), torch.rand(B1, B0, 5)),
            in_dims=(None, 1),
        )

        # both args are vmapped
        test(op, (torch.rand(B0, 2, 5), torch.rand(B0, 5)))
        test(
            vmap(op), (torch.rand(B1, B0, 2, 5), torch.rand(B0, B1, 5)), in_dims=(1, 0)
        )
        test(
            vmap(op, in_dims=(0, None)),
            (torch.rand(B1, 2, 5), torch.rand(B0, 5)),
            in_dims=(None, 0),
        )

    def test_narrow(self):
        op = torch.narrow
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13

        test(op, (torch.rand(B0, 2, 5), -1, 1, 3), in_dims=(0, None, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1, 3), in_dims=(1, None, None, None))
        test(
            vmap(op, in_dims=(0, None, None, None)),
            (torch.rand(B1, 2, B0, 5), 1, 0, 0),
            in_dims=(2, None, None, None),
        )
        test(
            vmap(
                vmap(op, in_dims=(2, None, None, None)), in_dims=(0, None, None, None)
            ),
            (torch.rand(B1, 2, B0, 5, B2), -1, 2, 3),
            in_dims=(2, None, None, None),
        )

    def test_new_empty(self):
        # Empty is non-deterministic so we just check that the shape of the
        # output tensor is what we expect and that the vmap fallback isn't used.
        op = Tensor.new_empty

        B0, B1 = 7, 11

        result = vmap(lambda x: op(x, [2, 3]))(torch.randn(B0))
        self.assertEqual(result.shape, [B0, 2, 3])

        result = vmap(lambda x: op(x, []))(torch.randn(B0))
        self.assertEqual(result.shape, [B0])

        result = vmap(vmap(lambda x: op(x, [2, 3])))(torch.randn(B0, B1))
        self.assertEqual(result.shape, [B0, B1, 2, 3])

    def test_new_empty_strided(self):
        # Empty is non-deterministic so we just check that the size and shape
        # of the output are what we expect and that the vmap fallback isn't used
        B0, B1 = 7, 11

        def _test_single_vmap(size, stride, B0):
            x = torch.randn(B0)
            result = vmap(lambda x: x.new_empty_strided(size, stride))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0] + size)
            self.assertEqual(result.stride(), [S] + stride)

        def _test_double_vmap(size, stride, B0, B1):
            x = torch.randn(B0, B1)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)))(x)
            S = torch.empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

            x = torch.randn(B1, B0)
            result = vmap(vmap(lambda x: x.new_empty_strided(size, stride)), in_dims=1)(
                x
            )
            S = x.new_empty_strided(size, stride).storage().size()
            self.assertEqual(result.shape, [B0, B1] + size)
            self.assertEqual(result.stride(), [B1 * S, S] + stride)

        # contiguous case
        _test_single_vmap([2, 3, 5], [3 * 5, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [3 * 5, 5, 1], B0, B1)

        # expanded
        _test_single_vmap([2, 3, 5], [0, 5, 1], B0)
        _test_double_vmap([2, 3, 5], [0, 5, 1], B0, B1)

        # some of these cases are pretty strange, just verifying that if
        # empty_strided allows them then BatchedTensor.new_empty_strided
        # can as well
        for shape in [[2, 3, 4], [0, 2, 0]]:
            for strides in [[12, 4, 1], [2, 4, 6], [0, 0, 0]]:
                _test_single_vmap(shape, strides, B0)
                _test_double_vmap(shape, strides, B0, B1)

    def test_new_zeros(self):
        op = Tensor.new_zeros
        test = functools.partial(self._vmap_test, check_propagates_grad=False)
        B0, B1 = 7, 11

        test(lambda x: op(x, 2, 3), (torch.rand(B0),))
        test(lambda x: op(x, []), (torch.rand(B0),))
        test(vmap(lambda x: op(x, 3, 5)), (torch.rand(B0, B1),))

    def test_select(self):
        op = torch.select
        test = self._vmap_view_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 0, 0), in_dims=(0, None, None))
        test(op, (torch.rand(2, B0, 5), 1, 1), in_dims=(1, None, None))
        test(vmap(lambda t: op(t, 1, 1)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 1, 1), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

    def test_roll_no_dims(self):
        op = torch.roll
        test = self._vmap_test
        B0, B1, B2 = 7, 11, 13
        test(op, (torch.rand(B0, 2, 5), 2), in_dims=(0, None))
        test(op, (torch.rand(2, B0, 5), 3), in_dims=(1, None))
        test(vmap(lambda t: op(t, 3)), (torch.rand(B1, 2, B0, 5),), in_dims=2)
        test(
            vmap(vmap(lambda t: op(t, 3), in_dims=1)),
            (torch.rand(B1, 2, B0, B2, 5),),
            in_dims=2,
        )

    def test_stack(self):
        test = self._vmap_test
        B0, B1 = 5, 7

        # Quick hack b/c vmap can't accept a list of tensors as an argument
        def get_op(dim):
            def op(*tensors):
                return torch.stack(tensors, dim=dim)

            return op

        test(get_op(0), (torch.rand(B0, 3), torch.rand(B0, 3)))
        test(get_op(0), (torch.rand(3), torch.rand(B0, 3)), in_dims=(None, 0))
        test(get_op(0), (torch.rand(2, 17)

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 13 class(es): EnableVmapFallbackWarnings, TestVmapAPI, TensorFactory, Namespace, TestVmapBase, TestVmapOperators, TestVmapBatchedGradient, TestVmapOperatorsOpInfo, TestRandomness, TestTransformFailure, Test, TestVmapDeviceType, TestVmapNestedTensor

### Functions
This file defines 472 function(s): get_platform_specific_sdpa, __enter__, __exit__, test_non_tensor_output_raises, multiple_outputs, test_different_map_dim_size_raises, test_func_with_no_inputs, foo, bar, test_func_with_no_tensors, foo, test_constant_function, test_single_input, square, test_multiple_inputs, test_multiple_outputs, foo, test_multiple_outputs2, returns_tuple_of_tensors, returns_tuple_of_tensors, returns_list_of_two_tensors, returns_list_of_one_tensor, test_nested_with_same_map_dim, test_nested_with_diag_embed, test_nested_with_different_map_dim, test_noop_in_inner_vmap, test_checkpoint, get_grad, get_loss, test_unsupported_op_err_msg


## Key Components

The file contains 20743 words across 6432 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 237111 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

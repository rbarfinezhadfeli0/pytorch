# Documentation: `docs/test/test_legacy_vmap.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_legacy_vmap.py_docs.md`
- **Size**: 53,783 bytes (52.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_legacy_vmap.py`

## File Metadata

- **Path**: `test/test_legacy_vmap.py`
- **Size**: 106,704 bytes (104.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: vmap"]
# ruff: noqa: F841

import functools
import itertools
import types
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch._vmap_internals import vmap
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


FALLBACK_REGEX = r"There is a performance drop"


class EnableVmapFallbackWarnings:
    def __enter__(self):
        self.prev_state = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
        torch._C._debug_only_display_vmap_fallback_warnings(True)

    def __exit__(self, *ignored):
        torch._C._debug_only_display_vmap_fallback_warnings(self.prev_state)


class TestVmapAPILegacy(TestCase):
    def test_non_tensor_output_raises(self):
        with self.assertRaisesRegex(
            ValueError, "got type <class 'float'> as the return"
        ):
            output = vmap(lambda x: 3.14)(torch.ones(3))

        def multiple_outputs(x):
            return x, 3

        with self.assertRaisesRegex(ValueError, "got type <class 'int'> for return 1"):
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

    def test_multiple_outputs_error_cases(self):
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

        # jax supports these, but we don't yet
        msg = "must only return Tensors, got type <class 'list'>"
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_two_tensors)(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(returns_list_of_one_tensor)(x)

    def test_nested_with_same_map_dim(self):
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        output = vmap(vmap(torch.mul))(x, y)
        self.assertEqual(output, x * y)

        output = vmap(vmap(vmap(torch.mul)))(x, y)
        self.assertEqual(output, x * y)

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

    def test_unsupported_op_err_msg(self):
        # Unsupported view op
        tensor = torch.randn(2, 3)
        msg = (
            r"Batching rule not implemented for aten::.+; the "
            r"fallback path doesn't work on out= or view ops"
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(torch.ravel)(tensor)

        def out_op(x, y):
            return torch.abs(x, out=y)

        with self.assertRaisesRegex(RuntimeError, msg):
            vmap(out_op)(tensor, tensor)

        tensor = torch.randn(2)
        # The fallback doesn't support TensorList
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(lambda t: torch.atleast_1d([t]))(tensor)

        # Don't support non-tensor returns. This is a limitation of vmap;
        # functions that don't return tensors must be special cased
        with self.assertRaisesRegex(RuntimeError, "Batching rule not implemented"):
            vmap(torch.Tensor.item)(tensor)

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

    def test_out_dims_must_be_int_or_tuple_of_int_err_msg(self):
        msg = "`out_dims` must be an int or a tuple of int"
        tensor = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims="lol")(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=("lol",))(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=None)(tensor)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda x: x, out_dims=(None,))(tensor)

    def test_out_dims_and_num_outputs_mismatch_err_msg(self):
        msg = "`out_dims` must have one dim per output"
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

    def test_accepts_nested_inputs(self):
        B0 = 2
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
        def foo(xy):
            return xy[0] * xy[1]

        def bar(x, yz):
            return x * yz[0] * yz[1]

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

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
            vmap(foo, in_dims=(-1,))(x)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(foo, in_dims=(2,))(y)
        with self.assertRaisesRegex(ValueError, msg):
            vmap(lambda z: z[0] + z[1], in_dims=([3, 0],))([x, y])
        # the following should not throw
        vmap(foo, in_dims=(0,))(torch.randn(2, 3))
        vmap(foo, in_dims=(1,))(torch.randn(2, 3))

    def test_fallback_does_not_warn_by_default(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            result = vmap(op)(x, y)
            # The single warning here is the "vmap is experimental"
            # warning, not a warning from the vmap fallback path.
            self.assertEqual(len(wa), 1)

    def test_fallback_warns_when_warnings_are_enabled(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2
        x = torch.randn(11)
        y = torch.randn(11)
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(op)(x, y)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def _assert_uses_vmap_fallback(self, vmap_args, inputs):
        with warnings.catch_warnings(record=True) as wa:
            with EnableVmapFallbackWarnings():
                result = vmap(*vmap_args)(*inputs)
            self.assertEqual(len(wa), 2)
            self.assertRegex(str(wa[-1].message), FALLBACK_REGEX)

    def test_fallback_zero_dim(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2
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

    def test_fallback_atan2(self):
        # NB: One day we will implement a batching rule for torch.atan2.
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = torch.atan2

        x = torch.randn(5, 7, 11)
        y = torch.randn(5, 7, 11)

        self._assert_uses_vmap_fallback((op,), (x, y))

        # fallback on torch.atan2
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(op, (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # fallback on torch.atan2, nested vmap
        x = torch.randn(7, 11, 5)
        y = torch.randn(5, 7, 11)
        result = vmap(vmap(op), (2, 0))(x, y)
        self.assertEqual(result, op(x.permute(2, 0, 1), y))

        # big batch size (total 10000)
        x = torch.randn(100, 10, 10, 5)
        y = torch.randn(100, 10, 10)
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(result, op(x, y.view(100, 10, 10, 1)))

    def test_fallback_masked_fill(self):
        # NB: One day we will implement a batching rule for masked_fill
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        def run_test(batch_size):
            B0 = batch_size
            x = torch.randn(B0, 7, 11, 13)
            dim = 0
            index = torch.tensor([0, 4, 2])
            values = torch.randn(B0, 3, 11, 13)

            self._assert_uses_vmap_fallback(
                (torch.index_add, (0, None, None, 0)), (x, dim, index, values)
            )

            result = vmap(torch.index_add, (0, None, None, 0))(x, dim, index, values)
            expected = torch.index_add(x, dim + 1, index, values.view(B0, 3, 11, 13))
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
        result = vmap(vmap(vmap(op)))(x, y)
        self.assertEqual(x, outplace_op(x_orig, y.view(B0, B1, B2, 1)))

    def test_inplace_fallback_nary_different_levels(self):
        # NB: One day we will implement a batching rule for atan2_
        # If/when we do, this test should be replaced to test the fallback
        # path on another operator to avoid bitrot.
        op = Tensor.atan2_
        outplace_op = torch.atan2
        B0, B1, B2 = 2, 3, 5

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
        err_msg = r"backward\(\) called inside torch.vmap"

        def backward_on_vmapped_tensor(x):
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_on_vmapped_tensor)(x)

        def backward_with_vmapped_grad(x, grad):
            x.backward(grad)

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(backward_with_vmapped_grad)(x, grad)

        def completely_unrelated_backward(y):
            x.sum().backward()

        with self.assertRaisesRegex(RuntimeError, err_msg):
            vmap(completely_unrelated_backward)(y)

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


def slice_inputs(inputs, bdims, i):
    result = []
    for inp, bdim in zip(inputs, bdims):
        if bdim is None:
            result.append(inp)
        else:
            result.append(inp.select(bdim, i))
    return tuple(result)


def reference_vmap(op, inputs, in_dims=0, out_dims=0):
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
        return torch.stack(results, dim=out_dims[0])

    assert all(isinstance(result, tuple) for result in results)
    num_returns = len(results[0])
    assert all(len(result) == num_returns for result in results)
    if isinstance(out_dims, int):
        out_dims = (out_dims,) * num_returns
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
    reference_result = reference_vmap(op, inputs, in_dims, out_dims)
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


# All tests of TestVmapBaseLegacy check that the slow vmap fallback is never invoked.
# This is so that we can incrementally add batching rules for operators to
# replace the slow vmap fallback path for said operators. To skip this check,
# please use the allowVmapFallbackUsage decorator.
#
# NB: Don't add tests to TestVmapBaseLegacy directly, unless you want them to run
# on every subclass of TestVmapBaseLegacy. Add them to e.g. TestVmapOperators.
#
# NB: TestVmapBaseLegacy is a nested class. This prevents test runners from picking
# it up and running it.
class Namespace:
    class TestVmapBaseLegacy(TestCase):
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
            msg = (
                "Expected the test to not invoke the vmap fallback path, i.e., "
                "all of the operators being tested in this test should have batching "
                "rules implemented. If you are intentionally testing something to "
                "do with the fallback path, use allowVmapFallbackUsage. Otherwise, "
                "please make sure that batching rules are implemented for the "
                "operator(s) being tested."
            )

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                with warnings.catch_warnings(record=True) as wa:
                    warnings.simplefilter("always")
                    with EnableVmapFallbackWarnings():
                        method(*args, **kwargs)
                    for captured_warning in wa:
                        self.assertNotRegex(
                            str(captured_warning.message), FALLBACK_REGEX, msg
                        )

            return types.MethodType(wrapper, self)

        @allowVmapFallbackUsage
        def test_vmap_fallback_check_ok(self):
            # One day we'll implement a batching rule for torch.var_mean.
            # When that happens, please change the example to use an
            # operator that doesn't have a batching rule implemented.
            op_using_fallback = torch.var_mean
            vmap(op_using_fallback)(torch.rand(3))

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


class TestVmapOperatorsLegacy(Namespace.TestVmapBaseLegacy):
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

    def test_unary_pointwise_ops(self):
        cases = [
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
        ]
        for op, getter in cases:
            self._test_unary(op, getter, "cpu")

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

    def test_binary_pointwise_ops(self):
        def get_number(getter):
            return getter([]).item()

        def make_case(op, input_getter=TensorFactory.randn):
            return (op, input_getter)

        cases = [
            # Basic arithmetic
            make_case(torch.add),
            make_case(lambda x, y: x + y),
            make_case(torch.sub),
            make_case(lambda x, y: x - y),
            make_case(torch.mul),
            make_case(lambda x, y: x * y),
            make_case(torch.div, input_getter=TensorFactory.randp1),
            make_case(lambda x, y: x / y, input_getter=TensorFactory.randp1),
            make_case(torch.pow, input_getter=TensorFactory.randp1),
            make_case(lambda x, y: x**y, input_getter=TensorFactory.randp1),
        ]
        test = self._vmap_test

        for op, getter in cases:
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
            test(
                op, (getter([2, 3], device), getter([B0, 3], device)), in_dims=(0, None)
            )

            # Nested vmap: op(Tensor, Tensor)
            test(
                vmap(op), (getter([B0, B1, 2, 3], device), getter([B0, B1, 3], device))
            )
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
                continue

            # TODO(rzou): fix the following
            # # Test cross-device scalars
            # number = get_number(getter)
            # self._test_unary(lambda t: op(t, number), getter, device='cuda')
            # self._test_unary(lambda t: op(number, t), getter, device='cuda')
            # self._test_unary(lambda t: op(t, torch.tensor(number)), getter, device='cuda')

    def test_as_strided(self):
        def _test(sizes, strides, offset, tensor, lambd):
            result = vmap(lambda t: t.as_strided(sizes, strides, offset))(tensor)
            expected = vmap(lambd)(tensor)
            self.assertTrue(result._base is expected._base)
            self.assertEqual(result, expected)

        # single vmap test
        B0 = 5
        tensors = [
            # contiguous
            torch.randn(B0, 2, 3),
            # non-contiguous
            torch.randn(B0, 3, 2).transpose(1, 2),
            # non-zero storage offset
            torch.randn(2, B0, 2, 3)[1],
            # non-contiguous strides, zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 0, :, 0],
            # non-contiguous strides, non-zero storage offset
            torch.randn(B0, 2, 4, 3, 7)[:, :, 2, :, 1],
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

        # Sanity check #1: we require the batch dims to be at the front of the
        # tensor (in memory layout).
        msg = "batch dims being vmapped over are at the front of the tensor"
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3).transpose(0, 1)
            vmap(lambda x: x.as_strided([2, 3], [B0 * 3, 1]))(x)
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(B0, 2, 3, B1).movedim(3, 1)
            vmap(vmap(lambda x: x.as_strided([2, 3], [B1 * 3, B1])))(x)

        # All the Sanity check #2{a,b,c} cases check that
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # doesn't index memory that is out of bounds of xs[i]. This condition
        # is important to the correctness of the as_strided batching rule
        # (see NOTE: [When will the as_strided_batching_rule fail?])

        # Sanity check #2a: The maximum indexable location of
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

        # Sanity check #2b: The min indexable location of
        # xs[i].as_strided(sizes, strides, offset + xs[i].offset() - xs.offset())
        # is greater than or equal to the min indexable location of xs[i].
        with self.assertRaisesRegex(RuntimeError, msg):
            x = torch.randn(2, B0, 3)[1]
            vmap(lambda
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
- **Context Manager**: Implements context manager protocol
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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_legacy_vmap.py_docs.md
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

- **File Documentation**: `test_legacy_vmap.py_docs.md_docs.md`
- **Keyword Index**: `test_legacy_vmap.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

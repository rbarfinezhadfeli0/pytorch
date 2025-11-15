# Documentation: `test/dynamo/test_subclasses.py`

## File Metadata

- **Path**: `test/dynamo/test_subclasses.py`
- **Size**: 165,139 bytes (161.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import functools
import itertools
import unittest
from functools import partial

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._functorch._aot_autograd.utils import make_boxed_compiler
from torch._functorch.compilers import min_cut_rematerialization_partition
from torch._higher_order_ops.wrap import wrap
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.nested._internal.nested_tensor import (
    jagged_from_list,
    jagged_from_tensor_and_lengths,
    nested_view_from_values_offsets,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    NestedTensorTestCase,
    parametrize,
    subtest,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import return_and_correct_aliasing


def nontraceable_subclass(c):
    return torch._dynamo.config.patch("nontraceable_tensor_subclasses", {c})


def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
    actual_recompiles = _recompiles_for_inputs(fn, inputs1, inputs2)
    self.assertEqual(actual_recompiles, expected_recompiles)


def get_jagged_tensor(nested_size, offsets, requires_grad=True):
    # Makes a jagged tensor with N constituent tensors with size
    # as specified ((S0, S1, S2), D)
    D = nested_size[1]
    out = []
    for s in nested_size[0]:
        out.append(torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64))
    return jagged_from_list(out, offsets)


def get_view_test_cases():
    # Test all cases with both an NT base and a dense base
    # Subclass -> Subclass
    # Dense -> Subclass

    # NB: Don't close over loop variables, they will not get copied into the
    # closure
    #
    # NB: These return functions so we don't generate tensors during test
    # collection time

    def mk_basic(base_is_nt):
        # There are three cases to consider here based on the logic in
        # meta_utils.py
        #
        # (1) basic case:
        # view is not a leaf and has the same requires grad as its basic case
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)
        x = x.clone() if base_is_nt else x
        assert not x.is_leaf
        return x.unsqueeze(-1)

    def mk_leaf(base_is_nt, requires_grad_1, requires_grad_2):
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=requires_grad_1)
        x = x.clone() if base_is_nt else x
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
            # The issue is this doesn't quite work
            x_view.requires_grad_(requires_grad_2)

        return x_view

    def mk_obscure(base_is_nt):
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=False)
        x = x.clone() if base_is_nt else x
        # intermediate leaf view
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
        x_view.requires_grad_(True)
        x_view_view = x_view.unsqueeze(-1)
        return x_view_view

    for base_is_nt in [False, True]:
        prefix = f"base_is_nt_{base_is_nt}"

        yield partial(mk_basic, base_is_nt), f"{prefix}_basic"

        # (2) leaf view case:
        # the view has to be a leaf (w/ requires_grad True or requires_grad False)
        # base w/ requires_grad True or requires_grad False
        for requires_grad_1, requires_grad_2 in itertools.product(
            [True, False], repeat=2
        ):
            yield (
                partial(mk_leaf, base_is_nt, requires_grad_1, requires_grad_2),
                f"{prefix}_leaf_{requires_grad_1}_{requires_grad_2}",
            )

        # (3) obscure case:
        # view is not a leaf (implies requires_grad True)
        # base w/ requires_grad False)
        yield partial(mk_obscure, base_is_nt), f"{prefix}_obscure"

    # Subclass -> Dense
    yield (
        lambda: get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone(),
        "subclass_dense",
    )

    # Dense -> Subclass -> Dense -> Subclass
    def mk_dense_subclass_dense_subclass():
        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 3, 6, 10])
        return nested_view_from_values_offsets(
            nested_view_from_values_offsets(values, offsets).values(), offsets
        )

    yield mk_dense_subclass_dense_subclass, "dense_subclass_dense_subclass"

    def mk_subclass_dense_subclass_dense():
        x = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone()
        offsets2 = x.offsets().detach().clone()
        nested_view_from_values_offsets(x.values(), offsets2).values()

    yield mk_subclass_dense_subclass_dense, "subclass_dense_subclass_dense"


VIEW_TEST_CASES = {k: v for v, k in get_view_test_cases()}


compile_full_eager = torch.compile(backend="eager", fullgraph=True)


class BaseTorchFunction(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)


class AttrSubclass(torch.Tensor):
    x: int = 10
    size: int = 10

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)


class DummyNDim(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func == torch.Tensor.ndim.__get__:
            return 10

        return super().__torch_function__(func, types, args, kwargs)


class WrapperSubclass:
    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, args)
        kwargs = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, kwargs)

        return func(*args, **kwargs)


class SigmoidToExpSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func == torch.Tensor.sigmoid:
            return super().__torch_function__(torch.Tensor.exp, types, args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)


# Wrapper subclass with two inner tensors: data and scale
# data has same shape as outer, and scale has single dim size
class ScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        *,
        constant: int = 0,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, constant: int = 0):
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        ctx = {"_constant": self._constant}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        return ScaledTensor(out, scaled_tensor._scale, constant=scaled_tensor._constant)

    def __repr__(self):
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"


class OptionalScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data,
        scale,
        *,
        constant: int = 0,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale, constant: int = 0):
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        ctx = {"_constant": self._constant}
        if self._scale is not None:
            return ["_data", "_scale"], ctx
        else:
            return ["_data"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return OptionalScaledTensor(
            inner_tensors["_data"],
            inner_tensors.get("_scale", None),
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        if scaled_tensor._scale is not None:
            out = out * scaled_tensor._scale
        return OptionalScaledTensor(
            out, scaled_tensor._scale, constant=scaled_tensor._constant
        )

    def __repr__(self):
        return (
            f"OptionalScaledTensor({self._data.__repr__()}\n{self._scale.__repr__()})"
        )


class CtxSubclassTensor(torch.Tensor):
    """
    Class used to verify guarding on the subclass metadata
    """

    @staticmethod
    def __new__(cls, a, constant):
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, a, constant):
        self.a = a
        self.constant = constant

    def __repr__(self):
        a_repr = repr(self.a)
        return f"CtxSubclassTensor({a_repr})"

    def __tensor_flatten__(self):
        return ["a"], (self.constant,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, sizes, strides):
        constant = meta[0]
        a = inner_tensors["a"]
        return CtxSubclassTensor(a, constant)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        biggest_constant = max(
            [
                x.constant
                for x in pytree.tree_flatten(args)[0]
                if isinstance(x, CtxSubclassTensor)
            ]
        )
        args_a = pytree.tree_map(
            lambda x: x.a if isinstance(x, CtxSubclassTensor) else x, args
        )
        kwargs_a = pytree.tree_map(
            lambda x: x.a if isinstance(x, CtxSubclassTensor) else x, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        out = pytree.tree_map(
            lambda x: (
                CtxSubclassTensor(x, biggest_constant)
                if isinstance(x, torch.Tensor)
                else x
            ),
            out_a,
        )

        if func == torch.ops.aten.mul.Tensor:
            out = out + out.constant

        return return_and_correct_aliasing(func, args, kwargs, out)


def func(a):
    return a.sin()


class EagerRecordGraphAndInputs:
    def __init__(self) -> None:
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


GLOBAL_TEST_SUBCLASSES = {
    MockSubclass,
    DummyNDim,
    SigmoidToExpSubclass,
    BaseTorchFunction,
}


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]

    def counter(gm, example_inputs):
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


class SubclassTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles)

    def test_no_call_to_new(self):
        class BadNewTorchFunction(torch.Tensor):
            def __new__(cls, *args, **kwargs):
                raise RuntimeError("Oops!")

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return torch.add(x, 1)

        input = torch.ones(2, 2).as_subclass(BadNewTorchFunction)

        res = fn(input)
        self.assertIsInstance(res, BadNewTorchFunction)

    def test_no_torch_function_recompiles(self):
        class NJT:
            def __repr__(self):
                return f"NJT(shape={self.shape})"

            def __init__(self, values, offsets):
                self._values = values
                self._offsets = offsets

            def sin(self):
                return torch.sin(self)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                if func == torch.sin:
                    self = args[0]
                    return NJT(func(self._values), self._offsets)
                raise AssertionError("should not get here")

        values1 = torch.randn(10, 3, 4, requires_grad=True)
        values2 = torch.randn(10, 3, 4, requires_grad=True)
        offsets = torch.tensor([0, 3, 10])
        njt1 = NJT(values1, offsets)
        njt2 = NJT(values2, offsets)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return torch.sin(x)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            f(njt1)
            f(njt2)

    def test_base_torch_function_tracing(self):
        def fn(x):
            return torch.add(x, 1)

        input = torch.ones(2, 2).as_subclass(BaseTorchFunction)
        out = fn(input)
        out_opt = compile_full_eager(fn)(input)
        self.assertIsInstance(out, BaseTorchFunction)
        self.assertEqual(out, out_opt)

    def test_torch_function_state_graph_break(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch._dynamo.graph_break()
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_disable_all_torch_function(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunction():
                torch._dynamo.graph_break()
                return (
                    torch._C._is_torch_function_enabled(),
                    torch._C._is_torch_function_all_disabled(),
                    torch.add(x, 1.0),
                )

        input = torch.ones(2, 2)
        res1, res2, _ = fn(input)
        self.assertFalse(res1)
        self.assertTrue(res2)

    def test_disable_all_torch_function_restore_values(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunction():
                x = torch._C._is_torch_function_all_disabled()

            return (
                x,
                torch._C._is_torch_function_all_disabled(),
                torch.add(x, 1.0),
            )

        input = torch.ones(2, 2)
        res1, res2, _ = fn(input)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_disable_all_torch_function_restore_values_graph_break(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunction():
                torch._dynamo.graph_break()
                x = torch._C._is_torch_function_all_disabled()

            return (
                x,
                torch._C._is_torch_function_all_disabled(),
                torch.add(x, 1.0),
            )

        input = torch.ones(2, 2)
        res1, res2, _ = fn(input)
        self.assertTrue(res1)
        self.assertFalse(res2)

    def test_torch_function_state_nested(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                with torch._C.DisableTorchFunctionSubclass():
                    x = x + 1
                # Should reset to the outer state (disabled) after exiting ctx manager
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_torch_function_state_tracing(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch.add(x, 1.0)

        input = torch.ones(2, 2)

        fn(input)

    def test_torch_function_state_guards(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            torch.add(x, 1.0)

        input = torch.ones(2, 2)

        with torch._C.DisableTorchFunctionSubclass():
            fn(input)

        fn(input)

        self.assertEqual(cnt.frame_count, 2)

    def test_return_subclass(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return MockSubclass(torch.add(x, 1.0)) * 2

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_as_subclass(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return torch.add(x, 1.0).as_subclass(MockSubclass) * 2

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_local_subclass(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return LocalSubclass(torch.add(x, 1.0)) * 2

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, LocalSubclass)

    def test_torch_function_list_args(self):
        HANDLED_FUNCTIONS = {}

        class MyClass:
            def __init__(self, foo):
                self.foo = foo

            @classmethod
            def __torch_function__(
                cls,
                func,
                types,
                args=(),
                kwargs=None,
            ):
                if kwargs is None:
                    kwargs = {}
                if func not in HANDLED_FUNCTIONS or not all(  # noqa: C419
                    [  # noqa: C419
                        issubclass(t, (torch.Tensor, MyClass)) for t in types
                    ]
                ):
                    return NotImplemented
                return HANDLED_FUNCTIONS[func](*args, **kwargs)

        def _stack(input, dim=0, *, out=None):
            return MyClass(sum([x.foo for x in input]))

        HANDLED_FUNCTIONS[torch.stack] = _stack

        @torch.compile(backend="eager", fullgraph=True)
        def fn(v0, v1):
            return torch.stack([v0, v1])

        ret = fn(MyClass(1), MyClass(1))
        self.assertEqual(ret.foo, 2)

    @parametrize(
        "comparison",
        [
            subtest(isinstance, "isinstance"),
            subtest(lambda instance, type_: type(instance) is type_, "equality"),
            subtest(lambda instance, type_: type(instance) is type_, "identity"),
        ],
    )
    @parametrize(
        "input_type",
        [
            subtest(torch.Tensor, "tensor"),
            subtest(DummyNDim, "subclass"),
        ],
    )
    def test_type_check(self, comparison, input_type):
        def fn(x):
            if comparison(x, DummyNDim):
                return torch.ones(1, 1)
            else:
                return torch.zeros(2, 2)

        input = torch.ones(2, 2).as_subclass(input_type)
        exp_res = fn(input)
        act_res = torch.compile(backend="eager", fullgraph=True)(fn)(input)
        self.assertEqual(exp_res, act_res)

    def test_torch_function_call_on_method(self):
        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = x.as_subclass(SigmoidToExpSubclass)
        wrapped2 = y.as_subclass(SigmoidToExpSubclass)

        def fn(w):
            return w.exp()

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped2)
        res_exp2 = z.exp()

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, res_exp2)

    def test_torch_function_call_on_method_arg(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if func == torch._C.TensorBase.add_:
                    func = torch._C.TensorBase.sub_

                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                return None

        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = y.as_subclass(LocalSubclass)
        wrapped2 = z.as_subclass(LocalSubclass)

        def fn(a, w):
            a.add_(w)
            return a

        fn_opt = torch.compile(fn)

        res_exp = fn(x, wrapped)
        res_act = fn_opt(y, wrapped2)

        self.assertEqual(res_exp, res_act)

    def test_user_overridden_method_unsupported(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                return None

        def fn(x):
            x.sigmoid()

        x = torch.ones(2, 2).as_subclass(LocalSubclass)
        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)

        self.assertEqual(res_exp, res_act)

    def test_user_overridden_attr_unsupported(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            ndim = 10

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.ndim

        msg = "`torch.compile` only support tracing certain types of overridden tensor subclass attributes"
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_user_overridden_property_unsupported(self):
        class LocalSubclass(torch.Tensor):
            def __init__(self, *args, **kwargs) -> None:
                self._ndim = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs)

            @property
            def ndim(self):
                return self._ndim

            @ndim.setter
            def ndim(self, value):
                self._ndim = value

        def fn(x):
            return x + x.ndim

        x = LocalSubclass(torch.ones(2, 2))
        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)

        self.assertEqual(res_exp, res_act)

    def test_overridden_method_guarding(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager")
        def fn(x):
            return x.sigmoid()

        with torch._dynamo.config.patch(error_on_recompile=True):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)
            fn(x)
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

        with self.assertRaisesRegex(TypeError, "'bool' object is not callable"):
            LocalSubclass.sigmoid = False
            fn(x)

    def test_torch_function_call_on_attr(self):
        x = torch.ones(2, 2)
        wrapped = x.as_subclass(DummyNDim)

        def fn(w):
            return w.ndim + torch.ones(2)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, torch.ones(2) + 10)

    def test_torch_function_wrapper_class(self):
        x = torch.ones(2, 2)
        wrapped = WrapperSubclass(x)

        def fn(w):
            return torch.add(w, 1.0)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)
        self.assertEqual(res_exp, res_act)

    def test_no_torch_function_on_size_bytecode(self):
        class TestTensor(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    out = func(*args, **kwargs)

                    if func == torch.clone:
                        return out * 2
                    else:
                        return out

        def fn(x):
            return torch.clone(x)

        inp = torch.ones(4, 4)
        x = inp.as_subclass(TestTensor)
        torch._dynamo.mark_dynamic(x, 0)
        compiled_fn = torch.compile(fn, fullgraph=True)
        out = compiled_fn(x)
        self.assertEqual(out, torch.ones(4, 4) * 2)

    def test_torch_function_wrapper_class_with_kwargs(self):
        x = torch.ones(2, 2)
        wrapped = WrapperSubclass(x)

        def fn(w):
            return torch.add(w, 1.0, alpha=2.0)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)
        self.assertEqual(res_exp, res_act)

    def test_tensor_subclass_with_non_classmethod_torch_function(self):
        class MySubclass(torch.Tensor):
            def __torch_function__(self, func, types, args, kwargs=None):
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

        def fn(x):
            return x + 1

        fn_opt = compile_full_eager(fn)

        x = torch.randn(2, 2).as_subclass(MySubclass)
        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_tensor_subclass_custom_attr(self):
        class AttrSubclass(torch.Tensor):
            x: int = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.x + torch.ones(2, 2)

        input = torch.ones(2, 2).as_subclass(AttrSubclass)
        fn_opt = compile_full_eager(fn)

        res_exp = fn(input)
        res_act = fn_opt(input)
        self.assertEqual(res_exp, res_act)

    def test_make_subclass(self):
        # Make sure `torch.Tensor._make_subclass` is traceable, and Dynamo
        # models its aliasing relationships correctly.
        class MySubclass(torch.Tensor):
            pass

        def fn(x):
            # Downcast then upcast
            y = torch.Tensor._make_subclass(MySubclass, x)
            z = torch.Tensor._make_subclass(torch.Tensor, x)
            # Now `x, y, z` should have the same underlying data.
            x += 1
            y += 2
            z += 3
            res = x * y + z
            return res

        x0 = torch.randn(2, 2)
        x1 = x0.clone()

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x0)
        res_act = fn_opt(x1)
        self.assertEqual(res_exp, res_act)
        self.assertEqual(x0, x1)

    def test_subclass_override_shape_and_to(self):
        # This is a slight variabtion of
        # https://github.com/huggingface/diffusers/blob/fbf6b856cc61fd22ad8635547bff4aafe05723f3/src/diffusers/quantizers/gguf/utils.py#L398-L435
        class MySubclass(torch.Tensor):
            def to(self, *args, **kwargs):
                new = super().to(*args, **kwargs)
                new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
                return new

            @property
            def shape(self):
                if not hasattr(self, "tensor_shape"):
                    self.tensor_shape = self.size()
                return self.tensor_shape

        def fn(x):
            x_shape = x.shape
            y = x.to("cpu")
            return x + 1, y + 2, x_shape, x.tensor_shape, y.tensor_shape

        x0 = torch.nn.Parameter(torch.randn(2, 2).as_subclass(MySubclass))
        x1 = torch.nn.Parameter(x0.clone().as_subclass(MySubclass))

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x0)
        res_act = fn_opt(x1)
        self.assertEqual(res_exp, res_act)
        self.assertEqual(x0, x1)
        self.assertEqual(x0.tensor_shape, x1.tensor_shape)

    def test_subclass_dont_invoke_torch_function_on_overridden_method(self):
        # We shouldn't fire `__torch_function__` for overridden tensor methods.
        class MySubclass(torch.Tensor):
            def to(self, device):
                return self * len(device)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if func is torch.Tensor.to:
                    torch._dynamo.graph_break()
                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            return x.to("cpu")

        x = torch.nn.Parameter(torch.randn(2, 2).as_subclass(MySubclass))

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_subclass_dont_invoke_torch_function_on_overridden_attr(self):
        from types import MethodWrapperType

        # We shouldn't fire `__torch_function__` for overridden tensor attrs.
        class MySubclass(torch.Tensor):
            def ndim(self):
                return 42

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if type(func) is MethodWrapperType and func.__name__ == "ndim":
                    torch._dynamo.graph_break()
                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            return x + x.ndim()

        x = torch.nn.Parameter(torch.randn(2, 2).as_subclass(MySubclass))

        fn_opt = compile_full_eager(fn)

        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_parameter_subclass_with_old_torch_function(self):
        class MySubclass(torch.nn.Parameter):
            pass

        def fn(x):
            x = x.t()
            x = x.T
            return x + 1

        fn_opt = compile_full_eager(fn)

        x = torch.randn(2, 2).as_subclass(MySubclass)
        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_subclass_with_disabled_torch_function(self):
        class MySubclass(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

        def fn(x):
            x = x.t()
            x = x.T
            return x + 1

        fn_opt = compile_full_eager(fn)

        x = torch.randn(2, 2).as_subclass(MySubclass)
        res_exp = fn(x)
        res_act = fn_opt(x)
        self.assertEqual(res_exp, res_act)

    def test_parameter_subclass_custom_torch_func_and_dynamic_attr(self):
        # This is a slight variation of
        # https://github.com/huggingface/diffusers/blob/fbf6b856cc61fd22ad8635547bff4aafe05723f3/src/diffusers/quantizers/gguf/utils.py#L398-L435
        # which basically
        # 1. uses tensor subclass to attach quantization metadata onto tensors
        # 2. preserve them across torch ops
        # 3. use the metadata to dequantize the tensor
        # 4. convert it to a regular tensor.
        #
        # The test is meant to make sure Dynamo won't graph break over it.
        class GGUFParameter(torch.nn.Parameter):
            def __new__(cls, data, requires_grad=False, quant_type=None):
                data = data if data is not None else torch.empty(0)
                self = torch.Tensor._make_subclass(cls, data, requires_grad)
                return self

            def __init__(self, *args, quant_type=None, **kwargs):
                self.quant_type = quant_type

            def as_tensor(self):
                return torch.Tensor._make_subclass(
                    torch.Tensor, self, self.requires_grad
                )

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                result = super().__torch_function__(func, types, args, kwargs)

                quant_type = None
                for arg in args:
                    if isinstance(arg, list) and isinstance(arg[0], GGUFParameter):
                        quant_type = arg[0].quant_type
                        break
                    if isinstance(arg, GGUFParameter):
                        quant_type = arg.quant_type
                        break
                if isinstance(result, torch.Tensor):
                    return cls(result, quant_type=quant_type)
                # Handle tuples and lists
                elif isinstance(result, (tuple, list)):
                    # Preserve the original type (tuple or list)
                    wrapped = [
                        (
                            cls(x, quant_type=quant_type)
                            if isinstance(x, torch.Tensor)
                            else x
                        )
                        for x in result
                    ]
                    return type(result)(wrapped)
                else:
                    return result

        def f(x):
            tmp = x * 2
            tmp = tmp + tmp.quant_type
            tmp = tmp.as_tensor()
            return tmp * 3

        opt_f = torch.compile(f, backend="eager", fullgraph=True)

        x = GGUFParameter(torch.ones(2), quant_type=42)
        res = f(x)
        ref = opt_f(x)
        self.assertEqual(res, ref)

    def test_newly_constructed_tensor_subclass_attr_mutation(self):
        # Make sure the attribute mutation for newly constructed tensor subclass
        # object (from constructor call) is handled both during Dynamo tracing
        # and codegen-ed to be visible outside `torch.compile`.
        class MySubclass(torch.Tensor):
            pass

        def f():
            x = MySubclass(torch.ones(2))
            x.bar = 42
            return x, x * x.bar

        opt_f = compile_full_eager(f)

        res = f()
        ref = opt_f()

        self.assertEqual(res, ref)
        self.assertEqual(res[0].bar, ref[0].bar)

    def test_as_subclass_attr_mutation(self):
        # Make sure the attribute mutation for newly constructed tensor subclass
        # object (from as_subclass call) is handled both during Dynamo tracing
        # and codegen-ed to be visible outside `torch.compile`.
        class MySubclass(torch.Tensor):
            pass

        def f():
            x = torch.ones(2).as_subclass(MySubclass)
            x.bar = 42
            return x, x * x.bar

        opt_f = compile_full_eager(f)

        res = f()
        ref = opt_f()

        self.assertEqual(res, ref)
        self.assertEqual(res[0].bar, ref[0].bar)

    def test_tensor_subclass_attr_codegen_tos(self):
        # This repros a very subtle interaction between
        # `TensorWithTFOverrideVariable` attribute mutation codegen and
        # `PyCodegen.top_of_stack`. It was uncovered from
        # `test_tensor_subclass_deepcopy`.
        class MySubclass(torch.Tensor):
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_subclass(cls, torch.ones(0))
                r.elem = elem
                return r

        def f(t):
            return MySubclass(t.elem.clone())

        opt_f = compile_full_eager(f)

        t = MySubclass(torch.ones(2))
        res = f(t)
        ref = opt_f(t)

        self.assertEqual(res, ref)
        self.assertEqual(res.elem, ref.elem)
        self.assertEqual(type(res), type(ref))

    def test_nontraceable_tensor_subclass(self):
        # This will error if Dynamo tries to wrap it as a tensor variable,
        # because that involves calling certain methods to inspect the tensor
        # property, which will blow up in the overridden `__torch_function__`.
        class MySubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("one shall not pass")

        def f(t):
            return t.foo + torch.ones(10)

        opt_f = torch.compile(f, backend="eager", fullgraph=False)

        t = MySubclass(torch.ones(2))
        t.foo = 42
        # Make sure the `nontraceable_tensor_subclasses` config prevents Dynamo
        # from wrapping `t`.
        with nontraceable_subclass(MySubclass):
            res = f(t)
            ref = opt_f(t)

        self.assertEqual(res, ref)

    def test_compile_with_fake_tensor_dynamic_dim(self):
        x = torch.randn([3, 4])

        def f(x):
            return torch.sin(x)

        def test_dynamic_dim(f, x, dim_dynamic, exp_frame_count, exp_op_count):
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()

            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            x1 = torch.rand_like(x)
            f(x)
            f(torch.randn([4, 3]))
            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                x_fake = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                x1_fake = fake_mode.from_tensor(
                    x1,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                opt_f(x_fake)
                opt_f(x1_fake)

            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        test_dynamic_dim(f, x, DimDynamic.DYNAMIC, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.DUCK, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.STATIC, 1, 1)

    def test_compile_with_fake_tensor_automatic_dynamic(self):
        def f(x):
            return torch.sin(x)

        def test_automatic_dynamic(f, inps, dim_dynamic, exp_frame_count, exp_op_count):
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                for inp in inps:
                    fake_inp = fake_mode.from_tensor(
                        inp,
                        symbolic_context=StatelessSymbolicContext(
                            [dim_dynamic for i in range(x.dim())]
                        ),
                    )
                    opt_f(fake_inp)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        x = torch.randn([3, 4])
        y = torch.randn([4, 5])
        z = torch.randn([5, 6])
        a = torch.randn([3, 5])
        b = torch.randn([4, 4])
        # When inputs' DimDynamic is DYNAMIC or DUCK, the inputs
        # to opt_f will be tensors with SymInt sizes. Dynamo will treat input
        # as dynamic automatically and will only compile once
        for dim_dynamic in [DimDynamic.DYNAMIC, DimDynamic.DUCK]:
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 1, 1)

        for dim_dynamic in [DimDynamic.STATIC]:
            # Recompile once, first with dim 0 and 1 become Dynamic
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 2, 2)
            # Recompile 2 times, first with dim 1 become Dynamic, second with dim 0 becomes Dynamic.
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 3, 3)
            # Recompile 2 times, first with dim 0 become Dynamic, second with dim 1 becomes Dynamic.
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 3, 3)

    def test_compile_with_functionalization(self):
        x = torch.randn([3, 4])
        x_clone = x.clone()
        x_clone2 = x.clone()
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return x.add_(1.0) + torch.nn.functional.relu_(x)

        f_out = f(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.example_inputs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )

        ff = torch.func.functionalize(f)
        ff_out = ff(x_clone)

        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(len(backend.example_inputs), 2)
        actual = normalize_gm(backend.graphs[1].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        # Cannot reuse the version from AOTAutograd, since that uses python functional tensors.
        def to_fun(x):
            x_functional = torch._to_functional_tensor(x)
            torch._mirror_autograd_meta_to(x, x_functional)
            return x_functional

        def aot_f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(to_fun, args)
                    func_kwargs = pytree.tree_map(to_fun, kwargs)
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        aot_ff = aot_f_wrapper(f)
        aot_ff_out = aot_ff(x_clone2)

        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 9)
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(len(backend.example_inputs), 3)
        actual = normalize_gm(backend.graphs[2].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        self.assertEqual(f_out, ff_out)
        self.assertEqual(f_out, aot_ff_out)

        try:
            torch._enable_functionalization(reapply_views=False)
            xf = pytree.tree_map(to_fun, x)
            x_view = xf.t()
            with self.assertRaisesRegex(RuntimeError, "Cannot safely fakify a view"):
                f(x_view)
        finally:
            torch._disable_functionalization()

    def test_compile_higher_order_with_functionalization(self):
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: x.add_(1.0), x)

        def check_count_and_graph(
            exp_frame_count, exp_op_count, exp_n_graph, exp_graph
        ):
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)
            self.assertEqual(len(backend.graphs), exp_n_graph)
            actual = normalize_gm(
                backend.graphs[exp_n_graph - 1].print_readable(print_output=False)
            )
            self.assertExpectedInline(actual, exp_graph, skip=1)

        t = torch.randn([3, 4])
        t_clone = t.clone()
        t_clone2 = t.clone()
        f(t)

        check_count_and_graph(
            1,
            2,
            1,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

        ff = torch.func.functionalize(f)
        ff_out = ff(t_clone)  # noqa: F841
        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(
            2,
            4,
            2,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

        try:
            x = torch._to_functional_tensor(t_clone2)
            torch._mirror_autograd_meta_to(t_clone2, x)
            torch._enable_functionalization(reapply_views=False)
            aot_f_out = f(x)  # noqa: F841
        finally:
            torch._disable_functionalization()

        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(
            3,
            6,
            3,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 
```



## High-Level Overview


This Python file contains 100 class(es) and 377 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BaseTorchFunction`, `MockSubclass`, `AttrSubclass`, `DummyNDim`, `WrapperSubclass`, `SigmoidToExpSubclass`, `ScaledTensor`, `OptionalScaledTensor`, `CtxSubclassTensor`, `EagerRecordGraphAndInputs`, `SubclassTests`, `BadNewTorchFunction`, `NJT`, `LocalSubclass`, `MyClass`, `LocalSubclass`, `LocalSubclass`, `LocalSubclass`, `LocalSubclass`, `LocalSubclass`

**Functions defined**: `nontraceable_subclass`, `_check_recompiles`, `get_jagged_tensor`, `get_view_test_cases`, `mk_basic`, `mk_leaf`, `mk_obscure`, `mk_dense_subclass_dense_subclass`, `mk_subclass_dense_subclass_dense`, `__torch_function__`, `__torch_function__`, `__torch_function__`, `__torch_function__`, `__init__`, `__torch_function__`, `__torch_function__`, `__new__`, `__init__`, `__tensor_flatten__`, `__tensor_unflatten__`

**Key imports**: functools, itertools, unittest, partial, torch, torch._dynamo.test_case, torch._dynamo.testing, torch._functorch.config, torch.utils._pytree as pytree, torch.utils.checkpoint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `itertools`
- `unittest`
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch._functorch.config`
- `torch.utils._pytree as pytree`
- `torch.utils.checkpoint`
- `torch._dynamo.backends.common`: aot_autograd
- `torch._functorch._aot_autograd.utils`: make_boxed_compiler
- `torch._functorch.compilers`: min_cut_rematerialization_partition
- `torch._higher_order_ops.wrap`: wrap
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton
- `torch.testing._internal.two_tensor`: TwoTensor
- `torch.utils._python_dispatch`: return_and_correct_aliasing
- `types`: MethodWrapperType
- `torch._dynamo.exc`
- `dataclasses`
- `collections`: namedtuple
- `typing`: Any
- `abc`
- `torch.fx._symbolic_trace`
- `collections.abc`: Callable
- `torch._dynamo.decorators`: mark_static_address
- `torch._inductor.compile_fx`: compile_fx


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/test_subclasses.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_subclasses.py_docs.md`
- **Keyword Index**: `test_subclasses.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

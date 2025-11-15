# Documentation: `test/test_dynamic_shapes.py`

## File Metadata

- **Path**: `test/test_dynamic_shapes.py`
- **Size**: 173,535 bytes (169.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
import contextlib
import copy
import itertools
import math
import operator
import unittest

import numpy as np
import pytest
import sympy

import torch
import torch.fx
import torch.nn.functional as F
from torch import sym_int, SymBool, SymFloat, SymInt
from torch._C import _disabled_torch_function_impl
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend
from torch._inductor.utils import fresh_cache
from torch.fx.experimental import sym_node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.sym_node import method_to_operator, SymNode, to_node
from torch.fx.experimental.symbolic_shapes import (
    _constrain_range_for_size,
    DimConstraints,
    DimDynamic,
    expect_true,
    guard_bool,
    guard_float,
    guard_int,
    GuardOnDataDependentSymNode,
    has_free_symbols,
    hint_int,
    is_symbolic,
    ShapeEnv,
    StatelessSymbolicContext,
    statically_known_false,
    statically_known_true,
)
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._sympy.functions import (
    CleanDiv,
    FloorDiv,
    IsNonOverlappingAndDenseIndicator,
    Mod,
)


aten = torch.ops.aten

meta_funcs = {}


def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f

        pytree.tree_map_(add_func, op)
        return f

    return decorator


@register_meta([aten.add.Tensor, aten.sub.Tensor])
def binary_meta(a, b):
    return a.new_empty(a.shape)


@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    concat_length = 0
    shape = tensors[0].shape
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length
    new_shape = list(shape)
    new_shape[dim] = concat_length
    return tensors[0].new_empty(new_shape)


@register_meta([aten.narrow_copy.default])
def narrow_copy_symint_meta(a, dim, start, length, **kwargs):
    shape = []
    for i, x in enumerate(a.shape):
        if i == dim:
            shape.append(length)
        else:
            shape.append(x)
    return a.new_empty(tuple(shape))


@register_meta([aten.expand.default])
def expand_symint_meta(a, size, implicit=False):
    return a.new_empty(size)


def create_contiguous(shape):
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))


class FakeSymbolicTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        sym_shape,
        sym_strides,
        dtype,
        layout,
        requires_grad,
        device,
        storage_offset=0,
    ):
        # TODO: this is wrong in general
        sym_stride = create_contiguous(sym_shape)
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sym_shape,
            sym_stride,
            storage_offset,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            device=device,
        )
        return r

    __torch_function__ = _disabled_torch_function_impl

    def new_empty(self, shape):
        return FakeSymbolicTensor(
            shape, None, self.dtype, self.layout, self.requires_grad, self.device
        )

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        if func_overload in meta_funcs:
            return meta_funcs[func_overload](*args, **kwargs)

        if func_overload == torch.ops.aten.new_empty.default:
            self = args[0]
            shape = args[1]
            return FakeSymbolicTensor(
                shape,
                self.stride(),
                self.dtype,
                self.layout,
                self.requires_grad,
                self.device,
            )

        raise RuntimeError(f"operator {func_overload} not supported")


def create_symbolic_tensor(name, arg, shape_env, source=None, dynamic_dims=None):
    from torch._dynamo.source import ConstantSource

    if source is None:
        source = ConstantSource(name)
    constraint_dims = [None] * arg.dim()
    if dynamic_dims is None:
        dynamic_dims = [DimDynamic.DUCK] * arg.dim()
    (
        sym_shapes,
        sym_strides,
        sym_storage_offset,
    ) = shape_env.create_symbolic_sizes_strides_storage_offset(
        arg,
        source=source,
        symbolic_context=StatelessSymbolicContext(
            dynamic_sizes=dynamic_dims, constraint_sizes=constraint_dims
        ),
    )
    return FakeSymbolicTensor(
        sym_shapes,
        sym_strides,
        arg.dtype,
        arg.layout,
        arg.requires_grad,
        arg.device,
        sym_storage_offset,
    )


def create_fake_tensor_with_dynamic_size(x, shape_env, dynamic_sizes, dynamic_strides):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(shape_env=shape_env) as fake_mode:
        return fake_mode.from_tensor(
            x,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=dynamic_sizes,
                dynamic_strides=dynamic_strides,
            ),
        )


def create_symtype(cls, pytype, shape_env, val, duck=True, **kwargs):
    from torch._dynamo.source import ConstantSource

    symbol = shape_env.create_symbol(
        val,
        source=ConstantSource(f"__testing_only{len(shape_env.var_to_val)}"),
        dynamic_dim=DimDynamic.DUCK if duck else DimDynamic.DYNAMIC,
        constraint_dim=None,
        **kwargs,
    )
    return cls(SymNode(symbol, shape_env, pytype, hint=val))


# TODO: default duck to False
def create_symint(shape_env, i: int, duck=True, **kwargs) -> SymInt:
    return create_symtype(SymInt, int, shape_env, i, duck=duck, **kwargs)


def create_symbool(shape_env, b: bool) -> SymBool:
    return create_symtype(SymBool, bool, shape_env, b)


def create_symfloat(shape_env, f: float) -> SymFloat:
    return create_symtype(SymFloat, float, shape_env, f)


@skipIfTorchDynamo(
    "Creating ShapeEnv fails for confusing reasons (also we never expect dynamo to see code like this)"
)
class TestPySymInt(TestCase):
    def test_arith_ops(self):
        shape_env = ShapeEnv()
        symints = []
        for i in range(2, 5):
            symints.append((i, create_symint(shape_env, i)))

        ops = [
            operator.add,
            operator.sub,
            operator.floordiv,
            operator.mul,
            operator.mod,
        ]

        for op in ops:
            for args in itertools.permutations(symints, 2):
                if not isinstance(args[0][1], int) and (
                    (op != operator.mod or op != operator.floordiv) and args[1][0] != 0
                ):
                    self.assertTrue(
                        op(args[0][1], args[1][1]) == op(args[0][0], args[1][0])
                    )

    def test_reverse_arith_ops(self):
        shape_env = ShapeEnv()

        a = create_symint(shape_env, 2)
        self.assertTrue(5 // a == 5 // 2)

        a = create_symint(shape_env, 2)
        self.assertTrue(5 * a == 5 * 2)

    def test_sympify_symint(self):
        shape_env = ShapeEnv()
        a = create_symint(shape_env, 2)
        self.assertIs(sympy.sympify(a), a.node.expr)
        b = create_symfloat(shape_env, 3.0)
        self.assertIs(sympy.sympify(b), b.node.expr)
        c = create_symbool(shape_env, True)
        self.assertIs(sympy.sympify(c), c.node.expr)

    def test_roundtrip(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)

        self.assertTrue(not isinstance(x.shape[0], SymNode))
        self.assertTrue(isinstance(x.shape[0], SymInt))

        self.assertTrue(x.shape[0] == 5)
        self.assertTrue(x.shape[1] == 4)
        self.assertTrue(x.shape[2], 3)

        self.assertTrue(x.size()[0], 5)
        self.assertTrue(x.size()[1], 4)
        # Should be simplifiable to an integer.
        # Ref: https://github.com/pytorch/pytorch/pull/107492
        self.assertTrue(isinstance(x.size()[1], SymInt))
        self.assertTrue(
            isinstance(x.size()[1].node.maybe_as_int(), int)
        )  # due to guard above
        self.assertTrue(x.size()[2] == 3)

        self.assertTrue(x.size(0) == 5)
        self.assertTrue(x.size(1) == 4)
        self.assertTrue(x.size(2) == 3)
        self.assertTrue(isinstance(x.size(2), SymInt))
        self.assertTrue(isinstance(x.size(2).node.maybe_as_int(), int))

        y = create_symbolic_tensor("y", torch.randn(5, 4, 3)[1:], shape_env)
        self.assertTrue(isinstance(y.storage_offset(), SymInt))
        self.assertTrue(y.storage_offset() == 12)

    def test_binary(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(5, 4, 3), shape_env)

        z = x + y
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # broadcasting
        y = create_symbolic_tensor("y2", torch.randn(1, 4, 1), shape_env)
        z = x + y
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

    def test_symint_args(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(5, 4, 1), shape_env)
        LAST_DIM = 2
        z = x.narrow_copy(LAST_DIM, 0, y.shape[LAST_DIM])
        self.assertTrue(z.shape[2] == y.shape[2])

        # arithmetic expr with two symints
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - y.shape[LAST_DIM])
        self.assertTrue(z.shape[2] == 2)

        # arithmetic expr with a symint and python int
        z = x.narrow_copy(LAST_DIM, 0, x.shape[LAST_DIM] - 1)
        self.assertTrue(z.shape[2] == 2)

    def test_symint_vargs(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 4, 3), shape_env)
        y = create_symbolic_tensor("y", torch.randn(1, 4, 1), shape_env)

        # varargs
        z = y.expand(x.shape[0], y.shape[1], x.shape[2])
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # shape list
        z = y.expand((x.shape[0], y.shape[1], x.shape[2]))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints
        z = y.expand(x.shape[0], y.shape[1], 3)
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints in a list
        z = y.expand((x.shape[0], y.shape[1], 3))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python symints and ints
        z = y.expand(5, y.shape[1], x.shape[2])
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        # mixed python ints and symints in a list
        z = y.expand((5, y.shape[1], x.shape[2]))
        self.assertTrue(z.shape[0] == 5)
        self.assertTrue(z.shape[1] == 4)
        self.assertTrue(z.shape[2] == 3)

        z = y.expand((y.shape[1],))
        z = y.expand(y.shape[1])

    def test_symint_bitwise_and(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_and = a0 & b0
        self.assertEqual(res_and, 0b1000)
        self.assertIsInstance(res_and, torch.SymInt, msg=type(res_and))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_and(s97, s26), 8)"""
        )

        a1 = create_symint(shape_env, 3)
        b1 = create_symbool(shape_env, True)
        self.assertEqual(a1 & b1, 1)

        a2 = create_symint(shape_env, 0b1100)
        self.assertEqual(a2 & 0b1010, 0b1000)

        a3 = create_symbool(shape_env, True)
        b3 = create_symbool(shape_env, True)
        self.assertEqual(a3 & b3, True)

    def test_symint_bitwise_or(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_or = a0 | b0
        self.assertEqual(res_or, 0b1110)
        self.assertIsInstance(res_or, torch.SymInt, msg=type(res_or))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_or(s97, s26), 14)"""
        )

    def test_symint_bitwise_xor(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 0b1100)
        b0 = create_symint(shape_env, 0b1010)
        res_xor = a0 ^ b0
        self.assertEqual(res_xor, 0b0110)
        self.assertIsInstance(res_xor, torch.SymInt, msg=type(res_xor))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(BitwiseFn_bitwise_xor(s97, s26), 6)"""
        )

    def test_stride(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5, 5), shape_env)
        self.assertIsInstance(x.stride()[0], SymInt)

    def test_size_expressions(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        expand_x = x.expand(x.shape[0], x.shape[0])
        if expand_x.shape[0] > 3:
            result = expand_x + expand_x
        else:
            result = expand_x + expand_x

        gt_op, _bt, is_size_obv = shape_env.guards[-1]
        self.assertTrue(isinstance(gt_op, sympy.core.relational.StrictGreaterThan))
        self.assertTrue(str(x.shape[0]), str(gt_op.args[0]))
        self.assertTrue(str(expand_x.shape[1]), str(x.shape[0]))
        self.assertTrue(str(expand_x.shape[1]), str(result.shape[0]))
        self.assertFalse(is_size_obv)

    def test_floordiv_static(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 8)
        # This was extracted from
        # python test/inductor/test_cuda_cpp_wrapper.py -k
        # DynamicShapesCudaWrapperCudaTests.test_insignificant_strides_cuda_dynamic_shapes_cuda_wrapper
        bool(s0 % 2 == 0)
        bool(s0 % (s0 // 2) == 0)
        bool(2 * (s0 // 2) == s0)
        self.assertTrue(statically_known_true(s0 // (s0 // 2) == 2))

    def test_numel(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        self.assertIsInstance(x.numel(), torch.SymInt)
        self.assertIsInstance(torch.numel(x), torch.SymInt)

        x = torch.rand(3, 3)
        self.assertIsInstance(x.numel(), int)
        self.assertIsInstance(torch.numel(x), int)

    def test_int_to_float(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        r = torch.sym_float(x.shape[0])
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))

    def test_aten_ops(self):
        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x", torch.randn(5), shape_env)
        torch.ops.aten.narrow_copy.default(x, 0, 0, x.shape[0])

        shape_env = ShapeEnv()
        x = create_symbolic_tensor("x2", torch.randn(5, 4, 3), shape_env)
        torch.ops.aten.expand.default(x, [x.shape[0], x.shape[1], x.shape[2]])

    def test_fx_trace_intlist(self):
        class CustomModule(torch.nn.Module):
            def forward(self, x):
                bs, c, h, w = x.shape
                return F.pad(x, (0, w % 2, 0, h % 2, 0, 0))

        m = CustomModule()
        x = torch.rand(1, 3, 4, 4)
        # should not TypeError: pad(): argument 'pad' (position 2) must be
        # tuple of ints, not tuple
        torch.fx.symbolic_trace(m)

    def test_meta_symint(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        r = torch.empty(a0, device="meta")
        self.assertIsInstance(r.shape[0], SymInt)

    def test_guard_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(a0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

    def test_sym_sum(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)
        self.assertEqual(
            (s0 + s1 + s2).node.expr, torch.sym_sum([s0, s1, s2]).node.expr
        )

    def test_prefer_deferred_runtime_assertions_over_guards(self):
        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        s0 = create_symint(shape_env, 2)
        self.assertEqual(guard_int(s0), 2)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

        shape_env = ShapeEnv(prefer_deferred_runtime_asserts_over_guards=True)
        s0 = create_symint(shape_env, 2)
        self.assertTrue(expect_true(s0 == 2))
        self.assertEqual(len(shape_env.guards), 0)
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[None]]),
            """[Eq(s97, 2)]""",
        )

    def test_sym_int(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = sym_int(a0)
        self.assertEqual(r, 5)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 5)""")

        a1 = create_symint(shape_env, 7)
        r = sym_int(a1 / 2)
        self.assertEqual(guard_int(r), 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]), """Eq(TruncToInt(IntTrueDiv(s26, 2)), 3)"""
        )

        a3 = create_symint(shape_env, 3)
        r = sym_int(2.0 * torch.sym_float(a3))
        self.assertEqual(guard_int(r), 6)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[2][0]), """Eq(TruncToInt(2.0*ToFloat(s57)), 6)"""
        )

    def test_sym_log2(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = torch._sym_log2(a0)
        self.assertEqual(r, 2.0)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_log2(ToFloat(s97)), 2.0)"""
        )

    def test_sym_sqrt(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 4)
        r = torch._sym_sqrt(a0)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymFloat, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(OpaqueUnaryFn_sqrt(ToFloat(s97)), 2.0)"""
        )

    def test_sym_floor(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.floor(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(FloorToInt(IntTrueDiv(s97, 2)), 2)""",
        )
        r = math.floor(3.0 * a0)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s97)), 15)""",
        )

    def test_sym_trunc(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.trunc(a0 / 2)
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]), """Eq(TruncToInt(IntTrueDiv(s97, 2)), 2)"""
        )
        r = torch.sym_int(torch.sym_sqrt(a0))
        self.assertEqual(r, 2)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(TruncToInt(OpaqueUnaryFn_sqrt(ToFloat(s97))), 2)""",
        )

    def test_sym_ceil(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = math.ceil(a0 / 2)
        self.assertEqual(r, 3)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(CeilToInt(IntTrueDiv(s97, 2)), 3)""",
        )
        r1 = 3.0 * a0
        r = math.floor(r1)
        self.assertEqual(r, 15)
        self.assertIsInstance(r, torch.SymInt, msg=type(r))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(FloorToInt(3.0*ToFloat(s97)), 15)""",
        )

    def test_sym_ite(self):
        shape_env = ShapeEnv()
        t = create_symint(shape_env, 5)
        f = create_symint(shape_env, 4)
        b1 = True
        r1 = torch.sym_ite(b1, t, f)
        self.assertTrue(r1 is t)
        b2 = False
        r2 = torch.sym_ite(b2, t, f)
        self.assertTrue(r2 is f)
        b3 = t == 5
        r3 = torch.sym_ite(b3, t, f)
        self.assertEqual(len(shape_env.guards), 0)
        self.assertEqual(r3, 5)
        self.assertEqual(type(t), type(r3))
        self.assertExpectedInline(
            str(shape_env.guards[0][0]),
            """Eq(Piecewise((s97, Eq(s97, 5)), (s26, True)), 5)""",
        )
        b4 = f == 5
        r4 = torch.sym_ite(b4, t, f)
        self.assertEqual(len(shape_env.guards), 1)
        self.assertEqual(r4, 4)
        self.assertEqual(type(f), type(r4))
        self.assertExpectedInline(
            str(shape_env.guards[1][0]),
            """Eq(Piecewise((s97, Eq(s26, 5)), (s26, True)), 4)""",
        )

    def test_tracing_sym_ite(self):
        def f(x):
            b = x.shape[0] == 5
            ret = torch.sym_ite(b, x.shape[0], x.shape[1])
            return ret

        gm = make_fx(f, tracing_mode="symbolic")(torch.ones(4, 5))
        self.assertEqual(len(gm.shape_env.guards), 0)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 5
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
    sym_ite = torch.sym_ite(eq, sym_size_int, sym_size_int_1);  eq = sym_size_int = sym_size_int_1 = None
    return sym_ite""",
        )
        r1 = gm(torch.ones(4, 5))
        self.assertIsInstance(r1, int)
        self.assertEqual(r1, 5)
        r2 = gm(torch.ones(5, 4))
        self.assertIsInstance(r2, int)
        self.assertEqual(r2, 5)

    def test_int_conversion(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        int(a0)
        self.assertExpectedInline(str(shape_env.guards[0][0]), """Eq(s97, 2)""")

    def test_data_dependent_guard(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        self.assertRaises(GuardOnDataDependentSymNode, lambda: bool(s0 == 0))

    def test_data_dependent_guard_propagate_real_tensors(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        shape_env.set_unbacked_var_to_val(s0.node.expr, 0)
        self.assertEqual(bool(s0 == 0), True)

    def test_expect_true_basic(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i0_sym = i0.node.expr
        # This doesn't error
        self.assertTrue(expect_true(i0 == 0))
        # This generates a deferred runtime assert via replacement
        self.assertEqual(shape_env.replacements[i0_sym], 0)
        # After expecting true, guards now resolve given the runtime assert
        bool(i0 == 0)

    def test_expect_true_with_s0(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5)
        i0 = shape_env.create_unbacked_symint()
        self.assertTrue(expect_true(i0 < s0))
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i0.node.expr]]),
            """[u0 < s97]""",
        )
        self.assertTrue(i0 < s0)
        self.assertTrue(i0 != s0)
        self.assertFalse(i0 > s0)
        self.assertFalse(i0 >= s0)

    def test_expect_true_prefer_later(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        i1_sym = i1.node.expr
        self.assertTrue(expect_true(i0 + i1 == 10))
        # Importantly, this is put in i1, not i0!
        self.assertExpectedInline(
            str([ra.expr for ra in shape_env.deferred_runtime_asserts[i1_sym]]),
            """[Eq(u0 + u1, 10)]""",
        )
        self.assertTrue(i0 + i1 == 10)
        # NB: We currently don't support deriving that we can substitute
        # i0 + i1 with 10; maybe we should, but this means our rewriting
        # system is no longer confluent (it's probably OK though, because
        # you're unlikely to get other equalities like this on the
        # unbacked SymInts.)

    def test_unbacked_substitution(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        i1 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i0)
        _constrain_range_for_size(i1)
        self.assertTrue(expect_true(i0 == i1 * 4))
        self.assertExpectedInline(str(i0), """u0""")

        i2 = shape_env.create_unbacked_symint()
        i3 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i2)
        _constrain_range_for_size(i3)
        self.assertTrue(expect_true(i2 * 4 == i3))
        self.assertExpectedInline(str(i3), """u3""")

    def test_avoid_unbacked_substitution(self):
        shape_env = ShapeEnv()
        i0 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i0)
        i1 = shape_env.create_unbacked_symint()
        _constrain_range_for_size(i1)
        self.assertTrue(expect_true(i0 == 10 - i1))
        self.assertExpectedInline(str(i0), """u0""")

    def test_expect_true_double_digits(self):
        shape_env = ShapeEnv()
        ia = [shape_env.create_unbacked_symint() for _ in range(11)]  # allocate 10
        self.assertEqual(str(ia[-1]), "u10")
        self.assertTrue(expect_true(sum(ia) == 20))
        self.assertEqual(len(shape_env.deferred_runtime_asserts[ia[-1].node.expr]), 1)

    def test_expect_true_refine_range(self):
        shape_env = ShapeEnv()
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = shape_env.create_unbacked_symint()
                self.assertTrue(expect_true(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertFalse(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 > 4))
                self.assertTrue(statically_known_true(i0 >= 5))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = shape_env.create_unbacked_symint()
                self.assertTrue(expect_true(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 2))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 < 4))
                self.assertTrue(statically_known_true(i0 <= 5))

    def test_guard_refine_range(self):
        shape_env = ShapeEnv()
        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 10, duck=False)
                self.assertTrue(bool(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertFalse(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 > 4))
                self.assertTrue(statically_known_true(i0 >= 5))

        for i, rel in enumerate(
            [lambda x: x > 4, lambda x: 4 < x, lambda x: x >= 5, lambda x: 5 <= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 2, duck=False)
                self.assertFalse(bool(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertFalse(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 != 6))
                self.assertTrue(statically_known_true(i0 <= 4))
                self.assertTrue(statically_known_true(i0 < 5))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 2, duck=False)
                self.assertTrue(bool(rel(i0)))
                self.assertFalse(statically_known_true(i0 != 2))
                self.assertFalse(statically_known_true(i0 != 3))
                self.assertTrue(statically_known_true(i0 != 4))
                self.assertTrue(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 < 4))
                self.assertTrue(statically_known_true(i0 <= 3))

        for i, rel in enumerate(
            [lambda x: x < 4, lambda x: 4 > x, lambda x: x <= 3, lambda x: 3 >= x]
        ):
            with self.subTest(f"i = {i}"):
                i0 = create_symint(shape_env, 10, duck=False)
                self.assertFalse(bool(rel(i0)))
                self.assertTrue(statically_known_true(i0 != 2))
                self.assertTrue(statically_known_true(i0 != 3))
                self.assertFalse(statically_known_true(i0 != 4))
                self.assertFalse(statically_known_true(i0 != 5))
                self.assertTrue(statically_known_true(i0 >= 4))
                self.assertTrue(statically_known_true(i0 > 3))

    def test_mul_int_oo_nan(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 5, duck=False)
        s1 = create_symint(shape_env, 6, duck=False)
        s2 = create_symint(shape_env, 5, duck=False)
        bool(s0 * (s1 // s0) == s2)

    def test_non_overlapping_and_dense_backed(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 5)
        r = torch.empty_strided((a0, 7), (1, a0), device="meta")
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(r))

    def test_non_overlapping_and_dense_unbacked(self):
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        cf = torch.ops.aten.is_non_overlapping_and_dense.default

        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 2, 2, 1), 1)
        self.assertEqual(IsNonOverlappingAndDenseIndicator(2, u0.node.expr, 1, 2), 1)
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))

        self.assertEqual(IsNonOverlappingAndDenseIndicator(u0.node.expr, 1), 1)
        self.assertEqual(IsNonOverlappingAndDenseIndicator(1, u0.node.expr), 1)
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        # NB: This only works because we're able to determine this tensor is
        # contiguous. transpose(0, 1) makes it stop working
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )

    def test_prims_non_overlapping_and_dense(self):
        shape_env = ShapeEnv()
        cf = torch._prims_common.is_non_overlapping_and_dense

        # backed case
        a0 = create_symint(shape_env, 5)
        self.assertTrue(cf(torch.empty_strided((a0, 7), (1, a0), device="meta")))

        # unbacked
        u0 = shape_env.create_unbacked_symint()
        self.assertTrue(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")))
        self.assertTrue(cf(torch.empty_strided((2, u0), (1, 2), device="meta")))
        self.assertTrue(cf(torch.empty_strided((u0,), (1,), device="meta")))
        self.assertTrue(cf(torch.empty_strided((1,), (u0,), device="meta")))

        Max = torch.sym_max
        self.assertTrue(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            )
        )
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (Max(1, u0), Max(1, u0), 1, 3 * Max(1, u0)),
                    device="meta",
                )
            )
        )

        # return False on arbitrary strides
        u1 = shape_env.create_unbacked_symint()
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2 * u0, u0, 1),
                    (u1, u0, u0 + u1),
                    device="meta",
                )
            )
        )
        self.assertFalse(
            cf(
                torch.empty_strided(
                    (2, 3, u0),
                    (u1, 3, 1),
                    device="meta",
                )
            )
        )

    def test_sympy_optimized_add_binary_search(self):
        import sympy

        from torch.fx.experimental.sym_node import _binary_search_insert_arg

        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        c = sympy.Symbol("c")

        args = []
        args = _binary_search_insert_arg([], b)
        self.assertEqual(args, [b])

        self.assertEqual(_binary_search_insert_arg(args, b), None)

        args = _binary_search_insert_arg(args, a)
        self.assertEqual(args, [a, b])

        self.assertEqual(_binary_search_insert_arg(args, b), None)
        self.assertEqual(_binary_search_insert_arg(args, a), None)

        args = _binary_search_insert_arg(args, c)
        self.assertEqual(args, [a, b, c])

        self.assertEqual(_binary_search_insert_arg(args, a), None)
        self.assertEqual(_binary_search_insert_arg(args, b), None)
        self.assertEqual(_binary_search_insert_arg(args, c), None)

        a1 = sympy.Symbol("a1")
        a2 = sympy.Symbol("a2")

        args = _binary_search_insert_arg(args, a1)
        self.assertEqual(args, [a, a1, b, c])

        args = _binary_search_insert_arg(args, a2)
        self.assertEqual(args, [a, a1, a2, b, c])

        c1 = sympy.Symbol("c1")
        args = _binary_search_insert_arg(args, c1)
        self.assertEqual(args, [a, a1, a2, b, c, c1])

        # insert to front
        _a = sympy.Symbol("_a")
        args = _binary_search_insert_arg(args, _a)
        self.assertEqual(args, [_a, a, a1, a2, b, c, c1])

    def test_floor_clean_div_axioms(self):
        # Test that if we add an axiom that have FloorDiv, after which the
        # shapeEnv changed such that it can be simplified it to CleanDiv, then
        # We still correctly replace CleanDiv with the axiom value of FloorDiv.
        shape_env = ShapeEnv()
        a = shape_env.create_unbacked_symint()

        shape_env.guard_or_defer_runtime_assert((a // 3 == 1).node.expr, " test")

        from sympy import Eq

        test1 = Eq(FloorDiv(a.node.expr, 3), 1)
        test2 = Eq(CleanDiv(a.node.expr, 3), 1)

        self.assertTrue(shape_env.evaluate_expr(test1))
        self.assertEqual(shape_env._maybe_evaluate_static(test2), None)

        # After this FloorDiv(a, 3) is simplified to CleanDiv(a, 3)
        shape_env.guard_or_defer_runtime_assert(Eq(Mod(a, 3), 0), " test")
        self.assertEqual(test2, shape_env.simplify(test1))

        self.assertTrue(shape_env.evaluate_expr(test1))
        self.assertTrue(shape_env.evaluate_expr(test2))

    def test_sympy_optimized_add(self):
        shape_env = ShapeEnv()
        s0 = create_symint(shape_env, 2)
        s1 = create_symint(shape_env, 3)
        s2 = create_symint(shape_env, 4)
        sum = s0 + s1

        self.assertTrue(sum.node._optimized_summation)

        def assert_optimized(sym):
            self.assertTrue(sym.node._optimized_summation)

        def assert_not_optimized(sym):
            self.assertFalse(getattr(sym.node, "_optimized_summation", False))

        assert_optimized(sum)

        # add duplicate symbol
        assert_not_optimized(sum + s0)

        # add constant.
        assert_not_optimized(sum + 1)

        # add new unique symbol, should maintain _optimized_summation property.
        assert_optimized(sum + s2)

        assert_optimized(s2 + sum)

        # add x + (a+b) with no  _optimized_summation on the rhs sum.
        a = create_symint(shape_env, 10)
        b = create_symint(shape_env, 11)
        two_sum = torch.sym_sum([a, b])
        assert_not_optimized(two_sum)
        assert_optimized(sum + two_sum)

        # adding two expressions of length >2 that are _optimized_summation.
        a = s0 + s1 + s2
        s3 = create_symint(shape_env, 10)
        s4 = create_symint(shape_env, 20)
        s5 = create_symint(shape_env, 30)
        b = s3 + s4 + s5
        assert_optimized(a)
        assert_optimized(b)
        assert_not_optimized(a + b)
        assert_not_optimized(b + a)
        assert_not_optimized(b + a + b)

    def test_max_of_unique_summation_opt(self):
        shape_env = ShapeEnv()
        s0 = shape_env.create_unbacked_symint()
        s1 = shape_env.create_unbacked_symint()
        s2 = shape_env.create_unbacked_symint()
        s3 = shape_env.create_unbacked_symint()
        s4 = shape_env.create_unbacked_symint()
        s5 = shape_env.create_unbacked_symint()
        s7 = shape_env.create_unbacked_symint()

        def assert_optimized(sym):
            self.assertTrue(sym.node.expr.unique_summations_symbols is not None)

        def assert_not_optimized(sym):
            getattr(sym.node.expr, "unique_summations_symbols", None)

        mx1 = torch.sym_max(s0, s1)
        assert_not_optimized(mx1)

        mx2 = torch.sym_max(s0 + s1, s2 + s3)
        assert_optimized(mx2)

        mx3 = torch.sym_max(mx2, s4 + s5)
        assert_optimized(mx3)
        assert_optimized(torch.sym_max(s4 + s5, mx2))

        assert_not_optimized(torch.sym_max(mx3, s7))
        assert_not_optimized(torch.sym_max(mx3, 10))
        assert_not_optimized(torch.sym_max(mx3, s3 + s7))
        assert_not_optimized(torch.sym_max(mx3, s7 * 2))

    def test_sym_max_multi_max_simplify(self):
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        self.assertTrue(
            statically_known_true(
                torch.sym_max(1, torch.sym_max(257, u0)) == torch.sym_max(257, u0)
            )
        )

    def test_numpy_sym_max(self):
        self.assertEqual(torch.sym_max(np.int64(10), 12), 12)
        self.assertEqual(torch.sym_max(np.int64(12), 10), 12)
        self.assertEqual(torch.sym_max(np.int64(10), 12.5), 12.5)
        self.assertEqual(torch.sym_max(np.int64(14), 12.5), 14.0)
        self.assertEqual(torch.sym_max(np.float64(14.0), 12), 14.0)
        self.assertEqual(torch.sym_max(np.float64(14.0), 16), 16.0)

    def test_numpy_sym_min(self):
        self.assertEqual(torch.sym_min(np.int64(10), 12), 10)
        self.assertEqual(torch.sym_min(np.int64(12), 10), 10)
        self.assertEqual(torch.sym_min(np.int64(10), 12.5), 10.0)
        self.assertEqual(torch.sym_min(np.int64(14), 12.5), 12.5)
        self.assertEqual(torch.sym_min(np.float64(14.0), 12), 12.0)
        self.assertEqual(torch.sym_min(np.float64(14.0), 16), 14.0)

    def test_debug_has_internal_overlap_unbacked(self):
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint()
        cf = torch._debug_has_internal_overlap
        self.assertEqual(cf(torch.empty_strided((u0, 2), (2, 1), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 2), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((u0,), (1,), device="meta")), 0)
        self.assertEqual(cf(torch.empty_strided((1,), (u0,), device="meta")), 2)
        Max = torch.sym_max
        self.assertEqual(
            cf(
                torch.empty_strided(
                    (2, 3, 1, u0),
                    (3 * Max(1, u0), Max(1, u0), Max(1, u0), 1),
                    device="meta",
                )
            ),
            2,
        )

        # Wobbling these to zero is OK too
        self.assertEqual(cf(torch.empty_strided((u0, 2), (3, 1), device="meta")), 2)
        self.assertEqual(cf(torch.empty_strided((2, u0), (1, 3), device="meta")), 2)

    def test_specialize_zero_one(self):
        shape_env = ShapeEnv(specialize_zero_one=True)
        a0 = create_symint(shape_env, 5)
        assert a0 != 1
        self.assertEqual(len(shape_env.guards), 0)

        shape_env = ShapeEnv(specialize_zero_one=False)
        a0 = create_symint(shape_env, 5)
        assert a0 != 1
        self.assertEqual(len(shape_env.guards), 1)

    def test_duck_shape(self):
        shape_env = ShapeEnv(duck_shape=True)
        a0 = create_symint(shape_env, 5)
        a1 = create_symint(shape_env, 5)
        assert a0 == a1
        self.assertEqual(len(shape_env.guards), 0)

        shape_env = ShapeEnv(duck_shape=False)
        a0 = create_symint(shape_env, 5)
        a1 = create_symint(shape_env, 5)
        assert a0 == a1
        self.assertEqual(len(shape_env.guards), 1)

    def test_int_bool(self):
        # See https://github.com/pytorch/pytorch/issues/95981
        shape_env = ShapeEnv(duck_shape=True)
        a0 = create_symint(shape_env, 5)
        assert a0
        self.assertEqual(len(shape_env.guards), 0)

    def test_symint_as_scalar(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)

        sym_int_encountered = False

        class TestSymInt(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                assert func == torch.ops.aten.add.Tensor

                nonlocal sym_int_encountered
                # WARNING: do not do identity tests on the outer
                # SymInt/SymFloat, they are NOT STABLE
                sym_int_encountered = kwargs["alpha"].node is a0.node
                kwargs["alpha"] = 0
                return func(*args)

        x = torch.rand([4, 4])
        with TestSymInt():
            y = torch.add(x, x, alpha=a0)

        self.assertTrue(sym_int_encountered)

    def test_deepcopy(self):
        shape_env = ShapeEnv()
        a0 = create_symint(shape_env, 2)
        assert a0 < 4
        new_shape_env = copy.deepcopy(shape_env)
        self.assertEqual(len(new_shape_env.guards), 1)

    def test_print_readable_with_symints(self):
        def f(a, b):
            dim0 = a.shape[0] + b.shape[0]
            dim1 = a.shape[1] + b.shape[1]
            d = a.new_empty(dim0, dim1)
            d = torch.ops.aten.native_dropout(d, 0.5, train=True)
            return d

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5, 3), torch.randn(4, 3))
        out = fx_g.print_readable(print_output=False)

        self.assertExpectedInline(
            out.strip(),
            """\
class f(torch.nn.Module):
    def forward(self, a_1: "f32[s75, s96]", b_1: "f32[s57, s96]"):
        # No stacktrace found for following nodes
        sym_size_int: "Sym(s75)" = torch.ops.aten.sym_size.int(a_1, 0)
        sym_size_int_1: "Sym(s57)" = torch.ops.aten.sym_size.int(b_1, 0)
        add: "Sym(s57 + s75)" = sym_size_int + sym_size_int_1;  sym_size_int = sym_size_int_1 = None
        sym_size_int_2: "Sym(s96)" = torch.ops.aten.sym_size.int(a_1, 1)
        sym_size_int_3: "Sym(s96)" = torch.ops.aten.sym_size.int(b_1, 1);  b_1 = None
        add_1: "Sym(2*s96)" = sym_size_int_2 + sym_size_int_3;  sym_size_int_2 = sym_size_int_3 = None
        new_empty: "f32[s57 + s75, 2*s96]" = torch.ops.aten.new_empty.default(a_1, [add, add_1], pin_memory = False);  a_1 = add = add_1 = None
        native_dropout = torch.ops.aten.native_dropout.default(new_empty, 0.5, True);  new_empty = None
        getitem: "f32[s57 + s75, 2*s96]" = native_dropout[0]
        getitem_1: "b8[s57 + s75, 2*s96]" = native_dropout[1];  native_dropout = None
        return (getitem, getitem_1)""",  # noqa: B950
        )

    def test_statically_known_true(self):
        shape_env = ShapeEnv()
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # Statically known true
        self.assertTrue(statically_known_true(True))
        self.assertTrue(statically_known_true(s2 == s2))
        self.assertTrue(statically_known_true(s2 * s3 > s3))
        self.assertTrue(statically_known_true(s3 * s4 > s4))
        self.assertTrue(statically_known_true((s3 + s3) % 2 == 0))

        # Statically known false
        self.assertFalse(statically_known_true(False))
        self.assertFalse(statically_known_true(s3 * s4 <= s4))
        self.assertFalse(statically_known_true((s3 + s3) % 2 == 1))

        # True for hints, but not known statically
        self.assertFalse(statically_known_true(s2 + s2 == s4))
        self.assertFalse(statically_known_true(s4 % s2 == 0))
        self.assertFalse(statically_known_true(s2 != s3))
        self.assertFalse(statically_known_true(s3 * s4 > s2))

        # False for hints, but not known statically
        self.assertFalse(statically_known_true(s2 == s3))
        self.assertFalse(statically_known_true(s2 > s3))
        self.assertFalse(statically_known_true(s3 + s3 == s4))

        # No guards should be generated
        self.assertEqual(len(shape_env.guards), 0)

    def test_statically_known_false(self):
        shape_env = ShapeEnv()
        s2, s3, s4 = (create_symint(shape_env, i) for i in range(2, 5))

        # Statically known true
        self.assertFalse(statically_known_false(True))
        self.assertFalse(statically_known_false(s2 == s2))
        self.assertFalse(statically_known_false(s2 * s3 > s3))
        self.assertFalse(statically_known_false(s3 * s4 > s4))
        self.assertFalse(statically_known_false((s3 + s3) % 2 == 0))

        # Statically known false
        self.assertTrue(statically_known_false(False))
        self.assertTrue(statically_known_false(s3 * s4 <= s4))
        self.assertTrue(statically_known_false((s3 + s3) % 2 == 1))

        # True for hints, but not known statically
        self.assertFalse(statically_known_false(s2 + s2 == s4))
        self.assertFalse(statically_known_false(s4 % s2 == 0))
        self.assertFalse(statically_known_false(s2 != s3))
        self.assertFalse(statically_known_false(s3 * s4 > s2))

        # False for hints, but not known statically
        self.assertFalse(statically_known_false(s2 == s3))
        self.assertFalse(statically_known_false(s2 > s3))
        self.assertFalse(statically_known_false(s3 + s3 == s4))

        # No guards should be generated
        self.assertEqual(len(shape_env.guards), 0)

    def test_ephemeral_source_simplification(self):
        from torch._dynamo.source import EphemeralSource

        # For full robustness, ensure the ephemeral source symbols are simplified out regardless
        # of construction order or check order.
        for construct_ephemeral_first, x_first_in_check in itertools.product(
            [False, True], [False, True]
        ):
            shape_env = ShapeEnv()
            shape = (5, 10)
            dynamic_dims = [DimDynamic.DYNAMIC for _ in shape]
            x = create_symbolic_tensor(
                "x",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if construct_ephemeral_first else None),
                dynamic_dims=dynamic_dims,
            )
            y = create_symbolic_tensor(
                "y",
                torch.randn(*shape),
                shape_env,
                source=(EphemeralSource() if not construct_ephemeral_first else None),
                dynamic_dims=dynamic_dims,
            )
            t_with_ephemeral = x if construct_ephemeral_first else y

            def _get_ephemeral_source_symbols(t):
                return [
                    s.node.expr
                    for s in itertools.chain(t.shape, t.stride(), (t.storage_offset(),))
                    if isinstance(s, torch.SymInt)
                    and s.node.expr in shape_env.var_to_sources
                    and any(
                        source.is_ephemeral()
                        for source in shape_env.var_to_sources[s.node.expr]
                    )
                ]

            # the
```



## High-Level Overview


This Python file contains 13 class(es) and 256 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FakeSymbolicTensor`, `TestPySymInt`, `CustomModule`, `TestSymInt`, `f`, `TestSymNumberMagicMethods`, `Foo`, `TestFloorDiv`, `TestDimConstraints`, `TestGuardsExpressions`, `TestUnbacked`, `TestUbackedOps`, `M`

**Functions defined**: `register_meta`, `decorator`, `add_func`, `binary_meta`, `cat_meta`, `narrow_copy_symint_meta`, `expand_symint_meta`, `create_contiguous`, `__new__`, `new_empty`, `__torch_dispatch__`, `create_symbolic_tensor`, `create_fake_tensor_with_dynamic_size`, `create_symtype`, `create_symint`, `create_symbool`, `create_symfloat`, `test_arith_ops`, `test_reverse_arith_ops`, `test_sympify_symint`

**Key imports**: contextlib, copy, itertools, math, operator, unittest, numpy as np, pytest, sympy, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `itertools`
- `math`
- `operator`
- `unittest`
- `numpy as np`
- `pytest`
- `sympy`
- `torch`
- `torch.fx`
- `torch.nn.functional as F`
- `torch._C`: _disabled_torch_function_impl
- `torch._dynamo.testing`: CompileCounter, CompileCounterWithBackend
- `torch._inductor.utils`: fresh_cache
- `torch.fx.experimental`: sym_node
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.fx.experimental.sym_node`: method_to_operator, SymNode, to_node
- `torch.testing._internal.common_dtype`: all_types_and
- `torch.testing._internal.logging_utils`: logs_to_string
- `torch.utils`: _pytree as pytree
- `torch.utils._python_dispatch`: TorchDispatchMode
- `torch._dynamo.source`: ConstantSource
- `torch._subclasses.fake_tensor`: FakeTensorMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/test_dynamic_shapes.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_dynamic_shapes.py_docs.md`
- **Keyword Index**: `test_dynamic_shapes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

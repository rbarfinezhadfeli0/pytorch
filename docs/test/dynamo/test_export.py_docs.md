# Documentation: test_export.py

## File Metadata
- **Path**: `test/dynamo/test_export.py`
- **Size**: 156906 bytes
- **Lines**: 4646
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_export_persist_assert)
"""

import copy
import functools
import inspect
import io
import operator
import unittest
from collections.abc import Sequence
from enum import Enum
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.experimental.control_flow import cond
from torch._dynamo import config
from torch._dynamo.exc import UserError
from torch._dynamo.testing import normalize_gm
from torch._higher_order_ops.out_dtype import out_dtype
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests


@torch._dynamo.assume_constant_result
def dynamo_assume_constant_result_global_function():
    return "test"


class ExportTests(torch._dynamo.test_case.TestCase):
    # TODO(voz): Refactor to a shared test function.
    # The tests in this file are a little redundant,
    # They all take a func, run it with eager, then export it, then compare
    def test_export(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for _ in range(4):
                bar2 = []
                for _ in range(3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        def func():
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
            state = [
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            i = torch.tensor(
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            return pre_attention_state_ops(i, mems, state)

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func()

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)()
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_no_tensor_computation_fail(self):
        with self.assertRaisesRegex(
            AssertionError,
            "Failed to produce a graph",
        ):
            inp = [torch.randn(3)]
            inp2 = 2
            inps = [inp, inp2]

            def func(x, y):
                return x

            torch._dynamo.export(func, same_signature=False)(*inps)

    def test_no_tensor_computation(self):
        inp = [torch.randn(3)]
        inp2 = 2
        inps = [inp, inp2]

        def func(x, y):
            return x

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        self.assertExpectedInline(
            out_graph.code.strip(),
            """\
def forward(self, x, y):
    arg0, arg1, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
    x = arg0
    return pytree.tree_unflatten([x], self._out_spec)""",
        )

    def test_no_tensor_computation_2(self):
        inp = torch.randn(3)
        inp2 = 2
        inps = [inp, inp2]

        def func(x, y):
            return y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        self.assertExpectedInline(
            out_graph.code.strip(),
            """\
def forward(self, x, y):
    arg0, arg1, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
    x = arg0
    return pytree.tree_unflatten([2], self._out_spec)""",
        )

    def test_export_mismatched_out(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(torch.tensor([[[1.3737, 0.1]]]))
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_shape_control_flow_1(self):
        def func(x):
            if x.shape[0] > 10:
                return x.cos()
            return x.sin()

        opt_func = torch.compile(func, backend="eager")
        real_result = opt_func(torch.ones(6, 4))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(torch.ones(6, 4))
        out_graph, out_guards = exported

        dynamo_result = out_graph(torch.ones(6, 4))

        from torch._guards import GuardSource

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        hit = False
        for guard in out_guards:
            if guard.source == GuardSource.SHAPE_ENV:
                hit = True
                self.assertExpectedInline(
                    guard.code_list,
                    """["L['x'].stride()[0] == L['x'].size()[1]", "L['x'].stride()[1] == 1", "L['x'].storage_offset() == 0", "2 <= L['x'].size()[0] and L['x'].size()[0] <= 10", "2 <= L['x'].size()[1]"]""",  # noqa: B950
                )
                break

        self.assertTrue(hit)

    def test_export_control_flow_with_getattr(self):
        class Animal(Enum):
            COW = "moo"

        class MyModule(torch.nn.Module):
            def __init__(self, a):
                super().__init__()
                self.a = a

            def forward(self, x):
                if self.a == Animal.COW.value:
                    return x * x
                else:
                    raise ValueError("bad")

        module = MyModule("moo")
        input = (torch.ones(4, 3),)
        resA = module(*input)
        graph, _ = torch._dynamo.export(module)(*input)
        resB = graph(*input)
        self.assertTrue(torch._dynamo.utils.same(resA, resB))

    def test_export_graph_bypass(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_unpack(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return x[0], first * second, x[1], x[2]

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_shallow_list_copy_wo_side_effects(self):
        def f(x):
            y = x.copy()
            return y[0] + y[1]

        inp = [torch.tensor([1.3, 3.77, 0.1]), torch.tensor([8.7, 6.23, 9.9])]
        gm = torch._dynamo.export(f, aten_graph=True, tracing_mode="symbolic")(
            inp
        ).graph_module
        self.assertTrue(torch._dynamo.utils.same(gm(inp), f(inp)))

    def test_export_with_shallow_list_copy_with_side_effects(self):
        def f(x):
            y = x.copy()
            x[0] = x[1]
            y.append(torch.tensor([[100]]))
            return x[0] + x[1], y[0] + y[1], y[2]

        inp = [torch.tensor([1.3, 3.77, 0.1]), torch.tensor([8.7, 6.23, 9.9])]
        gm = torch._dynamo.export(f, aten_graph=True, tracing_mode="symbolic")(
            inp
        ).graph_module
        res = gm(inp)
        ref = f(inp)
        self.assertTrue(torch._dynamo.utils.same(res, ref))
        self.assertEqual(res[0], res[1])

    def test_export_mismatched_out_2(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(torch.tensor([[[1.3737, 0.1]]]))
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_list(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second, x

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_complex_reorder(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[0]
            second = x[1]
            third = x[2]
            return third, first, second, first * second, first * third

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_immutable_list_dict(self):
        class M(torch.nn.Module):
            def forward(self, x1, x2):
                return [x1 + x2], {"moo1": x1 * x1, "moo2": x2 * x2}

        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        model = M()

        fx_model = make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[x1, x2])
        ep = torch.export.export(fx_model, (x1, x2))
        res = torch.compile(ep.module(), dynamic=True, fullgraph=True)(x1, x2)
        self.assertTrue(torch._dynamo.utils.same(res, M()(x1, x2)))

    def test_dupes(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_2(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        def func(x, z):
            y = x + 1
            return y, y, z

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y, y, z

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_reorder_with_non_tensor_arg(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return z, y, y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_dupes_and_bypass_with_non_tensor_output(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y[0].item(), y, z

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return a[0].item() + b[0].item() + c[0].item()

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out_permute(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return b[0].item() + c[0].item() + a[0].item() + a[0].item()

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_zeroes_in_new_shape_scalar_out_permute_dupe_and_bypass(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return a, b[0].item() + c[0].item() + a[0].item() + a[0].item(), a

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_func_return(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dict_return(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_aten_graph(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for _ in range(4):
                bar2 = []
                for _ in range(3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        def func():
            mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
            state = [
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
                torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            ]
            i = torch.tensor(
                [
                    [0.0313, -0.1487, -0.3846, -0.5321],
                    [-1.7073, 1.3331, -0.0890, -1.4935],
                    [-0.8314, -0.1862, -0.5935, 1.5232],
                ]
            )
            return pre_attention_state_ops(i, mems, state)

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func()

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)()
        out_graph = exported[0]

        dynamo_result = out_graph()
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_no_tensor_computation_with_aten_graph(self):
        inp = [torch.randn(3)]
        inp2 = 2
        inps = [inp, inp2]

        def func(x, y):
            return x

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        self.assertExpectedInline(
            out_graph.code.strip(),
            """\
def forward(self, x, y):
    arg0, arg1, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
    arg0_1 = arg0
    return pytree.tree_unflatten([arg0_1], self._out_spec)""",
        )

    def test_no_tensor_computation_2_with_aten_graph(self):
        inp = torch.randn(3)
        inp2 = 2
        inps = [inp, inp2]

        def func(x, y):
            return y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))
        self.assertExpectedInline(
            out_graph.code.strip(),
            """\
def forward(self, x, y):
    arg0, arg1, = fx_pytree.tree_flatten_spec(([x, y], {}), self._in_spec)
    arg0_1 = arg0
    return pytree.tree_unflatten([2], self._out_spec)""",
        )

    def test_export_mismatched_out_with_aten_graph(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(
            torch.tensor([[[1.3737, 0.1]]])
        )
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_bypass_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_unpack_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return x[0], first * second, x[1], x[2]

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_mismatched_out_2_with_aten_graph(self):
        def func(x):
            y = x + 1
            return ([x, x], (y, y))

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(torch.tensor([[[1.3737, 0.1]]]))

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(
            torch.tensor([[[1.3737, 0.1]]])
        )
        out_graph = exported[0]

        dynamo_result = out_graph(torch.tensor([[[1.3737, 0.1]]]))

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_list_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[2]
            second = x[2]
            return first * second, x

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_graph_with_complex_reorder_with_aten_graph(self):
        inp = [
            torch.tensor([0.1, 0.1]),
            torch.tensor([0.2, 0.2]),
            torch.tensor([0.3, 0.3]),
            torch.tensor([0.4, 0.4]),
        ]

        def func(x):
            first = x[0]
            second = x[1]
            third = x[2]
            return third, first, second, first * second, first * third

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_2_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            y = x + 1
            return y, y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(inp)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]

        dynamo_result = out_graph(inp)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.4, 0.4])
        inps = [inp, inp2]

        def func(x, z):
            y = x + 1
            return y, y, z

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_with_non_tensor_arg_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y, y, z

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dupes_and_bypass_reorder_with_non_tensor_arg_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return z, y, y

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_dupes_and_bypass_with_non_tensor_output_with_aten_graph(self):
        inp = torch.tensor([0.1, 0.1])
        inp2 = torch.tensor([0.1, 0.1])
        inp3 = 4
        inps = [inp, inp2, inp3]

        def func(x, z, k):
            y = x + k
            return y[0].item(), y, z

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_zeroes_in_and_out_different_shape_on_test_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            return [[a], [b, c], [a + b], [[c + c]]]

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_func_return_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c

            def func2(y):
                return x * y

            return func2(x)

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_dict_return_with_aten_graph(self):
        inp = torch.zeros(10)
        inp2 = torch.zeros(10)
        inp3 = torch.zeros(10)
        inps = [inp, inp2, inp3]

        inps_rand = [torch.randn(10), torch.randn(10), torch.randn(10)]

        def func(a, b, c):
            x = a + b + c
            return {"a": x}

        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps_rand)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps_rand)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_stack_trace(self):
        inp = torch.randn(4, 4)

        class MyBlock(torch.nn.Module):
            def forward(self, x):
                x = torch.nn.functional.linear(x, torch.randn(4, 4))
                return torch.cos(x).relu() + 1

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = MyBlock()

            def forward(self, x):
                out = self.block(x)
                return out

        exported = torch._dynamo.export(MyModule(), aten_graph=False)(inp)
        out_graph = exported[0]

        for node in out_graph.graph.nodes:
            if node.op not in {"placeholder", "output"}:
                self.assertTrue(node.stack_trace is not None)
                self.assertTrue(node.meta["nn_module_stack"] is not None)
                self.assertTrue(node.meta["source_fn_stack"] is not None)

        torch._dynamo.reset()

        exported = torch._dynamo.export(MyModule(), aten_graph=True)(inp)
        out_graph = exported[0]
        for node in out_graph.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(node.stack_trace is not None)
                self.assertTrue(node.meta["nn_module_stack"] is not None)
                self.assertTrue(node.meta["source_fn_stack"] is not None)
                self.assertTrue(node.meta["val"] is not None)
                self.assertTrue(node.meta["original_aten"] is not None)

    def test_export_preserves_nn_module_stack_for_get_attr(self):
        inp = torch.randn(4, 4)

        class MyBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 1))
                self.buffer = torch.nn.Buffer(torch.ones(1, 1))

            def forward(self, x):
                x = torch.nn.functional.linear(x, torch.randn(4, 4))
                return torch.cos(x).relu() + self.weight + self.buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = MyBlock()

            def forward(self, x):
                out = self.block(x)
                return out

        m = MyModule()
        exported = torch._dynamo.export(m, aten_graph=False)(inp)
        out_graph = exported[0]

        attr_access_count = 0
        for node in out_graph.graph.nodes:
            if node.op == "get_attr":
                attr_access_count += 1
                self.assertTrue(node.meta["nn_module_stack"] is not None)
        self.assertEqual(attr_access_count, 2)

        torch._dynamo.reset()

        exported = torch._dynamo.export(m, aten_graph=True)(inp)
        out_graph = exported[0]

        attr_access_count = 0
        for node in out_graph.graph.nodes:
            if node.op == "get_attr":
                attr_access_count += 1
                self.assertTrue(node.meta["nn_module_stack"] is not None)
        self.assertEqual(attr_access_count, 2)

    def test_export_compare_optimize_with_make_fx(self):
        inp = torch.tensor([0.1, 0.1])
        linear = torch.nn.Linear(2, 2)

        def func(x):
            x = x + 1
            y = x.t()
            y = y.relu()
            y = linear(y)
            return y

        exported = torch._dynamo.export(func, aten_graph=True)(inp)
        out_graph = exported[0]
        export_result = out_graph(inp)

        torch._dynamo.reset()

        def compiler(gm, sample_inputs):
            def fw(*args):
                aten_gm = make_fx(gm)(*args)
                return aten_gm(*args)

            return fw

        opt_func = torch.compile(func, backend=compiler, fullgraph=True, dynamic=True)
        make_fx_result_through_backend = opt_func(inp)

        fx_g = make_fx(func)(inp)
        make_fx_result_through_direct = fx_g(inp)

        self.assertTrue(
            torch._dynamo.utils.same(make_fx_result_through_backend, export_result)
        )
        self.assertTrue(
            torch._dynamo.utils.same(make_fx_result_through_direct, export_result)
        )

    def test_export_with_constant_method_on_module(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return torch.nonzero(x)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = self.helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_method_on_module_invoke_twice(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return torch.nonzero(x)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = self.helper_fn(x) + self.helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return torch.nonzero(x)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = helper_fn(x) + self.helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_global_function(self):
        class MyModule(torch.nn.Module):
            def forward(self):
                a = dynamo_assume_constant_result_global_function()
                b = dynamo_assume_constant_result_global_function()
                return a + b

        module = MyModule()
        graph, _ = torch._dynamo.export(module)()
        result = graph()
        self.assertEqual(result, "testtest")

    def test_export_with_constant_free_function_and_class_method(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                y = torch.sin(x)
                x = self.linear(x)
                y = helper_fn(x)
                return y

        module = MyModule()
        real_result = module(torch.tensor([[1.0, 0], [0, 0]]))
        module = MyModule()
        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        result = graph(torch.tensor([[1.0, 0.0], [0, 0]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(torch.tensor([[1, 0], [0.25, 0.25]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method_multiarg(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, z):
                y = torch.sin(x)
                x = self.linear(x)
                y = helper_fn(x) + helper_fn(z)
                return y

        module = MyModule()
        real_result = module(
            torch.tensor([[1.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        module = MyModule()
        graph, _ = torch._dynamo.export(module)(
            torch.tensor([[0.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        result = graph(
            torch.tensor([[1.0, 0.0], [0, 0]]), torch.tensor([[1.0, 0.0], [0, 0]])
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(
            torch.tensor([[1, 0], [0.25, 0.25]]), torch.tensor([[1, 0], [0.25, 0.25]])
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_free_function_and_class_method_multiarg_diff(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return torch.nonzero(x)

        class MyModule(torch.nn.Module):
            def forward(self, x, z):
                y = helper_fn(x) + helper_fn(z)
                return y

        module = MyModule()
        real_result = module(
            torch.tensor([[1.0, 0], [0, 0]]), torch.tensor([[1.0, 0], [0, 0]])
        )
        module = MyModule()
        graph, _ = torch._dynamo.export(module)(
            torch.tensor([[0.0, 0], [0, 0]]), torch.tensor([[0.0, 0], [0.5, 0]])
        )
        result = graph(
            torch.tensor([[1.0, 0.0], [0, 0]]), torch.tensor([[0.0, 1.0], [0, 0]])
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))
        result = graph(
            torch.tensor([[1, 0], [0.25, 0.25]]),
            torch.tensor([[0.33, 0.33], [0.25, 0.25]]),
        )
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_tuple_nonzero(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return (torch.nonzero(x), torch.nonzero(x))

            def forward(self, x):
                y = torch.tensor([0.5])
                elements = self.helper_fn(x)
                all_y = []
                for element in elements:
                    for item in element:
                        all_y.append(y * item)
                return all_y

        module = MyModule()
        real_result = module(torch.tensor([1.0, 1.0]))
        graph, _ = torch._dynamo.export(module)(torch.tensor([1.0, 1.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_list_nonzero(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return [torch.nonzero(x), torch.nonzero(x)]

            def forward(self, x):
                y = torch.tensor([0.5])
                elements = self.helper_fn(x)
                all_y = []
                for element in elements:
                    for item in element:
                        all_y.append(y * item)
                return all_y

        module = MyModule()
        real_result = module(torch.tensor([1.0, 1.0]))
        graph, _ = torch._dynamo.export(module)(torch.tensor([1.0, 1.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_list_nonzero_free_function(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            return [torch.nonzero(x), torch.nonzero(x)]

        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([0.5])
                elements = helper_fn(x)
                all_y = []
                for element in elements:
                    for item in element:
                        all_y.append(y * item)
                return all_y

        module = MyModule()
        real_result = module(torch.tensor([1.0, 1.0]))
        graph, _ = torch._dynamo.export(module)(torch.tensor([1.0, 1.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_dict_values(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return {"x": x, "x^2": x * x}

            def forward(self, x):
                y = torch.tensor([0.5])
                elements = self.helper_fn(x)
                y = y * elements["x"]
                y = y * elements["x^2"]
                return y

        module = MyModule()
        real_result = module(torch.tensor([2.0, 2.0]))
        graph, _ = torch._dynamo.export(module)(torch.tensor([2.0, 2.0]))

        # Tensor input can be almost anything here, and the result will capture what we
        # made constant at compile time.
        result = graph(torch.tensor([[[1.0, 0], [0, 0]], [[1.0, 0], [0, 0]]]))
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_none_control_flow(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([-1]))

        # X is negative, so .item() < 0, which means we return y
        self.assertEqual(real_result, torch.tensor([0.5]))

        graph, _ = torch._dynamo.export(module)(torch.tensor([-1]))
        result = graph(torch.tensor([2]))
        # X is positive, but we compiled helper_fn to return None, so it will still return y
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([2]))

        # X is positive, so .item() > 0, which means we return y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        graph, _ = torch._dynamo.export(module)(torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X is negative, but we compiled helper_fn to return x, so it will still return y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_none_control_flow_free_func(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            if x.item() < 0:
                return None
            else:
                return x

        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([0.5])
                x = helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([-1]))

        # X is negative, so .item() < 0, which means we return y
        self.assertEqual(real_result, torch.tensor([0.5]))

        graph, _ = torch._dynamo.export(module)(torch.tensor([-1]))
        result = graph(torch.tensor([2]))
        # X is positive, but we compiled helper_fn to return None, so it will still return y
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow_pos(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                if x.item() < 0:
                    return None
                else:
                    return x

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([2]))

        # X is positive, so .item() > 0, which means we return y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        graph, _ = torch._dynamo.export(module)(torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X is negative, but we compiled helper_fn to return x, so it will still return y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_none_control_flow_free_func(self):
        @torch._dynamo.assume_constant_result
        def helper_fn(x):
            if x.item() < 0:
                return None
            else:
                return x

        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor([0.5])
                x = helper_fn(x)
                if x is None:
                    return y
                return y * x

        module = MyModule()
        real_result = module(torch.tensor([2]))

        # X is positive, so .item() > 0, which means we return y * x
        self.assertEqual(real_result, torch.tensor([1.0]))

        graph, _ = torch._dynamo.export(module)(torch.tensor([2]))
        result = graph(torch.tensor([-0.5]))
        # X is negative, but we compiled helper_fn to return x, so it will still return y * x
        self.assertTrue(torch._dynamo.utils.same(result, real_result))

    def test_export_with_constant_not_return_const(self):
        class MyModule(torch.nn.Module):
            @torch._dynamo.assume_constant_result
            def helper_fn(self, x):
                return self.val

            def forward(self, x):
                y = torch.tensor([0.5])
                x = self.helper_fn(x)
                if x == "A":
                    return y
                return -1

        module = MyModule()
        module.val = "A"
        resA = module(torch.tensor([2]))
        graph, _ = torch._dynamo.export(module)(torch.tensor([2]))
        module.val = "B"
        resB = graph(torch.tensor([2]))
        self.assertTrue(torch._dynamo.utils.same(resA, resB))

    def test_export_with_builtin_op_on_assume_constant(self):
        @torch._dynamo.assume_constant_result
        def get_y(y) -> torch.Tensor:
            return y

        class Bob(torch.nn.Module):
            def __init__(self, p, val) -> None:
                super().__init__()
                self.p = p
                self.y = torch.nn.Parameter(torch.tensor(val))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # This only looks dynamic but it's actually a constant value
                if get_y(self.y) < self.p:
                    return torch.cat([x, x])
                else:
                    return x

        model = Bob(0.5, 0.3)
        inp = torch.ones(3, 4)
        graph, _ = torch._dynamo.export(model)(inp)
        self.assertEqual(model(inp), graph(inp))

    def test_export_with_constant_in_unspecialized_nn_module(self):
        class Module(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.y = y

            @torch._dynamo.assume_constant_result
            def check(self):
                return self.y[0].item() == 1

            def forward(self, x):
                # This line leads to module obj being tracked as UnspecializedNNModuleVariable in dynamo
                self.device = x.device

                if self.check():
                    return x + 1
                else:
                    return x + 2

        model = Module(torch.tensor([1]))
        inp = torch.ones(3, 4)
        graph, _ = torch._dynamo.export(model)(inp)
        self.assertEqual(model(inp), graph(inp))

    def test_export_decomp(self):
        def f(x):
            return x.t() + x.t()

        def nop(x):
            return x.cos()

        graph, _ = torch._dynamo.export(
            f,
            aten_graph=True,
            decomposition_table={torch.ops.aten.t.default: nop},
        )(torch.randn(5))
        self.assertEqual(
            len([n for n in graph.graph.nodes if n.target == torch.ops.aten.t.default]),
            0,
        )

        graph, _ = torch._dynamo.export(f, aten_graph=True, decomposition_table=None)(
            torch.randn(5)
        )
        self.assertEqual(
            len([n for n in graph.graph.nodes if n.target == torch.ops.aten.t.default]),
            2,
        )

    def test_export_decomp_asserts_bad_args(self):
        def f(x):
            return x.t() + x.t()

        def nop(x):
            return x.cos()

        with self.assertRaises(AssertionError):
            torch._dynamo.export(
                f,
                (torch.randn(5)),
                aten_graph=False,
                decomposition_table={torch.ops.aten.t.default: nop},
            )

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_module_layer(self):
        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, pred, x):
                def true_fn(val):
                    return self.linear(val) * torch.tensor(2)

                def false_fn(val):
                    return self.linear(val) * torch.tensor(-1)

                return cond(pred, true_fn, false_fn, [x])

        mod = Module()
        x = torch.randn([3, 3])
        pred = torch.tensor(x[0][0].item() < 0)
        real_result = mod.forward(pred, x)

        torch._dynamo.reset()

        exported = torch._dynamo.export(mod.forward)(pred, x)
        out_graph = exported[0]

        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

        # New X, just to show we did not specialize
        x = x * -1
        pred = torch.tensor(x[0][0].item() < 0)
        real_result_2 = mod.forward(pred, x)
        dynamo_result_2 = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result_2, dynamo_result_2))

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_cond_branches_calling_methods(self):
        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            # ok
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def t(self, val):
                return val + 1

            def f(self, val):
                return val - 1

            def true_fn(self, val):
                return self.linear(val) + self.t(val)

            def false_fn(self, val):
                return self.linear(val) - self.f(val)

            def forward(self, pred, x):
                return cond(pred, self.true_fn, self.false_fn, [x])

        mod = Module()
        x = torch.randn([3, 3])
        pred = torch.tensor(x[0][0].item() < 0)
        real_result = mod.forward(pred, x)
        out_graph, _ = torch._dynamo.export(mod.forward)(pred, x)
        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    @config.patch(capture_scalar_outputs=True)
    def test_export_with_cond_closure(self):
        from functorch.experimental.control_flow import cond

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, pred, x):
                def true_fn(x):
                    return x * 2

                def false_fn(x):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x])

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, pred, x):
                def true_fn(x):
                    return x * 2

                def false_fn(x):
                    return x - 2

                return cond(pred, true_fn, false_fn, [x + 1])

        class FooBar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, pred, x):
                y = x + x

                def true_fn(x, y):
                    return self.linear(x) * (x + y)

                def false_fn(x, y):
                    return x * (y - x)

                return cond(pred, true_fn, false_fn, [x, y])

        for Module in [Foo, Bar, FooBar]:
            mod = Module()
            x = torch.randn([3, 3], requires_grad=True)
            pred = torch.tensor(x[0][0].item() < 0)
            real_result = mod.forward(pred, x)
            out_graph, _ = torch._dynamo.export(mod.forward)(pred, x)
            dynamo_result = out_graph(pred, x)
            self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_cond_with_closed_function(self):
        def hello(x):
            return x + 1

        def hi(x):
            return x + 2

        def foo(pred, x):
            def true_fn(x):
                return hello(x)

            def false_fn(x):
                return hi(x)

            return cond(pred, true_fn, false_fn, [x])

        x = torch.randn(5)
        pred = x[0] > 0
        real_result = foo(pred, x)
        out_graph, _ = torch._dynamo.export(foo)(pred, x)
        dynamo_result = out_graph(pred, x)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_with_cond_dynamic_shape_pred(self):
        from functorch.experimental.control_flow import cond

        class Module(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return x + x

                def false_fn(x):
                    return x[:2].clone()

                return cond(x.shape[0] <= 2, true_fn, false_fn, [x])

        class Module2(torch.nn.Module):
            def forward(self, x):
                def true_fn(x):
                    return x + x

                def false_fn(x):
                    return x[:2].clone()

                return cond(x.shape[0] <= 2, true_fn, false_fn, (x,))

        mods = [Module(), Module2()]
        for mod in mods:
            x = torch.randn(2, 2)
            out_graph, _ = torch._dynamo.export(mod)(x)
            self.assertExpectedInline(
                out_graph.code.strip(),
                """\
def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    l_x_ = arg0
    sym_size_int = torch.ops.aten.sym_size.int(l_x_, 0)
    le = sym_size_int <= 2;  sym_size_int = None
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(le, cond_true_0, cond_false_0, (l_x_,));  le = cond_true_0 = cond_false_0 = l_x_ = None
    getitem_3 = cond[0]
    sym_size_int_1 = torch.ops.aten.sym_size.int(getitem_3, 0);  getitem_3 = None
    ge = sym_size_int_1 >= 2;  sym_size_int_1 = None
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 2 on node 'ge'");  ge = _assert_scalar_default = None
    getitem_2 = cond[0];  cond = None
    return pytree.tree_unflatten([getitem_2], self._out_spec)""",  # noqa: B950
            )
            self.assertExpectedInline(
                out_graph.cond_true_0.code.strip(),
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    add = l_x__1 + l_x__1;  l_x__1 = None
    return (add,)""",
            )
            self.assertExpectedInline(
                out_graph.cond_false_0.code.strip(),
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    getitem = l_x__1[slice(None, 2, None)];  l_x__1 = None
    clone = getitem.clone();  getitem = None
    return (clone,)""",
            )
            # We could successfully export branches that return different sizes
            torch._dynamo.export(mod)(torch.randn(3, 2))

            # We specialize into one of the branches since predicate is a python boolean.
            test_x = torch.randn(3, 2)
            mod(test_x)

    def test_export_with_map_cond(self):
        from functorch.experimental.control_flow import cond, map

        class Module(torch.nn.Module):
            def inner(self, x, pred):
                def true_fn(x):
                    return x + x

                def false_fn(x):
                    return x * x

                return cond(pred, true_fn, false_fn, [x])

            def forward(self, pred, xs):
                def body(x, pred):
                    return self.inner(x, pred)

                return map(body, xs, pred)

        mod = Module()
        x = torch.randn(3, 2, 1)
        pred_x = torch.tensor(True)

        y = torch.randn(4, 3, 2)
        pred_y = torch.tensor(False)
        real_result = mod(pred_y, y)

        out_graph, _ = torch._dynamo.export(mod)(pred_x, x)
        self.assertEqual(real_result, out_graph(pred_y, y))

    def test_export_with_map_zero_sized_tensor(self):
        from functorch.experimental.control_flow import map

        class Module(torch.nn.Module):
            def forward(self, xs):
                def body(x):
                    return x + 1

                return map(body, xs)

        mod = Module()
        xs = torch.randn(0, 2)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Observed exception",
        ):
            torch._dynamo.export(mod)(xs)

    def test_export_meta_val(self):
        def f(x, y, z):
            return x * y + z

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
        )(
            torch.ones(3, 2),
            torch.zeros(3, 2),
            torch.ones(3, 2),
        )
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                self.assertIn("val", node.meta)

    def test_input_container_type(self):
        def f(x: torch.Tensor, y: list[torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"a": x.sum() + sum(y).sum()}

        inp = (torch.randn(6, 5), [torch.randn(6, 5), torch.randn(6, 5)])

        gm, _ = torch._dynamo.export(f, aten_graph=True)(*inp)

        self.assertEqual(gm(*inp), f(*inp))

    @config.patch(assume_static_by_default=False)
    def test_export_symbolic_shape(self):
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.empty(x.shape[0] * 2)

        inp = (torch.randn(6, 5),)
        gm, _ = torch._dynamo.export(f, aten_graph=True)(*inp)

        has_sym_size = False
        for node in gm.graph.nodes:
            if node.target is torch.ops.aten.sym_size.int:
                has_sym_size = True

        self.assertTrue(has_sym_size)

    @config.patch(assume_static_by_default=False)
    def test_dynamic_slicing(self):
        def f(x):
            return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

        gm_aten_mode, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))

        inp = torch.randn(6, 7)
        self.assertEqual(gm_aten_mode(inp).shape, f(inp).shape)

        count = 0
        # aten graph should flatten getitem calls to actual
        # slice kernel call.
        for node in gm_aten_mode.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.slice.Tensor
            ):
                count += 1

        self.assertEqual(count, 2)

        gm_torch_mode, _ = torch._dynamo.export(f, aten_graph=False)(torch.randn(4, 5))

        # In torch mode, the graph should contain 3 getitem methods
        # one for x.shape[0]-2 and one for x.shape[1]-1 and one for slice
        # this is because Tensor class has its' own getitem method
        # which gets translated to aten.Slice later.
        count = 0
        for node in gm_torch_mode.graph.nodes:
            if node.op == "call_function" and node.target == operator.getitem:
                count += 1

        self.assertEqual(count, 1)
        self.assertEqual(gm_torch_mode(inp).shape, f(inp).shape)

    @config.patch(capture_scalar_outputs=True)
    def test_dynamic_slicing_simple(self):
        def f(x):
            return x[slice(None, None, None)]

        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.randn(4, 5))

        inp = torch.randn(6, 7)
        self.assertEqual(gm(inp), f(inp))

    def test_pre_dispatch_simple(self):
        def f(x):
            y = torch.ones_like(x)
            return torch.matmul(x, y)

        gm, _ = torch._dynamo.export(
            f,
            aten_graph=True,
            pre_dispatch=True,
            tracing_mode="fake",
        )(
            torch.randn(5, 5),
        )

        inp = torch.randn(6, 6)
        self.assertEqual(gm(inp), f(inp))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x):
    arg0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    arg0_1 = arg0
    ones_like = torch.ops.aten.ones_like.default(arg0_1, pin_memory = False)
    matmul = torch.ops.aten.matmul.default(arg0_1, ones_like);  arg0_1 = ones_like = None
    return pytree.tree_unflatten([matmul], self._out_spec)""",
        )

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_export_cond_in_aten_symbolic(self):
        class ConditionOp(torch.nn.Module):
            def true_fn(self, x, y):
                return x * y

            def false_fn(self, x, y):
                return x + y

            def forward(self, pred, x, y):
                return cond(pred, self.true_fn, self.false_fn, [x, y])

        model = ConditionOp()
        inp = (
            torch.tensor(False),
            torch.randn(4, 4),
            torch.randn(4, 4),
        )
        gm, _ = torch._dynamo.export(model, aten_graph=True)(*inp)

        gm.print_readable()

        self.assertEqual(gm(*inp), model(*inp))

    def test_export_with_kwargs(self):
        def fn_with_kwargs(pos0, tuple0, *myargs, mykw0=None, **mykwargs):
            out = pos0
            for arg in tuple0:
                out *= arg
            for arg in myargs:
                out *= arg
            out *= mykw0
            out *= mykwargs["input0"] * mykwargs["input1"]
            return out

        mykwargs = {"input0": torch.randn(4), "input1": torch.randn(4)}
        tuple0 = (torch.randn(4), torch.randn(4))
        mykw0 = torch.randn(4)
        pos0 = torch.randn(4)
        myargs = [torch.randn(4), torch.randn(4)]

        expected_argument_names = [
            "pos0",
            "tuple0",
            "myargs_0",
            "myargs_1",
            "mykw0",
            "input0",
            "input1",
        ]
        self._test_export_preserving_original_signature(
            fn_with_kwargs,
            expected_argument_names,
            pos0,
            tuple0,
            *myargs,
            mykw0=mykw0,
            **mykwargs,
        )

    def test_export_with_kwargs_and_empty_args(self):
        def fn_with_kwargs(mykw0=None, **mykwargs):
            out = mykw0
            out *= mykwargs["input0"] * mykwargs["input1"]
            return out

        mykwargs = {"input0": torch.randn(4), "input1": torch.randn(4)}
        mykw0 = torch.randn(4)

        expected_argument_names = ["mykw0"] + list(mykwargs.keys())
        self._test_export_preserving_original_signature(
            fn_with_kwargs, expected_argument_names, mykw0, **mykwargs
        )

    def test_export_with_args_and_empty_kwargs(self):
        def fn_with_kwargs(pos0, tuple0, *myargs):
            out = pos0
            for arg in tuple0:
                out *= arg
            for arg in myargs:
                out *= arg
            return out

        tuple0 = (torch.randn(4), torch.randn(4))
        pos0 = torch.randn(4)
        myargs = [torch.randn(4), torch.randn(4)]

        expected_argument_names = ["pos0", "tuple0", "myargs_0", "myargs_1"]
        self._test_export_preserving_original_signature(
            fn_with_kwargs, expected_argument_names, pos0, tuple0, *myargs
        )

    @common_utils.parametrize(
        "default_value",
        [
            common_utils.subtest(None, name="None"),
            common_utils.subtest(42.0, name="float"),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                torch.randn(4),
                name="tensor",
                decorators=[unittest.expectedFailure],
            ),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                (torch.randn(4),),
                name="tuple",
                decorators=[unittest.expectedFailure],
            ),
        ],
    )
    def test_export_with_args_with_default(self, default_value):
        def fn(pos0, pos1_default=default_value):
            out = pos0
            if pos1_default is None:
                pos1_default = torch.randn(4)
            if isinstance(pos1_default, tuple):
                pos1_default = pos1_default[0]
            out *= pos1_default
            return out

        pos0 = torch.randn(4)
        expected_argument_names = ["pos0"]
        self._test_export_preserving_original_signature(
            fn, expected_argument_names, pos0
        )

    @common_utils.parametrize(
        "default_value",
        [
            common_utils.subtest(None, name="None"),
            common_utils.subtest(42.0, name="float"),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                torch.randn(4),
                name="tensor",
                decorators=[unittest.expectedFailure],
            ),
            common_utils.subtest(
                # FIXME: AssertionError: Dynamo input and output is a strict subset of traced input/output
                (torch.randn(4),),
                name="tuple",
                decorators=[unittest.expectedFailure],
            ),
        ],
    )
    def test_export_with_kwargs_with_default(self, default_value):
        def fn(pos0, *, kw0, kw1_default=default_value, **kwargs):
            out = pos0
            out += kw0
            if kw1_default is None:
                kw1_default = torch.randn(4)
            elif isinstance(kw1_default, tuple):
                kw1_default = kw1_default[0]
            out += kw1_default
            out += kwargs["kw2"]
            return out

        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        kw2 = torch.randn(4)

        args = (pos0,)
        kwargs = {"kw0": kw0, "kw2": kw2}
        expected_argument_names = ["pos0", "kw0", "kw2"]
        self._test_export_preserving_original_signature(
            fn, expected_argument_names, *args, **kwargs
        )

    def test_export_with_wrapped_fn(self):
        # To ensure dynamo.export is robust to wrapped functions
        # when it cannot use `inspect` to retrieve original signature
        # info.
        def _fn(pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
            out = pos0
            out += pos1
            out += kw0
            out += kw1
            for arg in args:
                out += arg
            for kwarg in kwargs.values():
                out += kwarg
            return out

        def wrapped_fn(*args, **kwargs):
            return _fn(*args, **kwargs)

        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, torch.randn(4), torch.randn(4))
        kwargs = {"kw0": kw0, "kw2": torch.randn(4)}
        expected_argument_names = [f"args_{i}" for i in range(len(args))] + list(
            kwargs.keys()
        )

        self._test_export_preserving_original_signature(
            wrapped_fn, expected_argument_names, *args, **kwargs
        )

    def test_export_with_functools_wrapped_method(self):
        def test_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x

            @test_decorator
            def method_to_test(self, pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
                out = pos0
                out += pos1
                out += kw0
                out += kw1
                for arg in args:
                    out += arg
                for kwarg in kwargs.values():
                    out += kwarg
                return out

        pos0 = torch.randn(4)
        pos1 = torch.randn(4)
        unnamed_pos = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, pos1, unnamed_pos)
        kwargs = {"kw0": kw0, "kw2": torch.randn(4), "unnamed_kw": torch.randn(4)}
        expected_argument_names = [
            "pos0",
            "pos1",
            "args_0",  # 3rd unnamed positional argument
        ] + list(kwargs.keys())
        m = MyModule()

        self._test_export_preserving_original_signature(
            m.method_to_test, expected_argument_names, *args, **kwargs
        )

    def test_export_with_functools_wrapped_fn(self):
        def test_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        @test_decorator
        def _fn(pos0, pos1=1.0, *args, kw0, kw1=2.0, **kwargs):
            out = pos0
            out += pos1
            out += kw0
            out += kw1
            for arg in args:
                out += arg
            for kwarg in kwargs.values():
                out += kwarg
            return out

        def wrapped_fn(*args, **kwargs):
            return _fn(*args, **kwargs)

        pos0 = torch.randn(4)
        kw0 = torch.randn(4)
        args = (pos0, torch.randn(4), torch.randn(4))
        kwargs = {"kw0": kw0, "kw2": torch.randn(4)}
        expected_argument_names = [f"args_{i}" for i in range(len(args))] + list(
            kwargs.keys()
        )

        self._test_export_preserving_original_signature(
            wrapped_fn, expected_argument_names, *args, **kwargs
        )

    def _test_export_preserving_original_signature(
        self, fn, expected_argument_names: Sequence[str], *args, **kwargs
    ):
        torch._dynamo.reset()
        exported = torch._dynamo.export(
            fn,
            *args,
            **kwargs,
            aten_graph=False,
        )

        out_graph = exported[0]
        dynamo_result = out_graph(*args, **kwargs)
        real_result = fn(*args, **kwargs)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

        # Check that the exported graph preserves same argument names.
        self.assertEqual(
            inspect.getfullargspec(out_graph.forward).args[1:], expected_argument_names
        )

    def test_dataclass_input_output(self):
        from dataclasses import dataclass

        @dataclass
        class Tensors:
            x: torch.Tensor
            y: torch.Tensor

        def f(t):
            return t.x + t.y

        with self.assertRaisesRegex(
            UserError,
            "It looks like one of the inputs with type .*Tensors.* "
            "is not supported or pytree-flattenable",
        ):
            torch._dynamo.export(f, aten_graph=False)(
                Tensors(x=torch.randn(10), y=torch.randn(10))
            )

        def f(x, y):
            return Tensors(x=x.sin(), y=y.cos())

        with self.assertRaisesRegex(
            UserError,
            "It looks like one of the outputs with type .*Tensors.* "
            "is not supported or pytree-flattenable",
        ):
            torch._dynamo.export(f, aten_graph=False)(torch.randn(10), torch.randn(10))

    def test_empty(self):
        def f(x):
            return x

        exported = torch._dynamo.export(f)(torch.randn(3, 3))
        out_graph = exported[0]
        inp = torch.randn(3, 3)
        self.assertTrue(torch._dynamo.utils.same(inp, out_graph(inp)))

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(3, 3)

            def forward(self):
                return self.a

        exported = torch._dynamo.export(M())()
        out_graph = exported[0]
        self.assertTrue(torch._dynamo.utils.same(torch.ones(3, 3), out_graph()))

    def test_export_meta(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            def forward(self, x):
                return self.p + x

        with torch.device("meta"):
            m = MyModule()

        inp = torch.ones(2, 3, device="meta")
        exported = torch._dynamo.export(m)(inp)
        out_graph = exported[0]
        dynamo_result = out_graph(inp)
        self.assertEqual(dynamo_result, m(inp))

    def test_constraint_violation_error_messages(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] == x.shape[1] * 2:
                    return x + 1
                else:
                    return x + 2

        foo = Foo()

        t = torch.zeros([8, 4])
        dim0 = torch.export.Dim("dim0", min=3, max=10)
        dim1 = torch.export.Dim("dim1")
        dynamic_shapes = {"x": (dim0, dim1)}

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated .*!(.*\n)*.*"
            "by dim0 = 2\\*dim1(.*\n)*.*"
            "Not all values of dim1 .* satisfy the generated guard 2 <= .* and .* <= 5(.*\n)*.*",
        ):
            torch.export.export(foo, (t,), dynamic_shapes=dynamic_shapes, strict=True)

        class Bar(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] == 5:
                    return x + 1
                else:
                    return x + 2

        bar = Bar()

        t = torch.zeros([5])
        dim0 = torch.export.Dim("dim0", min=3, max=8)
        dynamic_shapes = {"x": (dim0,)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "You marked.*but your code specialized it to be a constant.*"
            "If you're using Dim.DYNAMIC, replace it with either Dim.STATIC or Dim.AUTO",
        ):
            torch.export.export(bar, (t,), dynamic_shapes=dynamic_shapes, strict=True)

        class Qux(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] > 5 and x.shape[0] < 10:
                    return x + 1
                else:
                    return x + 2

        qux = Qux()

        t = torch.zeros([7])
        dim0 = torch.export.Dim("dim0", min=3, max=8)
        dynamic_shapes = {"x": (dim0,)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Not all values.*satisfy the generated guard",
        ):
            torch.export.export(qux, (t,), dynamic_shapes=dynamic_shapes, strict=True)

    def test_untracked_inputs_in_constraints(self):
        from copy import copy

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return y + 1

        foo = Foo()

        x = torch.randn(2)
        y = torch.randn(5, 4)

        dim0_x, dim0_y = torch.export.dims("dim0_x", "dim0_y")
        dynamic_shapes = {"x": {0: dim0_x}, "y": {0: dim0_y}}

        example_inputs = (copy(x), y)
        ep = torch.export.export(
            foo, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
        )
        ep.module()(torch.randn(3), y)  # no specialization error

    def test_export_raise_guard_full_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] == 3:
                return x.sin()
            return x.cos()

        torch._dynamo.export(my_dyn_fn)(y)

        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(
                my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dimx")},)
            )(y)

    def test_export_module_specify_constraints_signature(self):
        y = torch.randn([3, 3, 3])

        class Mod(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] == 3:
                    return x.sin()
                return x.cos()

        mod = Mod()
        torch._dynamo.export(mod)(y)

        with self.assertRaisesRegex(ConstraintViolationError, "dimx = 3"):
            torch._dynamo.export(mod, dynamic_shapes=({0: torch.export.Dim("dimx")},))(
                y
            )

    def test_export_raise_guard_partial_constraint(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            if x.shape[0] > 3:
                return x.sin()
            return x.cos()

        torch._dynamo.export(my_dyn_fn)(y)

        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(
                my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dimx")},)
            )(y)

    def test_export_raise_on_relationship(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[0] == b.shape[1] == c.shape[2]:
                return a.sin()

            return a.cos()

        torch._dynamo.export(my_dyn_fn)(y, y, y)
        dim = torch.export.Dim("dim")
        dynamic_shapes = ({0: dim}, {0: dim}, {0: dim})
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(y, y, y)
        dynamic_shapes = ({0: dim}, {1: dim}, {2: dim})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(y, y, y)

    def test_export_no_raise(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[1] == 3:
                return a.cos()
            return a * b * c

        torch._dynamo.export(my_dyn_fn)(y, y, y)
        dim = torch.export.Dim("dim")
        dynamic_shapes = ({0: dim}, {0: dim}, {0: dim})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(y, y, y)

    def test_export_multi_dynamic_dim_unsafe_relationship(self):
        x = torch.randn([3, 3, 3])
        y = torch.randn([2, 2, 2])
        z = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[0] == c.shape[0]:
                return a.cos()
            return a * c, b

        torch._dynamo.export(my_dyn_fn)(x, y, z)
        dimx, dimy, dimz = torch.export.dims("dimx", "dimy", "dimz")
        dynamic_shapes = ({0: dimx}, {0: dimy}, {0: dimz})
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)
        dimz = dimx
        dynamic_shapes = ({0: dimx}, {0: dimy}, {0: dimz})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)

    def test_remove_redundant_dynamic_dim_in_error_message(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] == y["k"].shape[0]:
                    return x + 1
                else:
                    return x - 1

        foo = Foo()

        a = torch.randn(3)
        b = torch.randn(3)
        dim0_a, dim0_b = torch.export.dims("dim0_a", "dim0_b")
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "dim0_b = dim0_a"):
            torch.export.export(
                foo,
                (a, {"k": b}),
                dynamic_shapes={"x": {0: dim0_a}, "y": {"k": {0: dim0_b}}},
                strict=True,
            )

    def test_enforce_equalities(self):
        class Bar(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        bar = Bar()

        batch, size = torch.export.dims("batch", "size")
        dynamic_shapes = {"x": (batch, size, size), "y": (batch, size, size)}

        x = torch.randn(10, 3, 3)
        y = torch.randn(10, 3, 4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            ".*y.*size.*2.* = 4 is not equal to .*x.*size.*1.* = 3",
        ):
            with torch._export.config.patch(use_new_tracer_experimental=True):
                torch.export.export(
                    bar, (x, y), dynamic_shapes=dynamic_shapes, strict=True
                )
        y = torch.randn(10, 3, 3)
        with torch._export.config.patch(use_new_tracer_experimental=True):
            ebar = torch.export.export(
                bar, (x, y), dynamic_shapes=dynamic_shapes, strict=True
            )

        for node in ebar.graph_module.graph.nodes:
            if node.op == "placeholder":
                shape = node.meta["val"].shape
                self.assertEqual(shape[1], shape[2])

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    def test_export_preserve_constraints_as_metadata_tensor(self):
        def f(x):
            b = x.nonzero()
            torch._check(b.shape[0] >= 2)
            torch._check(b.shape[0] <= 5)
            return b

        y = torch.tensor([8, 8, 6])
        torch._dynamo.export(
            f,
            aten_graph=True,
            tracing_mode="symbolic",
        )(y)

    @config.patch(
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
        capture_scalar_outputs=True,
    )
    def test_exported_graph_serialization(self):
        def f(x, y):
            b = x.item()
            return torch.empty((b, y.shape[0]))

        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = [x, y]
        dynamic_shapes = (None, {0: torch.export.Dim("dimy", min=6, max=10)})
        gm, _ = torch._dynamo.export(
            f,
            dynamic_shapes=dynamic_shapes,
            aten_graph=True,
            tracing_mode="symbolic",
        )(*example_inputs)

        # Ensure the exported graph module with metadata is serializable,
        # metadata won't be saved in the serialized module
        buffer = io.BytesIO()
        torch.save(gm, buffer)

    def test_export_dynamic_dim_not_1(self):
        x = torch.randn([1, 1, 1])

        def my_dyn_fn(a):
            if a.shape[0] != 1:
                return a.cos()
            return a * a

        torch._dynamo.export(my_dyn_fn)(x)
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(
                my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dimx")},)
            )(x)

    def test_symbool(self):
        def f(x):
            a = torch.scalar_tensor(x.shape[0] > 4)
            return x.sin().sum() + a.sum()

        gm, _ = torch._dynamo.export(f, aten_graph=True)(torch.ones(6, 4))
        self.assertEqual(gm(torch.ones(3, 4)), f(torch.ones(3, 4)))

    def test_export_multi_dynamic_dim_constraint(self):
        x = torch.randn([3, 3, 3])
        y = torch.randn([2, 2, 2])
        z = torch.randn([3, 3, 3])

        def my_dyn_fn(a, b, c):
            if a.shape[0] == c.shape[0]:
                return a.cos()
            return a * c, b

        torch._dynamo.export(my_dyn_fn)(x, y, z)
        dimx_0, dimx_1, dimx_2 = torch.export.dims("dimx_0", "dimx_1", "dimx_2")
        dynamic_shapes = ({0: dimx_0, 1: dimx_1, 2: dimx_2}, None, None)
        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)
        dynamic_shapes = ({0: dimx_0, 1: dimx_1, 2: dimx_2}, None, {0: dimx_0})
        torch._dynamo.export(my_dyn_fn, dynamic_shapes=dynamic_shapes)(x, y, z)

    def test_export_dynamic_dim_range_constraint(self):
        x = torch.ones(6, 4, 4)
        dynamic_shapes = ({0: torch.export.Dim("dimx", min=5, max=6)},)

        def foo(x):
            if x.shape[0] > 3:  # ok
                return x.sin()
            return x.cos()

        torch._dynamo.export(
            foo,
            dynamic_shapes=dynamic_shapes,
            aten_graph=True,
        )(x)

        def bar(x):
            if x.shape[0] > 5:  # error
                return x.sin()
            return x.cos()

        with self.assertRaises(ConstraintViolationError):
            torch._dynamo.export(
                bar,
                dynamic_shapes=dynamic_shapes,
                aten_graph=True,
            )(x)

    def test_trivial_constraint(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                # complex divisibility condition
                if (2 * x.shape[0] + 3) % (x.shape[0] - 3) == 0:
                    return x + 1
                else:
                    return x - 1

        foo = Foo()

        class Bar(torch.nn.Module):
            def forward(self, x):
                # trivially true
                if (2 * x.shape[0] + 2) % (x.shape[0] + 1) == 0:
                    return x + 1
                else:
                    return x - 1

        bar = Bar()

        class Qux(torch.nn.Module):
            def forward(self, x):
                # simple divisibility condition (not trivially true)
                if (3 * x.shape[0]) % 2 == 0:
                    return x + 1
                else:
                    return x - 1

        qux = Qux()

        x = torch.randn(12)
        dim0 = torch.export.Dim("dim0", max=100)
        dynamic_shapes = {"x": (dim0,)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Constraints violated \(dim0\)",
        ):
            torch.export.export(foo, (x,), dynamic_shapes=dynamic_shapes, strict=True)

        torch.export.export(bar, (x,), dynamic_shapes=dynamic_shapes, strict=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Constraints violated \(dim0\)",
        ):
            torch.export.export(qux, (x,), dynamic_shapes=dynamic_shapes, strict=True)

    def test_list_contains(self):
        def func(x):
            assert x.size(-1) in [4, 5, 6], "bad"
            return x + x

        inps = (torch.randn(1, 5),)
        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_list_not_contains(self):
        def func(x):
            assert x.size(0) not in [4, 5, 6], "bad1"
            assert "monkey" not in ["cow", "pig"], "bad2"
            return x + x

        inps = (torch.randn(1, 5),)
        opt_func = torch.compile(func, backend="eager", fullgraph=True, dynamic=True)
        real_result = opt_func(*inps)

        torch._dynamo.reset()

        exported = torch._dynamo.export(func, aten_graph=True)(*inps)
        out_graph = exported[0]

        dynamo_result = out_graph(*inps)

        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_export_identity(self):
        inp = torch.tensor([0.1, 0.1])

        def func(x):
            return x

        torch._dynamo.reset()
        exported, _ = torch._dynamo.export(func)(inp)
        dynamo_result = exported(inp)
        self.assertTrue(torch._dynamo.utils.same(inp, dynamo_result))

    def test_export_specialized_int(self):
        class Foo(torch.nn.Module):
            def __init__(
                self,
                input_dim,
            ):
                super().__init__()
                self.torch_module = torch.nn.LayerNorm(
                    input_dim, eps=1e-5, elementwise_affine=True
                )
                self.int_val = 100

            def forward(self, input):
                return input.cos() * self.int_val * self.torch_module.eps

        mod = Foo(128)
        inp = torch.randn(3, 128)

        # In export, int & float in forward should always be specialized
        gm, _ = torch._dynamo.export(mod, aten_graph=True)(inp)
        count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                count += 1
        self.assertEqual(count, 1)

    def test_export_with_nonzero_static(self):
        class BasicModule(torch.nn.Module):
            def __init__(self, static_size):
                super().__init__()
                self.static_size = static_size

            def forward(self, x):
                return torch.nonzero_static(x, size=self.static_size)

        input_tensors = torch.tensor([6, 8]), torch.zeros(2, 3)
        static_sizes = 3, 4
        for input_tensor, static_size in zip(input_tensors, static_sizes):
            m = BasicModule(static_size)
            gm, _ = torch._dynamo.export(m, aten_graph=True)(input_tensor)
            res = gm(input_tensor)
            self.assertEqual(res.size(0), static_size)
            self.assertTrue(
                torch._dynamo.utils.same(
                    res, torch.nonzero_static(input_tensor, size=static_size)
                )
            )

    def test_export_pass_arg_by_name(self):
        class BasicModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.my_lin = torch.nn.Linear(3, 4, bias=True)

            def forward(self, x):
                return self.my_lin(x)

        mod, input_tensor = BasicModule(), torch.randn(2, 3)
        gm, _ = torch._dynamo.export(mod, aten_graph=True)(input_tensor)
        ref = mod(x=input_tensor)
        res = gm(x=input_tensor)
        self.assertTrue(torch._dynamo.utils.same(ref, res))

    def test_export_pass_arg_by_name_star_args(self):
        class BasicModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.my_lin = torch.nn.Linear(3, 4, bias=True)

            def forward(self, *args):
                return self.my_lin(args[0]) * self.my_lin(args[1])

        mod, input_tensor, input_tensor2 = (
            BasicModule(),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )
        gm, _ = torch._dynamo.export(mod, aten_graph=True)(input_tensor, input_tensor2)
        ref = mod(input_tensor, input_tensor2)
        res = gm(input_tensor, input_tensor2)
        self.assertTrue(torch._dynamo.utils.same(ref, res))

    def test_export_dynamic_dim_cleanup(self):
        y = torch.randn([3, 3, 3])

        def my_dyn_fn(x):
            return x.cos()

        torch._dynamo.export(my_dyn_fn, dynamic_shapes=({0: torch.export.Dim("dim")},))(
            y
        )

    @config.patch(capture_dynamic_output_shape_ops=True)
    def test_export_dynamic_control_flow_error(self):
        def f(x):
            if x.nonzero() > 3:
                return x.cos()
            return x.sin()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Data-dependent branching",
        ):
            torch._dynamo.export(f, aten_graph=True)(torch.randn(5, 6))

    @config.patch(assume_static_by_default=False)
    def test_export_persist_assert(self):
        def f(x):
            assert x[0].sum() > 4, "Shape must be more than 4"
            return x.cos() + x.sin()

        gm, _ = torch._dynamo.export(f, aten_graph=True, tracing_mode="symbolic")(
            torch.ones(5, 4, 6)
        )

        def has_aten_op(gm, op):
            for node in gm.graph.nodes:
                if node.target == op:
                    return True
            return False

        self.assertTrue(has_aten_op(gm, torch.ops.aten._assert_async.msg))

        gm.graph.eliminate_dead_code()
        gm.recompile()
   

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 99 class(es): ExportTests, Animal, MyModule, M, MyBlock, MyModule, MyBlock, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule, MyModule

### Functions
This file defines 569 function(s): dynamo_assume_constant_result_global_function, test_export, pre_attention_state_ops, func, test_no_tensor_computation_fail, func, test_no_tensor_computation, func, forward, test_no_tensor_computation_2, func, forward, test_export_mismatched_out, func, test_export_shape_control_flow_1, func, test_export_control_flow_with_getattr, __init__, forward, test_export_graph_bypass, func, test_list_unpack, func, test_export_with_shallow_list_copy_wo_side_effects, f, test_export_with_shallow_list_copy_with_side_effects, f, test_export_mismatched_out_2, func, test_export_graph_with_list


## Key Components

The file contains 12012 words across 4646 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 156906 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

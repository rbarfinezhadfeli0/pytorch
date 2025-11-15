# Documentation: `docs/test/dynamo/test_export.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_export.py_docs.md`
- **Size**: 54,244 bytes (52.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_export.py`

## File Metadata

- **Path**: `test/dynamo/test_export.py`
- **Size**: 156,906 bytes (153.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
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
                    ret
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/dynamo/test_export.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_export.py_docs.md_docs.md`
- **Keyword Index**: `test_export.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

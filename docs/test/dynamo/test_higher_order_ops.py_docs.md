# Documentation: `test/dynamo/test_higher_order_ops.py`

## File Metadata

- **Path**: `test/dynamo/test_higher_order_ops.py`
- **Size**: 274,774 bytes (268.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import enum
import functools
import pprint
import re
import unittest
import warnings
from copy import deepcopy

import functorch.experimental.control_flow as control_flow
import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    check_dynamic_shape_capture,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    empty_line_normalizer,
    normalize_gm,
)
from torch._dynamo.utils import counters, ifdynstaticdefault
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    munge_exc,
    parametrize,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def count_ops(gm, args, freq, op):
    actual = [node.target for node in gm.graph.nodes].count(op)
    assert actual == freq, f"expected={freq}, actual={actual}"
    return gm


class Obj:
    pass


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.existing = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        return self.existing * x


global_obj = Obj()
global_module = MyModule()
global_var = torch.randn(3)
global_num = 3.14
global_list = []


def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


# Checks that a dict matches a dict with "regex keys". That is,
# the keys are regex expressions.
def assert_dict_matches_regex(self, dct, dct_with_regex_keys):
    regex_keys = dct_with_regex_keys.keys()
    regex_key_to_actual_key = {}
    for regex_key in regex_keys:
        for key in dct:
            if re.match(regex_key, key):
                if regex_key in regex_key_to_actual_key:
                    raise AssertionError(
                        f"Single key regex mapped to multiple keys. Please improve your "
                        f"regex. Got: regex='{regex_key}' "
                        f"keys='{regex_key_to_actual_key[regex_key]}',"
                        f"'{key}'"
                    )
                regex_key_to_actual_key[regex_key] = key
    new_dct = {}
    for regex_key in regex_keys:
        if regex_key not in regex_key_to_actual_key:
            raise AssertionError(
                f"Got regex '{regex_key}' but could not match any key in dict with "
                f"keys {dct.keys()}"
            )
        new_dct[regex_key_to_actual_key[regex_key]] = dct_with_regex_keys[regex_key]
    self.assertEqual(dct, new_dct)


def default_args_generator(seed_value):
    flat_args, args_spec = pytree.tree_flatten(seed_value)
    for i in range(3):
        new_flat_arg = []
        for val in flat_args:
            if isinstance(val, torch.Tensor):
                new_val = val + 0.1 * i
            elif isinstance(val, int):
                new_val = val + 1 * i
            elif isinstance(val, float):
                new_val = val + 0.1 * i
            elif isinstance(val, enum.Enum):
                new_val = val
            else:
                raise AssertionError("unexpected arg type")

            new_flat_arg.append(new_val)
        new_args = pytree.tree_unflatten(new_flat_arg, args_spec)
        yield new_args


class HigherOrderOpTests(torch._dynamo.test_case.TestCaseWithNestedGraphBreaks):
    def _assert_wrap_fallback(self, func, args, setup=lambda: None):
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        setup()
        expected = func(*args)
        setup()
        result = torch.compile(func, backend=cnt, fullgraph=False)(*args)
        num_graph_breaks = len(counters["graph_break"].keys())
        self.assertGreater(num_graph_breaks, 0)

        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(node.target is wrap)

        self.assertEqual(result, expected)

    def _test_wrap_simple(
        self,
        func,
        args_generator,
        expected_num_wrap_args,
        expected_opcount=2,
        return_graph=False,
    ):
        # Given a `func` that has a single call to `wrap`,
        # we check that:
        # - there are no graph breaks
        # - eager vs torch.compile has the same result (correctness)
        # - other compilation metrics, e.g, # of ops in the dynamo captured graph,
        #   the wrap has the expected number of args, etc
        #
        # we have one or multiple runs through with each of the args from args_generator,
        # and we will check:
        # - correctness and no graph breaks for every run
        # - other compilation metrics only for the first run, since automatic_dynamic_shapes
        #   may compile another dynamic version graph for the later runs
        graph = None
        for i, args in enumerate(args_generator):
            backend = EagerAndRecordGraphs()
            cnt = CompileCounterWithBackend(backend)
            expected = func(*args)
            result = torch.compile(func, fullgraph=True, backend=cnt)(*args)
            # check correctness and no graph breaks
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(len(backend.graphs), 1)
            # check other compilation metrics
            if i == 0:
                self.assertEqual(cnt.op_count, expected_opcount)
                graph = backend.graphs[0]
                wrap_node = find_first_node(graph, wrap)
                self.assertEqual(len(wrap_node.args), expected_num_wrap_args)
        # We always return/check the graph from the first run if return_graph = True
        if return_graph:
            return normalize_gm(graph.print_readable(print_output=False))

    def test_error_message_sane(self):
        foo = []

        def inner(x):
            foo.append(x)
            return x.clone()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return wrap(inner, x)

        x = torch.randn(3)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"HigherOrderOperator: Mutating a variable not in the current scope \(SideEffects\)",
        ):
            f(x)

    def test_no_freevars(self):
        def f(x):
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_enum_arg(self):
        class SomeEnum(enum.Enum):
            A = 0
            B = 1

        def g(x, val):
            if val == SomeEnum.A:
                return torch.sin(x)
            return torch.cos(x)

        def f(x, val):
            return wrap(g, x, val)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x, SomeEnum.A)), arg_count)

    def test_return_captured_var(self):
        freevar = torch.randn(3)

        def test(x):
            return freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.

        # when testing with dynamic shape, symbols are lifted as input
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 1)

    def test_return_captured_vars(self):
        freevar1 = torch.randn(3)
        freevar2 = torch.randn(3)

        def test(x):
            return freevar1, freevar2, freevar1

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 1)

    def test_return_captured_var_used_multiple_times(self):
        freevar = torch.randn(3)

        def test(x):
            y = x + freevar
            return y, freevar

        def fn(x):
            return wrap(test, x)

        x = torch.randn(3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 2)

    def test_capture_untracked_global(self):
        def f(x):
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_allow_python_side_effects_utility(self):
        from torch._dynamo.utils import (
            _disable_side_effect_safety_checks_for_current_subtracer,
        )
        from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper

        def wrapper(fn):
            return fn

        count = 0

        def does_side_effect(x):
            nonlocal count
            count += 1
            return x.sin()

        def does_side_effect_wrapped(*args, **kwargs):
            return _disable_side_effect_safety_checks_for_current_subtracer(
                does_side_effect, *args, **kwargs
            )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return dynamo_bypassing_wrapper(wrapper, does_side_effect_wrapped, x)

        x = torch.tensor(1.0)
        fn(x)

        def inner_does_side_effect(x):
            nonlocal count
            count += 1
            return x

        # Test that any nested HOPs are unaffected
        def outer(x):
            return dynamo_bypassing_wrapper(wrapper, inner_does_side_effect, x)

        def outer_wrapped(*args, **kwargs):
            return _disable_side_effect_safety_checks_for_current_subtracer(
                outer, *args, **kwargs
            )

        @torch.compile(backend="eager", fullgraph=True)
        def fn_nested(x):
            return dynamo_bypassing_wrapper(wrapper, outer_wrapped, x)

        x = torch.tensor(1.0)
        with self.assertRaisesRegex(
            RuntimeError, "Mutating a variable not in the current scope"
        ):
            fn_nested(x)

    def test_symint_input(self):
        def f(x):
            i = x.size(0)
            return wrap(lambda x, i: x.view(i), x, i)

        x = torch.randn(3, 1)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
        )

    def test_symint_in_slice(self):
        def f(x):
            i = x.size(0) - 2
            j = x.size(1) - 3
            k = x.size(2)
            return wrap(lambda x: x[:i, :j, k:], x)

        x = torch.randn(3, 4, 5)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            # 3 basic symbols and 2 compound symbols
            ifdynstaticdefault(2, 7),
            # 2 more sym expression computation
            expected_opcount=ifdynstaticdefault(2, 4),
        )

    def test_wrap_pytree_args_nested(self):
        def f(x, y, z):
            def fn(d):
                return d["x"].sin() + d["y"][0].cos() - d["y"][1][2].sin()

            return wrap(fn, d)

        x = torch.tensor(1.5)
        y = torch.tensor(2.0)
        z = torch.tensor(3.0)
        d = {"x": x, "y": (y, [x, y, z])}

        def my_args_generator(t):
            yield t
            yield t[0] + 0.1, t[1], t[2]
            yield t[0], t[1] + 0.1, t[2]

        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator((x, y, z)),
            4,
            return_graph=True,
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_d_x_: "f32[]", L_d_y_0_: "f32[]", L_d_y_1_2_: "f32[]"):
        l_d_x_ = L_d_x_
        l_d_y_0_ = L_d_y_0_
        l_d_y_1_2_ = L_d_y_1_2_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);  wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None
        getitem: "f32[]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_d_x_: "f32[]", l_d_y_0_: "f32[]", l_d_y_1_2_: "f32[]"):
            sin: "f32[]" = l_d_x_.sin();  l_d_x_ = None
            cos: "f32[]" = l_d_y_0_.cos();  l_d_y_0_ = None
            add: "f32[]" = sin + cos;  sin = cos = None
            sin_1: "f32[]" = l_d_y_1_2_.sin();  l_d_y_1_2_ = None
            sub: "f32[]" = add - sin_1;  add = sin_1 = None
            return (sub,)
""",  # NOQA: B950
        )

    def test_wrap_pytree_args_with_symint_constant(self):
        def f(x, y):
            i = x.size(0)
            return wrap(lambda t: t[0].view(t[2]) + t[1], (x, y, i))

        x = torch.randn(3, 1)
        y = 0.5
        actual_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
            return_graph=True,
        )
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 1]"):
            view: "f32[3]" = l_x_.view(3);  l_x_ = None
            add: "f32[3]" = view + 0.5;  view = None
            return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_);  wrap_body_0 = s77 = l_x_ = None
        getitem: "f32[s77]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77, 1]"):
            view: "f32[s77]" = l_x_.view(s77);  l_x_ = s77 = None
            add: "f32[s77]" = view + 0.5;  view = None
            return (add,)
""",
            )

    def test_wrap_pytree_kwargs(self):
        def f(x, y, z):
            def fn(*, x, y, z):
                z1, _ = z
                return (x * 2) + y + z1

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        def my_args_generator(t):
            yield t
            x1 = t[0] + 0.1
            y1 = t[1] + 0.1
            yield (x1, y1, (x1, y1))
            x2 = t[0] + 0.2
            y2 = t[0] + 0.2
            yield (x2, y2, (x2, y2))

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, my_args_generator((x, y, (x, y))), arg_count)

    def test_wrap_pytree_args_not_const_symint_tensor(self):
        class MyClass:
            def __init__(self, x):
                self.val = x

        def f(x, y):
            return wrap(lambda z: z[0].sin() * z[1].val.cos(), (x, y))

        x = torch.tensor(1.2)
        y = MyClass(torch.tensor(3.4))
        self._test_wrap_simple(f, [(x, y)], 3)

    def test_capture_constants(self):
        x = torch.randn(3, 3)

        def fn(x, y, z):
            if z:
                return x + y
            return x * y

        def f(x, y, z):
            return wrap(fn, x, y, z)

        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

        # Ensure that we recompile here
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

    def test_capture_untracked_global_nested(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        x = torch.randn(3)
        result = f(x)

        self.assertEqual(result, x + global_var)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            def g(x):
                return wrap(lambda x: x + y, x)

            # when testing with dynamic shape, a symbol is lifted as input
            arg_count = ifdynstaticdefault(3, 4)
            self._test_wrap_simple(g, default_args_generator((x,)), arg_count)
            return g(x)

        f(x, y)

    def test_capture_tracked(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: x + y, x)

        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_capture_tracked_nested(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_inlined_functions(self):
        def g(x, y):
            return x + y

        def f(x, y):
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_same_freevar_twice(self):
        free = torch.randn(3)

        def g(x):
            y = free.sin()
            z = free.cos()
            return y, z

        def f(x):
            return wrap(g, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count, 3)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True,
    )
    def test_unbacked_symbol_closure(self):
        def f(x):
            c = x.sum().item()

            def g(x):
                def k(x):
                    return x + c

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        out_graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 4, return_graph=True
        )

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        item: "Sym(zuf0)" = sum_1.item();  sum_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, item);  wrap_body_1 = s77 = l_x_ = item = None
        getitem: "f32[s77]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", item: "Sym(zuf0)"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_, item);  wrap_body_0 = s77 = l_x_ = item = None
            getitem: "f32[s77]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", item: "Sym(zuf0)"):
                add: "f32[s77]" = l_x_ + item;  l_x_ = item = None
                return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        item: "Sym(zuf0)" = sum_1.item();  sum_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, item);  wrap_body_1 = l_x_ = item = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", item: "Sym(zuf0)"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, item);  wrap_body_0 = l_x_ = item = None
            getitem: "f32[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", item: "Sym(zuf0)"):
                add: "f32[3]" = l_x_ + item;  l_x_ = item = None
                return (add,)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_tensor_with_unbacked_shape_closure(self):
        def f(x):
            c = x.nonzero()

            def g(x):
                def k(x):
                    return x.sin(), c.sin()

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(4, 5)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0
        expected_op_count = ifdynstaticdefault(10, 8)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77]"):
        l_x_ = L_x_

        c: "i64[u0, 1]" = l_x_.nonzero()

        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        _check_is_size = torch._check_is_size(sym_size_int_1);  _check_is_size = None

        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, sym_size_int_1, c);  wrap_body_1 = s77 = l_x_ = sym_size_int_1 = c = None
        getitem: "f32[s77]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", u0: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_, u0, c);  wrap_body_0 = s77 = l_x_ = u0 = c = None
            getitem: "f32[s77]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", u0: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[s77]" = l_x_.sin();  l_x_ = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (sin, sin_1)
""",
            )
        else:
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        c: "i64[u0, 1]" = l_x_.nonzero()

        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        _check_is_size = torch._check_is_size(sym_size_int_1);  _check_is_size = None

        ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int_1 <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, sym_size_int_1, c);  wrap_body_1 = l_x_ = sym_size_int_1 = c = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", u0: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, u0, c);  wrap_body_0 = l_x_ = u0 = c = None
            getitem: "f32[3]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", u0: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[3]" = l_x_.sin();  l_x_ = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (sin, sin_1)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
        capture_scalar_outputs=True,
    )
    def test_tensor_to_list_closure(self):
        def f(x):
            li = x.tolist()

            def g(x):
                def k(x):
                    return li[0] + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.tensor([1, 2, 3], dtype=torch.int16)
        arg_count = ifdynstaticdefault(3, 3)
        out_graph = self._test_wrap_simple(f, ((x,),), arg_count, 4, return_graph=True)

        # tolist will specialize on input shapes, so dynamic and static tests
        # have the same graph
        self.assertExpectedInline(
            out_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "i16[3]"):
        l_x_ = L_x_

        getitem = l_x_[0]
        item: "Sym(u0)" = getitem.item();  getitem = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, item, l_x_);  wrap_body_1 = item = l_x_ = None
        getitem_3: "i16[3]" = wrap[0];  wrap = None
        return (getitem_3,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, item: "Sym(u0)", l_x_: "i16[3]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, item, l_x_);  wrap_body_0 = item = l_x_ = None
            getitem: "i16[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, item: "Sym(u0)", l_x_: "i16[3]"):
                add: "i16[3]" = item + l_x_;  item = l_x_ = None
                return (add,)
""",
        )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_tensor_and_unbacked_symbol_closure(self):
        def f(x):
            c = x.nonzero()
            sz = c.size(0)

            def g(x):
                def k(x):
                    return x.sin() + sz, c.sin()

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(4, 5)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0
        expected_op_count = ifdynstaticdefault(10, 8)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        # Note that u0 is accessed from sz and the shape of c
        # We cached via the symbol u0 and de-duplicate them.
        if not check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        c: "i64[u0, 1]" = l_x_.nonzero()

        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        _check_is_size = torch._check_is_size(sym_size_int);  _check_is_size = None

        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, sym_size_int, c);  wrap_body_1 = l_x_ = sym_size_int = c = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", size: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, size, c);  wrap_body_0 = l_x_ = size = c = None
            getitem: "f32[3]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", size: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[3]" = l_x_.sin();  l_x_ = None
                add: "f32[3]" = sin + size;  sin = size = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (add, sin_1)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_concat_unbacked_shape_tensor(self):
        def f(x, y):
            c = x.nonzero()
            d = y.nonzero()
            cat = torch.cat((c, d))

            def g(x):
                def k(x):
                    return cat.sum() + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        y = torch.randn(3)
        arg_count = ifdynstaticdefault(5, 6)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0 and u1
        expected_op_count = ifdynstaticdefault(17, 13)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        if not check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]", L_y_: "f32[3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        c: "i64[u0, 1]" = l_x_.nonzero()

        sym_size_int_2: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
        _check_is_size = torch._check_is_size(sym_size_int_2);  _check_is_size = None

        ge: "Sym(u0 >= 0)" = sym_size_int_2 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int_2 <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        d: "i64[u1, 1]" = l_y_.nonzero();  l_y_ = None

        sym_size_int_3: "Sym(u1)" = torch.ops.aten.sym_size.int(d, 0)
        _check_is_size_1 = torch._check_is_size(sym_size_int_3);  _check_is_size_1 = None

        ge_1: "Sym(u1 >= 0)" = sym_size_int_3 >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_2 = None
        le_1: "Sym(u1 <= 3)" = sym_size_int_3 <= 3
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u1 <= 3 on node 'le_1'");  le_1 = _assert_scalar_default_3 = None

        cat: "i64[u0 + u1, 1]" = torch.cat((c, d));  c = d = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, sym_size_int_2, sym_size_int_3, cat, l_x_);  wrap_body_1 = sym_size_int_2 = sym_size_int_3 = cat = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, u0: "Sym(u0)", u1: "Sym(u1)", cat: "i64[u0 + u1, 1]", l_x_: "f32[3]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, u0, u1, cat, l_x_);  wrap_body_0 = u0 = u1 = cat = l_x_ = None
            getitem: "f32[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, u0: "Sym(u0)", u1: "Sym(u1)", cat: "i64[u0 + u1, 1]", l_x_: "f32[3]"):
                sum_1: "i64[]" = cat.sum();  cat = None
                add: "f32[3]" = sum_1 + l_x_;  sum_1 = l_x_ = None
                return (add,)
""",
            )

    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        dynamic_shapes=True,
    )
    def test_lift_tensors_with_shared_symbols(self):
        def f(x, y):
            def g(x):
                def k(x):
                    return x @ y

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            6,
            2,
            return_graph=True,
        )
        self.assertExpectedInline(
            out_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]", s94: "Sym(s94)", L_y_: "f32[s27, s94]"):
        l_x_ = L_x_
        l_y_ = L_y_

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, s27, l_x_, s94, l_y_);  wrap_body_1 = s77 = s27 = l_x_ = s94 = l_y_ = None
        getitem: "f32[s77, s94]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", l_x_: "f32[s77, s27]", s94: "Sym(s94)", l_y_: "f32[s27, s94]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, s27, l_x_, s94, l_y_);  wrap_body_0 = s77 = s27 = l_x_ = s94 = l_y_ = None
            getitem: "f32[s77, s94]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", l_x_: "f32[s77, s27]", s94: "Sym(s94)", l_y_: "f32[s27, s94]"):
                matmul: "f32[s77, s94]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
                return (matmul,)
""",
        )

    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        dynamic_shapes=True,
        capture_dynamic_output_shape_ops=True,
    )
    def test_lift_tensors_with_compound_expressions(self):
        def f(x, y):
            x = x.view(-1, 2)
            c = y.nonzero()
            d = torch.concat((x, c))

            def g(x):
                def k(x):
                    return d.sum() + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        f(x, y)

        if not check_dynamic_shape_capture():
            out_graph = self._test_wrap_simple(
                f,
                default_args_generator((x, y)),
                6,
                9,
                return_graph=True,
            )
            self.assertExpectedInline(
                out_graph,
                """\
    class GraphModule(torch.nn.Module):
        def forward(self, s0: "Sym(s0)", s1: "Sym(s1)", L_x_: "f32[s0, s1]", s2: "Sym(s2)", L_y_: "f32[s1, s2]"):
            l_x_ = L_x_
            l_y_ = L_y_

            x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = l_x_.view(-1, 2);  l_x_ = None

            c: "i64[u0, 2]" = l_y_.nonzero();  l_y_ = None

            sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
            _check_is_size = torch._check_is_size(sym_size_int_1);  _check_is_size = None

            ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

            d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = torch.concat((x, c));  c = None

            wrap_body_1 = self.wrap_body_1
            wrap = torch.ops.higher_order.wrap(wrap_body_1, sym_size_int_1, s1, s0, d, x);  wrap_body_1 = sym_size_int_1 = s1 = s0 = d = x = None
            getitem: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_1(torch.nn.Module):
            def forward(self, u0: "Sym(u0)", s1: "Sym(s1)", s0: "Sym(s0)", d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]", x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]"):
                wrap_body_0 = self.wrap_body_0
                wrap = torch.ops.higher_order.wrap(wrap_body_0, u0, s1, s0, d, x);  wrap_body_0 = u0 = s1 = s0 = d = x = None
                getitem: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = wrap[0];  wrap = None
                return (getitem,)

            class wrap_body_0(torch.nn.Module):
                def forward(self, u0: "Sym(u0)", s1: "Sym(s1)", s0: "Sym(s0)", d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]", x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]"):
                    sum_1: "f32[]" = d.sum();  d = None
                    add: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = sum_1 + x;  sum_1 = x = None
                    return (add,)
    """,
            )

    def test_register_subclass(self):
        from torch._higher_order_ops.cond import cond_op
        from torch.testing._internal.two_tensor import TwoTensor

        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.randn(3)
        t = TwoTensor(a, b)
        with self.assertRaisesRegex(
            NotImplementedError,
            "no rule registered for HOP cond and subclass .*TwoTensor'>",
        ):
            res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))

        called = 0

        # Using cond.py_impl
        @cond_op.py_impl(TwoTensor)
        def _(pred, true_fn, false_fn, operands):
            nonlocal called
            called += 1
            assert len(operands) == 1
            a = cond_op(pred, true_fn, false_fn, (operands[0].a,))
            b = cond_op(pred, true_fn, false_fn, (operands[0].b,))
            return TwoTensor(a, b)

        res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))
        self.assertEqual(res.a, torch.sin(a))
        self.assertEqual(res.b, torch.sin(b))
        self.assertEqual(called, 1)

    def test_register_mode(self):
        from torch._higher_order_ops.cond import cond_op

        torch_dispatch_called = 0

        class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                nonlocal torch_dispatch_called
                torch_dispatch_called += 1
                return func(*args, **kwargs)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with self.assertRaisesRegex(
            NotImplementedError,
            "no rule registered for HigherOrderOperator cond and mode .*MyMode",
        ):
            with MyMode():
                res = cond_op(pred, torch.sin, torch.cos, (a,))

        py_impl_called = 0

        # Using cond.py_impl
        @cond_op.py_impl(MyMode)
        def _(mode, pred, true_fn, false_fn, operands):
            nonlocal py_impl_called
            py_impl_called += 1
            return cond_op(pred, true_fn, false_fn, operands)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with MyMode():
            res = cond_op(pred, torch.sin, torch.cos, (a,))
        self.assertEqual(res, a.sin())

    def test_capture_value_created_in_subgraph(self):
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 1)

        # No changes to args of outer wrap
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # z was lifted to arg of inner wrap
        body_function = getattr(gm, wrap_node.args[0].name)
        # addition + wrap + getitem
        self.assertEqual(op_count(body_function), 3)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # Innermost body function: z was also lifted to arg
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_side_effect_set_new_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()

        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()
            global_obj.foo = nn.Parameter(torch.tensor(4.0))

        def f(x):
            def h(x):
                def g(x):
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_obj(self):
        def setup():
            global global_obj
            global_obj = Obj()
            global_obj.foo = torch.tensor(4.0)

        def f(x):
            def h(x):
                def g(x):
                    del global_obj.foo
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_new_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                global_module.foo = nn.Parameter(x + 1)
                return x.clone()

            y = wrap(g, x)
            return y + global_module.foo

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                global_module.existing = nn.Parameter(torch.tensor(4.0))
                return global_module(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_module(self):
        def setup():
            global global_module
            global_module = MyModule()

        def h(x):
            def g(x):
                del global_module.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_mutate_global_num(self):
        def setup():
            global global_num
            global_num = 3.14

        def f(x):
            def g(x):
                global global_num
                global_num = global_num + 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_num_builtin(self):
        def setup():
            global global_num
            global_num = 3.14

        def f(x):
            def g(x):
                global global_num
                global_num += 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor(self):
        def setup():
            global global_var
            global_var = torch.ones(3)

        def f(x):
            def g(x):
                global global_var
                global_var = global_var + 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor_builtin(self):
        def setup():
            global global_var
            global_var = torch.ones(3)

        def f(x):
            def g(x):
                global global_var
                global_var += 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_list(self):
        def setup():
            global global_list
            global_list = []

        def f(x):
            def g(x):
                val = x + 1
                global_list.append(val)
                return global_list[-1]

            y = wrap(g, x)
            z = y + global_list[-1]
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_nonlocal_num(self):
        def f(x):
            def h(x):
                val = 1

                def g(x):
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()

                def g(x):
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_existing_attr_nonlocal_obj(self):
        def f(x):
            def h(x):
                obj = Obj()
                obj.val = 3

                def g(x):
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
      
```



## High-Level Overview


This Python file contains 111 class(es) and 798 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Obj`, `MyModule`, `HigherOrderOpTests`, `SomeEnum`, `GraphModule`, `wrap_body_0`, `GraphModule`, `wrap_body_0`, `GraphModule`, `wrap_body_0`, `MyClass`, `GraphModule`, `wrap_body_1`, `wrap_body_0`, `GraphModule`, `wrap_body_1`, `wrap_body_0`, `GraphModule`, `wrap_body_1`, `wrap_body_0`

**Functions defined**: `count_ops`, `__init__`, `forward`, `find_first_node`, `op_count`, `assert_dict_matches_regex`, `default_args_generator`, `_assert_wrap_fallback`, `_test_wrap_simple`, `test_error_message_sane`, `inner`, `f`, `test_no_freevars`, `f`, `test_enum_arg`, `g`, `f`, `test_return_captured_var`, `test`, `fn`

**Key imports**: enum, functools, pprint, re, unittest, warnings, deepcopy, functorch.experimental.control_flow as control_flow, torch, torch._dynamo.config as config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `enum`
- `functools`
- `pprint`
- `re`
- `unittest`
- `warnings`
- `copy`: deepcopy
- `functorch.experimental.control_flow as control_flow`
- `torch`
- `torch._dynamo.config as config`
- `torch._dynamo.test_case`
- `torch._functorch.config`
- `torch.nn as nn`
- `torch.utils._pytree as pytree`
- `torch.utils.checkpoint`
- `torch._dynamo.backends.common`: aot_autograd
- `torch._dynamo.utils`: counters, ifdynstaticdefault
- `torch._higher_order_ops.hints_wrap`: hints_wrapper
- `torch._higher_order_ops.wrap`: wrap
- `torch.testing._internal.hop_db`: hop_db
- `torch.testing._internal.logging_utils`: LoggingTestCase, make_logging_test
- `torch.testing._internal.triton_utils`: requires_cuda_and_triton
- `torch._higher_order_ops.cond`: cond_op
- `torch.testing._internal.two_tensor`: TwoTensor
- `torch._higher_order_ops.map`: map_dense
- `numpy as np`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/dynamo/test_higher_order_ops.py
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

- **File Documentation**: `test_higher_order_ops.py_docs.md`
- **Keyword Index**: `test_higher_order_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

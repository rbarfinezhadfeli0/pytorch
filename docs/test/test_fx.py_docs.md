# Documentation: `test/test_fx.py`

## File Metadata

- **Path**: `test/test_fx.py`
- **Size**: 182,825 bytes (178.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]
# ruff: noqa: F841
# flake8: noqa: E221

import builtins
import collections
import contextlib
import copy
import gc
import functools
import inspect
import io
import math
import numbers
import operator
import os
import pickle
import sys
import traceback
import types
import typing
import unittest
import weakref
import warnings
from math import sqrt
from torch.multiprocessing import Process
from torch.testing import FileCheck
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import ops, onlyCPU, instantiate_device_type_tests
import torch.utils._pytree as pytree
import torch.fx._pytree as fx_pytree
from torch.fx import symbolic_trace, Proxy, Node, GraphModule, Interpreter, Tracer, Transformer, Graph, wrap, PH, CodeGen
from torch.fx.node import Target, Argument, ArgumentT, _format_arg
from torch.fx.passes import shape_prop
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.operator_schemas import get_signature_for_torch_op
from copy import deepcopy
from collections import namedtuple
from typing import Any, NamedTuple, Optional, Union
from collections.abc import Callable

import torch

from functorch.experimental import control_flow

from fx.named_tup import MyNamedTup
from fx.test_common_passes import TestCommonPass  # noqa: F401
from fx.test_cse_pass import TestCSEPass  # noqa: F401
from fx.test_dce_pass import TestDCE  # noqa: F401
from fx.test_fx_const_fold import TestConstFold  # noqa: F401
from fx.test_fx_param_shape_control_flow import (  # noqa: F401
    TestConstParamShapeInControlFlow,
)

from fx.test_gradual_type import (  # noqa: F401  # noqa: F401
    AnnotationsTest,
    TypeCheckerTest,
)
from fx.test_matcher_utils import TestMatcher  # noqa: F401
from fx.test_pass_infra import TestPassManager  # noqa: F401
from fx.test_source_matcher_utils import TestSourceMatcher  # noqa: F401
from fx.test_subgraph_rewriter import TestSubgraphRewriter  # noqa: F401
from torch.fx._compatibility import _BACK_COMPAT_OBJECTS, _MARKED_WITH_COMPATIBILITY
from torch.fx._symbolic_trace import PHBase, PHWithMeta

from torch.fx.proxy import TraceError
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    skipIfRocm,
)
from torch.testing._internal.jit_utils import JitTestCase

import json
import tempfile
from torch.profiler import profile, ProfilerActivity
from torch.profiler._utils import map_recorded_events_to_aten_ops_with_stack_trace
from torch.autograd.profiler_util import _canonicalize_profiler_events

try:
    from torchvision import models as torchvision_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport


class SimpleTest(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3.0)


def a_non_torch_leaf(a, b):
    return a + b


# Used for test_autowrap_function. Autowrapped functions need to be global
def fx_int(x: float) -> int:
    return int(x)


def fx_int_x2(x: float) -> int:
    return int(x) * 2


# used in test_pytree. It's all the way out here because pickling a GraphModule
# that uses Point errors out if Point is local to the function
Point = namedtuple("Point", ["x", "y"])


# Test wrap() passing both a function name as well as a function
# directly
def a_lifted_leaf(a, b):
    return a[0] + a[1] + b


wrap("a_lifted_leaf")
# Test wrapping twice doesn't break anything
wrap("a_lifted_leaf")


def a_lifted_leaf2(a, b):
    return a[0] + a[1] + b


wrap(a_lifted_leaf2)

wrap("len")

wrap("getattr")


def wrapped_named_tup(p1, *, p2):
    return p1.x + p2.y


wrap(wrapped_named_tup)


@wrap
def wrapped_via_decorator(a):
    return a + 1


wrap("wrapped_with_submodule")


def wrapped_with_submodule(x: torch.Tensor, batchnorm1d: torch.nn.BatchNorm1d):
    return batchnorm1d(x)


def my_decorator(f):
    @functools.wraps(f)
    def wrapper_inside_decorator(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_inside_decorator


@wrap
@my_decorator
def wrapped_decorated_fn(x):
    return x


real_wrapped_via_decorator = wrapped_via_decorator
real_a_lifed_leaf = a_lifted_leaf
real_a_lifed_leaf2 = a_lifted_leaf2
_sqrt = sqrt

wrap("wrapper_fn")


def wrapper_fn(x):
    return torch.foo(x)


class Pair(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor

    def _custom_fx_repr_fn(self) -> str:
        return f"Pair(x={_format_arg(self.x)}, y={_format_arg(self.y)})"


# for testing pytrees
class Foo:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Add(torch.nn.Module):
    def forward(self, x):
        return x + x


@torch.fx.has_side_effect
@torch.fx.wrap
def side_effect_func(x: torch.Tensor):
    print(x)


def _enrich_profiler_traces(prof):
    """
    Helper function to extract and augment profiler events with stack traces.

    Args:
        prof: A torch.profiler.profile object

    Returns:
        A string representing enriched events
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        trace_file = f.name
        prof.export_chrome_trace(trace_file)

        with open(trace_file) as f:
            trace_data = json.load(f)

        map_recorded_events_to_aten_ops_with_stack_trace(
            trace_data
        )

        events = []
        for event in trace_data["traceEvents"]:
            if "args" in event and "stack_trace" in event["args"]:
                events.append(event)

        actual_traces = _canonicalize_profiler_events(events)
        return actual_traces


class TestFX(JitTestCase):
    def setUp(self):
        super().setUp()
        # Checking for mutable operations while tracing is feature flagged
        # Enable it in testing but not by default
        self.orig_tracer_mutable_flag = (
            torch.fx.proxy.TracerBase.check_mutable_operations
        )
        torch.fx.proxy.TracerBase.check_mutable_operations = True

        if not (IS_FBCODE or IS_WINDOWS or IS_MACOS):
            lib_file_path = find_library_location("libtorchbind_test.so")
            torch.ops.load_library(str(lib_file_path))

    def tearDown(self):
        super().tearDown()
        torch.fx.proxy.TracerBase.check_mutable_operations = (
            self.orig_tracer_mutable_flag
        )

    def checkGraphModule(self, m: torch.nn.Module, args, kwargs=None):
        """Check that an nn.Module's results match the GraphModule version
        for a given set of args/kwargs.
        """
        kwargs = kwargs if kwargs else {}
        ref_outs = m(*args, **kwargs)
        gm = symbolic_trace(m)
        gm.graph.lint()
        test_outs = gm(*args, **kwargs)
        self.assertEqual(ref_outs, test_outs)

    def test_graph_module(self):
        class MySub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                return self.w + x

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.sub_mod = MySub()
                self.w = torch.nn.Parameter(torch.rand(3))

            def forward(self, A, B, c):
                t = torch.sigmoid(A) + self.lin(c)
                return self.sub_mod(
                    t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3)
                )

        m = MyModule()
        gm = symbolic_trace(m)

        ms = torch.jit.script(gm)

        class M2(torch.nn.Module):
            def forward(self, A):
                m, idx = torch.max(A, 0)
                return m + 1, idx + 1

        m2 = M2()
        gm2 = symbolic_trace(m2)

        class T(torch.nn.Module):
            def forward(self, A, b=4, *args, c=5, **kwargs):
                x = A + 1 + args[0] + kwargs["3"]
                return x

        t = T()
        symbolic_trace(t)

        # test for issue described at https://github.com/pytorch/pytorch/issues/63883
        class M3(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        m3 = M3()
        gm3 = symbolic_trace(m3)
        new_instance = gm3.__new__(type(gm3))
        new_instance.__init__(gm3, gm3.graph)

        x = torch.randn(5, 3)
        torch.testing.assert_close(new_instance(x), torch.relu(x))

    def test_informative_co_filename(self):
        class MyModule(torch.nn.Module):
            def forward(self, a):
                return a * 2

        gm = symbolic_trace(MyModule())
        self.assertIn(os.path.basename(__file__), gm.forward.__code__.co_filename)

    def test_custom_import(self):
        graph = torch.fx.Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(a_non_torch_leaf, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        x, y = torch.rand(1), torch.rand(1)
        self.assertEqual(torch.sin(x + y), gm(x, y))

    def test_args_kwargs(self):
        class T(torch.nn.Module):
            def forward(self, *args, **kwargs):
                x = args[0] + kwargs["foo"]
                return x

        t = T()
        self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {"foo": torch.rand(1)})

    def test_varargs_concrete(self):
        class T(torch.nn.Module):
            def forward(self, *args, **kwargs):
                x = args[0] + args[1]
                return x

        args = (torch.rand(1), torch.rand(1))

        t = T()
        ref_outs = t(*args)
        gm = symbolic_trace(t, concrete_args=(torch.fx.PH, torch.fx.PH))
        gm.graph.lint()
        test_outs = gm(*args)
        self.assertEqual(ref_outs, test_outs)

    def test_args_kwargs_no_self(self):
        class T(torch.nn.Module):
            def forward(*args, **kwargs):  # noqa: B902
                self = args[0]
                return torch.relu(args[1])

        t = T()
        with self.assertRaisesRegex(
            RuntimeError, r"cannot be part of \*args expansion"
        ):
            self.checkGraphModule(
                t, (torch.rand(1), torch.rand(1)), {"foo": torch.rand(1)}
            )

    def test_fx_shifts(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x << 3, x >> 3

        input = torch.LongTensor(10).random_(0, 1024)

        m = MyModule()
        self.checkGraphModule(m, (input,))

    def test_fx_and_or(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x & x, x | x

        input = torch.LongTensor(10).random_(0, 1024)

        m = MyModule()
        self.checkGraphModule(m, (input,))

    def test_dict(self):
        class MyDictMod(torch.nn.Module):
            def forward(self, d):
                return d["3"].relu(), {"4": d["3"].neg()}

        input_dict = {"3": torch.rand(3, 4)}
        m = MyDictMod()

        self.checkGraphModule(m, (input_dict,))

    def test_matmul_tracing(self):
        const = torch.randn(3)

        def matmul_f(x):
            return x @ const

        mod = symbolic_trace(matmul_f)
        inp = torch.randn(3)
        self.assertEqual(mod(inp), matmul_f(inp))

        def rmatmul_f(x):
            return const @ x

        mod = symbolic_trace(rmatmul_f)
        inp = torch.randn(3)
        self.assertEqual(mod(inp), rmatmul_f(inp))

    @skipIfNoDynamoSupport
    def test_control_flow_tracing(self):
        def true(x, y):
            return x + y

        def false(x, y):
            return x - y

        def f(x, y):
            x = control_flow.cond(x[0] == 0, true, false, [x, y])

        with self.assertRaisesRegex(
            RuntimeError, r"Expected pred to be bool or tensor, but got Proxy\(eq\)"
        ):
            _ = symbolic_trace(f)

    def test_disallow_override(self):
        # Custom delegate to disallow in-place tensor operations
        class NoMutableCallTracer(Tracer):
            def create_node(
                self,
                kind: str,
                target: Union[str, Callable],
                args: tuple[Argument, ...],
                kwargs: dict[str, Any],
                name: Optional[str] = None,
                type_expr: Optional[Any] = None,
            ) -> Node:
                name = target if isinstance(target, str) else torch.typename(target)
                if name[-1] == "_":
                    raise RuntimeError("In-place operations are not supported")
                return super().create_node(kind, target, args, kwargs, name)

        # Test method
        class MyInplaceMod(torch.nn.Module):
            def forward(self, x):
                x.add_(3.0)
                return x

        m = MyInplaceMod()

        with self.assertRaisesRegex(RuntimeError, "In-place operations"):
            NoMutableCallTracer().trace(m)

        # Test free function
        class MyInplaceMod2(torch.nn.Module):
            def forward(self, x):
                torch.log_(x)
                return x

        m2 = MyInplaceMod2()
        with self.assertRaisesRegex(RuntimeError, "In-place operations"):
            NoMutableCallTracer().trace(m2)

        # Test symbolic node as an arg
        class MyInplaceMod3(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4)
                y.add_(x)
                return x

        m3 = MyInplaceMod3()
        with self.assertRaisesRegex(RuntimeError, "In-place operations"):
            NoMutableCallTracer().trace(m3)

    def test_leaf_module(self):
        # Custom delegate to make it so that there are no leaf modules, everything
        # should get traced through
        class NoLeafModulesTracer(Tracer):
            def is_leaf_module(self, m, qualname):
                return False

        class MyReluMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        mrm = MyReluMod()
        sym = NoLeafModulesTracer().trace(mrm)
        for node in sym.nodes:
            self.assertNotEqual(node.op, "call_module")
        sym.lint()

    def test_wrap(self):
        self.assertEqual(3 + 4 + 5, a_lifted_leaf((3, 4), 5))

        def to_trace(y):
            return (
                a_lifted_leaf((4, y), 3)
                + a_lifted_leaf((3, 4), 5)
                + a_lifted_leaf((y, y), y)
            )

        m = symbolic_trace(to_trace)
        self.assertIn("a_lifted_leaf", m.code)
        self.assertEqual(27, m(2))
        self.assertIs(a_lifted_leaf, real_a_lifed_leaf)

    def test_wrap_fn_directly(self):
        self.assertEqual(3 + 4 + 5, a_lifted_leaf2((3, 4), 5))

        def to_trace(y):
            return (
                a_lifted_leaf2((4, y), 3)
                + a_lifted_leaf2((3, 4), 5)
                + a_lifted_leaf2((y, y), y)
            )

        m = symbolic_trace(to_trace)
        self.assertIn("a_lifted_leaf2", m.code)
        self.assertEqual(27, m(2))
        self.assertIs(a_lifted_leaf2, real_a_lifed_leaf2)

    def test_wrapped_via_decorator(self):
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            return wrapped_via_decorator(y)

        m = symbolic_trace(to_trace)
        self.assertIn("wrapped_via_decorator", m.code)
        self.assertEqual(m(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

    def test_wrapped_via_decorator_and_transformed(self):
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            return wrapped_via_decorator(y)

        m = symbolic_trace(to_trace)
        self.assertIn("wrapped_via_decorator", m.code)
        self.assertEqual(m(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

        transformed = torch.fx.Transformer(m).transform()
        self.assertIn("wrapped_via_decorator", transformed.code)
        self.assertEqual(transformed(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

    def test_wrap_with_submodule(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)

            def forward(self, x: torch.Tensor):
                return wrapped_with_submodule(x, self.batchnorm1d)

        m = symbolic_trace(M())

        self.assertIn("wrapped_with_submodule", m.code)

        input = torch.rand(3, 2)
        ref_batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)
        self.assertEqual(ref_batchnorm1d(input), m(input))

    def test_wrapped_retrace(self):
        def to_trace(y):
            return wrapped_via_decorator(y)

        m = symbolic_trace(to_trace)
        self.assertIn("wrapped_via_decorator", m.code)
        self.assertEqual(m(0), 1)

        retraced = symbolic_trace(m)
        self.assertIn("wrapped_via_decorator", retraced.code)
        self.assertEqual(retraced(0), 1)

    def test_wrap_decorated_function(self):
        def to_trace(y):
            return wrapped_decorated_fn(y)

        m = symbolic_trace(to_trace)
        self.assertIn("wrapped_decorated_fn", m.code)
        self.assertEqual(m(1), 1)

    def test_graph_edit_with_proxy(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = M()
        g = symbolic_trace(m).graph
        new_g = torch.fx.Graph()
        val_map: dict[Node, Node] = {}
        output_val = new_g.graph_copy(g, val_map)
        t = Proxy(output_val)
        # test that we can use proxy objects to generate more graph code later for things that do not need to work with modules.
        new_g.output((t + t).node)
        gm = GraphModule(m, new_g)
        gm.graph.lint()
        self.assertEqual(gm(3, 4), 14)

    def test_proxy_deepcopy_without_tracer(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return 2 * x

        module = MyModule()
        traced = symbolic_trace(module)
        node = list(traced.graph.nodes)[-2]
        p = torch.fx.Proxy(node, None)
        node.proxy = p
        p2 = copy.deepcopy(p)
        self.assertTrue(isinstance(p2, torch.fx.Proxy))
        self.assertEqual(p2.node.name, node.name)
        self.assertEqual(p2.node.target, node.target)
        self.assertNotEqual(id(p2.node), id(node))

    def test_proxy_deepcopy_with_tracer(self):
        class TestTracer(Tracer):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def is_leaf_module(self, module, name):
                return True

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return 2 * x

        module = MyModule()
        tracer = TestTracer("mytracer")
        traced = symbolic_trace(module)
        node = list(traced.graph.nodes)[-2]
        p = torch.fx.Proxy(node, tracer)
        node.proxy = p
        p2 = copy.deepcopy(p)
        self.assertTrue(isinstance(p2, torch.fx.Proxy))
        self.assertTrue(isinstance(p2.tracer, torch.fx._symbolic_trace.Tracer))
        self.assertEqual(p2.tracer.name, "mytracer")
        self.assertEqual(p2.node.name, node.name)
        self.assertEqual(p2.node.target, node.target)
        self.assertNotEqual(id(p2.node), id(node))
        self.assertNotEqual(id(p2.tracer), id(tracer))

    def test_concrete_arg_none_assert(self):
        class Foo(torch.nn.Module):
            def forward(self, x, val=None):
                return x if val is None else x + val

        f = Foo()
        traced = torch.fx.symbolic_trace(f, concrete_args={"val": None})
        with self.assertRaisesRegex(
            AssertionError, "val has been specialized to have value None"
        ):
            traced(torch.randn(5), torch.randn(5))

        x = torch.randn(5)
        torch.testing.assert_close(traced(x), f(x))

    def test_trace_multiple_funcs(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

            def minus_forward(self, x, y):
                return x - y

            def multiply_forward(self, x, y):
                return x * y

        f = Foo()
        x, y = torch.randn(5), torch.randn(5)

        print(torch.__version__)

        tracer = Tracer()
        torch.testing.assert_close(GraphModule(f, tracer.trace(f))(x, y), f(x, y))

        tracer.traced_func_name = "minus_forward"
        torch.testing.assert_close(
            GraphModule(f, tracer.trace(f))(x, y),
            f.minus_forward(x, y),
        )

        tracer.traced_func_name = "multiply_forward"
        torch.testing.assert_close(
            GraphModule(f, tracer.trace(f))(x, y),
            f.multiply_forward(x, y),
        )

        tracer.traced_func_name = "add_forward"
        with self.assertRaisesRegex(AssertionError, "doesn't exist in"):
            tracer.trace(f)

    def test_graph_unique_names(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = M()
        g = symbolic_trace(m).graph
        new_g = torch.fx.Graph()
        val_map: dict[Node, Node] = {}
        output_val = new_g.graph_copy(g, val_map)
        t = Proxy(output_val)
        # test that we can use proxy objects to generate more graph code later for things that do not need to work with modules.
        new_g.output((t + t).node)
        gm = GraphModule(m, new_g)
        seen_names: set[str] = set()
        for node in gm.graph.nodes:
            assert node.name not in seen_names
            seen_names.add(node.name)

    def test_stack_traces(self):
        def foo(a, b):
            return a * b

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                c = a + b
                c = foo(a, c)
                return c

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True

        graph = tracer.trace(M())
        # saving the original list because we will insert new nodes as a part of a test
        stack_traces = "\n".join([node.meta.get("stack_trace", "") for node in graph.nodes])
        FileCheck().check_count(
            "c = a + b", 1, exactly=True
        ).run(stack_traces.strip())
        FileCheck().check_count(
            "c = foo(a, c)", 1, exactly=True
        ).run(stack_traces.strip())
        FileCheck().check_count(
            "return a * b", 1, exactly=True
        ).run(stack_traces.strip())

    def test_stack_traces_with_transformer(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True

        graph = tracer.trace(M())
        gm = GraphModule(tracer.root, graph)
        new_gm = Transformer(gm).transform()

        # nodes after Transformer should still preserve the original node's stack trace
        for node in new_gm.graph.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            self.assertTrue(node.stack_trace is not None)
            assert "test_fx.py" in node.stack_trace

    def test_lineno_map(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                a = torch.sin(a)
                b = torch.cos(b)
                return a + b

        tracer = torch.fx.Tracer()
        graph = tracer.trace(M())
        gm = GraphModule(tracer.root, graph)
        expected = {1: 2, 2: 3, 3: 4, 4: 5}
        self.assertTrue(set(expected.items()).issubset(set(gm._lineno_map.items())))
        self.assertEqual(gm._prologue_start, 4)

        # test custom codegen
        def transform_code(code):
            return ["print('hello!')\n", *code]

        gm.graph.on_generate_code(lambda _: transform_code)
        gm.recompile()
        expected = {2: 2, 3: 3, 4: 4, 5: 5}
        self.assertTrue(set(expected.items()).issubset(set(gm._lineno_map.items())))
        self.assertEqual(gm._prologue_start, 4)

    def test_graph_unique_names_manual(self):
        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.create_node("placeholder", "x")
        b: torch.fx.Node = graph.create_node(
            "call_module", "linear_mod", args=(a,), name="foo_1_1"
        )
        c: torch.fx.Node = graph.create_node("get_attr", "y_attr", name="foo_1")
        d: torch.fx.Node = graph.create_node("call_function", operator.add, args=(b, c))
        graph.output(d)
        graph2 = torch.fx.Graph()
        val_map: dict[Node, Node] = {}
        graph2.graph_copy(graph, val_map)
        seen_names: set[str] = set()
        for node in graph2.nodes:
            assert node.name not in seen_names
            seen_names.add(node.name)

    def test_unpack(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                c, d = a
                return c + d + b

        a = (torch.rand(1), torch.rand(1))
        b = torch.rand(1)
        m = M()
        self.checkGraphModule(m, (a, b))

    def test_native_callable(self):
        if IS_FBCODE or IS_WINDOWS or IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        # This test exercises the case where we use FX to translate from Python
        # code to some native callable object
        #
        # For the purposes of testing, we use ElementwiseInterpreter defined
        # in test_custom_class.cpp.
        #
        # We test that we can
        # 1) Construct a native callable from FX IR
        # 2) Construct a drop-in replacement module that delegates to the
        #    native callable rather than the original code
        # 3) Run both the original code and native callable wrapper with
        #    equivalent results
        # 4) TorchScript compile the native callable wrapper and confirm
        #    equivalent results with the reference
        # 5) TorchScript serialize and deserialize the native callable
        #    and confirm equivalent results with the reference

        # We use this simple Module as a reference computation
        class MySimpleMod(torch.nn.Module):
            def forward(self, x):
                return 3.0 * x + x

        msm = MySimpleMod()

        # This is what a lowering pass might look like: a function that takes
        # a valid nn.Module, symbolically traces it, lowers the Module to some
        # representation, and wraps that representation up into another
        # nn.Module instance that handles dispatch to the compiled/lowered code.
        def lower_to_elementwise_interpreter(
            orig_mod: torch.nn.Module,
        ) -> torch.nn.Module:
            # ===== Stage 1: Symbolic trace the module =====
            mod = symbolic_trace(orig_mod)

            # ===== Stage 2: Lower GraphModule representation to the C++
            #       interpreter's instruction format ======
            instructions = []
            constant_idx = 0
            constants = {}
            fn_input_names = []

            target_to_name = {operator.add: "add", operator.mul: "mul"}

            output_node: Optional[Node] = None
            # For each instruction, create a triple
            # (instruction_name : str, inputs : List[str], output : str)
            # to feed into the C++ interpreter
            for n in mod.graph.nodes:
                target, args, out_name = n.target, n.args, n.name
                assert len(n.kwargs) == 0, "kwargs currently not supported"

                if n.op == "placeholder":
                    # Placeholders specify function argument names. Save these
                    # for later when we generate the wrapper GraphModule
                    fn_input_names.append(target)
                elif n.op == "call_function":
                    assert target in target_to_name, "Unsupported call target " + target
                    arg_names = []
                    for arg in args:
                        if not isinstance(arg, Node):
                            # Pull out constants. These constants will later be
                            # fed to the interpreter C++ object via add_constant()
                            arg_name = f"constant_{constant_idx}"
                            constants[arg_name] = torch.tensor(
                                [arg] if isinstance(arg, numbers.Number) else arg
                            )
                            arg_names.append(arg_name)
                            constant_idx += 1
                        else:
                            arg_names.append(arg.name)
                    instructions.append((target_to_name[target], arg_names, out_name))
                elif n.op == "output":
                    if output_node is not None:
                        raise RuntimeError("Multiple output nodes!")
                    output_node = n
                else:
                    raise RuntimeError("Unsupported opcode " + n.op)

            interpreter = torch.classes._TorchScriptTesting._ElementwiseInterpreter()
            # Load constants
            for k, v in constants.items():
                interpreter.add_constant(k, v)
            # Specify names for positional input arguments
            interpreter.set_input_names(fn_input_names)
            # Load instructions
            interpreter.set_instructions(instructions)
            # Specify name for single output
            assert isinstance(output_node.args[0], torch.fx.Node)
            interpreter.set_output_name(output_node.args[0].name)

            # ===== Stage 3: Create a wrapper GraphModule around the interpreter =====
            class WrapperModule(torch.nn.Module):
                def __init__(self, interpreter):
                    super().__init__()
                    self.interpreter = interpreter

            wrapper = WrapperModule(interpreter)

            # Create a graph that: 1) Takes function arguments 2) Invokes the interpreter
            # 3) Returns the specified return value

            # FIXME: The following code could be greatly simplified by symbolic_trace'ing
            # the wrapper with a Tracer that considers the Wrapper instance a root
            # module, however, I can't get `__call__` exposed on TorchBind classes
            # without it messing up Python `hasattr` for some reason. More digging
            # into CPython's implementation of hasattr is probably in order...

            graph = torch.fx.Graph()
            # Add placeholders for fn inputs
            placeholder_nodes = []
            for name in fn_input_names:
                placeholder_nodes.append(graph.create_node("placeholder", name))

            # Get the interpreter object
            interpreter_node = graph.create_node("get_attr", "interpreter")

            # Add a node to call the interpreter instance
            output_node = graph.create_node(
                op="call_method",
                target="__call__",
                args=(interpreter_node, placeholder_nodes),
            )

            # Register output
            graph.output(output_node)

            graph.lint()

            # Return final GraphModule!!!
            return GraphModule(wrapper, graph)

        # Lower GraphModule to C++ interpreter
        lowered = lower_to_elementwise_interpreter(msm)

        # Compare correctness with original module
        x = torch.rand(3, 4)
        ref_out = msm(x)
        test_out = lowered(x)
        torch.testing.assert_close(test_out, ref_out)

        # Test TorchScript compilation
        scripted_lowered = torch.jit.script(lowered)
        script_out = scripted_lowered(x)
        torch.testing.assert_close(script_out, ref_out)

        # Test TorchScript Ser/De
        import_copy = self.getExportImportCopy(scripted_lowered)
        imported_out = import_copy(x)
        torch.testing.assert_close(imported_out, ref_out)

    def test_reserved_getattr(self):
        """Ensure that we do not name any nodes with a reserved builtin like `getattr`"""

        class M(torch.nn.Module):
            def forward(self, a):
                return a.foo.bar.baz

        m = M()
        m_g = symbolic_trace(m)
        m_g.graph.lint()
        for node in m_g.graph.nodes:
            self.assertTrue(node.name != "getattr")

    @unittest.skip("Hotfix for SEV remediation")
    def test_trace_buffer_slice(self):
        bs, d_hid = 10, 23

        class ExampleCode(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.lin = torch.nn.Linear(d_hid, d_hid)
                self.buffer = torch.nn.Buffer(torch.randn(bs + 100, d_hid))

            def forward(self, x):
                x = torch.mm(x, self.mm_param)
                skip_connection = x
                x = torch.relu(x)
                x = torch.mm(x, self.mm_param) + self.buffer[: x.shape[0]]
                x = self.lin(x)
                x = torch.relu(x)
                x = x + skip_connection
                x = torch.mm(x, self.mm_param2)
                x = self.lin(x)
                return x

        ec = ExampleCode()

        traced = torch.fx.symbolic_trace(ec)

        x = torch.randn(bs, d_hid)
        torch.testing.assert_close(ec(x), traced(x))

    def test_node_tagging(self):
        class TaggingTracer(Tracer):
            def create_node(
                self,
                kind: str,
                target: Union[str, Callable],
                args: tuple[Argument, ...],
                kwargs: dict[str, Any],
                name: Optional[str] = None,
                type_expr: Optional[Any] = None,
            ) -> Node:
                n = super().create_node(kind, target, args, kwargs, name)
                n.tag = "foo"
                return n

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = M()
        g = TaggingTracer().trace(m)
        g.lint()
        for n in g.nodes:
            self.assertTrue(hasattr(n, "tag"))
            self.assertEqual(n.tag, "foo")

    def test_tensor_attribute(self):
        class TensorAttribute(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tensor = torch.rand(3, 4)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.tensor)

        ta = TensorAttribute()
        traced = symbolic_trace(ta)
        traced(torch.rand(4, 4))

        class WrapperForQualname(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ta = TensorAttribute()

            def forward(self, x):
                return torch.nn.functional.linear(x, self.ta.tensor)

        wfq = WrapperForQualname()
        traced2 = symbolic_trace(wfq)
        traced2.graph.lint()
        traced2(torch.rand(4, 4))

    def test_tensor_attribute_coalseced(self):
        def count_attrs(fx_module):
            targets = set()
            for node in traced.graph.nodes:
                if node.op == "get_attr":
                    targets.add(node.target)
            return len(targets)

        val = torch.tensor(5)

        def f(x):
            return x + val + val

        traced = symbolic_trace(f)
        traced.graph.lint()
        self.assertEqual(count_attrs(traced), 1)

        val2 = torch.tensor(5)

        def f(x):
            val = torch.tensor(5)
            return x + val + val2

        traced = symbolic_trace(f)
        traced.graph.lint()
        self.assertEqual(count_attrs(traced), 2)

    def test_symbolic_trace_sequential(self):
        class Simple(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        seq = torch.nn.Sequential(Simple(), Simple(), Simple())
        traced = symbolic_trace(seq)
        traced.graph.lint()
        x = torch.rand(3, 4)
        self.assertEqual(traced(x), seq(x))

    def test_tensor_constant(self):
        class ConstTensor(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, torch.zeros(3, 4))

        ct = ConstTensor()
        traced = symbolic_trace(ct)
        traced.graph.lint()
        traced(torch.rand(4, 4))

    def test_pickle_graphmodule(self):
        class Nested(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.st = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.st(x)

        n = Nested()
        traced = symbolic_trace(n)
        traced.graph.lint()
        pickled = pickle.dumps(traced)
        loaded = pickle.loads(pickled)
        loaded.graph.lint()
        x = torch.rand(3, 4)
        self.assertEqual(loaded(x), traced(x))

    def test_pickle_custom_import(self):
        graph = torch.fx.Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(a_non_torch_leaf, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        pickled = pickle.dumps(gm)
        loaded = pickle.loads(pickled)
        loaded.graph.lint()
        x, y = torch.rand(1), torch.rand(1)
        self.assertEqual(loaded(x, y), gm(x, y))

    def test_all_input_nodes(self):
        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.placeholder("x")
        b: torch.fx.Node = graph.call_module("linear_mod", args=(a,))
        c: torch.fx.Node = graph.get_attr("y_attr")
        d: torch.fx.Node = graph.call_function(operator.add, args=(b, c))
        e: torch.fx.Node = graph.call_function(torch.unsqueeze, args=(d, 0))
        graph.output(e)
        graph.lint()

        self.assertEqual(b.all_input_nodes, [a])
        self.assertEqual(c.all_input_nodes, [])
        self.assertEqual(d.all_input_nodes, [b, c])
        self.assertEqual(e.all_input_nodes, [d])

    def test_deepcopy_graphmodule_with_transform(self):
        st = SimpleTest()
        traced = symbolic_trace(st)
        traced.graph.lint()

        def transform(traced):
            new_graph = torch.fx.Graph()
            val_map: dict[Node, Node] = {}
            output_value = new_graph.graph_copy(traced.graph, val_map)
            relu_out = new_graph.create_node(
                op="call_method", target="neg", args=(output_value,), kwargs={}
            )
            new_graph.output(relu_out)
            return GraphModule(traced, new_graph)

        transformed = transform(traced)
        transformed.graph.lint()
        copied = copy.deepcopy(transformed)
        self.assertNotEqual(id(type(transformed)), id(type(copied)))
        x = torch.randn(3, 4)
        self.assertEqual(copied(x), transformed(x))

    def test_deepcopy_with_submods_params(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))

            def forward(self, x):
                return torch.relu(x) + self.param

        class Baz(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.bar = Bar()

            def forward(self, x):
                return self.bar(x) - self.param

        baz = Baz()
        traced = symbolic_trace(baz)
        traced.graph.lint()
        copied = copy.deepcopy(traced)
        copied.graph.lint()

    def test_deepcopy_graph_with_tracer_cls(self):
        class TestTracer(Tracer):
            def is_leaf_module(self, module, name):
                return True

        g = Graph(tracer_cls=TestTracer)
        x = g.placeholder("x")
        g.output(x)

        h = copy.deepcopy(g)
        self.assertIsNotNone(h._tracer_cls)
        self.assertTrue(g._tracer_cls == h._tracer_cls)

    def test_unpack_list_better_error(self):
        class SomeArgs(torch.nn.Module):
            def forward(self, a, b):
                return torch.rand(3, 4)

        class UnpacksList(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sa = SomeArgs()

            def forward(self, x: list):
                return self.sa(*x)

        ul = UnpacksList()
        with self.assertRaisesRegex(TraceError, "Proxy object cannot be iterated."):
            symbolic_trace(ul)

    def test_unpack_dict_better_error(self):
        class SomeKwargs(torch.nn.Module):
            def forward(self, x=3, y=4):
                return torch.rand(3, 4)

        class UnpacksDict(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sk = SomeKwargs()

            def forward(self, x: dict):
                return self.sk(**x)

        ud = UnpacksDict()
        with self.assertRaisesRegex(TraceError, "Proxy object cannot be iterated."):
            symbolic_trace(ud)

    def test_pretty_print_targets(self):
        # Test that Graph pretty-print prints friendly name for targets
        # in `operator` and `builtins`

        class SomeMod(torch.nn.Module):
            def forward(self, x):
                return torch.add(x.foo + x.bar, 3.0)

        traced = symbolic_trace(SomeMod())
        graph_str = str(traced.graph)
        self.assertIn("builtins.getattr", graph_str)
        self.assertIn("operator.add", graph_str)
        self.assertIn("torch.add", graph_str)

    def test_pretty_print_node(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param: torch.nn.Parameter = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x: torch.Tensor, y: int = 2):
                return self.linear(x[y] + self.param).clamp(min=0.0, max=1.0)

        traced = symbolic_trace(M())

        all_formatted = "\n".join([n.format_node() for n in traced.graph.nodes])

        FileCheck().check("x").check("placeholder").check("y").check(
            "placeholder"
        ).check("getitem").check("call_function").check("param").check(
            "get_attr"
        ).check("add").check("call_function").check("linear").check(
            "call_module"
        ).check("clamp").check("call_method").run(all_formatted)

    def test_print_graph(self):
        op: torch._ops.OpOverload = torch.ops.aten.relu.default
        type_name: str = torch.typename(op)

        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.create_node("placeholder", "x")
        b: torch.fx.Node = graph.create_node("call_function", op, (a,), type_expr=type_name)
        c: torch.fx.Node = graph.create_node("call_function", op, (b,), type_expr=type_name)
        graph.output((b, c))

        gm: torch.fx.GraphModule = torch.fx.GraphModule(
            torch.nn.Module(), graph
        )
        gm.graph.lint()
        text = gm.print_readable(False)
        assert 2 == text.count("_torch__ops_aten_aten_relu_")

    def test_script_tensor_constant(self):
        # TorchScript seems to ignore attributes that start with `__`.
        # We used to call anonymous Tensor values `__tensor_constant*`, but
        # they were getting ignored by script. Now they're called
        # `_tensor_constant*`
        class IHaveATensorConstant(torch.nn.Module):
            def forward(self, x):
                return x + torch.rand(3, 4)

        traced = torch.fx.symbolic_trace(IHaveATensorConstant())
        torch.jit.script(traced)

    def test_autowrap_functions(self):
        class AutowrapFnTest(torch.nn.Module):
            def forward(self, x):
                return fx_int(x.shape[0] / 2)

        class AutowrapFnTest2(torch.nn.Module):
            def forward(self, x):
                return fx_int(x.shape[0] / 2) + fx_int_x2(x.shape[0] / 2)

        # Check function(s) are wrapped
        # `int` would normally throw a TypeError as argument can't be `Proxy`
        tracer = Tracer(autowrap_functions=(fx_int,))
        graph = tracer.trace(AutowrapFnTest())
        traced = GraphModule(tracer.root, graph, "test")
        tracer_2 = Tracer(autowrap_functions=(fx_int, fx_int_x2))
        tracer_2.trace(AutowrapFnTest2())

        # Test scriptability
        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(4)), 2)

    def test_tuple_no_subscript(self):
        def foo(x: tuple):
            return x[0]

        traced = torch.fx.symbolic_trace(foo)
        x = (torch.randn(5, 3),)
        torch.testing.assert_close(traced(x), x[0])

        bio = io.BytesIO()

        torch.save(traced, bio)

        bio.seek(0)

        # weights_only=False as this loads a GraphModule
        # GLOBAL torch.fx.graph_module.reduce_graph_module was not an allowed global by default
        loaded = torch.load(bio, weights_only=False)

        torch.testing.assert_close(loaded(x), x[0])

    def test_torch_fx_len(self):
        class FXLenTest(torch.nn.Module):
            def forward(self, x):
                return len(x)

        traced = symbolic_trace(FXLenTest())
        self.assertEqual(traced(torch.rand(3, 4)), 3)

        # Test scriptability
        scripted = torch.jit.script(FXLenTest())
        self.assertEqual(scripted(torch.rand(3)), 3)

        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(3)), 3)

        # Test non-proxy len
        class FXLenTest2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = [3, 4, 5]

            def forward(self, x):
                return x + len(self.l)

        traced2 = symbolic_trace(FXLenTest2())
        inp = torch.rand(3, 4)
        self.assertEqual(traced2(inp), inp + 3.0)
        self.assertIs(len, builtins.len)

    def test_torch_fx_getattr(self):
        class FXGetattrTest(torch.nn.Module):
            def forward(self, x):
                return getattr(x, "nonexistent_attr", torch.Tensor([2, 3]))

        traced = symbolic_trace(FXGetattrTest())
        self.assertEqual(traced(torch.rand(3, 4)), torch.Tensor([2, 3]))

    def test_sqrt(self):
        class Sqrt1(torch.nn.Module):
            def forward(self, x):
                return sqrt(x.size(0))

        class Sqrt2(torch.nn.Module):
            def forward(self, x):
                return math.sqrt(x.size(0))

        class Sqrt3(torch.nn.Module):
            def forward(self, x):
                return x + math.sqrt(2) + sqrt(2)

        self.checkGraphModule(Sqrt1(), [torch.zeros(8)])
        self.checkGraphModule(Sqrt2(), [torch.zeros(8)])
        self.checkGraphModule(Sqrt3(), [torch.zeros(8)])
        self.assertIs(sqrt, _sqrt)
        self.assertIs(math.sqrt, _sqrt)

    def test_torch_custom_ops(self):
        class M(torch.nn.Module):
            def forward(self, a):
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))

        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint()
        out = gm(input)
        self.assertEqual(out, ref_out)

    def test_torch_op_overloads(self):
        class M(torch.nn.Module):
            def forward(self, a):
                b = torch.ops.aten.add.Tensor(a, a)
                return b

        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint()
        out = gm(input)
        self.assertEqual(out, ref_out)

        for node in gm.graph.nodes:
            if node.op == "call_function":
                assert isinstance(node.target, torch._ops.OpOverload)
                assert node.target.__name__ == "add.Tensor"

    def test_pickle_torch_custom_ops(self):
        class M(torch.nn.Module):
            def forward(self, a):
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))

        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint()
        pickled = pickle.dumps(gm)
        loaded = pickle.loads(pickled)
        self.assertEqual(loaded(input), gm(input))

    def test_pretty_print(self):
        st = SimpleTest()
        traced = symbolic_trace(st)
        traced.graph.lint()
        printed = str(traced)
        assert "SimpleTest()" in printed
        assert "torch.relu" in printed

    def test_pretty_print_
```



## High-Level Overview


This Python file contains 193 class(es) and 575 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleTest`, `Pair`, `Foo`, `Add`, `TestFX`, `MySub`, `MyModule`, `M2`, `T`, `M3`, `MyModule`, `T`, `T`, `T`, `MyModule`, `MyModule`, `MyDictMod`, `NoMutableCallTracer`, `MyInplaceMod`, `MyInplaceMod2`

**Functions defined**: `forward`, `a_non_torch_leaf`, `fx_int`, `fx_int_x2`, `a_lifted_leaf`, `a_lifted_leaf2`, `wrapped_named_tup`, `wrapped_via_decorator`, `wrapped_with_submodule`, `my_decorator`, `wrapper_inside_decorator`, `wrapped_decorated_fn`, `wrapper_fn`, `_custom_fx_repr_fn`, `__init__`, `forward`, `side_effect_func`, `_enrich_profiler_traces`, `setUp`, `tearDown`

**Key imports**: builtins, collections, contextlib, copy, gc, functools, inspect, io, math, numbers


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `builtins`
- `collections`
- `contextlib`
- `copy`
- `gc`
- `functools`
- `inspect`
- `io`
- `math`
- `numbers`
- `operator`
- `os`
- `pickle`
- `sys`
- `traceback`
- `types`
- `typing`
- `unittest`
- `weakref`
- `warnings`
- `torch.multiprocessing`: Process
- `torch.testing`: FileCheck
- `torch.testing._internal.common_methods_invocations`: op_db
- `torch.testing._internal.common_device_type`: ops, onlyCPU, instantiate_device_type_tests
- `torch.utils._pytree as pytree`
- `torch.fx._pytree as fx_pytree`
- `torch.fx`: symbolic_trace, Proxy, Node, GraphModule, Interpreter, Tracer, Transformer, Graph, wrap, PH, CodeGen
- `torch.fx.node`: Target, Argument, ArgumentT, _format_arg
- `torch.fx.passes`: shape_prop


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_fx.py
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

- **File Documentation**: `test_fx.py_docs.md`
- **Keyword Index**: `test_fx.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

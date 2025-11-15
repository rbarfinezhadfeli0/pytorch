# Documentation: `docs/test/jit/test_tracer.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_tracer.py_docs.md`
- **Size**: 53,919 bytes (52.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_tracer.py`

## File Metadata

- **Path**: `test/jit/test_tracer.py`
- **Size**: 92,325 bytes (90.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import copy
import io
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import warnings

# Standard library
from collections import namedtuple
from itertools import chain
from typing import Dict, List, Optional, Tuple

from torch import Tensor
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    IS_SANDCASTLE,
    raise_on_run_directly,
    skipIfCompiledWithoutNumpy,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TemporaryFileName,
)
from torch.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,
    _trace,
    enable_cpu_fuser,
    JitTestCase,
    make_global,
    RUN_CUDA,
    RUN_CUDA_MULTI_GPU,
)


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
class TestTracer(JitTestCase):
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_large_nbr_kernel_args(self):
        class Recurrence(nn.Module):
            def __init__(self, seq_len):
                super().__init__()
                self.seq_len = seq_len

            def forward(self, input):
                input = input.transpose(0, 1)

                # Main loop
                output = []
                for i in range(self.seq_len):
                    b = input[i] * 2
                    output.append(b)

                output = torch.cat(output, 0).view(input.size(0), *output[0].size())
                output = output.transpose(0, 1)
                return output

        input_size = 8
        batch_size = 2
        seq_len = 130

        rec = Recurrence(seq_len)
        input = torch.rand(batch_size, seq_len, input_size)

        torch.cuda.set_device(0)
        rec = rec.cuda()
        input = input.cuda()

        traced_rec = torch.jit.trace(rec, (input))

    def test_trace_legacy_ctor(self):
        class MyModule(nn.Module):
            def forward(self, x):
                return (x + 1, torch.FloatTensor([0]))

        traced_rec = torch.jit.trace(MyModule(), torch.randn(2, 2))

    def test_simple(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        self.checkTrace(f, (x, y))

    def test_trace_checking_with_global_name(self):
        class MyClass(torch.nn.Module):
            def forward(self, xs: List[Tensor]):
                y = torch.cat(xs, dim=0)
                return y

        model = MyClass()
        # Simulate these inputs being in the globals, like they would be if,
        # e.g. they were defined outermost scope of a script
        global input1, input2
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 2)
        m2 = torch.jit.trace(model, ((input1, input2),))

    def test_trace_aliased_parameter(self):
        class M(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = nn.Parameter(x)

            def forward(self, y):
                return self.x + y

        m = M(torch.rand(3, 4))
        r = torch.jit.trace(m, m.x)
        t2 = torch.rand(3, 4)
        self.assertEqual(r(t2), m.x + t2)

    def test_trace_nested_fn(self):
        class TracedInlineDecision(torch.nn.Module):
            def forward(self, x, flag):
                @torch.jit.script
                def make_decision(flag, x):
                    if flag:
                        return x
                    else:
                        return torch.zeros_like(x)

                x = torch.neg(x)
                return make_decision(flag, x)

        decision = TracedInlineDecision()
        torch.jit.trace(
            decision,
            (torch.rand(3, 4), torch.tensor([True], dtype=torch.bool)),
            check_trace=True,
        )

    def test_trace_single_tuple(self):
        x = torch.tensor(2.0)

        def f2(x):
            return (x,)

        jit_f2 = torch.jit.trace(f2, x)
        assert f2(x) == jit_f2(x)  # fails

    def test_trace_out_operator_with_two_output(self):
        example_input = torch.rand(2, 8)
        out_1, out_2 = torch.cummax(example_input, 1)

        def run_cummax(example_input, out_1, out_2):
            output_1, output_2 = torch.cummax(example_input, 1, out=(out_1, out_2))
            return output_1, output_2

        trace_model = torch.jit.trace(run_cummax, (example_input, out_1, out_2))

    def test_trace_namedtuple(self):
        Point = namedtuple("point", ["x", "y"])

        def f(p):
            if type(p) is tuple:
                p = Point(*p)
            return p.x + p.y

        p = Point(torch.randn(1), torch.randn(1))
        traced = torch.jit.trace(f, (p,))
        self.assertEqual(f(p), traced(p))

    def test_trace_topk(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x.topk(y, dim=1)[1]

        mod = M()
        inputs = (torch.randint(0, 10, (20, 20)), torch.tensor(17))
        traced_func = torch.jit.trace(mod, inputs)

        test_inputs = (torch.randint(0, 9, (9, 9)), torch.tensor(8))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        self.assertEqual(eager_out, traced_out)

        test_inputs = (torch.randint(0, 50, (50, 50)), torch.tensor(12))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        self.assertEqual(eager_out, traced_out)

    def test_typeas_trace_check(self):
        a = torch.tensor([0.4], requires_grad=True)
        b = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            return x.type_as(y)

        trace = torch.jit.trace(f, (a, b))

    def test_trace_index(self):
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0], dtype=torch.int64)

        def fn(x, y):
            return x[y]

        fn_traced = torch.jit.trace(
            fn,
            (
                x,
                y,
            ),
        )

        self.assertEqual(fn(x, y), fn_traced(x, y))

    # Backwards tracing was broken for indexing by a constant,
    # because it's internally implemented using as_strided,
    # and we attempted to trace its derivative (which is not
    # currently supported.)  It currently works because
    # slice() is now not marked as traceable.
    def test_trace_index_constant(self):
        x = torch.tensor([0.4], requires_grad=True)

        def fn(x):
            return x[0]

        def run(f):
            y = f(x)
            grad = torch.autograd.grad(y, x)[0].clone()
            return y, grad

        traced_fn = torch.jit.trace(fn, torch.ones(1))
        self.assertEqual(run(fn), run(traced_fn))

    def test_index_put(self):
        ten = torch.zeros(3, 3)
        mask = torch.tensor(
            [[True, True, True], [True, False, False], [True, True, False]]
        )

        def test_fn(ten, mask):
            ten[mask] = torch.ones(6)
            return ten

        traced_test_fn = torch.jit.trace(test_fn, (ten, mask))

        ten = torch.rand(3, 3)
        self.assertEqual(test_fn(ten, mask), traced_test_fn(ten, mask))

    def test_canonicalize_tensor_iterator(self):
        x = torch.randn(4, 4)

        def f(x):
            x = x + 2
            x = x - 4
            x = x * 6
            x = x / 8
            return x

        traced = torch.jit.trace(f, (x,))
        f(x)
        graph = traced.graph_for(x)
        # There should be 4 int constants for the right sides of operators, plus one
        # for the alpha argument for add and sub
        self.assertTrue(str(traced.graph_for(x)).count(": int = prim::Constant") == 5)

    @suppress_warnings
    def test_constant(self):
        x = torch.randn(2, 2, requires_grad=True)

        def f(x):
            return x.matmul(torch.diag(torch.tensor([2.0, 2.0])))

        self.checkTrace(f, (x,), (torch.ones(2, 2, requires_grad=True),))

    def test_wrapped_number(self):
        # Scalar's get converted to 'wrapped' tensors of default tensor type.
        # Wrapped tensors behave differently in certain promotion operations:
        # float_tensor * double -> float but wrapped_float * double -> double.
        # This can cause issues in check-trace if not handled correctly in
        # `aten::isclose()`.

        def foobar():
            x = -10000.0
            result = x * torch.ones(1, dtype=torch.float)
            return result

        scripted = torch.jit.trace(foobar, (), check_trace=True)

    def test_inplace_transplant(self):
        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            y = x.clone()
            y.add_(2)
            y.add_(3)
            return y

        g, _ = torch.jit._get_trace_graph(fn, (x,))
        self.run_pass("dce", g)
        FileCheck().check_count("aten::clone", 1, exactly=True).check_count(
            "aten::add_", 2, exactly=True
        ).check_next("return").run(str(g))
        self.assertExportImport(g, (x,))

    def test_inplace_flags(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                return go

        class RegularFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                return go

        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            y = RegularFn.apply(x)
            y = InplaceFn.apply(y)
            y = InplaceFn.apply(y)
            y = RegularFn.apply(y)
            return y

        trace_graph, _ = torch.jit._get_trace_graph(fn, (x,), _force_outplace=True)
        self.run_pass("dce", trace_graph)
        ops = list(trace_graph.nodes())
        for op in ops:
            self.assertTrue(op.hasAttribute("inplace"))
        inplace_flags = [False, True, True, False]
        for op, is_inplace in zip(ops, inplace_flags):
            self.assertEqual(op.i("inplace"), is_inplace)

    def test_inplace_check(self):
        class MyInplaceFn(Function):
            @staticmethod
            def forward(self, x):
                x.add_(1)
                self.mark_dirty(x)
                return x

            @staticmethod
            def backward(self, grad):
                return grad

        def fn(x):
            return MyInplaceFn.apply(x)

        x = torch.randn(5, 5)
        ge = torch.jit.trace(fn, (x,), _force_outplace=True, check_trace=False)
        with self.assertRaisesRegex(RuntimeError, "inplace MyInplaceFn"):
            ge(x)

    def test_force_outplace_check_fill(self):
        def f(x):
            return torch.empty(x.shape).fill_(7)

        x = torch.randn(10, 15)
        ft = torch.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def test_force_outplace_check_zero(self):
        def f(x):
            return torch.empty(x.shape).zero_()

        x = torch.randn(10, 15)
        ft = torch.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def do_trace_size(self, requires_grad):
        def fn(x):
            return x.view(x.shape[1] * 2, x.size(0), 2)

        x = torch.randn(5, 2, 4, requires_grad=requires_grad)
        y = torch.randn(4, 8, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def test_trace_size(self):
        self.do_trace_size(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_size_with_grad(self):
        self.do_trace_size(True)

    def test_trace_numel(self):
        def fn(x):
            return x.numel()

        x = torch.randn(2, 3, 4)
        y = torch.randn(4, 5, 6)

        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def do_trace_arange(self, requires_grad):
        def arange(x):
            return torch.arange(x.shape[0])

        def arange_scalar(x):
            return torch.arange(12)

        def arange_start_end(x):
            return torch.arange(start=x.shape[0], end=x.shape[0] + 5)

        x = torch.randn(5, 3, 2, requires_grad=requires_grad)
        y = torch.randn(8, 2, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_arange = torch.jit.trace(arange, x)
        self.assertEqual(traced_arange(y), arange(y))
        self.assertEqual(traced_arange(x), arange(x))

        traced_arange_scalar = torch.jit.trace(arange_scalar, x)
        self.assertEqual(traced_arange_scalar(y), arange_scalar(y))
        self.assertEqual(traced_arange_scalar(x), arange_scalar(x))

        traced_arange_start_end = torch.jit.trace(arange_start_end, x)
        self.assertEqual(traced_arange_start_end(y), arange_start_end(y))
        self.assertEqual(traced_arange_start_end(x), arange_start_end(x))

    def test_trace_arange(self):
        self.do_trace_arange(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_arange_with_grad(self):
        self.do_trace_arange(True)

    # Test that a trace of torch.full(x.shape) doesn't store the shape as a constant
    def test_trace_full_dynamic_shape(self):
        def full_with_shape_like(x):
            return torch.full(x.shape, 2.0)

        x = torch.randn(3, 4)
        ge = torch.jit.trace(full_with_shape_like, example_inputs=x)
        y = torch.randn(2, 7)
        self.assertEqual(ge(y).shape, y.shape)
        self.assertEqual(ge(x).shape, x.shape)

    # Test that the trace of setitem doesn't store shapes as constants
    # Fix https://github.com/pytorch/pytorch/issues/43548
    def test_trace_slice_setitem_dynamic_shape(self):
        def slice_setitem(x, y):
            x[:, 2] = y + 1
            return x

        x = torch.randn(3, 4)
        traced = torch.jit.trace(slice_setitem, (x, x[:, 0]))
        x = torch.randn(10, 5)
        self.assertEqual(traced(x.clone(), x[:, 0]), slice_setitem(x.clone(), x[:, 0]))

    # Suppression: we are intentionally slicing a tensor, we don't care that it
    # will be constantified
    @suppress_warnings
    def do_trace_slice(self, requires_grad):
        def slice(x):
            results = []
            for i in range(4):
                results.append(x[: x.size(0) - i, i : x.size(2), i:3])
            return tuple(results)

        def slice_select(x):
            results = []
            for i in range(4):
                results.append(x[:, i:, x.size(2) - 5])
            return tuple(results)

        x = torch.randn(5, 6, 7, requires_grad=requires_grad)
        y = torch.randn(7, 8, 9, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_slice = torch.jit.trace(slice, x)
        self.assertEqual(traced_slice(y), slice(y))
        self.assertEqual(traced_slice(x), slice(x))

        traced_slice_select = torch.jit.trace(slice_select, x)
        self.assertEqual(traced_slice_select(y), slice_select(y))
        self.assertEqual(traced_slice_select(x), slice_select(x))

    def test_trace_slice(self):
        self.do_trace_slice(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_slice_with_grad(self):
        self.do_trace_slice(True)

    def test_trace_casts(self):
        casts = [
            lambda x: x.byte(),
            lambda x: x.float(),
            lambda x: x.cpu(),
            lambda x: x.to(device="cpu"),
            lambda x: x.to(dtype=torch.int64),
            lambda x: x.to(device="cpu", dtype=torch.float),
            lambda x: x.to(x),
        ]

        def assertContainsCast(trace):
            self.assertEqual(
                sum(n.kind() == "aten::to" for n in trace.graph.nodes()), 1
            )

        for cast in casts:
            trace = torch.jit.trace(cast, torch.randn(2, 2))
            assertContainsCast(trace)
            x = torch.randn(2, 2)
            self.assertEqual(trace(x), cast(x))

        def to_tensor(x, y):
            return x.to(y)

        to_tensor_trace = torch.jit.trace(
            to_tensor, (torch.randn(2, 2), torch.randn(1, 8))
        )
        assertContainsCast(to_tensor_trace)
        x, y = torch.randn(2, 2), torch.randn(1, 10)
        self.assertEqual(to_tensor_trace(x, y), to_tensor(x, y))

    @skipIfCompiledWithoutNumpy
    @skipIfCrossRef
    def test_trace_warn(self):
        def fn(x):
            int(x)  # Warning 1.
            y = x * 1
            if y:  # Warning 2.
                pass
            q = [x, x * 4]
            z = q[y]
            float(z)  # Warning 3.
            z.tolist()  # Warning 4.
            z.numpy()  # Warning 5.
            for _ in torch.ones(4, 4):  # Warning 6.
                pass
            return z + 4

        with warnings.catch_warnings(record=True) as warns:
            traced_fn = torch.jit.trace(fn, torch.tensor([1]))
        for warn in warns:
            self.assertIs(warn.category, torch.jit.TracerWarning)
        warns = [str(w.message) for w in warns]
        self.assertIn("a Python integer", warns[0])
        self.assertIn("a Python boolean", warns[1])
        self.assertIn("a Python float", warns[2])
        self.assertIn("a Python list", warns[3])
        self.assertIn("a NumPy array", warns[4])
        self.assertIn("Iterating over", warns[5])

    def test_trace_tuple(self):
        def fn(x, y):
            return x, (x * y[1], x * y[0])

        x, y = torch.randn(2, 2), (torch.ones(2, 2), torch.randn(2, 2))
        traced_fn = torch.jit.trace(fn, (x, y))
        self.assertEqual(traced_fn(x, y), fn(x, y))
        # should be a tuple nested within another tuple
        FileCheck().check_count("prim::TupleConstruct", 2, exactly=True).check_next(
            "return"
        ).run(str(traced_fn.graph))
        self.assertExportImport(traced_fn.graph, (x, y))

    def test_trace_random(self):
        def f(mean, std):
            return torch.normal(mean, std)

        traced = torch.jit.trace(
            f, (torch.zeros(2, 3), torch.ones(2, 3)), check_trace=False
        )
        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        with torch.random.fork_rng(devices=[]):
            output = f(mean, std)
        traced_output = traced(mean, std)
        self.assertEqual(output, traced_output)

    def test_trace_tensor_factory(self):
        def run(**kwargs):
            inputs_require_grads = kwargs.pop("inputs_require_grads", True)

            def fn(x):
                return x + torch.ones(2, 3, **kwargs)

            input_kwargs = kwargs.copy()
            if "out" in input_kwargs:
                del input_kwargs["out"]
            input = torch.ones(2, 3, **input_kwargs)
            self.checkTrace(fn, (input,), inputs_require_grads=inputs_require_grads)
            # check we recorded 'ones' and did not just record a constant
            tfn = torch.jit.trace(fn, input)
            self.assertTrue("ones" in str(tfn.graph))

        run()
        run(dtype=torch.int, inputs_require_grads=False)
        run(out=torch.tensor([]))
        if RUN_CUDA:
            run(device="cuda:0")
        if RUN_CUDA_MULTI_GPU:
            run(device="cuda:1")

    def test_trace_indexed_assignment(self):
        def stuff(x, y):
            x = x.clone()
            x[0] = y
            return x

        example = torch.rand(3, 4)
        self.checkTrace(stuff, (example, example[0] + 1))

    # TODO: implement
    @unittest.expectedFailure
    def test_output_unflatten(self):
        """Check that outputs of traced functions retain the original structure and nesting"""

        def fn(x):
            return (
                x * 2,
                (
                    x**2,
                    x + 4,
                    (x + 2,),
                ),
                x * 4,
            )

        self.checkTrace(fn, (torch.randn(2, 2),))

    def test_input_flatten(self):
        """Check that inputs to traced functions are flattened"""

        def fn(x, t):
            y, z = t
            return x * y * z

        inputs = (torch.randn(1), (torch.randn(1), torch.randn(1)))
        self.checkTrace(fn, inputs)

    def test_input_dict_empty(self):
        def test(d):
            pass

        with self.assertRaises(RuntimeError):
            self.checkTrace(test, {})

    def test_input_dict_remembers_keys(self):
        """Check that the trace remembers which keys were in a dict input"""

        class TestModule(torch.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"]

        input_1 = {"x": torch.tensor(1)}
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_1,))
        self.assertEqual(m_traced(input_1), torch.tensor(1))

        # should work to change the values and not the keys
        input_same_key_different_value = {"x": torch.tensor(2)}
        self.assertEqual(m_traced(input_same_key_different_value), torch.tensor(2))

        # error to use something that doesn't have `x`
        input_different_key = {"y": torch.tensor(3)}
        with self.assertRaises(RuntimeError):
            m_traced(input_different_key)

        # it's okay to have additional elements in the dictionary, so long as 'x' is there
        input_additional_key = {"x": torch.tensor(4), "y": torch.tensor(3)}
        self.assertEqual(m_traced(input_additional_key), torch.tensor(4))

    def test_input_dict_insertion_order(self):
        """Check that dictionary access doesn't care about insertion order"""

        class TestModule(torch.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"], dict_input["y"]

        input_x_then_y = {}
        input_x_then_y["x"] = torch.tensor(1)
        input_x_then_y["y"] = torch.tensor(2)

        m = TestModule()
        m_traced = torch.jit.trace(m, (input_x_then_y,))

        self.assertEqual(m_traced(input_x_then_y), (torch.tensor(1), torch.tensor(2)))

        input_y_then_x = {}
        input_y_then_x["y"] = torch.tensor(4)
        input_y_then_x["x"] = torch.tensor(3)

        self.assertEqual(m_traced(input_y_then_x), (torch.tensor(3), torch.tensor(4)))

    def test_input_dict_recursive(self):
        class TestModule(torch.nn.Module):
            def forward(self, dict_input):
                return dict_input["x"][1]

        input_1 = {"x": {1: torch.tensor(1)}}
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_1,))

        input_2 = {"x": {1: torch.tensor(2)}}
        self.assertEqual(m_traced(input_2), torch.tensor(2))

    def test_input_dict_checkTrace_mut(self):
        def test(d):
            d["x"].tanh_()
            return d["x"]

        inputs = {"x": torch.rand(3, 4), "y": torch.rand(3, 4)}
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_dict_unify(self):
        def test(d):
            return d["int"], d["float"]

        inputs = {
            "int": torch.ones((2, 2), dtype=torch.int32),
            "float": torch.ones((2, 2), dtype=torch.float32),
        }
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_tuple_of_dicts(self):
        def test(t):
            d = t[0]
            return d["x"]["y"]

        inputs = {"x": {"y": torch.rand(2, 3)}}
        self.checkTrace(test, ((inputs, inputs),), allow_unused=True)

    def test_input_dict_of_dicts(self):
        def test(d):
            return d["x"]["y"]

        nested_input = {"y": torch.rand(2, 3)}
        unified_nested = {"y": torch.rand(3, 2)}
        inputs = {"x": nested_input, "force_unify": unified_nested}
        self.checkTrace(test, (inputs,), allow_unused=True)

    def test_input_dict_of_lists(self):
        def test(d):
            return d["x"][0]

        inputs = {"x": [torch.rand(3, 2)]}
        self.checkTrace(test, (inputs,))

    def test_input_list_toplevel_flatten(self):
        def test(t1, t2):
            return torch.add(t1, t2)

        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        self.checkTrace(test, inputs)

    def test_input_list_toplevel_flatten_direct(self):
        class Test(torch.nn.Module):
            def forward(self, t1, t2):
                return torch.add(t1, t2)

        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        torch.jit.trace(Test(), inputs)

    def test_input_list_of_tuples(self):
        def test(l):
            return l[0][0]

        inputs = [(torch.ones(2, 2),)]
        self.checkTrace(test, (inputs,))

    def test_input_dict_empty_list(self):
        def test(d):
            pass

        inputs = {1: []}
        with self.assertRaisesRegex(RuntimeError, "List trace"):
            self.checkTrace(test, (inputs,))

    def test_input_list_mixed_type(self):
        def test(d):
            pass

        inputs = [torch.rand(2, 3), (torch.ones(2), torch.ones(2))]
        with self.assertRaisesRegex(RuntimeError, "consistent"):
            self.checkTrace(test, (inputs,))

    def test_conv(self):
        x = torch.ones(20, 16, 50, 40)
        g, outputs, inputs = torch.jit._get_trace_graph(
            nn.Conv2d(16, 13, 3, bias=False), x, return_inputs=True
        )
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))

    def test_max_pool(self):
        x = torch.rand(20, 16, 10, 10)

        def max_pool2d(x):
            return F.max_pool2d(x, 2) + 2

        trace = torch.jit.trace(max_pool2d, (x))
        graph = trace.graph_for(x)
        FileCheck().check("aten::max_pool2d(").run(graph)
        self.assertEqual(max_pool2d(x), trace(x))

    def test_nested_inplace(self):
        x = torch.randn(2, 2)
        g, outputs, inputs = torch.jit._get_trace_graph(
            lambda x: F.threshold(x, 0, 0, inplace=True), (x,), return_inputs=True
        )
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        FileCheck().check("threshold_").run(str(g))
        self.assertExportImport(g, (x,))

    def test_repeated_input(self):
        def fn(a, b):
            return a + b

        ge = self.checkTrace(fn, [torch.randn(2, 2)] * 2)
        inputs = set(ge.graph.inputs())
        # three instead of 2 because the export/import in checkTrace adds a
        # `self` module argument
        self.assertTrue(len(inputs) == 3)

    def test_repeated_output(self):
        def fn(a, b):
            z = a + b
            return z, z

        ge = self.checkTrace(fn, [torch.randn(2, 2) for _ in range(2)])
        tuple_output = list(ge.graph.outputs())[0]
        tuple_inputs = list(tuple_output.node().inputs())
        self.assertTrue(tuple_inputs[0] == tuple_inputs[1])

    def test_inplace_copy(self):
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            out = torch.zeros(x.size())
            out.copy_(x)
            return out

        g, outputs, inputs = torch.jit._get_trace_graph(f, (x,), return_inputs=True)
        self.run_pass("dce", g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))

    def test_inplace_copy_force_outplace(self):
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            out = torch.zeros(x.size())
            out.copy_(x)
            return out

        g, outputs, inputs = torch.jit._get_trace_graph(
            f, (x,), return_inputs=True, _force_outplace=True
        )
        self.run_pass("dce", g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))
        FileCheck().check("expand_as").run(str(g))

    def test_shared_param(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = self.a = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                return x * self.a + self.b

        m = MyModule()
        g, _ = torch.jit._get_trace_graph(m, (torch.randn(2, 2),))
        self.run_pass("dce", g)
        self.assertEqual(len(list(g.inputs())), 2)
        FileCheck().check("mul").check("add").run(str(g))

    def run_ge_tests(self, optimize, use_cuda):
        with enable_profiling_mode_for_profiling_tests():
            with torch.jit.optimized_execution(optimize):

                def rand(*args):
                    t = torch.rand(*args).float()
                    if use_cuda:
                        t = t.cuda()
                    return t

                self.checkTrace(
                    lambda a, b: a * b + b, [rand(1), rand(1)], [rand(2, 3), rand(2, 3)]
                )
                # trivial identity
                self.checkTrace(lambda a, b: (b, a), [rand(1), rand(1)])

                def foo(a):
                    t = a * a
                    return t * t, 4 * t

                self.checkTrace(foo, [rand(1)])
                # unused input
                self.checkTrace(
                    lambda a, b: a * a, [rand(1), rand(1)], allow_unused=True
                )
                # test outputs that do not get used in grad
                self.checkTrace(foo, [rand(1)], drop=1)
                # test autograd fallback
                self.checkTrace(
                    lambda a, b: a * b / (a - 2 * b) + b, [rand(1), rand(1)]
                )

    def test_ge_unoptimized(self):
        self.run_ge_tests(False, False)

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    @enable_cpu_fuser
    def test_ge_optimized(self):
        with enable_profiling_mode_for_profiling_tests():
            self.run_ge_tests(True, False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_ge_cuda(self):
        self.run_ge_tests(True, True)

    # more manual test of graph executor that can be used as a scratchpad
    def test_ge(self):
        def foo(a, b):
            return a * b / (a - b) + b

        V = Variable
        a, b = V(torch.rand(1)), V(torch.rand(1))
        ge = torch.jit.trace(foo, (a, b))
        a, b = (
            V(torch.rand(1), requires_grad=True),
            V(torch.rand(1), requires_grad=True),
        )
        (r,) = ge(a, b)
        da, db = torch.autograd.grad(r + 3, [a, b], create_graph=True)

        l2 = da * db + db * db
        g2result = torch.autograd.grad(l2, [da, db])

        r = foo(a, b)
        da2, db2 = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        l3 = da2 * db2 + db2 * db2
        g2result2 = torch.autograd.grad(l3, [da2, db2])
        self.assertEqual(g2result, g2result2)

    def test_trace_annotation(self):
        @_trace(torch.rand(1))
        def foo(a):
            return a + a + a

        x = torch.randn(5, 5)
        self.assertEqual(foo(x), x + x + x)

    @unittest.skipIf(not RUN_CUDA, "calls .cuda()")
    # By default, on Ampere or later GPUs, nn.Linear computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_traced_module_cuda(self):
        class Model(nn.Module):
            def __init__(self, num_features, num_layers):
                super().__init__()
                self.num_layers = num_layers
                layers = [
                    [nn.Linear(num_features, num_features), nn.Sigmoid()]
                    for _ in range(num_layers)
                ]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x

        model = Model(5, 3)
        x = torch.randn(2, 5)
        traced_model = torch.jit.trace(model, x)

        # We're missing some attributes these modules had initially. Make sure we can
        # still get the __repr__()
        model.__repr__()

        # XXX: indexing sequentials is broken
        linear_submodule = next(iter(traced_model.submodule._modules.values()))

        # All attributes that aren't parameters should raise
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        linear_submodule.weight
        linear_submodule.weight = nn.Parameter(
            torch.randn(linear_submodule.weight.shape)
        )
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight

        # Submodules can't be called
        with self.assertRaises(RuntimeError):
            linear_submodule(x)

        # Type casts
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to("cuda")
        cuda_out = traced_model(x.float().cuda())
        traced_model.to("cpu")
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to(torch.get_default_dtype())

        # state_dict + load_state_dict
        state = {k: v.clone() for k, v in traced_model.state_dict().items()}
        new_state = {k: v.clone().fill_(1) for k, v in state.items()}
        out = traced_model(x)
        traced_model.load_state_dict(new_state)
        out_ones = traced_model(x)
        traced_model.load_state_dict(state)
        out_state = traced_model(x)
        self.assertEqual(out, out_state)
        self.assertNotEqual(out, out_ones)

    @unittest.skipIf(not RUN_CUDA, "uses cuda")
    def test_type_same_device(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dtype = torch.float16

            def forward(self, x=None):
                h = x.type(self.dtype)
                return h

        a = Model()
        b = torch.jit.trace(
            a, example_inputs=(torch.ones([1], device=torch.device("cuda")),)
        )
        FileCheck().check_not("device").run(b.code)

    def test_export_no_reorder(self):
        def func(a, b):
            return a * b / (a - 2 * b) + b

        recording_inputs = [
            torch.tensor(
                [0.55619788169860839844], dtype=torch.float32, requires_grad=True
            ),
            torch.tensor(
                [0.25947844982147216797], dtype=torch.float32, requires_grad=True
            ),
        ]

        ge1 = torch.jit.trace(func, recording_inputs)
        ge2 = self.getExportImportCopy(ge1)

        outputs_ge1 = ge1(*recording_inputs)
        outputs_ge2 = ge2(*recording_inputs)

        grad_ge1 = torch.autograd.grad(outputs_ge1, recording_inputs)
        grad_ge2 = torch.autograd.grad(outputs_ge2, recording_inputs)
        self.assertTrue(outputs_ge1 == outputs_ge2)
        self.assertTrue(grad_ge1 == grad_ge2)

    def test_python_function(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            return MyFn.apply(x + 2) + 3

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_python_function_tup(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1, x - 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            a, b = MyFn.apply(x + 2)
            return a + b + 3

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_trace_detach(self):
        def foo(x, w):
            return torch.matmul(x, w).detach()

        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        FileCheck().check("matmul").check("detach").run(str(traced.graph))
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_redispatch(self):
        def foo(x, w):
            y = torch.matmul(x, w)
            assert y.requires_grad
            y = y.detach()
            # Make sure trace kernel redispatches to the right lower kernel.
            assert not y.requires_grad
            return y

        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # With `check_trace=True` it will run with `@torch.no_grad()` and break assert.
        torch.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_detach_inplace(self):
        def foo(x, w):
            y = torch.matmul(x, w)
            y.detach_()
            return y

        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        FileCheck().check("matmul").check("detach(").run(str(traced.graph))
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_inplace_redispatch(self):
        def foo(x, w):
            y = torch.matmul(x, w)
            assert y.requires_grad
            y.detach_()
            # Make sure trace kernel redispatches to the right lower kernel.
            assert not y.requires_grad
            return y

        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # With `check_trace=True` it will run with `@torch.no_grad()` and break assert.
        torch.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_slice_full_dim(self):
        def foo(x):
            return x[0:5, 0] + 1.0

        traced = torch.jit.trace(foo, (torch.rand(5, 4),))
        test_x = torch.rand(6, 3)
        self.assertEqual(foo(test_x), traced(test_x))

    def test_trace_dict_input(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self, a, b):
                return self.foo({"a": a, "b": b})["a"]

        class Foo(torch.nn.Module):
            def forward(self, x):
                return {"a": x["a"] * x["b"]}

        x = (torch.rand(3), torch.rand(3))
        model = Bar()
        self.checkTrace(model, x)

    def test_trace_dict_output(self):
        class TraceDictStrTensor(torch.nn.Module):
            def forward(self, a, b):
                return {"a": a, "b": b}

        class TraceDictTensorTensor(torch.nn.Module):
            def forward(self, a, b):
                return {a: b, b: a}

        x = (torch.rand(3), torch.rand(3))
        with self.assertRaisesRegex(RuntimeError, r"Encountering a dict at the output"):
            torch.jit.trace(TraceDictStrTensor(), x)

        traced_dict_str_mod = torch.jit.trace(TraceDictStrTensor(), x, strict=False)
        self.assertEqual(traced_dict_str_mod(*x), {"a": x[0], "b": x[1]})

        traced_dict_tensor_mod = torch.jit.trace(
            TraceDictTensorTensor(), x, strict=False
        )
        self.assertEqual(traced_dict_tensor_mod(*x), {x[0]: x[1], x[1]: x[0]})

    def test_trace_with_tensor_list_output(self):
        def f():
            return [torch.zeros(1), torch.zeros(5)]

        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            torch.jit.trace(f, [])
        traced_non_strict_f = torch.jit.trace(f, [], strict=False)
        self.assertEqual(traced_non_strict_f(), f())

    def test_trace_with_number_list_output(self):
        def f():
            return [1, 5]

        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_tensor_list_output(self):
        def f():
            return [[torch.zeros(1)], [torch.zeros(5)]]

        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_strided_tensor_output(self):
        @torch.jit.script
        def nt_construct(values, kv_lengths):
            kv_lengths_list: List[int] = kv_lengths.tolist()
            return torch._nested_tensor_from_tensor_list(
                list(values.split(kv_lengths_list, dim=0)), None, None, None, None
            )

        def f(x, offsets):
            kv_lengths = offsets[1:] - offsets[:-1]
            return nt_construct(x, kv_lengths).cos()

        x = torch.rand(5, 4)
        offsets = torch.tensor([0, 2, 5])
        ref = f(x, offsets)
        f_t = torch.jit.trace(f, (x, offsets))
        res = f_t(x, offsets)
        self.assertEqual(ref, res)
        x2 = torch.rand((8, 4))
        offsets2 = torch.tensor([0, 2, 4, 8])
        self.assertEqual(f(x2, offsets2), f_t(x2, offsets2))

    def test_trace_variable_instantiation(self):
        def random_foo(x):
            return Variable(Variable(x) + 1.0)

        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        x = torch.rand(5, 6)
        self.assertEqual(random_foo(x), random_foo_traced(x))

    def test_trace_slice_expr_complete_type(self):
        def random_foo(x):
            return x + 1.0

        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        @torch.jit.script
        def random_bar(x):
            return random_foo_traced(x)[0:1]

        x = torch.rand(3, 4)
        self.assertEqual(random_bar(x), (x + 1)[0:1])

    def test_trace_inline_shape(self):
        # testing peephole optimization of size is turned into a constant
        # in script fn

        @torch.jit.script
        def tensor_size(x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([x.size()[0]])

        self.assertEqual(
            tensor_size(
                torch.rand(
                    15,
                )
            ),
            torch.tensor([15]),
        )

        traced_tensor_size = torch.jit.trace(
            tensor_size,
            torch.rand(
                7,
            ),
        )

        self.assertEqual(
            traced_tensor_size(
                torch.rand(
                    15,
                )
            ),
            torch.tensor([15]),
        )

        @torch.jit.script
        def use_device(x):
            return torch.zeros_like(x, device=x.device)

        def foo(x):
            return use_device(x)

        traced_tensor_size = torch.jit.trace(
            foo,
            torch.rand(
                7,
            ),
        )
        self.run_pass("inline", traced_tensor_size.graph)
        FileCheck().check("prim::device").run(traced_tensor_size.graph)

    def test_trace_save(self):
        def fn(x):
            return x + 2

        def check(func):
            with TemporaryFileName() as fname:
                func.save(fname)
                loaded = torch.jit.load(fname)
                input = torch.randn(2, 2)
                self.assertEqual(func(input), loaded(input))

        out = torch.jit.trace(fn, (torch.ones(2, 2),))
        check(out)

    def test_trace_optioanl_dtype(self):
        class Test(torch.nn.Module):
            def forward(self):
                return torch.arange(5)

        traced = torch.jit.trace(Test(), ())
        torch.allclose(traced(), Test()())

    def test_trace_save_load_copy(self):
        class Test(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                return self.conv(x)

        traced = torch.jit.trace(Test(), torch.rand(1, 3, 224, 224))
        buffer = io.BytesIO()
        torch.jit.save(traced, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        # should work
        copy.copy(loaded)
        copy.deepcopy(loaded)

    def test_trace_export_fns(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3

            @torch.jit.export
            def __getstate__(self):
                return (3, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                return x + self.a

        f = Foo()

        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ["__getstate__", "__setstate__"]

        def check(mod):
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )

        check(traced)

        imported = self.getExportImportCopy(traced)
        check(imported)

    def test_trace_export_fns_recursive(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3

            @torch.jit.export
            def __getstate__(self):
                return (3, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                return x + self.a

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        f = Wrapper()

        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ["__getstate__", "__setstate__"]

        def check(mod):
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )

        check(traced.foo)

        imported = self.getExportImportCopy(traced)
        check(imported.foo)

        # Note that Bar's forward can only be traced, but not scripted
        class Bar(nn.Module):
            @torch.jit.export
            def addTwo(self, x):
                return x + 2

            def forward(self, input):
                return (lambda a: a + 1)(input)  # noqa: PLC3002

        # When tracing Bar as a submodule, we only want to script the
        # exported methods, and we want to keep the forwards still
        # being traced.
        class WrapperExports(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            @torch.jit.export
            def addOne(self, x):
                return x + 1

            def forward(self, x):
                return self.bar(x)

        f = WrapperExports()

        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ["addOne"]
        check(traced)

    def test_trace_autograd_function(self):
        class TestFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return torch.neg(input)

            @staticmethod
            def backward(ctx, grad_output):
                return torch.neg(grad_output)

        class TracedModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(TestFunc.apply(x))

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tm = TracedModule()

            def forward(self, x):
                return self.tm(x)

        traced = torch.jit.trace(Wrapper(), (torch.rand(3, 4),))

    def test_trace_multi_output_function(self):
        # An autograd.Function with two outputs.
        # It swaps inputs so we can check if shape
        # handling is correct in TorchScript.
        class Foo(torch.autograd.Function):
            @staticmethod
       
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/jit/test_tracer.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_tracer.py_docs.md_docs.md`
- **Keyword Index**: `test_tracer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

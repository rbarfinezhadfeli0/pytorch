# Documentation: test_control_flow.py

## File Metadata
- **Path**: `test/functorch/test_control_flow.py`
- **Size**: 382610 bytes
- **Lines**: 9752
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: functorch"]
import contextlib
import functools
import unittest

import torch
import torch.utils._pytree as pytree
from functorch.experimental import control_flow
from functorch.experimental.control_flow import cond
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm
from torch._higher_order_ops.associative_scan import (
    _fake_associative_scan,
    associative_scan,
)
from torch._higher_order_ops.map import _fake_map
from torch._higher_order_ops.scan import _fake_scan, scan
from torch._higher_order_ops.schema import HopSchemaGenerator
from torch._higher_order_ops.while_loop import while_loop
from torch._subclasses.functional_tensor import (
    CppFunctionalizeAPI,
    FunctionalTensor,
    FunctionalTensorMode,
    PythonFunctionalizeAPI,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    requires_cuda,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


# TODO: pull these helpers from AOTAutograd later
def to_fun(t):
    if isinstance(t, torch.Tensor):
        return FunctionalTensor.to_functional(t)
    return t


def from_fun(t):
    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)
        return t
    torch._sync(t)
    return torch._from_functional_tensor(t.elem)


def to_fun_old(t):
    if isinstance(t, torch.Tensor) and not torch._is_functional_tensor(t):
        out = torch._to_functional_tensor(t)
        torch._mirror_autograd_meta_to(t, out)
        return out
    return t


def from_fun_old(t):
    # quick sanity assert
    if isinstance(t, torch.Tensor):
        assert torch._is_functional_tensor(t)
        torch._sync(t)
        return torch._from_functional_tensor(t)
    return t


def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands


def compile_mode_helper(fct, compile_mode):
    if compile_mode == "compile":
        return torch.compile(fct, fullgraph=True, dynamic=False)
    elif compile_mode == "compile_dynamic_shape":
        return torch.compile(fct, fullgraph=True, dynamic=True)
    elif compile_mode == "eager":
        return torch.compile(fct, fullgraph=True, backend="eager")
    else:
        return fct


ALIAS_FN = [
    lambda x: x,
    lambda x: x.view(-1),
    lambda x: x.reshape(-1),
    lambda x: x.squeeze(0),
    lambda x: x.unsqueeze(0),
    lambda x: x.transpose(0, 1),
    lambda x: x.flatten(),
    lambda x: x.expand(1, *x.size()),
]


def get_scan_combine_fn(name, associative=True, parameters=None):
    def add(x: torch.Tensor, y: torch.Tensor):
        return x + y

    def adds(x: torch.Tensor, y: torch.Tensor):
        return x + x, y + y

    def mul(x: torch.Tensor, y: torch.Tensor):
        return x * y

    def div(x: torch.Tensor, y: torch.Tensor):
        return x / y

    def s5_operator(x: torch.Tensor, y: torch.Tensor):
        A_i, Bu_i = x
        A_j, Bu_j = y
        return A_j * A_i, A_j * Bu_i + Bu_j

    def different_input_size_operator(x: torch.Tensor, y: torch.Tensor):
        x_o, dA_o, dB_o, C_o, y_o = x
        x_n, dA_n, dB_n, C_n, y_n = y

        x_new = x_n + x_o
        y_new = torch.einsum("bdn,bn->bd", x_new, C_n)

        return x_new, dA_n + 0.0, dB_n + 0.0, C_n + 0.0, y_new

    def tuple_fct(x, y):
        return (x[0] + y[0], x[1] * y[1])

    def complex_pointwise(x, y):
        return {
            "i": x["i"] * y["i"],
            "j": (
                [x["j"][0][0] * y["j"][0][0]],
                [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
            ),
        }

    def non_pointwise(x: torch.Tensor, y: torch.Tensor):
        W = torch.arange(4, dtype=torch.float, device=x.device).view(2, 2)
        return x @ W + y @ W

    def RNN(x: torch.Tensor, y: torch.Tensor):
        c_new = y @ parameters[0] + parameters[1]
        h_new = torch.tanh(c_new + x @ parameters[2] + parameters[3])
        return h_new, h_new.clone()

    def fct_c1_no_grad(x: torch.Tensor, y: torch.Tensor):
        h_new = torch.tanh(x[0] + x[1] + y)
        c2 = x[1] + y
        with torch.no_grad():
            c1 = x[0] + y
        return (c1, c2), h_new

    if name == "add":
        fct = add
    elif name == "adds":
        fct = adds
    elif name == "mul":
        fct = mul
    elif name == "div":
        fct = div
    elif name == "s5_operator":
        fct = s5_operator
    elif name == "different_input_size_operator":
        fct = different_input_size_operator
    elif name == "tuple_fct":
        fct = tuple_fct
    elif name == "complex_pointwise":
        fct = complex_pointwise
    elif name == "non_pointwise":
        fct = non_pointwise
    elif name == "RNN":
        fct = RNN
    elif name == "fct_c1_no_grad":
        fct = fct_c1_no_grad
    else:
        raise ValueError("Combine_fn name unknown!")

    if not associative:
        return lambda x, y: (fct(x, y), fct(x, y))
    else:
        return fct


def _while_loop_tests():
    def simple(x):
        def cond_fn(x):
            return x.sum() < 10

        def body_fn(x):
            return (x + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def simple_with_mutation(x):
        def cond_fn(x):
            y = x.clone().add_(1).add_(-1)
            return y.sum() < 10

        def body_fn(x):
            y = x.clone().add_(1).add_(-1)
            return (y + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def nested(out_iter, it, y):
        def cond_fn(out_iter, it, y):
            return it.sum() < 10

        def body_fn(out_iter, it, y):
            return (out_iter.clone(), it + y, y + 1)

        def outer_cond_fn(out_iter, it, y):
            return out_iter.sum() < 2

        def outer_body_fn(out_iter, it, y):
            out_iter, it, y = while_loop(cond_fn, body_fn, (out_iter, it, y))
            return (out_iter + 1, it, y)

        return while_loop(outer_cond_fn, outer_body_fn, (out_iter, it, y))

    class Nested(torch.nn.Module):
        def forward(self, ci, cj, a, b):
            def cond_fn(i1, j1, x1, y1):
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                def cond_fn_nested(i2, j2, x2, y2):
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                i1, j1, x1, y1 = while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )
                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            return while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class SimpleWithLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.dec = torch.nn.Buffer(torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                return it - 1, self.linear(x)

            return while_loop(cond_fn, body_fn, (iter, x))

    class SimpleWithPytreeCarry(torch.nn.Module):
        def forward(self, it, pytree_input):
            def cond_fn(it, pytree_input):
                return it > 0

            def body_fn(it, pytree_input):
                x = pytree_input[0][0]
                y = pytree_input[1]["x"]
                z = pytree_input[1]["y"]
                new_x = y.sin()
                new_y = z.cos()
                new_z = x + 1
                return it - 1, ([new_x], {"x": new_y, "y": new_z})

            return while_loop(cond_fn, body_fn, (it, pytree_input))

    class NestedWithLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod = SimpleWithLinear()
            self.outer_linear = torch.nn.Linear(2, 2)
            self.dec = torch.nn.Buffer(torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                return it - 1, self.outer_linear(self.mod(it, x)[1])

            return while_loop(cond_fn, body_fn, (iter, x))

    class PytreeIntCarry(torch.nn.Module):
        def forward(self, x):
            a = x.shape[0]
            b = x.shape[1]

            def cond_fn(shapes, const_int_dict, x):
                a, b = shapes
                c1, c2, c3 = const_int_dict["int_carry"]
                return c1 * c2 * c3 < a * b

            def body_fn(shapes, const_int_dict, x):
                a, b = shapes
                c1, c2, c3 = const_int_dict["int_carry"]
                return (
                    [a + 1, b + 1],
                    {"int_carry": (c1 + 1, c2 + 1, c3 + 1)},
                    x + 1,
                )

            carry = ([a, b], {"int_carry": (2, 2, 3)}, x.sin())
            out_shapes, out_it, out_x = while_loop(cond_fn, body_fn, carry)
            out_inc = pytree.tree_map(lambda x: x + 1, out_it)
            out_add = pytree.tree_map(lambda x: x + out_x, out_it)
            return (out_shapes, out_inc, out_add, out_x)

    class IntCarry(torch.nn.Module):
        def forward(self, x):
            def cond_fn(it, x):
                return it < x.shape[0]

            def body_fn(it, x):
                x_clone = x.clone()
                # Need these checks to select from x
                torch._check(it >= 0)
                torch._check(it < x.shape[0])
                x_clone.select(0, it).copy_(x_clone.select(0, it) + it)
                return it + 1, x_clone

            # We invoke the hop directly to avoid triggering dyanmo tracing
            out_it, out_x = torch.ops.higher_order.while_loop(
                cond_fn, body_fn, (0, x), tuple()
            )
            # We need torch._check to use it in torch.ones call
            torch._check(out_it > 0)
            return (
                out_it + 1,
                out_it + out_x,
                out_it < x.shape[0],
                torch.ones(out_it * 2),
            )

    class ConstAndSymIntOutput(torch.nn.Module):
        def forward(self, t):
            a = t.shape[0]
            b = t.shape[1]

            def cond_fn(a, b, c1, c2, c3, c0, u0, x):
                return c1 * c2 * c3 < a * b

            def body_fn(a, b, c1, c2, c3, c0, u0, x):
                return b, c1, c2, c3, a, 0, u0 + 1, x + 1

            carry = (a, b, 1, 1, 1, a + 1, t.sum().to(torch.int64).item(), t.sin())
            out_it = torch.ops.higher_order.while_loop(cond_fn, body_fn, carry, tuple())
            out_inc = pytree.tree_map(lambda x: x + 1, out_it)
            out_add = pytree.tree_map(lambda x: x + t, out_it)
            return out_inc, out_add

    nested2 = Nested()
    simple_with_linear = SimpleWithLinear()
    simple_with_pytree_carry = SimpleWithPytreeCarry()
    nested_with_linear = NestedWithLinear()
    int_carry = IntCarry()
    pytree_int_carry = PytreeIntCarry()
    const_and_symint_output = ConstAndSymIntOutput()

    x = torch.zeros(1)
    y = torch.zeros(1)
    z = torch.zeros(1)
    return {
        "simple": (simple, (x,)),
        "nested": (nested, (x, y, z)),
        "nested2": (
            nested2,
            (torch.tensor(2), torch.tensor(2), torch.ones(2, 2), torch.ones(2, 2)),
        ),
        "simple_with_mutation": (simple_with_mutation, (x,)),
        "simple_with_linear": (
            simple_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
        "nested_with_linear": (
            nested_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
        "simple_with_pytree_carry": (
            simple_with_pytree_carry,
            (
                torch.tensor(3),
                ([torch.randn(3, 3)], {"x": torch.randn(3, 3), "y": torch.randn(3, 3)}),
            ),
        ),
        "int_carry": (int_carry, (torch.randn(2, 3),)),
        "pytree_int_carry": (
            pytree_int_carry,
            (torch.randn(2, 3),),
        ),
        "const_and_symint_output": (
            const_and_symint_output,
            (torch.randn(2, 3),),
        ),
    }


WHILE_LOOP_TESTS = _while_loop_tests()


def collect_meta_for_filtered_nodes(
    gm: torch.fx.GraphModule, node_names, meta_field_name
):
    ret = []
    for mod in gm.modules():
        for node in mod.graph.nodes:
            if node.name in node_names:
                for field_name in meta_field_name:
                    ret.append(node.meta.get(field_name))
    return ret


def reduce_func(*operands):
    acc = 0
    for operand in operands:
        acc += operand
    return acc


class ReduceObj:
    def __call__(self, *operands):
        return reduce_func(*operands)


class ReduceMod(torch.nn.Module):
    def _reduce(self, *operands):
        return reduce_func(*operands)

    def forward(self, *operands):
        return self._reduce(*operands)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class TestControlFlow(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    def check_autograd(self, result, result_exp, params):
        params_flatten = pytree.tree_leaves(params)
        result_flatten = pytree.tree_leaves(result)
        result_exp_flatten = pytree.tree_leaves(result_exp)
        grad_exp_init = [torch.ones_like(el) for el in result_exp_flatten]
        expected_grads = torch.autograd.grad(
            result_exp_flatten, params_flatten, grad_exp_init
        )
        grad_init = [torch.ones_like(el) for el in result_flatten]
        grads = torch.autograd.grad(result_flatten, params_flatten, grad_init)
        self.assertEqual(grads, expected_grads, atol=6e-05, rtol=6e-06)

    def test_cond_no_trace(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4)
        result = cond(False, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_cond_gpu(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        x = torch.randn(4, device="cuda")
        pred = torch.tensor(False, device="cuda")
        result = cond(pred, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    def test_cond_autograd_simple(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_complex(self):
        def true_fn(x):
            return torch.abs((x**2).sin())

        def false_fn(x):
            return (x + 42).cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_nested(self):
        class Nested(torch.nn.Module):
            def forward(self, p0, p1, p2, a, b, c):
                def true_fn(x0, y0, z0):
                    def true_true_fn(x1, y1, z1):
                        return (x1 - y1 * z1) * 3.14

                    def true_false_fn(x1, y1, z1):
                        def true_false_true_fn(x2, y2, z2):
                            return (x2 * y2 * z2) / 2.71

                        def true_false_false_fn(x2, y2, z2):
                            return (x2 + y2 + z2) * 1.23

                        return torch.cond(
                            p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                        )

                    return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])

                def false_fn(x0, y0, z0):
                    def false_true_fn(x1, y1, z1):
                        def false_true_true_fn(x2, y2, z2):
                            return (x2 - y2 - z2) + 1.23

                        def false_true_false_fn(x2, y2, z2):
                            return (x2 / y2 / z2) - 3.14

                        return torch.cond(
                            p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                        )

                    def false_false_fn(x1, y1, z1):
                        return (x1 - y1 * z1) / 2.71

                    return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])

                return torch.cond(p0, true_fn, false_fn, [a, b, c])

        nn_module = Nested()

        def true_fn(x):
            return nn_module(
                torch.tensor(False), torch.tensor(True), torch.tensor(False), x, x, x
            )

        def false_fn(x):
            return nn_module(
                torch.tensor(True), torch.tensor(False), torch.tensor(True), x, x, x
            )

        x = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_mixed_require_grad(self):
        def true_fn(x, y, z):
            return x * y * z

        def false_fn(x, y, z):
            return x + y + z

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=False)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x, y, x))
            self.assertEqual(result, fn(x, y, x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x, y, x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x, y, z):
            result = cond(pred, true_fn, false_fn, (x, y, z))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x, y, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1, y_1, z_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (z_1, y_1));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (z_1, y_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = z_1 = y_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1];  cond_1 = getitem_2 = None
    return (getitem_1,)""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_grad_through_cond(self):
        nn_module = torch.nn.Linear(4, 4)

        def true_fn(x):
            return nn_module(x)

        def false_fn(X):
            return x * nn_module(x)

        x = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (nn_module.weight,), grad_out)
            expected_grads = torch.autograd.grad(
                fn(
                    x,
                ),
                (nn_module.weight,),
                grad_out,
            )
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (nn_module.weight,), grad_out)

        # need to set _allow_non_fake_inputs = True because model parameters don't
        # get fakified.
        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    _tensor_constant0 = self._tensor_constant0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_param_constant0, _param_constant1, x_1, sym_size_int, _tensor_constant0));  true_graph_0 = false_graph_0 = _param_constant0 = _param_constant1 = _tensor_constant0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _param_constant0_1 = self._param_constant0
    _param_constant1_1 = self._param_constant1
    _tensor_constant0_1 = self._tensor_constant0
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (_param_constant0_1, _param_constant1_1, x_1, sym_size_int, _tensor_constant0_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = _param_constant0_1 = _param_constant1_1 = x_1 = sym_size_int = _tensor_constant0_1 = ones_like = None
    getitem_1 = cond_1[0];  getitem_1 = None
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  getitem_3 = None
    getitem_4 = cond_1[3];  cond_1 = getitem_4 = None
    return (getitem_2,)""",  # noqa: B950
        )

    def test_cond_in_forloop(self):
        def for_loop_fake(x):
            for _ in range(3):
                x = x * x + 1
            return x

        def for_loop_test(x):
            for i in range(3):
                pred = i < 3

                def true_fn(x):
                    return x * x + 1

                def false_fn(x):
                    return x

                x = cond(pred, true_fn, false_fn, (x,))

            return x

        x = torch.ones(4, requires_grad=True)
        x_new = for_loop_test(x)
        x_exp = for_loop_fake(x)

        self.assertEqual(x_new, x_exp)

        grad_out = torch.ones_like(x_new)
        grads = torch.autograd.grad(x_new, (x,), grad_out)
        expected_grads = torch.autograd.grad(x_exp, (x,), grad_out)
        self.assertEqual(expected_grads, grads)

        def f(x):
            x_new = for_loop_test(x)
            grad_out = torch.ones_like(x_new)
            return torch.autograd.grad(x_new, (x,), grad_out)

        gm = make_fx(f, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    mul = torch.ops.aten.mul.Tensor(x_1, x_1)
    add = torch.ops.aten.add.Tensor(mul, 1);  mul = None
    mul_1 = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(add_1, add_1)
    add_2 = torch.ops.aten.add.Tensor(mul_2, 1);  mul_2 = None
    ones_like = torch.ops.aten.ones_like.default(add_2, pin_memory = False);  add_2 = None
    mul_3 = torch.ops.aten.mul.Tensor(ones_like, add_1)
    mul_4 = torch.ops.aten.mul.Tensor(ones_like, add_1);  ones_like = add_1 = None
    add_3 = torch.ops.aten.add.Tensor(mul_4, mul_3);  mul_4 = mul_3 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_3, add)
    mul_6 = torch.ops.aten.mul.Tensor(add_3, add);  add_3 = add = None
    add_4 = torch.ops.aten.add.Tensor(mul_6, mul_5);  mul_6 = mul_5 = None
    mul_7 = torch.ops.aten.mul.Tensor(add_4, x_1)
    mul_8 = torch.ops.aten.mul.Tensor(add_4, x_1);  add_4 = x_1 = None
    add_5 = torch.ops.aten.add.Tensor(mul_8, mul_7);  mul_8 = mul_7 = None
    return (add_5,)""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_pytree_not_all_inputs_used(self):
        def true_fn(x):
            return x["t"][0] + x["t"][1]["b"]

        def false_fn(x):
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            self.assertEqual(result, fn({"t": [a, {"b": b}, (c,)]}))

            grad_out = torch.ones_like(result)
            if pred:
                with self.assertRaisesRegex(Exception, r"."):
                    grads = torch.autograd.grad(result, (a, b, c), grad_out)
                    expected_grads = torch.autograd.grad(
                        fn({"t": [a, {"b": b}, (c,)]}), (a, b, c), grad_out
                    )
                    self.assertEqual(expected_grads, grads)

        def f(pred, a, b, c):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (a, b), grad_out)

        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            pred, a, b, c
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, a_1, b_1, c_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(b_1, 0)
    sym_size_int_2 = torch.ops.aten.sym_size.int(c_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (a_1, b_1, sym_size_int, sym_size_int_1, c_1, sym_size_int_2));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (a_1, b_1, sym_size_int, sym_size_int_1, c_1, sym_size_int_2, ones_like));  pred_1 = true_graph_1 = false_graph_1 = a_1 = b_1 = sym_size_int = sym_size_int_1 = c_1 = sym_size_int_2 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  cond_1 = getitem_3 = None
    return (getitem_1, getitem_2)""",  # noqa: B950
        )
        # Forward
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )
        # Backward
        self.assertExpectedInline(
            gm.true_graph_1.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = add = None
    zeros_like = torch.ops.aten.zeros_like.default(arg4_1, pin_memory = False);  arg4_1 = None
    clone = torch.ops.aten.clone.default(arg6_1)
    clone_1 = torch.ops.aten.clone.default(arg6_1);  arg6_1 = None
    return [clone, clone_1, zeros_like]""",
        )

    def test_cond_autograd_pytree_input(self):
        def true_fn(x):
            return x["t"][0] + x["t"][1]["b"] * x["t"][2][0]

        def false_fn(x):
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            self.assertEqual(result, fn({"t": [a, {"b": b}, (c,)]}))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (a, b), grad_out)
            expected_grads = torch.autograd.grad(
                fn({"t": [a, {"b": b}, (c,)]}), (a, b), grad_out
            )
            self.assertEqual(expected_grads, grads)

        def f(pred):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (a, b), grad_out)

        # need to set _allow_non_fake_inputs = True because model parameters don't
        # get fakified.
        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(pred)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    _tensor_constant2 = self._tensor_constant2
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_tensor_constant0, _tensor_constant1, _tensor_constant2));  true_graph_0 = false_graph_0 = _tensor_constant0 = _tensor_constant1 = _tensor_constant2 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _tensor_constant0_1 = self._tensor_constant0
    _tensor_constant1_1 = self._tensor_constant1
    _tensor_constant2_1 = self._tensor_constant2
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (_tensor_constant0_1, _tensor_constant1_1, _tensor_constant2_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = _tensor_constant0_1 = _tensor_constant1_1 = _tensor_constant2_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  cond_1 = getitem_3 = None
    return (getitem_1, getitem_2)""",  # noqa: B950
        )

    def test_cond_autograd_different_pytree_output(self):
        def true_fn(x):
            return x["t"][0], {"r": x["t"][2][0] / x["t"][1]["b"]}, [x["t"][2][0]]

        def false_fn(x):
            return {"res": [x["t"][0] * x["t"][1]["b"], x["t"][2][0]]}

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred in [torch.tensor(False), torch.tensor(True)]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                "Cond doesn't work unless it is captured completely with torch.compile",
            ):
                cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_same_pytree_output(self):
        def true_fn(x):
            return {"res": [x["t"][0].clone(), (x["t"][2][0].clone(),)]}

        def false_fn(x):
            return {"res": [x["t"][1]["b"].clone(), (x["t"][2][0].clone(),)]}

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            result_exp = fn({"t": [a, {"b": b}, (c,)]})
            self.assertEqual(result, result_exp)

            result_flat, _ = pytree.tree_flatten(result)
            result_exp_flat, _ = pytree.tree_flatten(result_exp)

            grad_out = [torch.ones_like(g) for g in result_flat]
            expected_grads = torch.autograd.grad(result_exp_flat, (c,), grad_out)
            grads = torch.autograd.grad(result_flat, (c,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            return result

        gm = make_fx(f, tracing_mode="real", _allow_non_fake_inputs=True)(pred)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    _tensor_constant2 = self._tensor_constant2
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_tensor_constant0, _tensor_constant1, _tensor_constant2));  pred_1 = true_graph_0 = false_graph_0 = _tensor_constant0 = _tensor_constant1 = _tensor_constant2 = None
    getitem = cond[0]
    getitem_1 = cond[1];  cond = None
    return {'res': [getitem, (getitem_1,)]}""",  # noqa: B950
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_torch_nn_module(self):
        nn_module_true = torch.nn.Linear(4, 4)

        def true_fn(x):
            return nn_module_true(torch.abs((x**2).sin()))

        nn_module_false = torch.nn.GRUCell(4, 4)

        def false_fn(x):
            return nn_module_false((x + 42).cos())

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    _param_constant2 = self._param_constant2
    _param_constant3 = self._param_constant3
    _param_constant4 = self._param_constant4
    _param_constant5 = self._param_constant5
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1, _param_constant0, _param_constant1, _param_constant2, _param_constant3, _param_constant4, _param_constant5));  true_graph_0 = false_graph_0 = _param_constant0 = _param_constant1 = _param_constant2 = _param_constant3 = _param_constant4 = _param_constant5 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _param_constant0_1 = self._param_constant0
    _param_constant1_1 = self._param_constant1
    _param_constant2_1 = self._param_constant2
    _param_constant3_1 = self._param_constant3
    _param_constant4_1 = self._param_constant4
    _param_constant5_1 = self._param_constant5
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, _param_constant0_1, _param_constant1_1, _param_constant2_1, _param_constant3_1, _param_constant4_1, _param_constant5_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = _param_constant0_1 = _param_constant1_1 = _param_constant2_1 = _param_constant3_1 = _param_constant4_1 = _param_constant5_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1];  getitem_2 = None
    getitem_3 = cond_1[2];  getitem_3 = None
    getitem_4 = cond_1[3];  getitem_4 = None
    getitem_5 = cond_1[4];  getitem_5 = None
    getitem_6 = cond_1[5];  getitem_6 = None
    getitem_7 = cond_1[6];  cond_1 = getitem_7 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_user_nn_module(self):
        class User_nn_module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, input):
                return input * input

        nn_module_true = User_nn_module()

        def true_fn(x):
            return nn_module_true(torch.abs((x**2).sin()))

        nn_module_false = torch.nn.ReLU(inplace=False)

        def false_fn(x):
            return nn_module_false((x + 42).cos())

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_inner_fn(self):
        def true_fn(x):
            return torch.abs((x**2).sin())

        def false_fn(x):
            def inner_fn(x):
                return x**2

            return torch.abs(inner_fn(x).sin())

        x = torch.randn(4, requires_grad=True)
        pred = torch.tensor(False)
        fn = false_fn
        result_false = cond(pred, true_fn, false_fn, (x,))
        self.assertEqual(result_false, fn(x))

        grad_out = torch.ones_like(result_false)
        grads_false = torch.autograd.grad(result_false, (x,), grad_out)
        expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads_false)

        pred = torch.tensor(True)
        fn = true_fn
        result_true = cond(pred, true_fn, false_fn, (x,))
        self.assertEqual(result_true, fn(x))
        self.assertEqual(result_false, result_true)

        grad_out = torch.ones_like(result_true)
        grads_true = torch.autograd.grad(result_true, (x,), grad_out)
        expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads_true)
        self.assertEqual(grads_false, grads_true)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    def test_cond_autograd_inner_tensor(self):
        def true_fn(x):
            return torch.abs((x**2).sin())

        def false_fn(x):
            y = torch.ones(4, requires_grad=False) * 42
            return (x * y).cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",  # noqa: B950
        )

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_cond_autograd_gpu(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        for pred, fn in zip(
            [torch.tensor(False, device="cuda"), torch.tensor(True, device="cuda")],
            [false_fn, true_fn],
        ):
            x = torch.randn(4, requires_grad=True, device="cuda")
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def _test_cond_autograd(self, cond_fct, pred_fn, true_fn, false_fn, operands):
        from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

        # This is a helper function that extracts the metadata from the tensor and
        # sets the requires_grad flag to false. This is needed as we compare the
        # metadata of the operands and the gradients
        def _extract_tensor_metadata_except_requires_grad(arg):
            metadata = _extract_tensor_metadata(arg)
            metadata = TensorMetadata(
                metadata.shape,
                metadata.dtype,
                False,
                metadata.stride,
                metadata.memory_format,
                metadata.is_quantized,
                metadata.qparams,
            )
            return metadata

        # Comparison of FWD path
        cond_outputs = cond_fct(pred_fn(*operands), true_fn, false_fn, operands)
        operands_forced_grad = [o.clone().detach() for o in operands]
        for o in operands_forced_grad:
            o.requires_grad = True
        cond_outputs_exp = (
            true_fn(*operands_forced_grad)
            if pred_fn(*operands_forced_grad)
            else false_fn(*operands_forced_grad)
        )
        self.assertEqual(cond_outputs, cond_outputs_exp)

        # Comparison of BWD path
        cond_inputs = [o for o in operands if o.requires_grad]
        cond_inputs_exp = [o for o in operands_forced_grad if o.requires_grad]

        # Check if at least some operators require grads
        if len(cond_inputs) > 0:
            grad_inputs = torch.autograd.grad(
                cond_outputs, cond_inputs, allow_unused=True, retain_graph=True
            )
            grad_inputs_exp = torch.autograd.grad(
                cond_outputs_exp,
                cond_inputs_exp,
                allow_unused=True,
                materialize_grads=True,
            )

            grad_exp_masked = [
                g for g, o in zip(grad_inputs_exp, operands) if o.requires_grad
            ]
            self.assertEqual(grad_exp_masked, grad_inputs)

            # Extraction and comparison of Metadata of operands and gradients
            operands_metadata = [
                _extract_tensor_metadata_except_requires_grad(o) for o in cond_inputs
            ]
            grad_metadata = [
                _extract_tensor_metadata_except_requires_grad(o) for o in grad_inputs
            ]
            self.assertTrue(
                all(op == g for op, g in zip(operands_metadata, grad_metadata))
            )

        return cond_outputs, cond_inputs

    @skipIfTorchDynamo("don't test compile on compile")
    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["compile_dynamic_shape"])
    @parametrize("scalar", [False])
    def test_cond_autograd_zeros_unused_branch_complex_compile_fail(
        self, compile_mode, scalar
    ):
        device = torch.device("cuda")
        cond_fct = compile_mode_helper(torch.cond, compile_mode)

        autograd = [False, True, True, True, True]

        if scalar:
            # These operands work
            x = torch.randn((), device=device, requires_grad=bool(autograd[0]))
            w1 = torch.randn((), device=device, requires_grad=bool(autograd[1]))
            b1 = torch.randn((), device=device, requires_grad=bool(autograd[2]))
            w2 = torch.randn((), device=device, requires_grad=bool(autograd[3]))
            b2 = torch.randn((), device=device, requires_grad=bool(autograd[4]))
        else:
            # These operands do not work
            x = torch.randn(4, 5, device=device, requires_grad=bool(autograd[0]))
            w1 = torch.randn(2, 4, device=device, requires_grad=bool(autograd[1]))
            b1 = torch.randn(2, 1, device=device, requires_grad=bool(autograd[2]))
            w2 = torch.randn(2, 4, device=device, requires_grad=bool(autograd[3]))
            b2 = torch.randn(1, 5, device=device, requires_grad=bool(autograd[4]))

        operands = [x, w1, b1, w2, b2]

        def true_fn(x, w1, b1, w2, b2):
            if scalar:
                # This works
                return ((w1 * x + b1),)
            else:
                # This does not work
                return ((w1 @ x + b1).sum(),)

        def false_fn(x, w1, b1, w2, b2):
            if scalar:
                # This works
                return ((w2 * x + b2),)
            else:
                # This does not work
                return ((w2 @ x + b2).sum(),)

        def pred_fn(x, w1, b1, w2, b2):
            return x.mean() > 0

        cond_outputs, cond_inputs = self._test_cond_autograd(
            cond_fct, pred_fn, true_fn, false_fn, operands
        )

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_map_gpu(self):
        def f(x, y):
            return x + y

        xs = torch.ones(3, 2, 2, device="cuda")
        y = torch.ones(2, device="cuda")
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(expected, res)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_while_loop_gpu(self):
        def cond_fn(x):
            return x.sum() < 10

        def body_fn(x):
            return (x + 1,)

        x = torch.zeros(1, device="cuda")
        res = while_loop(cond_fn, body_fn, (x,))
        expected = _fake_while_loop(cond_fn, body_fn, (x,))
        self.assertEqual(expected, res)

    def test_map_illegal_inputs(self):
        def f(x, y):
            return x[0] + x[1] + y

        with self.assertRaisesRegex(
            RuntimeError,
            r"Mapped xs can only consist of tensors\. Got xs \[3, tensor\(\[1\., 1\.\]\)\]\.",
        ):
            _ = control_flow.map(f, (3, torch.ones(2)), torch.ones(2))

        with self.assertRaisesRegex(
            RuntimeError, r"Leading dimensions of mapped xs cannot be 0\."
        ):
            _ = control_flow.map(
                f, (torch.ones(0, 1, 2), torch.ones(0, 1, 2)), torch.ones(2)
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Leading dimensions of mapped xs must be consistent\. "
            r"Got shapes \[torch\.Size\(\[3, 4, 5\]\), torch\.Size\(\[4, 4, 5\]\)\]\.",
        ):
            _ = control_flow.map(
                f, (torch.ones(3, 4, 5), torch.ones(4, 4, 5)), torch.ones(5)
            )

    def test_map_illegal_outputs(self):
        def f(x, y):
            return x.item()

        def f1(x, y):
            return y.size()

        def f2(x, y):
            return None

        x = torch.ones([3])
        y = torch.ones([1, 2, 3])
        with self.assertRaisesRegex(
            RuntimeError, "map doesn't work unless it is captured completely"
        ):
            control_flow.map(f, x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # "Expected all leaves to be of torch.Tensor type.*",
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "map doesn't work unless it is captured completely with torch.compile.*",
        ):
            control_flow.map(f1, x, y)

        # return None is OK
        control_flow.map(f2, x, y)

    def test_map_list_in_out(self):
        def f(x, y):
            return [[x[0][0] + y]]

        xs = [[torch.ones(3, 2, 2)]]
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(expected, res)

    def test_map_dict_in_out(self):
        def f(x, y):
            return {"c": x["a"]["b"] + y}

        xs = {"a": {"b": torch.ones(3, 2, 2)}}
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertTrue("c" in res)
        self.assertEqual(expected, res)

    def test_map_autograd_simple(self):
        def f(x, y):
            return x.sin().cos() * y.cos().sin()

        xs = torch.ones(3, 2, 2, requires_grad=True)
        y = torch.ones(2, requires_grad=True)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs, y), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs, y), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_simple_partial_grad(self):
        def f(x, y):
            return x.sin().cos() * y.cos().sin()

        xs = torch.ones(3, 2, 2, requires_grad=True)
        # Disable the gradient computation for y
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs,), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs,), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_no_grad_output(self):
        def f(x, y):
            return x[0].sin().cos() + y, y.cos().sin()

        xs = [torch.ones(3, 2, 2, requires_grad=True), torch.ones(3, 3)]
        # Disable the gradient computation for y
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res[0])
        grads = torch.autograd.grad(res[0], (xs[0],), grad_out)
        expected_grads = torch.autograd.grad(expected_res[0], (xs[0],), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_nested_list(self):
        import torch.utils._pytree as pytree

        def f(x, y):
            a, b = x
            c, d = a
            return [[b.sin() * c.cos()], d.sin() * y.cos()]

        def fwbw(map_op, f, x, y):
            z = map_op(f, x, y)
            flat_x = pytree.tree_leaves(x)
            flat_z = pytree.tree_leaves(z)
            grads = torch.autograd.grad(
                flat_z, flat_x, [torch.ones_like(z) for z in flat_z]
            )
            return z, grads

        x = [
            [
                torch.randn(3, 2, 2, requires_grad=True),
                torch.randn(3, 2, 1, requires_grad=True),
            ],
            torch.ones(3, 1, 2, requires_grad=True),
        ]
        y = torch.ones(1, requires_grad=True)
        true_outs = fwbw(control_flow.map, f, x, y)
        fake_outs = fwbw(_fake_map, f, x, y)
        self.assertEqual(true_outs, fake_outs)

    def test_map_autograd_higher_order(self):
        from torch.autograd.functional import hessian as hes, jacobian as jac

        def f(x, y):
            return x.sin().cos() + y

        def wrapper_jac(x, y):
            return control_flow.map(f, x, y)

        def wrapper_jac_fake(x, y):
            return _fake_map(f, x, y)

        def wrapper_hes(x, y):
            return control_flow.map(f, x, y).sum()

        def wrapper_hes_fake(x, y):
            return _fake_map(f, x, y).sum()

        for g_fct, (wrap, wrap_fake) in [
            (jac, [wrapper_jac, wrapper_jac_fake]),
            (hes, [wrapper_hes, wrapper_hes_fake]),
        ]:
            xs = torch.ones(3, 2, 2, requires_grad=True)
            # Disable the gradient computation for y
            y = torch.ones(2, requires_grad=False)
            res = control_flow.map(f, xs, y)
            expected_res = _fake_map(f, xs, y)
            self.assertEqual(expected_res, res)

            expected_grads = g_fct(wrap_fake, (xs, y))
            grads = g_fct(wrap, (xs, y))
            self.assertEqual(expected_res, res)
            self.assertEqual(expected_grads, grads)

    def test_scan_y_less_ndim_then_dim(self):
        def combine_fn(carry, x):
            return carry @ x, (carry @ x).sum()

        init = torch.randn(4, 3)
        xs = torch.randn(3, 3, 2)
        dim = 2
        out = scan(combine_fn, init, xs, dim=dim)
        exp_out = _fake_scan(combine_fn, init, xs, dim=dim)
        self.assertEqual(out, exp_out)

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_compile(self, reverse, compile_mode, device, autograd):
        def add2(x: torch.Tensor, y: torch.Tensor):
            return x * y, x + y

        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        scan_fct = compile_mode_helper(scan, compile_mode)

        for op, op_pt, init in [
            (
                get_scan_combine_fn("add", False),
                torch.cumsum,
                torch.zeros(10, 2, device=device, requires_grad=autograd),
            ),
            (
                get_scan_combine_fn("mul", False),
                torch.cumprod,
                torch.ones(10, 2, device=device, requires_grad=autograd),
            ),
        ]:
            result = scan_fct(op, init, x, dim=0, reverse=reverse)
            result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
            self.assertEqual(result, result_exp)
            if not reverse:
                result_exp_PT = op_pt(x, 0)
                self.assertEqual(result[1], result_exp_PT)

            if autograd:
                self.check_autograd(result, result_exp, (init, x))

        # Jax Examples
        x = torch.arange(0, 4, device=device, dtype=torch.int64)
        init = torch.zeros(1, device=device, dtype=torch.int64)
        cumsum1 = scan_fct(
            get_scan_combine_fn("add", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        cumsum_exp = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )
        if not reverse:
            self.assertEqual(
                cumsum1[1],
                torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=torch.int64),
            )
            self.assertEqual(cumsum1[0], torch.tensor([6.0], dtype=torch.int64))
        else:
            self.assertEqual(
                cumsum1[1],
                torch.tensor([[6.0], [6.0], [5.0], [3.0]], dtype=torch.int64),
            )
            self.assertEqual(cumsum1[0], torch.tensor([6.0], dtype=torch.int64))
        self.assertEqual(cumsum1, cumsum_exp)

        # Different carry computation as output computation
        x = torch.arange(1, 5, device=device, dtype=torch.int64)
        init = torch.ones(1, device=device, dtype=torch.int64)
        result = scan_fct(add2, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(add2, init=init, xs=x, dim=0, reverse=reverse)
        if not reverse:
            self.assertEqual(
                result[1],
                torch.tensor([[2.0], [3.0], [5.0], [10.0]], dtype=torch.int64),
            )
            self.assertEqual(result[0], torch.tensor([24.0], dtype=torch.int64))
        else:
            self.assertEqual(
                result[1],
                torch.tensor([[25.0], [14.0], [7.0], [5.0]], dtype=torch.int64),
            )
            self.assertEqual(result[0], torch.tensor([24.0], dtype=torch.int64))
        self.assertEqual(result, result_exp)

        # Non associative operation
        x = torch.arange(
            0, 5, device=device, dtype=torch.float32, requires_grad=autograd
        )
        init = torch.ones(1, device=device, dtype=torch.float32, requires_grad=autograd)
        result = scan_fct(
            get_scan_combine_fn("div", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("div", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result, result_exp, (init, x))

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize(
        "dtype",
        [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.int64,
            torch.complex64,
        ],
    )
    def test_scan_dtype(self, reverse, compile_mode, device, dtype):
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Check all outputs and carries on the correct device and with torch.float32
        x = torch.randn(3, 10, 2, device=device).to(dtype=dtype)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=dtype),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.device.type for r in res] for res in result],
            [[device.type for _ in res] for res in result],
        )
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [[dtype for _ in res] for res in result],
        )

        # Check all outputs and carries on the correct device and
        # carry.dtype torch.float32 and output.dtype torch.float16
        x = torch.randn(3, 10, 2, device=device).to(dtype=dtype)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=torch.float32),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [
                [torch.float32 for _ in range(len(result[0]))],
                [dtype for _ in range(len(result[1]))],
            ],
        )

        # Check all outputs and carries on the correct device and
        # carry.dtype torch.int64 and output.dtype torch.float32
        x = torch.randn(3, 10, 2, device=device)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=dtype),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [
                [dtype for _ in range(len(result[0]))],
                [torch.float32 for _ in range(len(result[1]))],
            ],
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_dim(self, reverse, compile_mode, device, autograd):
        import random

        scan_fct = compile_mode_helper(scan, compile_mode)

        num_dims = [random.randint(2, 5) for _ in range(5)]
        for num_dim in num_dims:
            shapes = [random.randint(1, 10) for _ in range(num_dim)]
            rnd_scan_dim = random.randint(0, num_dim - 1)
            x = torch.randn(*shapes, device=device, requires_grad=autograd)
            init_shapes = shapes[:rnd_scan_dim] + shapes[rnd_scan_dim + 1 :]

            for op, op_pt, init in [
                (
                    get_scan_combine_fn("add", False),
                    torch.cumsum,
                    torch.zeros(*init_shapes, device=device, requires_grad=autograd),
                ),
                (
                    get_scan_combine_fn("mul", False),
                    torch.cumprod,
                    torch.ones(*init_shapes, device=device, requires_grad=autograd),
                ),
            ]:
                result = scan_fct(op, init, x, dim=rnd_scan_dim, reverse=reverse)
                result_exp = _fake_scan(
                    op, init=init, xs=x, dim=rnd_scan_dim, reverse=reverse
                )
                self.assertEqual(result, result_exp)
                if not reverse:
                    result_exp_PT = op_pt(x, rnd_scan_dim)
                    res_list = list(result)
                    res_list[1] = res_list[1].movedim(0, rnd_scan_dim)
                    self.assertEqual(res_list[1], result_exp_PT)

                if autograd:
                    self.check_autograd(result, result_exp, (init, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_binary_operator(self, reverse, compile_mode, device, autograd):
        state_dim = 20
        timesteps = 10
        scan_fct = compile_mode_helper(scan, compile_mode)

        projected_inputs = torch.randn(
            timesteps, state_dim, requires_grad=autograd, device=device
        )
        A = torch.randn(state_dim, requires_grad=autograd, device=device)
        elements = (A.repeat((timesteps, 1)), projected_inputs)
        init = tuple(
            [
                torch.ones_like(
                    torch._ops.ops.aten.slice(elements[0], 0, 0, 1, 1),
                    requires_grad=autograd,
                )
            ]
            + [
                torch.zeros_like(
                    torch._ops.ops.aten.slice(projected_inputs, 0, 0, 1, 1),
                    requires_grad=autograd,
                )
            ]
        )

        init_clone = [i.clone() for i in init]
        init_clone2 = [i.clone() for i in init]
        elements_clone = [ele.clone() for ele in elements]
        elements_clone2 = [ele.clone() for ele in elements]
        result = scan_fct(
            get_scan_combine_fn("s5_operator", False),
            init_clone,
            elements_clone,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("s5_operator", False),
            init_clone2,
            elements_clone2,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            result_flatten, _ = pytree.tree_flatten(result)
            result_exp_flatten, _ = pytree.tree_flatten(expected_result)

            grad_out = [torch.ones_like(el) for el in result_exp_flatten]
            expected_grads = torch.autograd.grad(
                result_exp_flatten, (*init_clone2, *elements_clone2), grad_out
            )
            grads = torch.autograd.grad(
                result_flatten, (*init_clone, *elements_clone), grad_out
            )
            self.assertEqual(grads, expected_grads)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_tuple(self, reverse, compile_mode, device, autograd):
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = (x, y)
        init = tuple(torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp)

        scan_fct = compile_mode_helper(scan, compile_mode)

        result_same = scan_fct(
            get_scan_combine_fn("tuple_fct", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("tuple_fct", False),
            init=init,
            xs=inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result_same, expected_result)

        if autograd:
            self.check_autograd(result_same, expected_result, (init, inp))

        def fct_different_output_tuple(x, y):
            return ((x[0] + y[0], x[1] * y[1]), (x[1] * y[1]))

        inp = (x, y)
        init = tuple(torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp)

        result_diff = scan(
            fct_different_output_tuple, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_different_output_tuple, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result_diff, expected_result)
        self.assertEqual(result_diff[1], result_same[1][1])

        if autograd:
            self.check_autograd(result_diff, expected_result, (init, inp))

    def test_scan_wrong_pytree(self):
        # Init and input have same pytree
        def fct_wrong_pytree(x, y):
            return (
                {
                    "i": x["i"] * y["j"][0][0],
                    "k": torch.tensor(0.0),
                    "j": (
                        [x["j"][1][0]["o"].clone()],
                        [{"o": torch.sin(x["i"])}],
                    ),
                },
                {
                    "i": x["i"] * y["j"][0][0],
                    "k": torch.tensor(0.0),
                    "j": ([x["j"][1][0]["o"].clone()], [{"o": torch.sin(x["i"])}]),
                },
            )

        x = torch.randn(3, 2, 2)
        y = torch.randn(3, 2, 2)
        z = torch.randn(3, 2, 2)
        inp = {"i": x, "j": ([y], [{"o": z}])}
        inp_flat, inp_spec = pytree.tree_flatten(inp)
        init_flat = [torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp_flat]
        init = pytree.tree_unflatten(init_flat, inp_spec)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # r"The tree structure of the inits and the carries are not identical.*",
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan(fct_wrong_pytree, init, inp, dim=0)

    def test_scan_float_output(self):
        # Init and input have same pytree
        def fct_float_output(x, y):
            return 0.0, x + y

        x = torch.randn(3, 2, 2)
        init = torch._ops.ops.aten.slice(x, 0, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be:
            # torch._dynamo.exc.Unsupported,
            # "HigherOrderOperator body's output must consist of tensors or ints only"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "scan must be captured completely.*",
        ):
            scan(fct_float_output, init, x, dim=0)

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_complex_pytree(self, reverse, compile_mode, device, autograd):
        # Init and input have same pytree

        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}
        inp_flat, inp_spec = pytree.tree_flatten(inp)
        init_flat = [torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp_flat]
        init = pytree.tree_unflatten(init_flat, inp_spec)

        result = scan_fct(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("complex_pointwise", False),
            init=init,
            xs=inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, inp))

    # TODO: Does not work because of the usage of vmap within associative_scan
    # The paT206899919 rameterization is commented out for the moment and the test is marked with expected fail
    # Fails with: AssertionError: scan is not an OpOverload
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_associative_scan(self):
        combine_mode = "generic"
        compile_mode_scan = "compile"
        compile_mode_associative_scan = "none"
        reverse = True
        reverse_associative_scan = True
        device = torch.device("cuda")

        scan_fct = compile_mode_helper(scan, compile_mode_scan)
        associative_scan_fct = compile_mode_helper(
            associative_scan, compile_mode_associative_scan
        )
        init = torch.randn(10, 5, device=device)
        inp = torch.randn(3, 10, 5, device=device)

        def body(x, y):
            val = associative_scan_fct(
                get_scan_combine_fn("add", True),
                y,
                0,
                reverse=reverse_associative_scan,
                combine_mode=combine_mode,
            )
            return x + y, x + val

        result = scan_fct(body, init, inp, dim=0, reverse=reverse)
        expected_result = _fake_scan(
            body,
            init,
            inp,
            0,
            reverse=reverse,
        )

        self.assertEqual(result, expected_result)

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_downstream_scan_matmul(self, compile_mode, reverse, device, autograd):
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        for ind in range(2):
            # Chain with matmul
            def chain_fct(inp):
                W = torch.ones(2, 5, device=device)
                o = scan(
                    get_scan_combine_fn("add", False),
                    init,
                    inp,
                    dim=1,
                    reverse=reverse,
                )
                return o[ind] @ W

            fct_cmp = compile_mode_helper(chain_fct, compile_mode)

            expected_result = _fake_scan(
                get_scan_combine_fn("add", False),
                init=init,
                xs=inp,
                dim=1,
                reverse=reverse,
            )[ind] @ torch.ones(2, 5, device=device)
            result = fct_cmp(inp)
            self.assertEqual(result, expected_result)

            if autograd:
                self.check_autograd(result, expected_result, (init, inp))

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_downstream_scan_scan_dim(
        self, compile_mode, reverse, device, autograd
    ):
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        # Chain with scan on different dim
        init2 = torch.randn(1, 10, 2, device=device, requires_grad=autograd)

        def chain_fct_different_dim(inp):
            o1 = scan(
                get_scan_combine_fn("add", False),
                init,
                inp,
                dim=1,
                reverse=reverse,
            )
            o1 = pytree.tree_map(lambda t: t.movedim(0, 1), o1)
            o2 = scan(
                get_scan_combine_fn("add", False),
                init2,
                o1[1],
                dim=0,
                reverse=reverse,
            )
            return o2

        fct_cmp = compile_mode_helper(chain_fct_different_dim, compile_mode)

        xs = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init,
            xs=inp,
            dim=1,
            reverse=reverse,
        )[1]
        xs = pytree.tree_map(lambda t: t.movedim(0, 1), xs)
        expected_result = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init2,
            xs=xs,
            dim=0,
            reverse=reverse,
        )
        result = fct_cmp(inp)
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, init2, inp))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_non_pointwise(self, reverse, compile_mode, device, autograd):
        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(10, 2, device=device, requires_grad=autograd)
        expected_result = _fake_scan(
            get_scan_combine_fn("non_pointwise", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )

        result = scan_fct(
            get_scan_combine_fn("non_pointwise", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, x))

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_compile_cnt(self, reverse, device):
        dim = 1

        from torch._dynamo.testing import CompileCounter

        # Tests rely on automatic_dynamic = True
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True):
            cnt = CompileCounter()
            x = torch.randn(3, 2, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # First compilation step
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 1)

            x = torch.randn(3, 20, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # Recompilation due to first different size
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 2)

            x = torch.randn(3, 40, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # No recompilation, because of dynamic shape
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 2)

            x = torch.randn(3, 40, 5, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of dim change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 3)

            x = torch.randn(3, 40, 20, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation due to first different size on new dim
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 4)

            x = torch.randn(3, 40, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # No recompilation, because of dynamic shape on new dim
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 4)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of dim change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 5)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of reverse change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=not reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # No recompilation, as nothing changed
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=not reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

            x = torch.randn(3, 120, 80, device=device)
            init = torch.randn(3, 80, device=device)
            # No recompilation, final test
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_scanned_0(self):
        # Only init and no input
        x = torch.randn(3, 1, 2, device=torch.device("cpu"))
        init = torch.randn(3, 2, device=torch.device("cpu"))
        dim = 1

        # Scan dimension is 0
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)
        inp = torch._ops.ops.aten.slice(x, dim, 1, None, 1)
        with self.assertRaisesRegex(
            RuntimeError,
            "All xs leaves must at least have.*",
        ):
            scan(
                get_scan_combine_fn("add", False),
                init,
                inp,
                dim=dim,
            )

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_non_tensor(self):
        x = torch.randn(3, 1, 2, device=torch.device("cpu"))
        dim = 1

        # Init is a float and not a tensor
        init = 1.0
        with self.assertRaisesRegex(RuntimeError, "All init leaves must be a Tensor.*"):
            scan(get_scan_combine_fn("add", False), init, x, dim=dim, reverse=False)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_shape(self):
        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong shape (Other dim different)
        init = torch.randn(1, 2)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same metadata.*",
        ):
            scan_fct(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
            )

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_init_longer_carry(self):
        def init_longer_carry(x: torch.Tensor, y: torch.Tensor):
            return x[0] + 1.0, y + 1.0

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = (
            torch._ops.ops.aten.slice(x, dim, 0, 1, 1),
            torch._ops.ops.aten.slice(x, dim, 0, 1, 1),
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan_fct(init_longer_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_init_shorter_carry(self):
        def init_shorter_carry(x: torch.Tensor, y: torch.Tensor):
            return (x + 1, x + 2), x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # The tree structure of the inits and the carries are not identical!
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan_fct(init_shorter_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_carry_shape(self):
        def wrong_carry_shape(x: torch.Tensor, y: torch.Tensor):
            return x[0, :], x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "scan must be captured completely with torch.compile.*",
        ):
            scan_fct(wrong_carry_shape, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_one_return(self):
        def no_carry(x: torch.Tensor, y: torch.Tensor):
            return x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # combine_fn needs to produce two pytrees, one for the carries and one for the outputs.
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "scan must be captured completely with.*",
        ):
            scan_fct(no_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_init(self, reverse, compile_mode, device, autograd):
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Only init and no input
        x = torch.randn(3, 1, 2, device=device, requires_grad=autograd)
        dim = 1
        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)

        # Only init given
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)
        result = scan_fct(op, init, [], dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=[], dim=dim, reverse=reverse)
        result_init = scan_fct(op, init, [], dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0], init)

        if autograd:
            self.check_autograd(result, result_exp, (init,))

        x = torch.randn(3, 5, 2, device=device, requires_grad=autograd)
        dim = 0

        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)
        inp = torch._ops.ops.aten.slice(x, dim, 1, None, 1)

        # Init tensor scalar
        init = torch.ones(1, device=device, requires_grad=autograd)

        def add_scalar_carry(x: torch.Tensor, y: torch.Tensor):
            return x + 1.0, x + y

        result_init = scan_fct(add_scalar_carry, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            add_scalar_carry, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0], torch.tensor([3.0], device=device))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        # Init tensor entirely different shape than inp
        init = torch.randn(7, 8, device=device, requires_grad=autograd)

        def add_scalar_carry2(x: torch.Tensor, y: torch.Tensor):
            return x + 1.0, x[: y.shape[0], : y.shape[1]] + y

        result_init = scan_fct(add_scalar_carry2, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            add_scalar_carry2, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)

        # Init with two timestep on dim axis. Should work as y has always 1 on dim axis and
        # hence automatic broadcasting should work
        # I.e., the input shape is 2x5x2, but the carry at each iteration is 2x5x2,
        # thus the output of each iteration is 2x5x2, which results in the total output
        # to be 4x5x2
        init = torch._ops.ops.aten.slice(x, dim, 0, 2, 1)
        result_init = scan_fct(op, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=inp, dim=dim, reverse=reverse)
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0].shape, torch.Size([2, 5, 2]))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        init = torch.tile(init, (1, 2, 1))

        def add_scalar_carry_sliced_out(x: torch.Tensor, y: torch.Tensor):
            return x + 1.0, x[:, :1, :] + y

        result_init = scan_fct(
            add_scalar_carry_sliced_out, init, inp, dim=dim, reverse=reverse
        )
        result_exp = _fake_scan(
            add_scalar_carry_sliced_out, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0].shape, torch.Size([2, 10, 2]))
        self.assertEqual(result_init[1].shape, torch.Size([2, 2, 5, 2]))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        # Correct case
        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        init = torch.zeros(3, 2, device=device, requires_grad=autograd)
        dim = 2

        result = scan_fct(op, init, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=dim, reverse=reverse)

        self.assertEqual(result, result_exp)
        if not reverse:
            result_exp_PT = op_pt(x, dim)
            result = list(result)
            result[1] = pytree.tree_map(lambda t: torch.movedim(t, 0, dim), result[1])
            self.assertEqual(result[1], result_exp_PT)

        if autograd:
            self.check_autograd(result, result_exp, (init, x))

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_init_wrong_pytree_complex(self, reverse, device):
        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        z = torch.randn(3, 2, 2, device=device)

        # Wrong pytree fed to the function
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, 0, 1, 1),
            "j": (
                {"a": torch._ops.ops.aten.slice(x, 0, 0, 1, 1)},
                [torch._ops.ops.aten.slice(y, 0, 0, 1, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, 0, 1, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, 0, None, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, 0, None, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, 0, None, 1)}],
            ),
        }
        with self.assertRaisesRegex(
            Exception,
            ".*",
        ):
            scan(
                get_scan_combine_fn("complex_pointwise", False),
                init,
                inp,
                dim=0,
                reverse=reverse,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_init_pytree_complex(self, reverse, compile_mode, device, autograd):
        def fct_pointwise_different_output(x, y):
            return (
                {
                    "i": x["i"] * y["i"],
                    "j": (
                        [x["j"][0][0] * y["j"][0][0]],
                        [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
                    ),
                },
                (
                    y["i"] * 2,
                    {
                        "o": x["i"] * y["i"],
                        "j": (
                            [x["j"][0][0] * y["j"][0][0]],
                            [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
                        ),
                    },
                ),
            )

        def fct_pointwise_different_carry(x, y):
            return (
                {
                    "i": x["i"] * y["i"],
                    "j": (
                        x["i"] * 2,
                        [x["j"][1][0] * y["j"][0][0]],
                        [{"o": x["j"][2][0]["o"] + y["j"][1][0]["o"]}],
                    ),
                },
                (
                    y["i"] * 2,
                    {
                        "o": x["i"] * y["i"] + x["j"][0][0],
                        "j": (
                            [x["j"][1][0] * y["j"][0][0]],
                            [{"o": x["j"][2][0]["o"] + y["j"][1][0]["o"]}],
                        ),
                    },
                ),
            )

        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        if reverse:
            init_start, init_end = -1, None
            inp_start, inp_end = 0, -1
        else:
            init_start, init_end = 0, 1
            inp_start, inp_end = 1, None

        # Regular case
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, init_start, init_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, init_start, init_end, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, inp_start, inp_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, inp_start, inp_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, inp_start, inp_end, 1)}],
            ),
        }
        result = scan_fct(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
      

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 86 class(es): Nested, SimpleWithLinear, SimpleWithPytreeCarry, NestedWithLinear, PytreeIntCarry, IntCarry, ConstAndSymIntOutput, ReduceObj, ReduceMod, TestControlFlow, Nested, User_nn_module, GraphModule, scan_combine_fn_0, RNNLoop, RNNScanList, RNNScanTensor, in, AssociativeScanModels, CombineFn

### Functions
This file defines 922 function(s): to_fun, from_fun, to_fun_old, from_fun_old, _fake_while_loop, compile_mode_helper, get_scan_combine_fn, add, adds, mul, div, s5_operator, different_input_size_operator, tuple_fct, complex_pointwise, non_pointwise, RNN, fct_c1_no_grad, _while_loop_tests, simple, cond_fn, body_fn, simple_with_mutation, cond_fn, body_fn, nested, cond_fn, body_fn, outer_cond_fn, outer_body_fn


## Key Components

The file contains 32327 words across 9752 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 382610 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

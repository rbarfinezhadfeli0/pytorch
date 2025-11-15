# Documentation: `docs/test/higher_order_ops/test_invoke_subgraph.py_docs.md`

## File Metadata

- **Path**: `docs/test/higher_order_ops/test_invoke_subgraph.py_docs.md`
- **Size**: 53,838 bytes (52.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/higher_order_ops/test_invoke_subgraph.py`

## File Metadata

- **Path**: `test/higher_order_ops/test_invoke_subgraph.py`
- **Size**: 103,859 bytes (101.42 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950
# flake8: noqa: E731

import unittest
import unittest.mock as mock

from parameterized import parameterized_class

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
import torch.utils._pytree as pytree
from functorch.compile import aot_function, nop
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    empty_line_normalizer,
    InductorAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.schema import find_hop_schema
from torch._inductor import config as inductor_config
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton, requires_gpu


nested_compile_region = torch.compiler.nested_compile_region

if HAS_GPU:
    import triton


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraph(TestCase):
    def test_simple(self):
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            return nested_compile_region(gn)(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function(self):
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            return nested_compile_region(gn)(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_multiple(self):
        @nested_compile_region
        def cos(x):
            return torch.cos(x)

        @nested_compile_region
        def sin(x):
            return torch.sin(x)

        def fn(x):
            a = cos(x)
            b = sin(a)
            return cos(b)

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        self.assertEqual(ref, res)


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraphCompile(TestCase):
    def count_unique_get_attr_nodes(self, gm, args, expected):
        subgraph_attr_names = set()
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                subgraph_attr_names.add(node.target)
        self.assertEqual(len(subgraph_attr_names), expected)

    def test_simple(self):
        @nested_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            return gn(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_module_forward(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x, y):
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_gen_schema(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x, y):
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)
        res.sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        fw_schema = find_hop_schema(
            backend.fw_graphs[0], torch.ops.higher_order.invoke_subgraph
        )
        bw_schema = find_hop_schema(
            backend.bw_graphs[0], torch.ops.higher_order.invoke_subgraph
        )
        self.assertExpectedInline(
            str(fw_schema[0]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1) -> (Tensor, Tensor, Tensor)""",
        )
        self.assertExpectedInline(
            str(fw_schema[1]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1) -> (Tensor, Tensor, Tensor)""",
        )
        self.assertExpectedInline(
            str(bw_schema[0]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1, Tensor arg2) -> (Tensor, Tensor)""",
        )
        self.assertExpectedInline(
            str(bw_schema[1]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1, Tensor arg2) -> (Tensor, Tensor)""",
        )

    def test_gen_schema_with_buffer_mutation(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 5
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                self.buf.add_(1)
                return torch.mul(x, y).sin() + self.c + self.buf

        mod_ref = Mod()
        mod = Mod()

        def fn(mod, x, y):
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(mod_ref, x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = EagerAndRecordGraphs()
        with (
            torch.no_grad(),
        ):
            res = torch.compile(fn, backend=backend, fullgraph=True)(
                mod, x_clone, y_clone
            )

        self.assertEqual(len(backend.graphs), 1)
        fw_schema = find_hop_schema(
            backend.graphs[0], torch.ops.higher_order.invoke_subgraph
        )
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]", L_mod_buffers_buf_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_
        l_mod_buffers_buf_ = L_mod_buffers_buf_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_mod_buffers_buf_, l_x_, l_y_);  subgraph_0 = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_mod_buffers_buf_, l_x_, l_y_);  subgraph_1 = l_mod_buffers_buf_ = l_x_ = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        add: "f32[8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_mod_buffers_buf_: "f32[8]", l_x_: "f32[8]", l_y_: "f32[8]"):
            add_: "f32[8]" = l_mod_buffers_buf_.add_(1);  add_ = None

            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            sin: "f32[8]" = mul.sin();  mul = None
            add: "f32[8]" = sin + 5;  sin = None
            add_1: "f32[8]" = add + l_mod_buffers_buf_;  add = l_mod_buffers_buf_ = None
            return (add_1,)
""",
            )
        self.assertExpectedInline(
            str(fw_schema[0]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor(a2!) arg0, Tensor arg1, Tensor arg2) -> ((Tensor))""",
        )
        self.assertExpectedInline(
            str(fw_schema[1]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor(a2!) arg0, Tensor arg1, Tensor arg2) -> ((Tensor))""",
        )
        self.assertEqual(res, ref)
        self.assertEqual(mod.buf, mod_ref.buf)

    def test_auto_functionalize(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 5
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                return torch.mul(x, y).sin() * self.c * self.buf

        mod_ref = Mod()
        mod = Mod()

        def fn(mod, x, y):
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(mod_ref, x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(mod, x_clone, y_clone)
        res.sum().backward()
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertEqual(ref, res)
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]", primals_2: "f32[8]", primals_3: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2, primals_3);  partitioned_fw_subgraph_0_0 = None
        getitem_12: "f32[8]" = invoke_subgraph_4[3]
        getitem_11: "f32[8]" = invoke_subgraph_4[2]
        getitem_10: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, primals_2, primals_3);  partitioned_fw_subgraph_0_1 = primals_1 = primals_2 = primals_3 = None
        getitem_15: "f32[8]" = invoke_subgraph_6[3]
        getitem_14: "f32[8]" = invoke_subgraph_6[2]
        getitem_13: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add, getitem_12, getitem_11, getitem_10, getitem_15, getitem_14, getitem_13)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]", primals_2: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            sin: "f32[8]" = torch.ops.aten.sin.default(mul);  mul = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(sin, 5);  sin = None
            mul_2: "f32[8]" = torch.ops.aten.mul.Tensor(mul_1, primals_2);  mul_1 = None
            return (mul_2, primals_0, primals_1, primals_2)
""",
            ignore_empty_lines=True,
        )
        self.assertExpectedInline(
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_12: "f32[8]", getitem_11: "f32[8]", getitem_10: "f32[8]", getitem_15: "f32[8]", getitem_14: "f32[8]", getitem_13: "f32[8]", tangents_1: "f32[8]"):
        partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_0', getitem_13, getitem_14, getitem_15, tangents_1);  partitioned_bw_subgraph_0_1 = getitem_13 = getitem_14 = getitem_15 = None
        getitem_2: "f32[8]" = invoke_subgraph_7[0]
        getitem_3: "f32[8]" = invoke_subgraph_7[1];  invoke_subgraph_7 = None
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_10, getitem_11, getitem_12, tangents_1);  partitioned_bw_subgraph_0_0 = getitem_10 = getitem_11 = getitem_12 = tangents_1 = None
        getitem_6: "f32[8]" = invoke_subgraph_5[0]
        getitem_7: "f32[8]" = invoke_subgraph_5[1];  invoke_subgraph_5 = None
        add_1: "f32[8]" = torch.ops.aten.add.Tensor(getitem_2, getitem_6);  getitem_2 = getitem_6 = None
        add_2: "f32[8]" = torch.ops.aten.add.Tensor(getitem_3, getitem_7);  getitem_3 = getitem_7 = None
        return (add_1, add_2, None)

    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]", primals_2: "f32[8]", tangents_0: "f32[8]"):
            mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(tangents_0, primals_2);  tangents_0 = primals_2 = None
            mul_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_3, 5);  mul_3 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            cos: "f32[8]" = torch.ops.aten.cos.default(mul);  mul = None
            mul_5: "f32[8]" = torch.ops.aten.mul.Tensor(mul_4, cos);  mul_4 = cos = None
            mul_6: "f32[8]" = torch.ops.aten.mul.Tensor(mul_5, primals_0);  primals_0 = None
            mul_7: "f32[8]" = torch.ops.aten.mul.Tensor(mul_5, primals_1);  mul_5 = primals_1 = None
            return (mul_7, mul_6, None)
""",
            ignore_empty_lines=True,
        )

    def test_buffer_mutation_works_under_no_grad(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                self.buf.add_(1)
                return torch.mul(x, y).sin() * self.buf

        mod_ref = Mod()
        mod = Mod()

        def fn(mod, x, y):
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(mod_ref, x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        with torch.no_grad():
            res = torch.compile(fn, fullgraph=True)(mod, x_clone, y_clone)
        self.assertEqual(ref, res)
        self.assertEqual(mod_ref.buf, mod.buf)

        mod = Mod()
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        with torch.inference_mode():
            res = torch.compile(fn, fullgraph=True)(mod, x_clone, y_clone)
        self.assertEqual(ref, res)
        self.assertEqual(mod_ref.buf, mod.buf)

        mod = Mod()
        x_clone = x.detach().clone().requires_grad_(False)
        y_clone = y.detach().clone().requires_grad_(False)
        res = torch.compile(fn, fullgraph=True)(mod, x_clone, y_clone)
        self.assertEqual(ref, res)
        self.assertEqual(mod_ref.buf, mod.buf)

    def test_buffer_mutation_errors_under_training(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                self.buf.add_(1)
                return torch.mul(x, y).sin() * self.buf

        mod = Mod()

        def fn(mod, x, y):
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "does not currently support training with in-place input or buffer mutations",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(mod, x, y)

    def test_list(self):
        @nested_compile_region
        def gn(x, y):
            return [torch.mul(x, y), torch.add(x, y)]

        def fn(x, y):
            lst = gn(x, y)
            lst.append(torch.sin(x))
            return lst[0] + lst[1] + lst[2]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_tuple_of_tuple(self):
        @nested_compile_region
        def gn(x, y):
            return ((torch.mul(x, y),), torch.add(x, y))

        def fn(x, y):
            tup = gn(x, y)
            return tup[0][0] + tup[1]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    @unittest.skip("FunctionCtx ops is not cacheable right now")
    def test_differing_strides_for_grad_outs(self):
        class CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_out):
                a = grad_out.view(12, 5)
                return torch.cos(torch.reshape(a, (3, 4, 5)))

        @nested_compile_region
        def gn(x):
            return CustomOp.apply(x)

        def fn(x):
            a = gn(x)
            # Force stride changes so that backward view causes a failure if
            # contiguous not called.
            b = torch.permute(a, (0, 2, 1))
            return b

        x = torch.randn(3, 4, 5, requires_grad=True)
        ref = torch.permute(gn(x), (0, 2, 1))

        x_clone = x.clone().detach().requires_grad_(True)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    @requires_cuda_and_triton
    def test_sdpa(self):
        @nested_compile_region
        def gn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        def fn(q, k, v):
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
            ):
                return gn(q, k, v)

        q = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        ref = fn(q, k, v)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(q, k, v)
        res.sum().backward()
        self.assertEqual(ref, res)

        res = opt_fn(q, k, v)
        res.sum().backward()

    def test_symint_from_fwd_to_bwd(self):
        @nested_compile_region
        def gn(x, y):
            a = torch.sum(x, (1,), keepdim=True).view(y.shape[1], y.shape[0])
            return torch.matmul(a, y)

        def fn(x, y):
            return gn(x, y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(64, 1, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

        x = torch.randn(256, 1, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
        res.sum().backward()

        x = torch.randn(16, 1, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
        res.sum().backward()

    @inductor_config.patch("fx_graph_cache", False)
    def test_dropout_checks_joint_graph(self):
        # `dropout` tests that joint graph passes (not just partitioner) is ran
        # on the hop graphs. Inductor rng functionalization happens in the joint
        # graph passes. Without running joint graph passes, we would get an
        # error like AssertionError: should have been handled in
        # replace_random.py
        @nested_compile_region
        def gn(x):
            return torch.nn.functional.dropout(torch.sin(x), p=0.5)

        @nested_compile_region
        def hn(x):
            return torch.sin(x)

        def fn(x):
            return gn(x) + hn(x)

        x = torch.randn(8, requires_grad=True)
        # Difficult to check the results here because we random does not match
        # between eager and Triton.
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x)  # noqa: F841

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x)
        res.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1);  partitioned_fw_subgraph_0_0 = None
        getitem_7: "b8[8]" = invoke_subgraph_4[2]
        getitem_6: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_1_0 = self.partitioned_fw_subgraph_1_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_1_0, 'partitioned_fw_subgraph_1_0', primals_1);  partitioned_fw_subgraph_1_0 = primals_1 = None
        getitem_8: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None

        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add, getitem_7, getitem_6, getitem_8)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(primals_0)

            inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))

            inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
            inductor_random_default: "f32[8]" = torch.ops.prims.inductor_random.default([8], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
            gt: "b8[8]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(gt, sin);  sin = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
            return (mul_1, primals_0, gt)

    class partitioned_fw_subgraph_1_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(primals_0)
            return (sin, primals_0)
""",
                ignore_empty_lines=True,
            )

    @inductor_config.patch("fx_graph_cache", False)
    def test_dropout_checks_joint_graph_inference(self):
        # Checks that joint graph results in inductor seeds for just the inference graph
        @nested_compile_region
        def gn(x):
            return torch.nn.functional.dropout(torch.sin(x), p=0.5)

        def fn(x):
            return gn(x)

        backend = InductorAndRecordGraphs()
        x = torch.randn(8, requires_grad=False)
        torch.compile(fn, backend=backend, fullgraph=True)(x)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]"):
            inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))

            inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
            inductor_random_default: "f32[8]" = torch.ops.prims.inductor_random.default([8], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
            gt: "b8[8]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
            sin: "f32[8]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(gt, sin);  gt = sin = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
            return (mul_1,)
""",
                ignore_empty_lines=True,
            )

    def test_dedupe(self):
        @nested_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            a = gn(x, y)
            return gn(a, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Check that the Dynamo and AOT graphs have just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 1)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        a: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', a, l_y_);  subgraph_1 = a = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            return (mul,)
""",
            )

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]", primals_2: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = primals_1 = None
        getitem_9: "f32[8]" = invoke_subgraph_4[2]
        getitem_8: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', getitem, primals_2);  partitioned_fw_subgraph_0_1 = getitem = primals_2 = None
        getitem_11: "f32[8]" = invoke_subgraph_6[2]
        getitem_10: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        return (getitem_1, getitem_9, getitem_8, getitem_11, getitem_10)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            return (mul, primals_0, primals_1)
""",
            ignore_empty_lines=True,
        )

    def test_dce(self):
        @nested_compile_region
        def gn(x):
            x = torch.sin(x)
            # should be dce'd
            y = torch.cos(x)  # noqa: F841
            return x

        def fn(x):
            return gn(x)

        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.randn(4, requires_grad=False)
        )

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[4]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4]"):
            sin: "f32[4]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return (sin,)
""",
            )

    def test_nonlocal_update(self):
        counter = 2

        @nested_compile_region
        def gn(x, y):
            nonlocal counter
            return (torch.mul(x, y) * counter,)

        def fn(x, y):
            nonlocal counter
            counter = 2
            a = gn(x, y)[0]
            counter = 3
            return gn(a, y)[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        torch._dynamo.reset()
        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        a: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', a, l_y_);  subgraph_1 = a = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            mul_1: "f32[8]" = mul * 2;  mul = None
            return (mul_1,)

    class subgraph_1(torch.nn.Module):
        def forward(self, a: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(a, l_y_);  a = l_y_ = None
            mul_1: "f32[8]" = mul * 3;  mul = None
            return (mul_1,)
""",
            )

    @inductor_config.patch("fx_graph_cache", False)
    def test_view_to_reshape(self):
        @nested_compile_region
        def gn(x):
            x = torch.sin(x)
            x = x.view(1, 8)
            return torch.sin(x)

        def fn(x):
            return gn(x)

        x = torch.randn(8, requires_grad=False)

        torch._dynamo.reset()
        backend = InductorAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[1, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

            view: "f32[1, 8]" = torch.ops.aten.reshape.default(sin, [1, 8]);  sin = None

            sin_1: "f32[1, 8]" = torch.ops.aten.sin.default(view);  view = None
            return (sin_1,)
""",
            )

    def test_normalize_gm(self):
        @nested_compile_region
        def gn(x, y):
            # Different graph give different names to intermediate nodes
            for _ in range(5):
                x = x * y
            return x

        def fn(x, y):
            for _ in range(5):
                x = gn(x, y)
            return x

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        opt_fn(x, y)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', getitem, l_y_);  subgraph_1 = getitem = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        subgraph_2 = self.subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_2, 'subgraph_0', getitem_1, l_y_);  subgraph_2 = getitem_1 = None
        getitem_2: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        subgraph_3 = self.subgraph_0
        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(subgraph_3, 'subgraph_0', getitem_2, l_y_);  subgraph_3 = getitem_2 = None
        getitem_3: "f32[8]" = invoke_subgraph_3[0];  invoke_subgraph_3 = None
        subgraph_4 = self.subgraph_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(subgraph_4, 'subgraph_0', getitem_3, l_y_);  subgraph_4 = getitem_3 = l_y_ = None
        getitem_4: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        return (getitem_4,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            x: "f32[8]" = l_x_ * l_y_;  l_x_ = None
            x_1: "f32[8]" = x * l_y_;  x = None
            x_2: "f32[8]" = x_1 * l_y_;  x_1 = None
            x_3: "f32[8]" = x_2 * l_y_;  x_2 = None
            x_4: "f32[8]" = x_3 * l_y_;  x_3 = l_y_ = None
            return (x_4,)
""",
            )

    def test_input_mutation(self):
        @nested_compile_region
        def gn(x, y):
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            return gn(x, y)

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x_clone = x.clone()
        self.assertEqual(opt_fn(x, y), fn(x_clone, y))

    def test_input_mutation_mutiple_times(self):
        @nested_compile_region
        def gn(x, y):
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            z = gn(x, y)
            for _ in range(16):
                z += gn(x, y)
            return z

        x = torch.randn(8, requires_grad=False)
        x_clone = x.clone()
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with (
            torch.no_grad(),
        ):
            out = opt_fn(x, y)
        exp_out = fn(x_clone, y)
        self.assertEqual(exp_out, out)
        self.assertEqual(x_clone, x)

    def test_input_mutation_mutiple_times_fake_tensor_cahche_hit(self):
        @nested_compile_region
        def gn(x, y):
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            z = gn(x, y)
            for _ in range(16):
                z += gn(x, y)
            return z

        x = torch.randn(8, requires_grad=False)
        x_clone = x.clone()
        y = torch.randn(8, requires_grad=False)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        fake_prop_count = 0

        def _mock_invoke_subgraph(mode, subgraph, identifier, *operands):
            nonlocal fake_prop_count
            fake_prop_count += 1
            return (operands[0].clone(),)

        with (
            mock.patch(
                "torch._higher_order_ops.utils.registered_hop_fake_fns",
                {torch.ops.higher_order.invoke_subgraph: _mock_invoke_subgraph},
            ),
            torch.no_grad(),
        ):
            out = opt_fn(x, y)

        # Fake propagation occurs only twice, with subsequent calls using cached results.
        #
        # First fake propagation (in collect_metadata_analysis of AOT):
        #   - Uses the original Dynamo graph
        #   - Flow: functionalization -> fake tensor
        #
        # Second fake propagation (in _create_graph of AOT):
        #   - Uses a materialized graph that includes epilogue operations
        #   - Flow: functionalization -> proxy -> fake tensor
        #
        # The key difference: the second time we materialize the graph with epilogue
        # operations included in the proxy key. Since the dynamo graph module is not
        # in the functional + epilogue format, the cache key should be different,
        # preventing cache reuse between these two phases.
        self.assertEqual(fake_prop_count, 2)
        exp_out = fn(x_clone, y)
        self.assertEqual(exp_out, out)
        self.assertEqual(x_clone, x)

    def test_input_mutation_inference_mode(self):
        @nested_compile_region
        def gn(x, y):
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            z = torch.cos(x)
            with torch.inference_mode():
                return gn(torch.cos(z), y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        with self.assertRaisesRegex(
            RuntimeError,
            "Inplace update to inference tensor outside InferenceMode is not allowed",
        ):
            opt_fn(x, y)

    def test_simple_module(self):
        mod = torch.nn.Linear(8, 8)

        @nested_compile_region
        def gn(x):
            return torch.cos(x), mod(x)

        def fn(x):
            out = gn(x)
            return out[0] + out[1]

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        # requires_grad is False deliberately to force None the joint_graph
        # outputs
        x = torch.randn(8, 8, requires_grad=False)
        x_clone = x.detach().clone().requires_grad_(False)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    def test_fail_with_direct_invoke_subgraph(self):
        from torch._higher_order_ops import invoke_subgraph

        def gn(x):
            return torch.sin(x)

        def fn(x):
            return invoke_subgraph(gn, None, (x,))

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Directly using invoke_subgraph is not"
        ):
            opt_fn(x)

    def test_input_output_aliasing(self):
        @nested_compile_region
        def gn(x, y):
            return (x, torch.mul(x, y))

        def fn(x, y):
            outs = gn(x, y)
            return outs[0] * outs[1]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x, y)

    def test_input_input_aliasing(self):
        @nested_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x):
            return gn(x, x.view(1, 8))

        x = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x)

    def test_output_output_aliasing(self):
        @nested_compile_region
        def gn(x):
            z = torch.cos(x)
            return z, z.view(1, 8)

        def fn(x):
            return gn(x)

        x = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x)

    def test_mod_attr_aliasing(self):
        class MutateParam(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(8)

            def forward(self, x):
                self.a.add_(1)
                return torch.mul(x, self.a)

        @nested_compile_region
        def gn(x):
            return mod(x)

        def fn(x, y):
            return gn(x) * y

        mod = MutateParam()
        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        compiled_out = opt_fn(x, y)
        # reset constant attr
        mod.a = torch.ones(8)
        self.assertEqual(compiled_out, fn(x, y))

    def test_redundant_compile_region(self):
        @nested_compile_region
        @nested_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x):
            return gn(x) + gn(x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_);  subgraph_1 = l_x_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)
""",
            )

    def test_kwargs_only(self):
        @nested_compile_region
        def gn(x, *, y):
            return x * y

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        def fn(x, y):
            return gn(x, y=y)

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_module_method(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            @nested_compile_region
            def helper(self, x):
                return self.linear(x)

            def forward(self, x):
                return x + self.helper(x) * self.helper(x) + x

        mod = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_mod = torch.compi
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/higher_order_ops`, which is part of the **testing infrastructure**.



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
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/higher_order_ops/test_invoke_subgraph.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/higher_order_ops`):

- [`test_local_map.py_kw.md_docs.md`](./test_local_map.py_kw.md_docs.md)
- [`test_invoke_quant.py_docs.md_docs.md`](./test_invoke_quant.py_docs.md_docs.md)
- [`test_invoke_quant.py_kw.md_docs.md`](./test_invoke_quant.py_kw.md_docs.md)
- [`test_invoke_subgraph.py_kw.md_docs.md`](./test_invoke_subgraph.py_kw.md_docs.md)
- [`test_print.py_docs.md_docs.md`](./test_print.py_docs.md_docs.md)
- [`test_with_effects.py_kw.md_docs.md`](./test_with_effects.py_kw.md_docs.md)
- [`test_print.py_kw.md_docs.md`](./test_print.py_kw.md_docs.md)
- [`test_with_effects.py_docs.md_docs.md`](./test_with_effects.py_docs.md_docs.md)
- [`test_local_map.py_docs.md_docs.md`](./test_local_map.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_invoke_subgraph.py_docs.md_docs.md`
- **Keyword Index**: `test_invoke_subgraph.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

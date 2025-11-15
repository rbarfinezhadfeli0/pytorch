# Documentation: `test/export/test_serialize.py`

## File Metadata

- **Path**: `test/export/test_serialize.py`
- **Size**: 84,387 bytes (82.41 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_sym_bool)
"""

# Owner(s): ["oncall: export"]
import copy
import io
import math
import tempfile
import unittest
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


if HAS_GPU:
    import triton
    import triton.language as tl

    from torch.library import wrap_triton
    from torch.utils._triton import has_triton

import torch
import torch._dynamo as torchdynamo
import torch._export.serde.schema as schema
import torch.export._trace
import torch.utils._pytree as pytree
from torch._export.db.case import ExportCase, SupportLevel
from torch._export.db.examples import all_examples
from torch._export.serde.schema import ArgumentKind
from torch._export.serde.serialize import (
    _dict_to_dataclass,
    _to_json_bytes,
    canonicalize,
    deserialize,
    ExportedProgramDeserializer,
    ExportedProgramSerializer,
    GraphModuleSerializer,
    serialize,
    SerializeError,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim, export, load, save, unflatten
from torch.export.pt2_archive.constants import ARCHIVE_VERSION_PATH
from torch.fx.experimental.symbolic_shapes import is_concrete_int, ValueRanges
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TemporaryFileName,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


def get_filtered_export_db_tests():
    return [
        (name, case)
        for name, case in all_examples().items()
        if case.support_level == SupportLevel.SUPPORTED
    ]


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    def test_export_with_extension_op_serialization(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        class FooExtensionOp:
            def __hash__(self):
                return 0

            def __eq__(self, other):
                return type(other) is type(self)

            def __call__(self, *args, **kwargs):
                return torch.ops.aten.add.Tensor(*args, **kwargs)

            @property
            def __name__(self):
                return "foo.my_op"

        class ExtensionVerifier(torch._export.verifier.Verifier):
            dialect = "FOO"

            def allowed_op_types(self):
                return super().allowed_op_types() + (FooExtensionOp,)

        class FooExtensionHandler(torch._export.serde.serialize.ExtensionHandler):
            @classmethod
            def namespace(cls):
                return "foo"

            @classmethod
            def to_op_name(cls, op):
                return "my_op"

            @classmethod
            def from_op_name(cls, name: str):
                self.assertEqual(name, "my_op")
                return FooExtensionOp()

            @classmethod
            def op_schema(cls, op):
                return torch.ops.aten.add.Tensor._schema

        inp = (torch.ones(10),)
        ep = export(TestModule(), inp, strict=True)

        # Register the custom op handler.
        foo_custom_op = FooExtensionOp()
        torch._export.serde.serialize.register_extension(
            FooExtensionOp, FooExtensionHandler
        )

        new_gm = copy.deepcopy(ep.graph_module)
        # Inject the custom operator.
        for node in new_gm.graph.nodes:
            if node.name == "add":
                node.target = foo_custom_op

        new_ep = ep._update(new_gm, ep.graph_signature, verifiers=[ExtensionVerifier])
        serialized = serialize(new_ep)
        deserialized = deserialize(serialized)
        self.assertEqual(
            len(
                deserialized.graph.find_nodes(op="call_function", target=foo_custom_op)
            ),
            1,
        )

    def test_predispatch_export_with_autograd_op(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                with torch.enable_grad():
                    return x + x

        inp = (torch.ones(10),)
        with torch.no_grad():
            from torch.export._trace import _export

            ep = _export(Foo(), inp, pre_dispatch=True)

        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)
        self.assertEqual(exp_out.requires_grad, actual_out.requires_grad)

    def test_export_example_inputs_preserved(self):
        class MyModule(torch.nn.Module):
            """A test module with that has multiple args and uses kwargs"""

            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            def forward(self, x, y, use_p=False):
                out = x + y
                if use_p:
                    out += self.p
                return out

        model = MyModule().eval()
        random_inputs = (torch.rand([2, 3]), torch.rand([2, 3]))
        exp_program = export(model, random_inputs, {"use_p": True}, strict=True)

        output_buffer = io.BytesIO()
        # Tests that example inputs are preserved when saving and loading module.
        torch.export.save(exp_program, output_buffer)
        loaded_model = torch.export.load(output_buffer)
        # Extract the example inputs from before and after saving.
        orig_args, orig_kwargs = exp_program.example_inputs
        loaded_args, loaded_kwargs = loaded_model.example_inputs
        # Run both modules and confirm that outputs match.
        orig_out = exp_program.module()(*orig_args, **orig_kwargs)
        loaded_out = loaded_model.module()(*loaded_args, **loaded_kwargs)
        self.assertEqual(orig_out, loaded_out)

    def test_metadata_run_decomp_serder(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        exp_program = export(M(), (torch.randn(4, 4),), strict=True)

        output_buffer = io.BytesIO()
        # Tests that example forward arg names are preserved when saving and loading module.
        torch.export.save(exp_program, output_buffer)
        loaded_model = torch.export.load(output_buffer)

        ep = loaded_model.run_decompositions({})
        # We should preserve the original module name
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x);  x = None
    return (sin,)""",
        )

    def test_metadata_parsing_with_layer_split(self):
        # Tests that modules with more complicated layer patterns can be serialized
        # and deserialized correctly.
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                # Splitting layers of a sequential stack introduces commas and parens
                # into metadata trace.
                out_start, out_rest = self.layers[0], self.layers[1:]
                h = out_start(x)
                h = out_rest(h)
                return h

        inp = (torch.ones(10),)
        # Module will only be able to roundtrip if metadata
        # can be correctly parsed.
        ep = export(MyModule(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        # Check that both modules run to confirm load was successful.
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_nested_layer_split(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                out_start, out_rest = self.layers[0], self.layers[1:]
                h = out_start(x)
                h = out_rest(h) + 2
                return h

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_module("a[(1)]", Bar())
                self.register_module("b[(2)]", Bar())
                self.register_buffer("c:[22]", torch.randn(1))

            def forward(self, x):
                out_a, out_b = getattr(self, "a[(1)]"), getattr(self, "b[(2)]")
                out_c = getattr(self, "c:[22]")
                h = out_a(x)
                h = out_b(h)
                return h + out_c

        inp = (torch.ones(10),)
        ep = export(Foo(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        # Check that both modules run to confirm load was successful.
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_param_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(4, 4))

            def forward(self, x):
                with torch.no_grad():
                    self.parameter.div_(2)
                return x + self.parameter

        foo = Foo()
        ep = torch.export.export(foo, (torch.rand(4, 4),)).run_decompositions()
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)
        val = loaded_ep.graph_signature.parameters_to_mutate
        self.assertEqual({"div": "parameter"}, val)

    def test_serialize_constant_outputs(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                # Along with tensor output, return Nonetype
                # and constant. Although these outputs aren't
                # very useful, they do show up in graphs.
                return x + 1, None, 1024

        # Check that module can be roundtripped, thereby confirming proper deserialization.
        inp = (torch.ones(10),)
        ep = export(MyModule(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_multiple_returns_from_node(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        exported_module = export(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),
                torch.ones([512]),
                torch.ones([512]),
            ),
            strict=True,
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.native_layer_norm.default")
        # aten::native_layer_norm returns 3 tensors
        self.assertEqual(len(node.outputs), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_sym_int(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dim1_bc = torch.export.Dim("dim1_b")
        dynamic_shapes = {
            "a": {0: dim0_ac},
            "b": {1: dim1_bc},
            "c": {0: dim0_ac, 1: dim1_bc},
        }
        exported_module = export(
            DynamicShapeSimpleModel(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)
        sym_size_nodes = [
            node
            for node in serialized.exported_program.graph_module.graph.nodes
            if node.target == "torch.ops.aten.sym_size.int"
        ]
        for node in sym_size_nodes:
            self.assertEqual(node.inputs[0].name, "self")
            self.assertEqual(node.inputs[1].name, "dim")

    def test_serialize_sym_float(self) -> None:
        # TODO(rec): This doesn't seem to test anything!

        class DynamicFloatSimpleModel(torch.nn.Module):
            def __init__(self, multiplier: torch.SymFloat):
                super().__init__()
                self.multiplier = multiplier

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                e = d * self.multiplier
                e_s0 = e.shape[0]
                e_s1 = e.shape[1]
                e_s3 = e_s0 * e_s1
                f = e.view(e_s3)
                return torch.cat([f, f])

        multiplier_sym = torch.SymFloat("multiplier_sym")
        _model = DynamicFloatSimpleModel(multiplier_sym)
        _inputs = (
            torch.randn(2, 4),
            torch.randn(4, 7),
            torch.randn(2, 7),
        )
        _dim0_ac = Dim("dim0_ac")
        _dim1_bc = Dim("dim1_b")

    def test_serialize_infinite_sym_int(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dim1_bc = torch.export.Dim("dim1_b")
        dynamic_shapes = {
            "a": {0: dim0_ac},
            "b": {1: dim1_bc},
            "c": {0: dim0_ac, 1: dim1_bc},
        }
        exported_module = export(
            DynamicShapeSimpleModel(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)
        for v in serialized.exported_program.range_constraints.values():
            self.assertEqual(v.max_val, None)

    def test_symint_list(self):
        # This reflects the behavior from inductor's ExternFallbackNode
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        symint = shape_env.create_unbacked_symint()
        serializer = GraphModuleSerializer(None, None)  # type: ignore[arg-type]
        res = serializer.serialize_inputs(
            torch.ops.aten.ones.default, ([1, symint, 3],), {}
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].arg._type, "as_sym_ints")

    def test_serialize_list_returns(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.split(x, 2)

        input = torch.arange(10.0).reshape(5, 2)
        exported_module = export(MyModule(), (input,), strict=True).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        # split.Tensor gets decomposed to split_with_sizes by the core ATen decomposition table
        self.assertEqual(node.target, "torch.ops.aten.split_with_sizes.default")
        self.assertEqual(len(node.outputs), 1)
        # Input looks like:
        # tensor([[0, 1],
        #         [2, 3],
        #         [4, 5],
        #         [6, 7],
        #         [8, 9]])
        # Output looks like:
        # (tensor([[0, 1],
        #          [2, 3]]),
        #  tensor([[4, 5],
        #          [6, 7]]),
        #  tensor([[8, 9]]))
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_nonfinite_inputs(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                x = torch.ops.aten.add.Scalar(x, math.inf)
                x = torch.ops.aten.add.Scalar(x, -math.inf)
                return torch.ops.aten.add.Scalar(x, math.nan)

        fn = Module()
        ep = torch.export.export(
            fn,
            (torch.randn(3, 2),),
        )
        json_bytes = _to_json_bytes(
            ExportedProgramSerializer().serialize(ep).exported_program
        )
        import json

        def parse_constant(x):
            raise RuntimeError(f"Invalid JSON float: {x}")

        json.loads(json_bytes, parse_constant=parse_constant)

    def test_multi_return_some_unused(self) -> None:
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        exported_module = export(
            MyModule(), (torch.ones([512, 512], requires_grad=True),), strict=True
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.var_mean.correction")
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_rational_ranges(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return x + x

        ep = export(
            M(), (torch.randn(4),), dynamic_shapes=({0: Dim("temp")},), strict=True
        )

        range_constraints = list(ep.range_constraints.keys())
        assert len(range_constraints) == 1
        symint = range_constraints[0]

        import sympy

        upper_range = sympy.Rational(10, 3)
        lower_range = sympy.Rational(10, 6)
        ep.range_constraints[symint] = ValueRanges(lower=lower_range, upper=upper_range)

        serialized = ExportedProgramSerializer().serialize(ep)
        self.assertEqual(
            serialized.exported_program.range_constraints[symint.name].min_val, 2
        )
        self.assertEqual(
            serialized.exported_program.range_constraints[symint.name].max_val, 3
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not has_triton(), "requires cuda and triton"
    )
    def test_triton_hop(self) -> None:
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            fval,
            ival,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y + fval + ival
            tl.store(out_ptr + offsets, output, mask=mask)

        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrap_triton(add_kernel)[grid](
                x, y, output, n_elements, 3.14, 42, BLOCK_SIZE=16
            )

            return output

        class MyModel(torch.nn.Module):
            def forward(self, x, y):
                return custom_add(x, y)

        def custom_add_autotune(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrap_triton(add_kernel)[grid](
                x, y, output, n_elements, 3.14, 42, BLOCK_SIZE=16, num_warps=8
            )

            return output

        class MyModelAutotune(torch.nn.Module):
            def forward(self, x, y):
                return custom_add_autotune(x, y)

        device = "cuda"

        for m in [MyModel().to(device), MyModelAutotune().to(device)]:
            args = (torch.randn(3, device=device), torch.randn(3, device=device))
            ep = torch.export.export(m, args=args)
            ep = ep.run_decompositions(decompose_custom_triton_ops=False)
            assert torch.allclose(m(*args), ep.module()(*args))

            serialized = ExportedProgramSerializer().serialize(ep)

            for node in serialized.exported_program.graph_module.graph.nodes:
                if (
                    node.target
                    == "torch.ops.higher_order.triton_kernel_wrapper_functional"
                ):
                    triton_node = node

            self.assertIsNotNone(triton_node)

            args = []
            kwargs = {}

            for arg in triton_node.inputs:
                if arg.kind == ArgumentKind.POSITIONAL:
                    args.append(arg.arg)
                elif arg.kind == ArgumentKind.KEYWORD:
                    kwargs[arg.name] = arg.arg

            self.assertEqual(len(args), 6)
            # Always: name, grid, output_indices and num_warps are
            # Triton version dependent: num_cpu_threads, shared_memory_bytes
            self.assertTrue(len(kwargs) >= 4)

            for i in range(3):
                self.assertIsNotNone(args[i].as_tensor)

            self.assertEqual(args[3].as_int, 3)
            self.assertAlmostEqual(args[4].as_float, 3.14, places=2)
            self.assertEqual(args[5].as_int, 42)
            kernel_name = kwargs["name"].as_string
            symbol_name = kernel_name.rpartition("_")[0]
            self.assertEqual(symbol_name, "add_kernel")
            self.assertEqual(kwargs["grid"].as_ints, [1, 1, 1])
            self.assertEqual(kwargs["output_indices"].as_ints, [2])
            self.assertEqual(
                kwargs["num_warps"].as_int, 8 if isinstance(m, MyModelAutotune) else 4
            )

            if "num_cpu_threads" in kwargs:
                self.assertEqual(kwargs["num_cpu_threads"].as_int, 0)
            if "shared_memory_bytes" in kwargs:
                self.assertEqual(kwargs["shared_memory_bytes"].as_int, 0)

            self.assertEqual(len(triton_node.outputs), 1)
            self.assertIsNotNone(triton_node.outputs[0].as_tensors)
            self.assertEqual(
                len(triton_node.outputs[0].as_tensors),
                len(kwargs["output_indices"].as_ints),
            )
            self.assertEqual(triton_node.outputs[0].as_tensors[0].name, "getitem")

            with self.assertRaisesRegex(
                SerializeError,
                "deserialize nyi for torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional",
            ):
                ExportedProgramDeserializer().deserialize(
                    serialized.exported_program,
                    serialized.state_dict,
                    serialized.constants,
                    serialized.example_inputs,
                )

    def test_kwargs_default(self) -> None:
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values = torch.randn(3, 2)
                return torch.searchsorted(x, values, side="right", right=True)

        f = Foo()

        x, _ = torch.sort(torch.randn(3, 4))
        exported_module = export(f, (x,), strict=True).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)

        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(node.inputs[2].name, "right")
        self.assertEqual(node.inputs[2].arg.as_bool, True)
        self.assertEqual(node.inputs[3].name, "side")
        self.assertEqual(node.inputs[3].arg.as_string, "right")

    def test_canonicalize(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                a = y + x
                b = x + y
                return b + a

        ep = export(Module(), (torch.randn(3, 2), torch.randn(3, 2)), strict=True)
        s = ExportedProgramSerializer().serialize(ep)
        c = canonicalize(s.exported_program)
        g = c.graph_module.graph
        self.assertLess(
            g.nodes[0].inputs[0].arg.as_tensor.name,
            g.nodes[1].inputs[0].arg.as_tensor.name,
        )

    def test_int_list(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, [])

        ep = torch.export.export(M(), (torch.randn(3, 2),), strict=True)
        serialized = ExportedProgramSerializer().serialize(ep)
        for node in serialized.exported_program.graph_module.graph.nodes:
            if "aten.sum.dim_IntList" in node.target:
                self.assertEqual(node.inputs[1].arg.type, "as_ints")

    def test_empty_constant(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        m = M()
        sample_inputs = (torch.randn(1, 4),)
        eager_out = m(*sample_inputs)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)
        ep_out = loaded_ep.module()(*sample_inputs)
        self.assertTrue(torch.allclose(eager_out, ep_out))
        self.assertEqual(len(loaded_ep.constants), 0)

    def test_empty_state_dict(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.randn(4, 4)

            def forward(self, x):
                return x + self.const

        m = M()
        sample_inputs = (torch.randn(4, 4),)
        eager_out = m(*sample_inputs)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)
        ep_out = loaded_ep.module()(*sample_inputs)
        self.assertTrue(torch.allclose(eager_out, ep_out))
        self.assertEqual(len(loaded_ep.state_dict), 0)

    def test_preserve_aliasing(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(8, 8)
                self.linear2 = self.linear1  # alias of linear1
                self.register_buffer("buffer1", torch.randn(8, 8))
                self.register_buffer("buffer2", torch.randn(8, 8), persistent=False)
                self.const1 = torch.ones(8, 8)
                self.const2 = self.const1.diagonal()  # a partial view of const1

            def forward(self, x):
                return (
                    self.linear1(x)
                    + self.linear2(x)
                    + self.buffer1
                    + self.buffer2
                    + self.const1
                    + self.const2
                )

        m = M()
        sample_inputs = (torch.randn(1, 8),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)
        eager_out = m(*sample_inputs)
        epm = loaded_ep.module()
        ep_out = epm(*sample_inputs)
        self.assertTrue(torch.allclose(eager_out, ep_out))

        # loaded_ep should preserve the aliasing info
        self.assertEqual(
            loaded_ep.state_dict["linear1.weight"].untyped_storage(),
            loaded_ep.state_dict["linear2.weight"].untyped_storage(),
        )
        self.assertEqual(
            loaded_ep.state_dict["linear1.bias"].untyped_storage(),
            loaded_ep.state_dict["linear2.bias"].untyped_storage(),
        )
        self.assertEqual(
            loaded_ep.constants["const1"].untyped_storage(),
            loaded_ep.constants["const2"].untyped_storage(),
        )
        # verify const1 and const2 share the same storage
        loaded_ep.constants["const1"][0][0] = 123
        self.assertEqual(loaded_ep.constants["const2"][0], 123)
        loaded_ep.constants["const2"][-1] = 321
        self.assertEqual(loaded_ep.constants["const1"][-1][-1], 321)

        # unlifted module should also preserve the aliasing info
        epm = loaded_ep.module()
        epm_state_dict = epm.state_dict()
        self.assertEqual(
            epm_state_dict["linear1.weight"].untyped_storage(),
            epm_state_dict["linear2.weight"].untyped_storage(),
        )
        self.assertEqual(
            epm_state_dict["linear1.bias"].untyped_storage(),
            epm_state_dict["linear2.bias"].untyped_storage(),
        )
        self.assertEqual(
            epm.const1.untyped_storage(),
            epm.const2.untyped_storage(),
        )
        # verify const1 and const2 share the same storage
        epm.const1[0][0] = 123
        self.assertEqual(epm.const2[0], 123)
        epm.const2[-1] = 321
        self.assertEqual(epm.const1[-1][-1], 321)

    def test_storage_offset(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.arange(8)[:4]
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x) + self.const

        m = M()
        sample_inputs = (torch.randn(1, 4),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_1D_tensor_slicing(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.arange(8)[::2]

            def forward(self, x):
                return x + self.const

        m = M()
        sample_inputs = (torch.randn(4),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_2D_tensor_slicing(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.randn(4, 4)[:2, :2]

            def forward(self, x):
                return x + self.const

        m = M()
        sample_inputs = (torch.randn(2, 2),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_non_float_weight(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(
                    torch.ones(2, 2, dtype=torch.int8), requires_grad=False
                )

            def forward(self, x):
                return x + self.p

        m = M()
        sample_inputs = (torch.randn(2, 2),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    @requires_gpu
    def test_weight_sharing_gpu(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c2 = torch.ones(2, 4, device=GPU_TYPE)
                self.c1 = self.c2[0, :]
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x) + self.c1 + self.c2

        m = M().to(GPU_TYPE)
        sample_inputs = (torch.randn(2, 4, device=GPU_TYPE),)
        ep = torch.export.export(m, sample_inputs)
        # Check that c1 and c2 share the same storage
        self.assertEqual(
            ep.constants["c1"].untyped_storage(), ep.constants["c2"].untyped_storage()
        )
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        # Check that c1 and c2 share the same storage after serdes
        self.assertEqual(
            loaded_ep.constants["c1"].untyped_storage(),
            loaded_ep.constants["c2"].untyped_storage(),
        )
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))

    def test_complex_constant(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                s = torch.sin(x)
                y = (1 + 1j) * s
                z = 1j * s
                return y, z

        m = M()
        sample_inputs = (torch.randn(2, 2),)
        ep = torch.export.export(m, sample_inputs)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        self.assertEqual(m(*sample_inputs), loaded_ep.module()(*sample_inputs))


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestDeserialize(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def _check_graph_nodes(self, gm1, gm2, _check_meta=True):
        # TODO: The _check_meta flag bypasses checking for
        # source_fn/nn_module_stack as there is an issue with
        # roundtripping the source_fn value on torch.ops.map nodes
        # original source_fn: <functorch.experimental._map.MapWrapper object at 0x7f80a0549930>
        # deserialized source_fn: 'functorch.experimental._map.map'

        self.assertEqual(len(gm1.graph.nodes), len(gm2.graph.nodes))

        for node1, node2 in zip(gm1.graph.nodes, gm2.graph.nodes):
            self.assertEqual(node1.op, node2.op)
            if node1.op == "call_function":
                # Check "val" metadata
                val1 = node1.meta.get("val", None)
                val2 = node2.meta.get("val", None)
                self.assertEqual(len(node1.args), len(node2.args))
                self.assertEqual(set(node1.kwargs.keys()), set(node2.kwargs.keys()))
                if val1 is None or val2 is None:
                    # Either both are None
                    self.assertEqual(val1, val2)
                elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                    # Or both are fake tensors with the same shape/dtype
                    self.assertEqual(len(val1.shape), len(val2.shape))
                    for s1, s2 in zip(val1.shape, val2.shape):
                        if is_concrete_int(s1) and is_concrete_int(s2):
                            self.assertEqual(s1, s2)
                        else:
                            self.assertEqual(str(s1), str(s2))
                    self.assertEqual(val1.dtype, val2.dtype)
                elif isinstance(val1, (list, tuple)) and isinstance(
                    val2, (list, tuple)
                ):
                    # Or both are fake tensors lists with one element and with the
                    # same shape/dtype
                    for v1, v2 in zip(
                        pytree.tree_leaves(val1), pytree.tree_leaves(val2)
                    ):
                        if isinstance(v1, FakeTensor):
                            self.assertEqual(v1.shape, v2.shape)
                            self.assertEqual(v1.dtype, v2.dtype)
                else:
                    # For expressions like 's0 < 10' can only compare through string
                    self.assertEqual(str(val1), str(val2))

                # Check "stack_trace" metadata
                self.assertEqual(
                    node1.meta.get("stack_trace", None),
                    node2.meta.get("stack_trace", None),
                )

                if node1.target == torch.ops.higher_order.cond:
                    true_graph1 = getattr(gm1, node1.args[1].target)
                    true_graph2 = getattr(gm2, node2.args[1].target)
                    self._check_graph_nodes(true_graph1, true_graph2)

                    false_graph1 = getattr(gm1, node1.args[2].target)
                    false_graph2 = getattr(gm2, node2.args[2].target)
                    self._check_graph_nodes(false_graph1, false_graph2)
                elif node1.target == torch.ops.higher_order.map_impl:
                    map_graph1 = getattr(gm1, node1.args[0].target)
                    map_graph2 = getattr(gm2, node2.args[0].target)
                    self._check_graph_nodes(map_graph1, map_graph2, False)

            if _check_meta and node1.op not in ("get_attr", "placeholder", "output"):
                # Check "nn_module_stack" metadata
                self.assertEqual(
                    node1.meta.get("nn_module_stack", None),
                    node2.meta.get("nn_module_stack", None),
                )
                # Check "source_fn_stack" metadata
                self.assertEqual(
                    node1.meta.get("source_fn_stack", None),
                    node2.meta.get("source_fn_stack", None),
                )

    def check_graph(
        self,
        fn,
        inputs,
        dynamic_shapes=None,
        _check_meta=True,
        use_pre_dispatch=True,
        strict=True,
    ) -> None:
        """Export a graph, serialize it, deserialize it, and compare the results."""

        def _deepcopy_inputs(inputs):
            # copy.deepcopy(deepcopy) can fail if tensor inputs have attribute (i.e. __dict__).
            # we remove __dict__ when deepcopying.
            dict_mapping = dict()
            inputs_clone = ()
            for idx, i in enumerate(inputs):
                if isinstance(i, torch.Tensor) and hasattr(inputs[0], "__dict__"):
                    dict_mapping[idx] = i.__dict__
                    i.__dict__ = {}
                inputs_clone += (copy.deepcopy(i),)

            # Add __dict__ back.
            for k, v in dict_mapping.items():
                inputs[k].__dict__ = v
                inputs_clone[k].__dict__ = v
            return inputs_clone

        def _check_graph(pre_dispatch):
            if pre_dispatch:
                ep = torch.export.export(
                    fn,
                    _deepcopy_inputs(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
            else:
                # We should have this branch because
                # PT2 Inference goes through this private
                # export API.
                ep = torch.export._trace._export(
                    fn,
                    _deepcopy_inputs(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    pre_dispatch=False,
                )
            ep.graph.eliminate_dead_code()

            serialized_artifact = serialize(ep, opset_version={"aten": 0})
            deserialized_ep = deserialize(
                serialized_artifact, expected_opset_version={"aten": 0}
            )
            deserialized_ep.graph.eliminate_dead_code()

            orig_outputs = ep.module()(*_deepcopy_inputs(inputs))
            loaded_outputs = deserialized_ep.module()(*_deepcopy_inputs(inputs))

            flat_orig_outputs = pytree.tree_leaves(orig_outputs)
            flat_loaded_outputs = pytree.tree_leaves(loaded_outputs)

            for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
                self.assertEqual(type(orig), type(loaded))
                # torch.allclose doesn't work for float8
                if isinstance(orig, torch.Tensor) and orig.dtype not in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ]:
                    if orig.is_meta:
                        self.assertEqual(orig, loaded)
                    else:
                        self.assertTrue(torch.allclose(orig, loaded))
                else:
                    self.assertEqual(orig, loaded)
            self._check_graph_nodes(
                ep.graph_module, deserialized_ep.graph_module, _check_meta
            )

        if use_pre_dispatch:
            _check_graph(pre_dispatch=True)
            _check_graph(pre_dispatch=False)
        else:
            _check_graph(pre_dispatch=False)

    def test_optional_tuple(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b, Tensor? c) -> (Tensor, Tensor?)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo")
            def foo_impl(a, b, c):
                res2 = None
                if c is not None:
                    res2 = c + a + b
                return a + b, res2

            class M(torch.nn.Module):
                def forward(self, a, b, c):
                    return torch.ops.mylib.foo(a, b, c)

            self.check_graph(M(), (torch.randn(3), torch.randn(3), torch.randn(3)))

    def test_unbacked_bindings_serialize(self):
        from torch._export.utils import _get_shape_env_from_gm
        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class M(torch.nn.Module):
            def forward(self, x, y):
                x += 2
                n = x.item()
                n = n * 2 + y.item()
                return n + 2

        inps = (
            torch.tensor(4),
            torch.tensor(5),
        )
        for _strict in [True, False]:
            ep = torch.export.export(M(), inps, strict=_strict).run_decompositions()

            # check bindings after deserialization
            buffer = io.BytesIO()
            save(ep, buffer)
            buffer.seek(0)
            loaded_ep = load(buffer)
            bound = set()
            for old_node, new_node in zip(ep.graph.nodes, loaded_ep.graph.nodes):
                self.assertEqual(
                    "unbacked_bindings" in old_node.meta,
                    "unbacked_bindings" in new_node.meta,
                )
                bound.update(new_node.meta.get("unbacked_bindings", {}))

            # check ShapeEnv counters
            shape_env = _get_shape_env_from_gm(loaded_ep.graph_module)
            next_index = shape_env.unbacked_symint_counter
            shape_env.unbacked_symint_counter += 1
            for symbol in bound:
                self.assertTrue(symbol_is_type(symbol, SymT.UNBACKED_INT))
                self.assertTrue(
                    int(str(symbol)[len(prefix_str[SymT.UNBACKED_INT]) :]) < next_index
                )

    def test_sym_bool_dynamic_shapes(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                z = x[:, -y.shape[0] :, :]
                return z

        inputs = (torch.ones(4, 5, 10), torch.ones(3))
        dynamic_shapes = {"x": {}, "y": {0: Dim("seqlen", max=4)}}
        # Compile with dynamic_shapes set to get operator.neg involved
        self.check_graph(MyModule(), inputs, dynamic_shapes=dynamic_shapes)

    def test_auto_functionalize(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo1",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::foo2",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::foo3",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo1", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo1")
            def foo1_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return n + n

            @torch.library.impl("mylib::foo2", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo2")
            def foo2_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return (n + n, n * n)

            @torch.library.impl("mylib::foo3", "cpu", lib=lib)
            @torch.library.register_fake("mylib::foo3")
            def foo3_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return

            class M(torch.nn.Module):
                def forward(self, x, y, z, n):
                    n = torch.ops.mylib.foo1(x, y, z, 2, n)
                    torch.ops.mylib.foo3(x, y, z, 2, n)
                    return torch.ops.mylib.foo2(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            # TODO Auto_functionalize is not supported on pre_dispatch IR
            self.check_graph(M(), orig_args, use_pre_dispatch=False)

    def test_hoo_symint_input(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                num = c.item()
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda a, b: a + b + num,
                    false_fn=lambda a, b: a - b - num,
                    operands=(a, b),
                )

        inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(2))
        self.check_graph(Mod(), inp, use_pre_dispatch=False)

    def test_none_input(self):
        """
        Testing a backwards-compatibility breakage where old models do not have
        an input spec with the node name.
        """

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + z

        ep = torch.export.export(M(), (torch.ones(3, 3), None, torch.ones(3, 3)))

        serialized_program = ExportedProgramSerializer(None, 2).serialize(ep)
        serialized_program.exported_program.graph_module.signature.input_specs[1] = (
            schema.InputSpec.create(
                user_input=schema.UserInputSpec(
                    arg=schema.Argument.create(as_none=True)
                )
            )
        )
        ep = ExportedProgramDeserializer(None
```



## High-Level Overview

"""PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interfereswith test_sym_bool)

This Python file contains 92 class(es) and 239 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSerialize`, `TestModule`, `FooExtensionOp`, `ExtensionVerifier`, `FooExtensionHandler`, `Foo`, `MyModule`, `M`, `MyModule`, `Bar`, `Foo`, `Foo`, `MyModule`, `MyModule`, `DynamicShapeSimpleModel`, `DynamicFloatSimpleModel`, `DynamicShapeSimpleModel`, `MyModule`, `Module`, `MyModule`

**Functions defined**: `get_filtered_export_db_tests`, `test_export_with_extension_op_serialization`, `forward`, `__hash__`, `__eq__`, `__call__`, `__name__`, `allowed_op_types`, `namespace`, `to_op_name`, `from_op_name`, `op_schema`, `test_predispatch_export_with_autograd_op`, `__init__`, `forward`, `test_export_example_inputs_preserved`, `__init__`, `forward`, `test_metadata_run_decomp_serder`, `forward`

**Key imports**: copy, io, math, tempfile, unittest, zipfile, namedtuple, Path, NamedTuple, GPU_TYPE, HAS_GPU


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `io`
- `math`
- `tempfile`
- `unittest`
- `zipfile`
- `collections`: namedtuple
- `pathlib`: Path
- `typing`: NamedTuple
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU
- `torch.testing._internal.triton_utils`: requires_gpu
- `triton`
- `triton.language as tl`
- `torch.library`: wrap_triton
- `torch.utils._triton`: has_triton
- `torch`
- `torch._dynamo as torchdynamo`
- `torch._export.serde.schema as schema`
- `torch.export._trace`
- `torch.utils._pytree as pytree`
- `torch._export.db.case`: ExportCase, SupportLevel
- `torch._export.db.examples`: all_examples
- `torch._export.serde.schema`: ArgumentKind
- `torch._higher_order_ops.torchbind`: enable_torchbind_tracing
- `torch._subclasses.fake_tensor`: FakeTensor, FakeTensorMode
- `torch.export`: Dim, export, load, save, unflatten
- `torch.export.pt2_archive.constants`: ARCHIVE_VERSION_PATH
- `torch.fx.experimental.symbolic_shapes`: is_concrete_int, ValueRanges
- `torch.testing._internal.torchbind_impls`: init_torchbind_implementations


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/export/test_serialize.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_serialize.py_docs.md`
- **Keyword Index**: `test_serialize.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

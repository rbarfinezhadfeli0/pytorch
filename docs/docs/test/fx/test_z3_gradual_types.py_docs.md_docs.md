# Documentation: `docs/test/fx/test_z3_gradual_types.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_z3_gradual_types.py_docs.md`
- **Size**: 53,760 bytes (52.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_z3_gradual_types.py`

## File Metadata

- **Path**: `test/fx/test_z3_gradual_types.py`
- **Size**: 91,084 bytes (88.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]
import operator
import unittest

import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.experimental.meta_tracer import symbolic_trace as meta_symbolic_trace
from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintT,
    DVar,
    T,
    TVar,
)
from torch.fx.experimental.migrate_gradual_types.constraint_generator import (
    ConstraintGenerator,
)
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import (
    transform_constraint,
)
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_consistency,
    op_matching,
    op_precision,
)
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (
    evaluate_conditional_with_constraints,
    transform_all_constraints,
)
from torch.fx.experimental.migrate_gradual_types.z3_types import D, tensor_type, z3_dyn
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.tensor_type import Dyn, TensorType


try:
    import z3  # type: ignore[import]

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


try:
    from torchvision import models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class TorchDynamoUseCases(unittest.TestCase):
    def test_dim(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 2])):
                y = x.dim()
                return y

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        y_res = z3.z3.Int(2)
        self.assertEqual(s.model()[y_res], 2)

    def test_reshape(self):
        """
        In this example, we prove that some nodes must
        always have a fixed shape regardless of the input
        """

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                y = x.view(100)
                tmp = y.size()[0]
                return tmp

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        dim = z3.Int(4)
        self.assertEqual(s.model()[dim], 100)
        # print(s.model()[dim])


class HFOperations(unittest.TestCase):
    def test_eq_dim(self):
        """
        test dimensions and equalities
        """

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([32, 4, 4])):
                eq = x.dim() == 3
                return eq

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.eq:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.unsat)

    def test_conditional_ne_1(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([32, 4, 4]), y: TensorType([32, 4, 4])):
                size_5 = x.size()
                getitem_7 = size_5[0]
                getitem_8 = size_5[1]
                getitem_9 = size_5[2]
                ne_1 = y != (getitem_7, getitem_8, getitem_9)
                return ne_1

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.ne:
                node = n

        # since x and y are equal, the requirement that x != y cannot be true, so we should get unsat
        # for the positive condition and sat for the negative condition
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.unsat)
        self.assertEqual(negative, z3.sat)

    def test_bmm(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 2, 3]), y: TensorType([1, 3, 2])):
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        output = z3.Const(3, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])

    def test_bmm2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: TensorType([1, 3, 2])):
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)

        output = z3.Const(3, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(0), 0)
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])

    def test_bmm3(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 3, 3]), y: TensorType([1, 3, 2])):
                bmm = torch.bmm(x, y)
                return bmm

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

    def test_transpose(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 2, 3, 4])):
                transpose = x.transpose(0, 1)
                return transpose

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3, 4))

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        output = z3.Const(2, tensor_type)
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])
        self.assertEqual(s.model()[output].arg(3).arg(1), b.shape[3])

        # change the annotation to Dyn
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_index_select(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2050, 1024]), y: Dyn):
                index_select = x.index_select(0, y)
                return index_select

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # print(symbolic_traced)
        b = BasicBlock().forward(torch.rand(2050, 1024), torch.ones(8).int())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        index_select = z3.Const(3, tensor_type)

        # the second dimension of the result should not be affected since
        # the index is 0
        self.assertEqual(s.model()[index_select].arg(1).arg(1), b.shape[1])

        replacement_vector = z3.Const(2, tensor_type)

        # we set the vector to Dyn
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        index_select = z3.Const(3, tensor_type)
        s.add(replacement_vector == z3_dyn)
        self.assertEqual(s.check(), z3.sat)

        # this implies that the index at 0 should be dyn
        self.assertEqual(s.model()[index_select].arg(0).arg(0), 0)

    def test_get_attr(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 2, 3])):
                getattr = x.device
                to = x.to(getattr)
                return to

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        b = BasicBlock().forward(torch.rand(1, 2, 3))
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        attr_res = z3.Const(3, tensor_type)
        assert s.model()[attr_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[attr_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[attr_res].arg(2).arg(1) == b.shape[2]

    def test_expand(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([1, 4])):
                size = x.size()
                getitem = size[-1]
                expand = x.expand(getitem, 4)
                return expand

        b = BasicBlock().forward(torch.rand(1, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        expand_res = z3.Const(4, tensor_type)
        assert s.model()[expand_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[expand_res].arg(1).arg(1) == b.shape[1]

        # change the annotation on the input to Dyn.
        # the last dimension should still be 4
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        assert s.model()[expand_res].arg(1).arg(1) == b.shape[1]

    def test_getitem_tensor(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([4, 4])):
                getitem = x[
                    (None, None, slice(None, None, None), slice(None, None, None))
                ]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]

        # change the annotation on the input to make sure it propagates
        # to the output
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, 4])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # dyn check
        assert s.model()[get_item_res].arg(2).arg(0) == 0

    def test_getitem_tensor2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([4, 4])):
                getitem = x[(None, None)]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]

    def test_getitem_tensor_3(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([4, 4])):
                getitem = x[
                    (None, slice(None, None, None), None, slice(None, None, None))
                ]
                return getitem

        B = BasicBlock()
        b = B.forward(torch.rand(4, 4))
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        get_item_res = z3.Const(2, tensor_type)
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]

    def test_layer_norm(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.LayerNorm((1024,))

            def forward(self, x: Dyn):
                return self.l(x)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the output a size 1 tensor which should result
        # in the migration of the input

        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # input shape = output shape
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

        # change annotation to the wrong shape
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([10, 10])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

        # fix the annotation
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([10, 1024])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        s.check()
        b = BasicBlock().forward(torch.rand(10, 1024)).shape
        self.assertEqual(s.model()[output].arg(0).arg(1), b[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b[1])

    def test_layer_norm_functional(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                return torch.nn.functional.layer_norm(x, (1024,))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # make the output a size 1 tensor which should result
        # in the migration of the input

        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # input shape = output shape
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

    def test_ne_int_long_type_as(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn, Dyn])):
                ne_int = torch.ne(x, y).int()
                type_as = ne_int.type_as(y)
                long = type_as.long()
                return long

        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # migrate one of the parameters to a fully static shape so we can compare

        input = z3.Const(1, tensor_type)
        input_2 = z3.Const(2, tensor_type)
        s1, s2 = z3.Ints("s1 s2")

        output_long = z3.Const(8, tensor_type)
        s.add(input == tensor_type.tensor2(D(1, 2), D(1, 4)))
        s.add(input_2 == tensor_type.tensor2(D(1, s1), D(1, s2)))

        self.assertEqual(s.check(), z3.sat)
        actual_shape = BasicBlock().forward(torch.rand(2, 4), torch.rand(2, 4)).shape
        self.assertEqual(s.model()[output_long].arg(0).arg(1), actual_shape[0])
        self.assertEqual(s.model()[output_long].arg(1).arg(1), actual_shape[1])

    def test_ne(self):
        s1, s2 = z3.Ints("s1 s2")
        s11, s22 = z3.Ints("s11 s22")
        d1, d2 = D(s11, s1), D(0, s2)

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.ne(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # change the annotations
        for n in graph.nodes:
            if n.name == "x":
                n.type = TensorType([1, 2])
            if n.name == "y":
                n.type = TensorType([2, Dyn])

        # resulting type should be TensorType([2, 2])
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # force the second dimension to be Dyn
        # output should still be TensorType([2, 2])
        input = z3.Const(2, tensor_type)
        s.add(input == tensor_type.tensor2(d1, d2))
        self.assertEqual(s.check(), z3.sat)
        B = BasicBlock().forward(torch.rand(1, 2), torch.rand(2, 1))
        output = z3.Const(3, tensor_type)
        self.assertEqual(s.model()[output].arg(0).arg(1), B.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), B.shape[0])

    def test_cumsum(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4, 3])):
                t = torch.cumsum(x, 3)
                return t

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        # should be unsat since the index is not valid for this annotation
        self.assertEqual(s.check(), z3.unsat)

        # modify the annotation to Dyn which should give sat
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # # modify the annotation to the right tensor size
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([1, 2, 3, 4])

        # verify that the input is equal to the output
        B = BasicBlock().forward(torch.rand(1, 2, 3, 4))
        res_shape = B.shape
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        # confirm the output matches the expected tensor
        result = z3.Const(2, tensor_type)
        self.assertEqual(s.model()[result].arg(0).arg(1), res_shape[0])
        self.assertEqual(s.model()[result].arg(1).arg(1), res_shape[1])
        self.assertEqual(s.model()[result].arg(2).arg(1), res_shape[2])
        self.assertEqual(s.model()[result].arg(3).arg(1), res_shape[3])

        # confirm the output is not dyn
        self.assertNotEqual(s.model()[result].arg(0).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(1).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(2).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(3).arg(0).as_long(), 0)

    def test_cumsum_kwargs(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4, 3])):
                t = torch.cumsum(x, dim=3)
                return t

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        # should be unsat since the index is not valid for this annotation
        self.assertEqual(s.check(), z3.unsat)

        # modify the annotation to Dyn which should give sat
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_arange(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                return arange

        B = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        arange_result = z3.Const(5, tensor_type)
        self.assertNotEqual(s.model()[arange_result].arg(0).arg(0).as_long(), 0)
        self.assertEqual(s.model()[arange_result].arg(0).arg(1).as_long(), B.size()[0])

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_scalar_add(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                add = arange + 1
                return add

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        arange_result = z3.Const(5, tensor_type)
        add_result = z3.Const(6, tensor_type)
        self.assertEqual(s.model()[arange_result], s.model()[add_result])

    def test_regular_add_2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                to = x.to()
                size = to.size()
                getitem = size[-1]
                add = getitem + 1
                return add

        b = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Int(5)
        self.assertEqual(s.model()[res], b)

    def test_regular_add_3(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                to = x.to()
                size = to.size()
                getitem = size[-1]
                add = 1 + getitem
                return add

        b = BasicBlock().forward(torch.rand(2, 4))

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Int(5)
        self.assertEqual(s.model()[res], b)

    def test_embedding(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                return self.embedding(x)

        B = BasicBlock().forward(torch.ones([2, 4], dtype=torch.long)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        embedding_result = z3.Const(2, tensor_type)

        assert s.model()[embedding_result].arg(0).arg(1) == B[0]
        assert s.model()[embedding_result].arg(1).arg(1) == B[1]
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

        # change the type. This should still be satisfiable
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn])

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        assert s.model()[embedding_result].arg(0).arg(0) == 0
        assert s.model()[embedding_result].arg(1).arg(0) == 0
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

        # change the type to Dyn. Here, we will get an arbitrary migration
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

    def test_embedding_2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4]), y: TensorType([Dyn, 1024])):
                return torch.nn.functional.embedding(x, y)

        B = (
            BasicBlock()
            .forward(torch.ones([2, 4], dtype=torch.long), torch.rand(256008, 1024))
            .size()
        )
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        embedding_result = z3.Const(5, tensor_type)

        assert s.model()[embedding_result].arg(0).arg(1) == B[0]
        assert s.model()[embedding_result].arg(1).arg(1) == B[1]
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

    def test_size_two_args(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 2, Dyn])):
                size = x.size(-1)
                return size

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        d1, d2 = z3.Int(39), z3.Int(2)
        d4, d5 = z3.Int("input_d1"), z3.Int("input_d2")

        # migrate the third dimension
        s.add(d1 != 0)

        self.assertEqual(s.check(), z3.sat)
        input = z3.Const(1, tensor_type)
        s.add(input == tensor_type.tensor3(D(3, 39), D(1, 2), D(d4, d5)))

        # check if the item we got is the right one
        self.assertEqual(s.check(), z3.sat)
        self.assertEqual(s.model()[d5], s.model()[d2])
        self.assertEqual(s.model()[d1], s.model()[d4])

    def test_size_getitem(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                size = x.size()
                getitem = size[-1]
                return getitem

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)

        # force the input to be of size 4

        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        input = z3.Const(1, tensor_type)
        s.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # check if the model is still SAT
        self.assertEqual(s.check(), z3.sat)

        s1, s2 = z3.Int(23), z3.Int(3)

        # check that the item is correct
        self.assertEqual(s.model()[s1], s.model()[s2])

        # invalid index but should still be SAT because input will be Dyn
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                size = x.size()
                getitem = size[-10]
                return getitem

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)

        self.assertEqual(s.check(), z3.sat)
        s.add(input != z3_dyn)
        self.assertEqual(s.check(), z3.unsat)

    def test_view_mul(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                embed_tokens = self.embed_tokens(view)
                mul = embed_tokens * 32.0
                return mul

        # print(B)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # print(traced)

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        # print(s.model())

        embedding_result = z3.Const(6, tensor_type)

        # note that the view output will be: tensor3(dim(0, 0), dim(1, 4), dim(1, 1024))
        # this is due to the reshape constraints. This can be lifted
        # but would require revising the type rules accordingly so we leave it for now
        assert (s.model()[embedding_result].arg(1).arg(1)) == 4
        assert (s.model()[embedding_result].arg(2).arg(1)) == 1024

        mul_result = z3.Const(13, tensor_type)
        assert s.model()[mul_result] == s.model()[embedding_result]

    def test_gt(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4])):
                size = x.size()
                getitem_1 = size[-1]
                gt = getitem_1 > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        res = z3.Bool(4)
        self.assertEqual(s.model()[res], True)

    def test_view(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                view = x.view(-1, 8)
                return view

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_lt_tensor(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4]), y: Dyn):
                lt = x > y
                return lt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_conditional_wrong_assumption(self):
        """
        Test condition after making the wrong assumption about the input
        """

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                gt = x > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.gt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

    def test_conditional(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """

        class BasicBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                _embed_tokens = self.embed_tokens(view)
                getitem_1 = size[-1]
                gt = getitem_1 > 1
                return gt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.gt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.unsat)

        # change the annotation to Dyn
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # here, both should be SAT since the input is Dyn
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

        # change the annotation to TensorType[Dyn, Dyn]
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn])

        # here, both should be SAT as well
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        self.assertEqual(positive, z3.sat)
        self.assertEqual(negative, z3.sat)

    def test_conditional_2(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns the opposite result of the above testcase

        """

        class BasicBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                size = x.size()
                getitem = size[-1]
                view = x.view(-1, getitem)
                _embed_tokens = self.embed_tokens(view)
                getitem_1 = size[-1]
                lt = getitem_1 < 1
                return lt

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())

        # The node we are considering is the gt node
        for n in graph.nodes:
            if n.target == operator.lt:
                node = n

        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        self.assertEqual(positive, z3.unsat)
        self.assertEqual(negative, z3.sat)


class ComposeOperationsGradualTypes(unittest.TestCase):
    def test_masked_fill(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                size = x.size()
                getitem = size[-1]
                arange = torch.arange(getitem)
                view = x.view(-1, getitem)
                lt = arange > view
                masked_fill = x.masked_fill_(lt, 0)
                return masked_fill

        B = BasicBlock().forward(torch.rand(2, 4))
        # print(B.shape)

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # print(symbolic_traced)
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)
        masked_fill_res = z3.Const(10, tensor_type)
        self.assertEqual(
            s.model()[masked_fill_res].arg(0).arg(1).as_long(), B.size()[0]
        )
        self.assertEqual(
            s.model()[masked_fill_res].arg(1).arg(1).as_long(), B.size()[1]
        )

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_add_reshape_1(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)

        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_add_reshape_2(self):
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.add(torch.reshape(x, (-1, 2)), torch.reshape(y, (2, 2, 2)))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(BasicBlock())
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.sat)

    def test_conv_reshape_add_0(self):
        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn, y: Dyn):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)

    def test_conv_reshape_add_0_2(self):
        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn, y: TensorType([4, 1])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)

        #        4,1
        # 1, 2, 4, 8
        res = B.forward(torch.rand(20, 20), torch.rand(1, 2, 4, 8)).size()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.sat)

        conv_result = z3.Const(4, tensor_type)
        add_result = z3.Const(9, tensor_type)
        input_2 = z3.Const(2, tensor_type)

        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        solver.check()
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        solver.add(input_2 == tensor_type.tensor2(D(1, 4), D(1, 1)))
        self.assertEqual(solver.check(), z3.sat)
        solver.add(add_result == tensor_type.tensor4(d1, d2, d3, d4))
        self.assertEqual(solver.check(), z3.sat)

        # first dimension could be anything because we have broadcasting
        assert solver.model()[s1] == res[0]
        assert solver.model()[s2] == res[1]
        assert solver.model()[s3] == res[2]
        assert solver.model()[s4] == res[3]

    def test_conv_reshape_add_0_3(self):
        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn, y: TensorType([11, 1])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv_reshape_add_1(self):
        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn, y: TensorType([1, 2, 10, 20])):
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        new_transformed_c = transform_all_constraints(traced)
        solver = z3.Solver()
        solver.add(new_transformed_c)
        self.assertEqual(solver.check(), z3.unsat)


class GradualTypes(unittest.TestCase):
    def test_conv_reshape_unsat(self):
        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn):
                return self.conv1(torch.reshape(x, (1, 2, 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/fx/test_z3_gradual_types.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/fx`):

- [`named_tup.py_kw.md_docs.md`](./named_tup.py_kw.md_docs.md)
- [`test_dynamism.py_kw.md_docs.md`](./test_dynamism.py_kw.md_docs.md)
- [`test_fx_traceback.py_docs.md_docs.md`](./test_fx_traceback.py_docs.md_docs.md)
- [`test_fx_xform_observer.py_docs.md_docs.md`](./test_fx_xform_observer.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_fx_xform_observer.py_kw.md_docs.md`](./test_fx_xform_observer.py_kw.md_docs.md)
- [`test_fx_node_hook.py_kw.md_docs.md`](./test_fx_node_hook.py_kw.md_docs.md)
- [`test_partitioner_order.py_docs.md_docs.md`](./test_partitioner_order.py_docs.md_docs.md)
- [`test_subgraph_rewriter.py_kw.md_docs.md`](./test_subgraph_rewriter.py_kw.md_docs.md)
- [`test_fx_split.py_docs.md_docs.md`](./test_fx_split.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_z3_gradual_types.py_docs.md_docs.md`
- **Keyword Index**: `test_z3_gradual_types.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `test/onnx/test_utility_funs.py`

## File Metadata

- **Path**: `test/onnx/test_utility_funs.py`
- **Size**: 71,094 bytes (69.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: onnx"]

import copy
import io
import re
import warnings

import onnx

import parameterized
import pytorch_test_common
import torchvision
from autograd_helper import CustomFunction as CustomFunction2
from pytorch_test_common import (
    skipIfNoCuda,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
)

import torch
import torch.onnx
import torch.utils.cpp_extension
from torch.onnx import _constants, OperatorExportTypes, TrainingMode, utils
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.onnx.symbolic_helper import _unpack_list, parse_args
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


def _remove_test_environment_prefix_from_scope_name(scope_name: str) -> str:
    """Remove test environment prefix added to module.

    Remove prefix to normalize scope names, since different test environments add
    prefixes with slight differences.

    Example:

        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "test_utility_funs.M"
        >>> )
        "M"
        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "test_utility_funs.test_abc.<locals>.M"
        >>> )
        "M"
        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "__main__.M"
        >>> )
        "M"
    """
    prefixes_to_remove = ["test_utility_funs", "__main__"]
    for prefix in prefixes_to_remove:
        scope_name = re.sub(f"{prefix}\\.(.*?<locals>\\.)?", "", scope_name)
    return scope_name


class _BaseTestCase(pytorch_test_common.ExportTestCase):
    def _model_to_graph(
        self,
        model,
        input,
        do_constant_folding=True,
        training=TrainingMode.EVAL,
        operator_export_type=OperatorExportTypes.ONNX,
        input_names=None,
        dynamic_axes=None,
    ):
        torch.onnx.utils._setup_trace_module_map(model, False)
        if training == torch.onnx.TrainingMode.TRAINING:
            model.train()
        elif training == torch.onnx.TrainingMode.EVAL:
            model.eval()
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)
        graph, params_dict, torch_out = utils._model_to_graph(
            model,
            input,
            do_constant_folding=do_constant_folding,
            _disable_torch_constant_prop=True,
            operator_export_type=operator_export_type,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
        )
        return graph, params_dict, torch_out


@parameterized.parameterized_class(
    [
        {"opset_version": opset}
        for opset in range(
            _constants.ONNX_BASE_OPSET,
            _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET + 1,
        )
    ],
    class_name_func=lambda cls,
    num,
    params_dict: f"{cls.__name__}_opset_{params_dict['opset_version']}",
)
class TestUtilityFuns(_BaseTestCase):
    opset_version = None

    def test_is_in_onnx_export(self):
        test_self = self

        class MyModule(torch.nn.Module):
            def forward(self, x):
                test_self.assertTrue(torch.onnx.is_in_onnx_export())
                raise ValueError
                return x + 1

        x = torch.randn(3, 4)
        f = io.BytesIO()
        try:
            torch.onnx.export(
                MyModule(), x, f, opset_version=self.opset_version, dynamo=False
            )
        except ValueError:
            self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_validate_dynamic_axes_invalid_input_output_name(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils._validate_dynamic_axes(
                {"input1": {}, "output": {}, "invalid_name1": {}, "invalid_name2": {}},
                None,
                ["input1", "input2"],
                ["output"],
            )
            messages = [str(warning.message) for warning in w]
        self.assertIn(
            "Provided key invalid_name1 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertIn(
            "Provided key invalid_name2 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertEqual(len(messages), 2)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_to_slice(self):
        class SplitModule(torch.nn.Module):
            def forward(self, x, y, t):
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        graph, _, _ = self._model_to_graph(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::SplitToSequence")

    def test_constant_fold_transpose(self):
        class TransposeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.transpose(a, 1, 0)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(3, 2)
        graph, _, __ = self._model_to_graph(
            TransposeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    @skipIfUnsupportedMaxOpsetVersion(17)
    def test_constant_fold_reduceL2(self):
        class ReduceModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=2, dim=-2, keepdim=False)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ReduceModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL2")

    @skipIfUnsupportedMaxOpsetVersion(17)
    def test_constant_fold_reduceL1(self):
        class NormModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=1, dim=-2)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            NormModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL1")

    def test_constant_fold_slice(self):
        class NarrowModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.narrow(a, 0, 0, 1)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            NarrowModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_index_exceeds_dim(self):
        class SliceIndexExceedsDimModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[1:10]  # index exceeds dimension
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceIndexExceedsDimModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_negative_index(self):
        class SliceNegativeIndexModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[0:-1]  # index relative to the end
                c = torch.select(a, dim=-1, index=-2)
                d = torch.select(a, dim=1, index=0)
                return b + x, c + d

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceNegativeIndexModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")

    def test_constant_fold_gather(self):
        class GatherModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.select(a, dim=1, index=-2)
                c = torch.index_select(a, dim=-2, index=torch.tensor([0, 1]))
                return b + 1, c + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        model = GatherModule()
        model(x)
        graph, _, __ = self._model_to_graph(
            GatherModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Gather")

    def test_constant_fold_unsqueeze(self):
        class UnsqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.unsqueeze(a, -2)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 2, 3)
        graph, _, __ = self._model_to_graph(
            UnsqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_unsqueeze_multi_axies(self):
        class PReluModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                a = torch.randn(2, 3, 4, 5, 8, 7)
                return self.prelu(x) + a

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(2, 3, 4, 5, 8, 7)
        graph, _, __ = self._model_to_graph(
            PReluModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3, 4, 5]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 5)

    def test_constant_fold_squeeze_without_axes(self):
        class SqueezeModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a) + x + torch.squeeze(a)

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_squeeze_with_axes(self):
        class SqueezeAxesModule(torch.nn.Module):
            def forward(self, x):
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a, dim=-3) + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeAxesModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_concat(self):
        class ConcatModule(torch.nn.Module):
            def forward(self, x):
                # Why did I insert a Cast here?  There appears to be intentional
                # behavior in ONNX constant folding where constant tensors which
                # are not attached to any known to be foldable onnx
                # operations don't get extracted into the initializer graph.  So
                # without these casts, we will actually fail to pull out one of
                # the constants, thus failing constant folding.  I think the
                # test is wrong but I don't have time to write a more correct
                # test (I think the right way to go about the test is to setup
                # a predicate for what invariant graphs should hold after
                # constant folding, and then verify this predicate holds.
                # I think the asserts below are an attempt at this predicate,
                # but it is not right!)
                #
                # More commentary at
                # https://github.com/pytorch/pytorch/pull/18698/files#r340107552
                a = torch.tensor([[1.0, 2.0, 3.0]]).to(torch.float)
                b = torch.tensor([[4.0, 5.0, 6.0]]).to(torch.float)
                c = torch.cat((a, b), 0)
                d = b + c
                return x + d

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ConcatModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Concat")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_lstm(self):
        class GruNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                return self.mygru(input, initial_state)

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = self._model_to_graph(
            GruNet(),
            (input, h0),
            input_names=["input", "h0"],
            dynamic_axes={"input": [0, 1, 2], "h0": [0, 1, 2]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Concat")
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")

        if self.opset_version <= 12:
            self.assertEqual(len(list(graph.nodes())), 3)
        else:
            # Unsqueeze op parameter "axes" as an input instead of as an attribute when opset version >= 13
            self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_transpose_matmul(self):
        class MatMulNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        A = torch.randn(2, 3)
        graph, _, __ = self._model_to_graph(
            MatMulNet(), (A,), input_names=["A"], dynamic_axes={"A": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_reshape(self):
        class ReshapeModule(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                b = self.weight.reshape(1, -1, 1, 1)
                return x * b

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(4, 5)
        graph, _, __ = self._model_to_graph(
            ReshapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Reshape")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_div(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                div = self.weight.div(torch.tensor([1, 2, 3, 4, 5]))
                return div * x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Div")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_mul(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                mul = self.weight.mul(torch.tensor([1, 2, 3, 4, 5]))
                return mul / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Mul")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_add(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                add = self.weight + torch.tensor([1, 2, 3, 4, 5])
                return add - x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        for node in graph.nodes():
            self.assertTrue(node.kind() != "onnx::Add")
        self.assertEqual(len(list(graph.nodes())), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        self.assertEqual(weight, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_constant_fold_sub(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                sub = self.weight - torch.tensor([1, 2, 3, 4, 5])
                return sub + x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sub")
        self.assertEqual(len(list(graph.nodes())), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        self.assertEqual(weight, torch.tensor([0.0, -1.0, -2.0, -3.0, -4.0]))

    def test_constant_fold_sqrt(self):
        class Module(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                sqrt = torch.sqrt(self.weight)
                return sqrt / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sqrt")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_shape(self):
        class ShapeModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                shape = self.weight.shape[0]
                return x + shape

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            ShapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Shape")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_upsample_scale_fold_as_constant(self):
        # upsample scale is a constant, not a model parameter,
        # therefore should not be added as initializer after constant folding.
        model = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.randn(1, 32, 224, 224)
        f = io.BytesIO()
        torch.onnx.export(model, x, f, dynamo=False)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(len(onnx_model.graph.initializer), 0)

    def test_verbose(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                return torch.exp(input)

        x = torch.randn(3, 4)

        def is_model_stripped(f, verbose=None):
            if verbose is None:
                torch.onnx.export(
                    MyModule(), x, f, opset_version=self.opset_version, dynamo=False
                )
            else:
                torch.onnx.export(
                    MyModule(),
                    x,
                    f,
                    verbose=verbose,
                    opset_version=self.opset_version,
                    dynamo=False,
                )
            model = onnx.load(io.BytesIO(f.getvalue()))
            model_strip = copy.copy(model)
            onnx.helper.strip_doc_string(model_strip)
            return model == model_strip

        # test verbose=False (default)
        self.assertTrue(is_model_stripped(io.BytesIO()))
        # test verbose=True
        self.assertFalse(is_model_stripped(io.BytesIO(), True))

    # NB: remove this test once DataParallel can be correctly handled
    def test_error_on_data_parallel(self):
        model = torch.nn.DataParallel(torch.nn.ReflectionPad2d((1, 2, 3, 4)))
        x = torch.randn(1, 2, 3, 4)
        f = io.BytesIO()
        with self.assertRaisesRegex(
            ValueError,
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try ",
        ):
            torch.onnx.export(
                model, x, f, opset_version=self.opset_version, dynamo=False
            )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_sequence_dim(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return [x, y]

        model = Module()
        # Export with scripting to keep output as Sequence type.
        # Tracing unpacks the list.
        script_model = torch.jit.script(model)
        x = torch.randn(2, 3)

        # Case 1: dynamic axis
        f = io.BytesIO()
        y = torch.randn(2, 3)
        torch.onnx.export(
            script_model,
            (x, y),
            f,
            opset_version=self.opset_version,
            input_names=["x", "y"],
            dynamic_axes={"y": [1]},
            dynamo=False,
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        loop_output_value_info_proto = onnx_model.graph.output[0]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, None]
        )
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

        # Case 2: no dynamic axes.
        f = io.BytesIO()
        y = torch.randn(2, 3)
        torch.onnx.export(
            script_model, (x, y), f, opset_version=self.opset_version, dynamo=False
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        loop_output_value_info_proto = onnx_model.graph.output[0]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, 3]
        )
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

    def test_export_mode(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                return y

        model = MyModule()
        x = torch.randn(10, 3, 128, 128)
        f = io.BytesIO()

        # set mode to in inference mode and export in training mode
        model.eval()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        # verify that the model state is preserved
        self.assertEqual(model.training, old_state)

        # set mode to training mode and export in inference mode
        model.train()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
        # verify that the model state is preserved
        self.assertEqual(model.training, old_state)

    def test_export_does_not_fail_on_frozen_scripted_module(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                if x > 0:
                    return x
                else:
                    return x * x

        class Outer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = torch.jit.script(Inner())

            def forward(self, x):
                return self.inner(x)

        x = torch.zeros(1)
        # Freezing is only implemented in eval mode. So we need to call eval()
        outer_module = Outer().eval()
        module = torch.jit.trace_module(outer_module, {"forward": (x)})
        # jit.freeze removes the training attribute in the module
        module = torch.jit.freeze(module)

        torch.onnx.export(
            module, (x,), io.BytesIO(), opset_version=self.opset_version, dynamo=False
        )

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function(self):
        class N(torch.nn.Module):
            def __init__(self, prob):
                super().__init__()
                self.dropout = torch.nn.Dropout(prob)

            def forward(self, x):
                return self.dropout(x)

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)]
                )
                self.celu1 = torch.nn.CELU(1.0)
                self.celu2 = torch.nn.CELU(2.0)
                self.dropout = N(0.5)

            def forward(self, x, y, z):
                res1 = self.celu1(x)
                res2 = self.celu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.dropout(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        # Export specified modules. Test against specifying modules that won't
        # exist in the exported model.
        # Model export in inference mode will remove dropout node,
        # thus the dropout module no longer exist in graph.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={
                torch.nn.CELU,
                torch.nn.Dropout,
                torch.nn.LayerNorm,
            },
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))

        # Check function definition
        funcs = onnx_model.functions
        celu_funcs = [f for f in funcs if f.name == "CELU"]
        self.assertEqual(len(celu_funcs), 1)
        self.assertEqual(celu_funcs[0].domain, "torch.nn.modules.activation")
        self.assertEqual(len(celu_funcs[0].attribute), 3)
        ln_funcs = [f for f in funcs if f.name == "LayerNorm"]
        self.assertEqual(len(ln_funcs), 1)
        self.assertEqual(ln_funcs[0].domain, "torch.nn.modules.normalization")
        self.assertEqual(len(ln_funcs[0].attribute), 3)

        # Check local function nodes
        nodes = onnx_model.graph.node
        celu_ns = [n for n in nodes if n.op_type == "CELU"]
        ln_ns = [n for n in nodes if n.op_type == "LayerNorm"]
        self.assertEqual(len(celu_ns), 2)
        self.assertEqual(celu_ns[0].domain, "torch.nn.modules.activation")
        self.assertEqual(len(celu_ns[0].attribute), 3)
        self.assertEqual(len(ln_ns), 3)
        self.assertEqual(ln_ns[0].domain, "torch.nn.modules.normalization")
        self.assertEqual(len(ln_ns[0].attribute), 3)

        # Export specified modules.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={torch.nn.CELU},
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "CELU")

        # Export with empty specified modules. Normal export.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions=set(),
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 0)

        # Export all modules. Should contain {M, CELU, LayerNorm}.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions=True,
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 3)

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_overloads(self):
        class NWithOverloads(torch.nn.Module):
            def forward(self, x, y=None, z=None):
                if y is None:
                    return x + 1
                elif z is None:
                    return x + y
                else:
                    return x + y, x + z

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.n = NWithOverloads()

            def forward(self, x, y, z):
                return self.n(x), self.n(x, y), self.n(x, y, z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={NWithOverloads},
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 3)
        func_names = [f.name for f in funcs]
        self.assertIn("NWithOverloads", func_names)
        self.assertIn("NWithOverloads.1", func_names)
        self.assertIn("NWithOverloads.2", func_names)

    # Failing after ONNX 1.13.0
    @skipIfUnsupportedMaxOpsetVersion(1)
    def test_local_function_infer_scopes(self):
        class M(torch.nn.Module):
            def forward(self, x):
                # Concatenation of scalars inserts unscoped tensors in IR graph.
                new_tensor_shape = x.size()[:-1] + (1, 1, -1)
                tensor = x.view(*new_tensor_shape)
                return tensor

        x = torch.randn(4, 5)
        f = io.BytesIO()
        torch.onnx.export(
            M(),
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            do_constant_folding=False,
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertIn("M", [f.name for f in funcs])

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_predefined_attributes(self):
        class M(torch.nn.Module):
            num_layers: int

            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            def forward(self, x):
                for ln in self.lns:
                    x = ln(x)
                return x

        x = torch.randn(2, 3)
        f = io.BytesIO()
        model = M(3)
        torch.onnx.export(
            model,
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        m_funcs = [fn for fn in funcs if fn.name == "M"]
        self.assertEqual(m_funcs[0].attribute, ["num_layers"])
        ln_funcs = [fn for fn in funcs if fn.name == "LayerNorm"]
        self.assertEqual(ln_funcs[0].attribute, ["eps", "elementwise_affine"])

        from onnx import helper

        m_node = [n for n in onnx_model.graph.node if n.op_type == "M"]
        self.assertEqual(
            m_node[0].attribute[0],
            helper.make_attribute("num_layers", model.num_layers),
        )

        ln_nodes = [n for n in m_funcs[0].node if n.op_type == "LayerNorm"]
        expected_ln_attrs = [
            helper.make_attribute(
                "elementwise_affine", model.lns[0].elementwise_affine
            ),
            helper.make_attribute("eps", model.lns[0].eps),
        ]
        for ln_node in ln_nodes:
            self.assertIn(ln_node.attribute[0], expected_ln_attrs)
            self.assertIn(ln_node.attribute[1], expected_ln_attrs)

    # This test cases checks the issue where an object does not have an attribute.
    # When enabling `export_modules_as_functions = True`, the exporter could return an
    # AttributeError. With this test case, we check that the export passes successfully
    # without any AttributeError exceptions.
    # See https://github.com/pytorch/pytorch/pull/109759 for an example. The exception that
    # this test tries to avoid is `AttributeError: 'Embedding' object has no attribute 'freeze'`.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_subset_of_predefined_attributes(self):
        class M(torch.nn.Module):
            num_layers: int

            def __init__(self, num_layers):
                super().__init__()
                self.embed_layer = torch.nn.Embedding.from_pretrained(
                    torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
                )
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            def forward(self, x):
                e = self.embed_layer(torch.LongTensor([1]))
                for ln in self.lns:
                    x = ln(x)
                return x, e

        x = torch.randn(2, 3)
        f = io.BytesIO()
        model = M(3)
        torch.onnx.export(
            model,
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            verbose=True,  # Allows the test case to print `Skipping module attribute 'freeze'`
            dynamo=False,
        )

    def test_node_scope(self):
        class N(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        class M(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=float(i)) for i in range(num_layers)]
                )
                self.gelu1 = torch.nn.GELU()
                self.gelu2 = torch.nn.GELU()
                self.relu = N()

            def forward(self, x, y, z):
                res1 = self.gelu1(x)
                res2 = self.gelu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.relu(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        model = M(3)
        expected_scope_names = {
            "M::/torch.nn.modules.activation.GELU::gelu1",
            "M::/torch.nn.modules.activation.GELU::gelu2",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.0",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.1",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.2",
            "M::/N::relu/torch.nn.modules.activation.ReLU::relu",
            "M::",
        }

        graph, _, _ = self._model_to_graph(
            model, (x, y, z), input_names=[], dynamic_axes={}
        )
        for node in graph.nodes():
            self.assertIn(
                _remove_test_environment_prefix_from_scope_name(node.scopeName()),
                expected_scope_names,
            )

        graph, _, _ = self._model_to_graph(
            torch.jit.script(model), (x, y, z), input_names=[], dynamic_axes={}
        )
        for node in graph.nodes():
            self.assertIn(
                _remove_test_environment_prefix_from_scope_name(node.scopeName()),
                expected_scope_names,
            )

    def test_scope_of_constants_when_combined_by_cse_pass(self):
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant):
                super().__init__()
                self.constant = constant

            def forward(self, x):
                # 'self.constant' is designed to be the same for all layers,
                # hence it is common sub expression.
                return x + self.constant

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [M(constant=torch.tensor(1.0)) for i in range(layers)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )

        # NOTE: Duplicated constants are populated due to implicit casting in scalar_type_analysis,
        #       so we expect 3 constants with different scopes. The 3 constants are for the 3 layers.
        #       If CSE in exporter is improved later, this test needs to be updated.
        #       It should expect 1 constant, with same scope as root.
        expected_root_scope_name = "N::"
        expected_layer_scope_name = "M::layers"
        expected_constant_scope_name = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        constant_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Constant":
                constant_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
        self.assertEqual(constant_scope_names, expected_constant_scope_name)

    def test_scope_of_nodes_when_combined_by_cse_pass(self):
        layer_num = 3

        class M(torch.nn.Module):
            def __init__(self, constant, bias):
                super().__init__()
                self.constant = constant
                self.bias = bias

            def forward(self, x):
                # 'constant' and 'x' is designed to be the same for all layers,
                # hence `x + self.constant` is common sub expression.
                # 'bias' is designed to be different for all layers,
                # hence `* self.bias` is not common sub expression.
                return (x + self.constant) * self.bias

        class N(torch.nn.Module):
            def __init__(self, layers: int = layer_num):
                super().__init__()

                self.layers = torch.nn.ModuleList(
                    [
                        M(constant=torch.tensor([1.0]), bias=torch.randn(1))
                        for i in range(layers)
                    ]
                )

            def forward(self, x):
                y = []
                for layer in self.layers:
                    y.append(layer(x))
                return y[0], y[1], y[2]

        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )
        expected_root_scope_name = "N::"
        expected_layer_scope_name = "M::layers"
        expected_add_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.0"
        ]
        expected_mul_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        add_scope_names = []
        mul_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Add":
                add_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
            elif node.kind() == "onnx::Mul":
                mul_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
        self.assertEqual(add_scope_names, expected_add_scope_names)
        self.assertEqual(mul_scope_names, expected_mul_scope_names)

    def test_aten_fallthrough(self):
        # Test aten export of op with no symbolic
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.erfc(x)

        x = torch.randn(2, 3, 4)
        GLOBALS.export_onnx_opset_version = self.opset_version
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::erfc")

    def test_custom_op_fallthrough(self):
        # Test custom op
        op_source = """
        #include <torch/script.h>

        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
          return self + other;
        }

        static auto registry =
          torch::RegisterOperators("custom_namespace::custom_op", &custom_add);
        """

        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        class FooModel(torch.nn.Module):
            def forward(self, input, other):
                # Calling custom op
                return torch.ops.custom_namespace.custom_op(input, other)

        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)
        model = FooModel()
        graph, _, __ = self._model_to_graph(
            model,
            (x, y),
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "custom_namespace::custom_op")

    # gelu is exported as onnx::Gelu for opset >= 20
    @skipIfUnsupportedMaxOpsetVersion(19)
    def test_custom_opsets_gelu(self):
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::gelu", 9)

        def gelu(g, self, approximate):
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("::gelu", gelu, 9)
        model = torch.nn.GELU(approximate="none")
        x = torch.randn(3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)

    # gelu is exported as onnx::Gelu for opset >= 20
    @skipIfUnsupportedMaxOpsetVersion(19)
    def test_register_aten_custom_op_symbolic(self):
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "aten::gelu", 9)

        def gelu(g, self, approximate):
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("aten::gelu", gelu, 9)
        model = torch.nn.GELU(approximate="none")
        x = torch.randn(3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model, (x,), f, opset_version=self.opset_version, dynamo=False
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))

        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        self.assertEqual(graph.opset_import[1].domain, "com.microsof
```



## High-Level Overview

"""Remove test environment prefix added to module.    Remove prefix to normalize scope names, since different test environments add    prefixes with slight differences.    Example:        >>> _remove_test_environment_prefix_from_scope_name(        >>>     "test_utility_funs.M"        >>> )        "M"        >>> _remove_test_environment_prefix_from_scope_name(        >>>     "test_utility_funs.test_abc.<locals>.M"        >>> )        "M"        >>> _remove_test_environment_prefix_from_scope_name(        >>>     "__main__.M"        >>> )        "M"

This Python file contains 65 class(es) and 167 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_BaseTestCase`, `TestUtilityFuns`, `MyModule`, `SplitModule`, `TransposeModule`, `ReduceModule`, `NormModule`, `NarrowModule`, `SliceIndexExceedsDimModule`, `SliceNegativeIndexModule`, `GatherModule`, `UnsqueezeModule`, `PReluModel`, `SqueezeModule`, `SqueezeAxesModule`, `ConcatModule`, `GruNet`, `MatMulNet`, `ReshapeModule`, `Module`

**Functions defined**: `_remove_test_environment_prefix_from_scope_name`, `_model_to_graph`, `test_is_in_onnx_export`, `forward`, `test_validate_dynamic_axes_invalid_input_output_name`, `test_split_to_slice`, `forward`, `test_constant_fold_transpose`, `forward`, `test_constant_fold_reduceL2`, `forward`, `test_constant_fold_reduceL1`, `forward`, `test_constant_fold_slice`, `forward`, `test_constant_fold_slice_index_exceeds_dim`, `forward`, `test_constant_fold_slice_negative_index`, `forward`, `test_constant_fold_gather`

**Key imports**: copy, io, re, warnings, onnx, parameterized, pytorch_test_common, torchvision, CustomFunction as CustomFunction2, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/onnx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `io`
- `re`
- `warnings`
- `onnx`
- `parameterized`
- `pytorch_test_common`
- `torchvision`
- `autograd_helper`: CustomFunction as CustomFunction2
- `torch`
- `torch.onnx`
- `torch.utils.cpp_extension`
- `torch.onnx._internal.torchscript_exporter._globals`: GLOBALS
- `torch.onnx.symbolic_helper`: _unpack_list, parse_args
- `torch.testing._internal`: common_utils
- `torch.testing._internal.common_utils`: skipIfNoLapack


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/onnx/test_utility_funs.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/onnx`):

- [`test_lazy_import.py_docs.md`](./test_lazy_import.py_docs.md)
- [`onnx_test_common.py_docs.md`](./onnx_test_common.py_docs.md)
- [`pytorch_test_common.py_docs.md`](./pytorch_test_common.py_docs.md)
- [`test_pytorch_onnx_shape_inference.py_docs.md`](./test_pytorch_onnx_shape_inference.py_docs.md)
- [`test_onnxscript_no_runtime.py_docs.md`](./test_onnxscript_no_runtime.py_docs.md)
- [`test_models_onnxruntime.py_docs.md`](./test_models_onnxruntime.py_docs.md)
- [`test_custom_ops.py_docs.md`](./test_custom_ops.py_docs.md)
- [`test_models.py_docs.md`](./test_models.py_docs.md)
- [`test_onnxscript_runtime.py_docs.md`](./test_onnxscript_runtime.py_docs.md)
- [`test_pytorch_onnx_onnxruntime_cuda.py_docs.md`](./test_pytorch_onnx_onnxruntime_cuda.py_docs.md)


## Cross-References

- **File Documentation**: `test_utility_funs.py_docs.md`
- **Keyword Index**: `test_utility_funs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

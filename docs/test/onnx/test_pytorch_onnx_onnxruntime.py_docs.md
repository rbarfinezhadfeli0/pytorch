# Documentation: test_pytorch_onnx_onnxruntime.py

## File Metadata
- **Path**: `test/onnx/test_pytorch_onnx_onnxruntime.py`
- **Size**: 492969 bytes
- **Lines**: 13932
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: onnx"]
# ruff: noqa: F841

from __future__ import annotations

import functools
import io
import itertools
import os
import unittest
from collections import OrderedDict
from typing import Optional, Union

import numpy as np

import onnx
import onnx_test_common
import parameterized
import torchvision
from model_defs import (
    lstm_flattening_result,
    rnn_model_with_packed_sequence,
    word_language_model,
)
from pytorch_test_common import (
    BATCH_SIZE,
    RNN_BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
    skipDtypeChecking,
    skipIfQuantizationBackendQNNPack,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedOpsetVersion,
    skipScriptTest,
    skipShapeChecking,
    skipTraceTest,
)

import torch
from torch import Tensor
from torch.nn.utils import rnn as rnn_utils
from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import verification
from torch.onnx._internal.torchscript_exporter._type_utils import JitScalarType
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


def _init_test_generalized_rcnn_transform():
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size, max_size, image_mean, image_std
    )
    return transform


def _init_test_rpn():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    out_channels = 256
    rpn_head = torchvision.models.detection.rpn.RPNHead(
        out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
    )
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7
    rpn_score_thresh = 0.0

    rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
        rpn_anchor_generator,
        rpn_head,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        score_thresh=rpn_score_thresh,
    )
    return rpn


def _construct_tensor_for_quantization_test(
    shape: tuple[int, ...],
    offset: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
) -> Tensor:
    """Helper function to generate weights and test inputs in a deterministic way.

    Due to difference in implementation details between PyTorch and ONNXRuntime, randomly generated
    test data for quantization tests can be flaky. To help stablize the test, this helper function is
    used to generate weights and test inputs in a deterministic way.

    Args:
        shape (Tuple[int]): Shape for tensor to construct.
        offset (Optional[Union[int, float]]): Offset to be added to the generated tensor.
        max_val (Optional[Union[int, float]]): If any element within tensor has a larger absolute value than
            max_val, the tensor will be scaled by max_val / tensor.abs().max(). This step is done after
            applying offset.
    """
    tensor = torch.arange(np.prod(shape), dtype=torch.float).view(shape)
    if offset is not None:
        tensor = tensor + offset
    if max_val is not None and tensor.abs().max() > max_val:
        tensor = tensor * max_val / tensor.abs().max()
    return tensor


def _parameterized_class_attrs_and_values(
    min_opset_version: int, max_opset_version: int
):
    attrs = ("opset_version", "is_script", "keep_initializers_as_inputs")
    input_values = []
    input_values.extend(itertools.product((7, 8), (True, False), (True,)))
    # Valid opset versions are defined in torch/onnx/_constants.py.
    # Versions are intentionally set statically, to not be affected by changes elsewhere.
    if min_opset_version < 9:
        raise ValueError("min_opset_version must be >= 9")
    input_values.extend(
        itertools.product(
            range(min_opset_version, max_opset_version + 1),
            (True, False),
            (True, False),
        )
    )
    return {"attrs": attrs, "input_values": input_values}


def _parametrize_rnn_args(arg_name):
    options = {
        "layers": {1: "unilayer", 3: "trilayer"},
        "bidirectional": {True: "bidirectional", False: "forward"},
        "initial_state": {True: "with_initial_state", False: "no_initial_state"},
        "packed_sequence": {
            0: "without_sequence_lengths",
            1: "with_variable_length_sequences",
            2: "with_batch_first_sequence_lengths",
        },
        "dropout": {0.2: "with_dropout", 0.0: "without_dropout"},
    }

    return {
        "arg_str": arg_name,
        "arg_values": options[arg_name].keys(),
        "name_fn": lambda val: options[arg_name][val],
    }


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        onnx_test_common.MIN_ONNX_OPSET_VERSION, onnx_test_common.MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
@common_utils.instantiate_parametrized_tests
class TestONNXRuntime(onnx_test_common._TestONNXRuntime):
    def test_fuse_conv_bn1d(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = torch.nn.BatchNorm1d(33)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn2d(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv3d(
                    3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False
                )
                self.bn = torch.nn.BatchNorm3d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-6)

    def test_fuse_conv_in_block(self):
        class Fuse(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    in_channels=5,
                    out_channels=5,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=1,
                )
                self.bn = torch.nn.BatchNorm1d(5)

            def forward(self, x):
                results_available = True

                if x.sum() > -1:
                    results_available = False

                if results_available:
                    x = self.conv(x)
                    x = self.bn(x)

                return x

        model = Fuse()
        x = torch.randn(2, 5, 9, requires_grad=True)
        self.run_test(
            torch.jit.script(model),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            rtol=1e-3,
            atol=1e-6,
        )

    def test_conv_tbc(self):
        from torch.nn.modules.utils import _single

        class ConvTBC(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = _single(kernel_size)
                self.padding = _single(padding)

                self.weight = torch.nn.Parameter(
                    Tensor(self.kernel_size[0], in_channels, out_channels)
                )
                self.bias = torch.nn.Parameter(Tensor(out_channels))
                self.reset_parameters()

            def reset_parameters(self):
                torch.nn.init.xavier_normal_(self.weight)
                torch.nn.init.zeros_(self.bias)

            def conv_tbc(self, input):
                return torch.conv_tbc(
                    input.contiguous(), self.weight, self.bias, self.padding[0]
                )

            def forward(self, input):
                return self.conv_tbc(input)

        in_channels = 3
        out_channels = 5
        kernel_size = 5
        model = ConvTBC(in_channels, out_channels, kernel_size, padding=0)
        x = torch.randn(10, 7, in_channels, requires_grad=True)
        self.run_test(model, (x,), atol=1e-5)

    def test_reshape_constant_fold(self):
        class Reshape(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1

        x = torch.randn(4, 5)
        self.run_test(Reshape(), (x,), rtol=1e-3, atol=1e-5)

    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        if model_name == "GRU":
            model = word_language_model.RNNModelWithTensorHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        elif model_name == "LSTM":
            model = word_language_model.RNNModelWithTupleHidden(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        else:
            model = word_language_model.RNNModel(
                model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
            )
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(model, (x, model.hidden))

    def get_image(self, rel_path: str, size: tuple[int, int]) -> Tensor:
        from PIL import Image
        from torchvision import transforms

        data_dir = os.path.join(os.path.dirname(__file__), "assets")
        path = os.path.join(data_dir, *rel_path.split("/"))
        image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)

        return transforms.ToTensor()(image)

    def get_test_images(self) -> tuple[list[Tensor], list[Tensor]]:
        return (
            [self.get_image("grace_hopper_517x606.jpg", (100, 320))],
            [self.get_image("rgb_pytorch.png", (250, 380))],
        )

    def test_paste_mask_in_image(self):
        masks = torch.rand(10, 1, 26, 26)
        boxes = torch.rand(10, 4)
        boxes[:, 2:] += torch.rand(10, 2)
        boxes *= 50
        o_im_s = (100, 100)
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        out = paste_masks_in_image(masks, boxes, o_im_s)
        jit_trace = torch.jit.trace(
            paste_masks_in_image,
            (masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]),
        )
        out_trace = jit_trace(
            masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]
        )

        assert torch.all(out.eq(out_trace))

        masks2 = torch.rand(20, 1, 26, 26)
        boxes2 = torch.rand(20, 4)
        boxes2[:, 2:] += torch.rand(20, 2)
        boxes2 *= 100
        o_im_s2 = (200, 200)
        from torchvision.models.detection.roi_heads import paste_masks_in_image

        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        out_trace2 = jit_trace(
            masks2, boxes2, [torch.tensor(o_im_s2[0]), torch.tensor(o_im_s2[1])]
        )

        assert torch.all(out2.eq(out_trace2))

    def test_heatmaps_to_keypoints(self):
        maps = torch.rand(10, 1, 26, 26)
        rois = torch.rand(10, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints

        out = heatmaps_to_keypoints(maps, rois)
        jit_trace = torch.jit.trace(heatmaps_to_keypoints, (maps, rois))
        out_trace = jit_trace(maps, rois)

        assert torch.all(out[0].eq(out_trace[0]))
        assert torch.all(out[1].eq(out_trace[1]))

        maps2 = torch.rand(20, 2, 21, 21)
        rois2 = torch.rand(20, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints

        out2 = heatmaps_to_keypoints(maps2, rois2)
        out_trace2 = jit_trace(maps2, rois2)

        assert torch.all(out2[0].eq(out_trace2[0]))
        assert torch.all(out2[1].eq(out_trace2[1]))

    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    @skipScriptTest()  # scripting prim::unchecked_cast prim::setattr
    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_sliceint(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_neg_slice(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:-1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.uint8)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_data(self):
        class Data(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())

        x = torch.randn(3, 4)
        self.run_test(Data(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(Data(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_mask_nd(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[input > 0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipScriptTest()
    def test_dict(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(
                    x_in[list(x_in.keys())[0]],  # noqa: RUF015
                    list(x_in.keys())[0],  # noqa: RUF015
                )
                return x_out

        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()
    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()  # User-defined class not supported
    def test_dict_output(self):
        class DictModelOutput(OrderedDict):
            tensor_out: Tensor
            tuple_out: Optional[tuple[Tensor]] = None
            list_out: Optional[list[Tensor]] = None

        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return DictModelOutput(
                    tensor_out=a,
                    tuple_out=(b, c),
                    list_out=[d],
                )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return a, (b, c), d

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return a, ((b,), (c, d))

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_input(self):
        class TupleModel(torch.nn.Module):
            def forward(self, a: tuple[Tensor, Tensor]):
                return a

        x = (torch.randn(3, 4), torch.randn(4, 3))
        self.run_test(TupleModel(), input_args=(x,))

    def test_tuple_primitive_input(self):
        class TupleModel(torch.nn.Module):
            def forward(self, a: tuple[int, Tensor], b):
                return a[0], a[1] + b

        x = (3, torch.randn(4, 3))
        y = torch.randn(4, 3)
        self.run_test(TupleModel(), input_args=(x, y))

    def test_nested_tuple_input(self):
        class NestedTupleModel(torch.nn.Module):
            def forward(self, a, b: tuple[Tensor, tuple[Tensor, Tensor]]):
                return a + b[0] + b[1][0] + b[1][1]

        x = torch.randn(4, 5)
        y = (torch.randn(4, 5), (torch.randn(1, 5), torch.randn(4, 1)))
        self.run_test(NestedTupleModel(), input_args=(x, y))

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_none(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = None,
                z: Optional[Tensor] = None,
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()
        # Without kwargs dict.
        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))
        # With kwargs dict.
        self.run_test(model, (x,), {"y": y, "z": None})
        self.run_test(model, (x,), {"y": None, "z": z})
        self.run_test(model, (x,), {"z": z})
        self.run_test(model, (x,), {"y": y})

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = torch.ones(2, 3),
                z: Optional[Tensor] = torch.zeros(2, 3),
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()

        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor_script(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Optional[Tensor] = torch.ones(2, 3),
                z: Optional[Tensor] = torch.zeros(2, 3),
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = torch.jit.script(Model())

        self.run_test(model, (x, y, z), input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y, "z": z}, input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y}, input_names=("x", "y"))

        for example_inputs, example_kwargs in (
            ((x, y, None), {}),
            ((x, None, z), {}),
            ((x,), {"y": y, "z": None}),
            ((x,), {"y": None, "z": z}),
        ):
            with self.assertRaisesRegex(
                ValueError, "args contained 1 None's after flattening."
            ):
                self.run_test(
                    model, example_inputs, example_kwargs, input_names=("x", "y", "z")
                )

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_none(self):
        class Model(torch.nn.Module):
            def forward(self, x: Optional[Tensor] = None, y: Optional[Tensor] = None):
                if x is not None:
                    return x
                if y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(
            model,
            (),
            {"x": x, "y": None},
            # y disappears in tracing.
            input_names=("x",),
        )

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x: Optional[Tensor] = torch.ones(2, 3),
                y: Optional[Tensor] = torch.zeros(2, 3),
            ):
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (None, y))
        # tracing means y is never used so it's removed from the exported model inputs,
        # and we fail when trying to run ORT.
        with self.assertRaisesRegex(ValueError, "got too many positional inputs"):
            self.run_test(model, (x, y))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor_script(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x: Optional[Tensor] = torch.ones(2, 3),
                y: Optional[Tensor] = torch.zeros(2, 3),
            ):
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = torch.jit.script(Model())

        # Optional supports None inputs
        self.run_test(model, (x,))
        # NOTE: default value is not supported on ONNX, so torch and ONNX has
        # different behavior
        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            self.run_test(model, (), {"y": y}, input_names=["y"])

        self.run_test(model, (x, y))
        self.run_test(model, (), {"x": x, "y": y}, input_names=("x", "y"))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logit(self):
        class Logit(torch.nn.Module):
            def __init__(self, eps):
                super().__init__()
                self.eps = eps

            def forward(self, x):
                return x.logit(self.eps)

        model = Logit(eps=1e-6)
        self.run_test(model, torch.randn(1, 3, 640, 640))

    class Atleast1d(torch.nn.Module):
        def forward(self, t, w, x, y, z):
            return torch.atleast_1d((t, w, x, y, z))

    class Atleast2d(torch.nn.Module):
        def forward(self, t, w, x, y, z):
            return torch.atleast_2d((t, w, x, y, z))

    class Atleast3d(torch.nn.Module):
        def forward(self, t, w, x, y, z):
            return torch.atleast_3d((t, w, x, y, z))

    class Atleast1dTensor(torch.nn.Module):
        def forward(self, x):
            return torch.atleast_1d(x)

    class Atleast2dTensor(torch.nn.Module):
        def forward(self, x):
            return torch.atleast_2d(x)

    class Atleast3dTensor(torch.nn.Module):
        def forward(self, x):
            return torch.atleast_3d(x)

    @skipScriptTest()  # tracing uses prim::ListUnpack to avoid onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize("module_class", (Atleast1d, Atleast2d, Atleast3d))
    def test_atleast_nd_list_input(self, module_class: torch.nn.Module):
        inputs = (
            torch.tensor(1.0),
            torch.randn(2),
            torch.randn(2, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5),
        )
        self.run_test(module_class(), inputs)

    @skipScriptTest()  # tracing uses prim::ListUnpack to avoid onnx::SequenceConstruct
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize(
        "module_class", (Atleast1dTensor, Atleast2dTensor, Atleast3dTensor)
    )
    @common_utils.parametrize(
        "inputs",
        [
            torch.tensor(1.0),
            torch.randn(2),
            torch.randn(2, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4, 5),
        ],
    )
    def test_atleast_nd_single_tensor_input(
        self, module_class: torch.nn.Module, inputs: torch.Tensor
    ):
        self.run_test(module_class(), inputs)

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y: Optional[Tensor]):
                if y is not None:
                    return x + y
                return x

        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (x, x))

    @skipScriptTest()  # Needs https://github.com/pytorch/rfcs/pull/21
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y: tuple[Optional[Tensor], Optional[Tensor]]):
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipScriptTest()  # tracing eliminates None inputs so it works differently. See _script version below.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: tuple[Optional[Tensor], Optional[Tensor]] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipTraceTest()  # tracing is verified with different set of inputs. See above.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor_script(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: tuple[Optional[Tensor], Optional[Tensor]] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x = torch.randn(2, 3)
        y0 = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        model = torch.jit.script(Model())
        with self.assertRaisesRegex(
            ValueError, "args contained 1 None's after flattening."
        ):
            self.run_test(model, (x, (None, y1)))
        self.run_test(model, (x, (y0, y1)))
        # export succeeds, but running ORT through run_test would fail because the exported model
        # has the inputs flattened into 3 inputs.
        torch.onnx.export(
            model,
            (x, {"y": (y0, y1)}),
            io.BytesIO(),
            opset_version=self.opset_version,
            dynamo=False,
        )

    def test_primitive_input_integer(self):
        class Model(torch.nn.Module):
            def forward(self, x: int, y):
                return x + y

        x = 3
        y = torch.randint(10, (2, 3, 4))
        self.run_test(Model(), (x, y))

    @skipDtypeChecking
    def test_primitive_input_floating(self):
        class Model(torch.nn.Module):
            def forward(self, x: float, y):
                return x + y

        x = 3.0
        y = torch.randn(2, 3, 4)
        self.run_test(Model(), (x, y))

    def test_primitive_input_bool(self):
        class Model(torch.nn.Module):
            def forward(self, flag: bool, x, y):
                if flag:
                    return x
                else:
                    return y

        flag = True
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(torch.jit.script(Model()), (flag, x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cste_script(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.zeros(x.size(0)), torch.ones(
                    (x.size(1), x.size(0)), dtype=torch.int64
                )

        x = torch.randn(3, 4)
        self.run_test(MyModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]})
        self.run_test(MyModel(), x, remained_onnx_input_idx=[])

    def test_scalar_tensor(self):
        class test(torch.nn.Module):
            def forward(self, input):
                return torch.scalar_tensor(input.size(0)), torch.scalar_tensor(
                    input.size(1), dtype=torch.int64
                )

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        model = test()
        self.run_test(
            model,
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            dynamic_axes={"input_1": [0, 1, 2]},
        )

    def test_tensor(self):
        class ScalarInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor(input.shape[1])

        x = torch.randn(3, 4)
        self.run_test(
            ScalarInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(ScalarInputModel(), x, remained_onnx_input_idx=[])

        class TensorInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor([input.shape[0], input.shape[1]])

        x = torch.randn(3, 4)
        self.run_test(
            TensorInputModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(TensorInputModel(), x, remained_onnx_input_idx=[])

        class FloatInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor([float(input)])

        x = torch.randn(1)
        self.run_test(FloatInputModel(), x)

        class InputWithDtypeModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor(input.shape[1], dtype=torch.long)

        x = torch.randn(3, 4)
        self.run_test(
            InputWithDtypeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        self.run_test(InputWithDtypeModel(), x, remained_onnx_input_idx=[])

        class MixedInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor([input.shape[0], int(input)])

        x = torch.randn(1)
        self.run_test(MixedInputModel(), x)

    def test_hardtanh(self):
        model = torch.nn.Hardtanh(-1.5, 2.5)
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardtanh_script_with_default_values(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.nn.functional.hardtanh(x)

        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardswish(self):
        model = torch.nn.Hardswish()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardswish_script(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.nn.functional.hardswish(x)

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardsigmoid(self):
        model = torch.nn.Hardsigmoid()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # corner cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_tanhshrink(self):
        model = torch.nn.Tanhshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink(self):
        model = torch.nn.Hardshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink_dtype(self):
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Hardshrink(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink(self):
        model = torch.nn.Softshrink()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink_dtype(self):
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Softshrink(), x)

    def test_clamp(self):
        class ClampModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampModel(), x)

        class ClampMinModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(min=-0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMinModel(), x)

        class ClampMaxModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(max=0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMaxModel(), x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_clamp_dyn(self):
        class ClampMaxModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.clamp(None, x.size(0))

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), x)

        class ClampMinModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.clamp(x.size(0), None)

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), x)

        class ClampMinMaxModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.clamp(x.size(0), x.size(1))

        x = torch.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), x)

        class ClampTensorModel(torch.nn.Module):
            def forward(self, x, min, max):
                return x.clamp(min, max)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        z = torch.randn(3, 4)
        self.run_test(ClampTensorModel(), (x, y, z))

        class ClampTensorMinModel(torch.nn.Module):
            def forward(self, x, min):
                return x.clamp(min=min)

        self.run_test(ClampTensorMinModel(), (x, y))

        class ClampTensorMaxModel(torch.nn.Module):
            def forward(self, x, max):
                return x.clamp(max=max)

        self.run_test(ClampTensorMaxModel(), (x, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_trace(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_script(self):
        class FullModelScripting(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_fuse_addmm(self):
        class AddmmModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddmmModel(), x)

    def test_maxpool(self):
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_conv(self):
        class TraceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_str_padding(self):
        class TraceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, padding="valid")
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=1, padding="valid", dilation=(3, 1)
                )
                self.conv3 = torch.nn.Conv3d(
                    16, 33, (3, 5, 2), stride=1, padding="same"
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_shape_inference(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2 = torch.nn.Conv2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )

            def forward(self, input):
                return self.conv2(input) + 2

        x = torch.randn(20, 16, 50, 100)
        self.run_test(
            Model(), x, atol=10e-5, input_names=["x"], dynamic_axes={"x": [0]}
        )

    def test_conv_transpose(self):
        class TraceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
                )
                self.conv3 = torch.nn.ConvTranspose3d(
                    16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0)
                )

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 10)
        x2 = torch.randn(20, 16, 10, 10)
        x3 = torch.randn(20, 16, 10, 10, 10)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_numpy_T(self):
        class NumpyTranspose(torch.nn.Module):
            def forward(self, x):
                return x.T

        self.run_test(NumpyTranspose(), torch.randn(4, 7))

    # Conversion of Transpose depends on input shape to be known.
    # The following test only works when onnx shape inference is enabled.
    def test_transpose_infer_shape(self):
        class TransposeModule(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test(
            TransposeModule(),
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 2]},
            additional_test_inputs=[y],
        )

    def squeeze_model_tests(self, d, x1, x2):
        class Squeeze(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = d

            def forward(self, x):
                if self.d is not None:
                    return torch.squeeze(x, dim=self.d)
                else:
                    return torch.squeeze(x)

        x2 = [] if x2 is None else [x2]
        if len(x2) > 0:
            self.run_test(
                Squeeze(d),
                x1,
                input_names=["input"],
                dynamic_axes={"input": {0: "0", 1: "1", 2: "2"}},
                additional_test_inputs=x2,
            )
        else:
            self.run_test(Squeeze(d), x1)

    def test_squeeze_without_no_op(self):
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(1, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_dynamic(self):
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(1, x_squeeze, x_noop)

    def test_squeeze_neg_without_no_op(self):
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(-2, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_neg(self):
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(-2, x_squeeze, x_noop)

    def test_squeeze_all_dims(self):
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(None, x_squeeze, x_noop)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_no_op(self):
        x_noop = torch.randn(2, 1, 4)
        x_squeeze = torch.randn(2, 2, 1)
        self.squeeze_model_tests(2, x_noop, x_squeeze)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_runtime_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test(Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)])
        self.run_test(Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)])

    def test_squeeze(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=-2)

        x = torch.randn(2, 1, 4)
        self.run_test(Squeeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_squeeze_dynamic_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x, dim: int):
                return torch.squeeze(x, dim)

        x = torch.randn(2, 1, 4)
        dim = 1
        self.run_test(Squeeze(), (x, dim))

    def test_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, dim=-2)

        x = torch.randn(2, 3, 4)
        self.run_test(Unsqueeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_unsqueeze_dynamic_dim(self):
        class Unsqueeze(torch.nn.Module):
            def forward(self, x, dim: int):
                return torch.unsqueeze(x, dim)

        x = torch.randn(2, 1, 4)
        dim = -1
        self.run_test(Unsqueeze(), (x, dim))

    def test_maxpool_default_stride(self):
        class MaxPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, 2)

        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_adaptive(self):
        model = torch.nn.AdaptiveMaxPool1d((5), return_indices=False)
        x = torch.randn(20, 16, 50, requires_grad=True)
        y = torch.randn(32, 16, 50, requires_grad=True)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[y],
        )

    def test_maxpool_2d(self):
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_1d_ceil(self):
        model = torch.nn.MaxPool1d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_2d_ceil(self):
        model = torch.nn.MaxPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil(self):
        model = torch.nn.MaxPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dynamic(self):
        class test(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        model = test(8, 16)
        inputs = torch.randn(2, 8, 64, 64)
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    # TODO: Enable maxpool-ceil family after ONNX 1.15.1+ is bumped
    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_1d_ceil_corner(self):
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=False
        )
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_2d_ceil_corner(self):
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_3d_ceil_corner(self):
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=False,
        )
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_1d_ceil_corner_with_indices(self):
        model = torch.nn.MaxPool1d(
            kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=True
        )
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_2d_ceil_corner_with_indices(self):
        model = torch.nn.MaxPool2d(
            kernel_size=[1, 1],
            dilation=[1, 1],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_3d_ceil_corner_with_indices(self):
        model = torch.nn.MaxPool3d(
            kernel_size=[7, 8, 4],
            dilation=[1, 1, 1],
            stride=[10, 11, 3],
            padding=[2, 2, 2],
            ceil_mode=True,
            return_indices=True,
        )
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_default_stride(self):
        class AvgPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(x, 2)

        model = AvgPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_1d_ceil(self):
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7)
        self.run_test(model, x)

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and PyTorch has different calculation for ceil_mode (the last value).
    @common_utils.parametrize(
        "padding",
        (0, 1),
    )
    @common_utils.parametrize(
        "count_include_pad",
        (True, False),
    )
    def test_avgpool_2d(self, padding, count_include_pad):
        model = torch.nn.AvgPool2d(
            3,
            3,
            padding=padding,
            count_include_pad=count_include_pad,
        )
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    # TODO: ceil_mode is not included in the test, because of
    # https://github.com/microsoft/onnxruntime/issues/16203
    # The ORT and PyTorch has different calculation for ceil_mode (the last value).
    # the issue requires fix in onnx(21) (https://github.com/onnx/onnx/issues/5711)
    # a fix in ORT is planned. After the fixes in place, we can add ceil_mode to the test.
    @skipIfUnsupportedMinOpsetVersion(21)
    def test_avgpool_3d_ceil(self):
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        y = torch.randn(32, 8, 50, 44, 31)
        self.run_test(
            model,
            x,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_avgpool_dynamic(self):
        class test(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.AvgPool2d(
                    (2, 2), stride=2, ceil_mode=True, count_include_pad=False
                )
                self.conv = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                )
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                return self.norm(self.conv(self.avgpool(x)))

        model = test(8, 16)
        inputs = torch.randn(2, 8, 64, 64)
        self.run_test(
            model,
            inputs,
            input_names=["input_0"],
            dynamic_axes={"input_0": {3: "x", 2: "y"}, "output_0": {3: "x", 2: "y"}},
            output_names=["output_0"],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floating_point(self):
        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

    # Operator rank mismatch between outputs of two branches for opsets below 11.
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_floating_point_infer_dtype(self):
        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloatingPoint(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x

        x = torch.randn(2, 3, 4).to(torch.int32)
        self.run_test(FloatingPoint(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_prim_min(self):
        @torch.jit.script
        def list_append(boxes: list[Tensor]):
            temp = []
            for i, b in enumerate(
                boxes
            ):  # enumerate is creating a prim::min op in torch graph
                temp.append(torch.full_like(b[:, 1], i))
            return temp[0]

        class Min(torch.nn.Module):
            def forward(self, x):
                boxes = [x for _ in range(3)]
                return list_append(boxes)

        x = torch.rand(5, 5)
        self.run_test(Min(), (x,))

        class M(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                i = 3
                return min(x[i], i)

        x = torch.arange(6, dtype=torch.int64)
        self.run_test(M(), (x,))

    def test_arithmetic(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_arithmetic_prim_long(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[0]

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_float(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: float):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2.5
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[1] / 2

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_bool(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int, z: bool, t: float):
                x = x + y
                x = x - y
                if z:
                    x = x * (y * 3)
                    x = x / (y * 4)
                return x / t, z

        x = torch.randn(2, 3, 4)
        y = 2
        z = False
        t = 2.5
        self.run_test(ArithmeticModule(), (x, y, z, t))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x: int, y: int):
                return x == y

        x = 3
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

    @skipScriptTest(
        15,
        reason="In trace: Outputs that are always None are removed. \
                In script: Outputs that are always None are removed before opset 15. \
                After opset 15, we replace the None in output with Optional node.",
    )
    def test_tuple_with_none_outputs(self):
        class TupleModel(torch.nn.Module):
            def forward(self, x):
                return (x, (x, None, (x, None)))

        x = torch.randn(3, 4)
        self.run_test(TupleModel(), (x,))

    # In scripting the first transpose node do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_arithmetic_infer_dtype(self):
        class ArithmeticModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3)
        self.run_test(ArithmeticModule(), x)

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    def test_floor_div(self):
        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x // 3,
                    x // 2.0,
                    x.to(dtype=torch.float64) // 3,
                    x.to(dtype=torch.float64) // 2.0,
                    x.to(dtype=torch.int64) // 3,
                    x.to(dtype=torch.int64) // 2.0,
                    x // (y + 1.0).to(dtype=torch.int64),
                    x // y,
                    x.to(dtype=torch.float64) // y.to(dtype=torch.int64),
                    x.to(dtype=torch.float64) // y.to(dtype=torch.float64),
                    x.to(dtype=torch.int64) // y.to(dtype=torch.int64),
                    x.to(dtype=torch.int64) // y,
                )

        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    def test_floor_div_script(self):
        class FloorDivModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                return x // 3, x // 2.0, x // y

        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip("Floor division on ONNX is inconsistent with eager (see #78411)")
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floordiv(self):
        class FloordivModule(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.size(2) // x.size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(
            FloordivModule(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )
        self.run_test(FloordivModule(), (x,), remained_onnx_input_idx=[])

    def test_div(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    # Note: div cannot (generally) be exported via scripting
    # since its type promotion logic is dependent on knowing the scalar types
    # of the input tensors. That is, the ONNX graph is dependent on the
    # data type of the inputs. This makes it appropriate for tracing only.
    def test_div_promotion_trace(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

    # In scripting x, y do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    def test_div_promotion_script(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                # Add transpose to hide shape/type information
                # Otherwise shape and type are still available from input.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        # 1. x,y are int, and output is float.
        #    This can be handled by the default case, where both are cast to float.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.script(DivModule()), (x, y))

        # 2. x,y are int, and output is double.
        #    This can be handled by the default case, where both are cast to double.
        #    It works even if type of x, y are unknown.
        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.script(DivModule()), (x, y))

        # 3. x is int, y is double, and output is double.
        #    This can only be handled when both type of x and y are known.
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test(torch.jit.script(DivModule()), (x, y))

    @skipDtypeChecking
    def test_div_rounding_mode(self):
        class TrueDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode=None),
                    torch.div(x, y, rounding_mode=None),
                )

        class TruncDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode="trunc"),
                    torch.div(x, y, rounding_mode="trunc"),
                )

        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (
                    x.div(y, rounding_mode="floor"),
                    torch.div(x, y, rounding_mode="floor"),
                )

        modules = [TrueDivModule(), TruncDivModule(), FloorDivModule()]

        x = (torch.randn(2, 3, 4) * 100).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

        x = torch.randn(2, 3, 4)
        y = torch.rand(2, 3, 4) * 10.0 + 0.1

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

    def test_slice_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_neg(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[-1:]

        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, -3:-1, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large_negone(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_index(self):
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y):
                x[: y.size(0), 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Torchscript doesn't support 1d index.
    def test_slice_with_1d_input_index(self):
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y):
                x[:y, 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.tensor([5], dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_step_size(self):
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y, z):
                x[:y:z, 0::z, :] = 1
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.tensor(5, dtype=torch.int64)
        z = torch.tensor(2, dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1"],
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_script(self):
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1 : x.size(1)]

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_shape_script(self):
        class DynamicSliceModel(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1 : x.size(2)])

        x = torch.rand(1, 2, 3, 4)
        self.run_test(
            DynamicSliceModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3]}
        )
        self.run_test(DynamicSliceModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()  # scripting tuple/list append
    def test_slice_dynamic_to_end(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_test(
            DynamicSliceExportMod(),
            x,
            dynamic_axes={"input_1": [0, 1, 2], "output_1": [0, 1, 2]},
        )

    def test_square(self):
        class Square(torch.nn.Module):
            def forward(self, x):
                return torch.square(x)

        x = torch.randn(2, 3, 4)
        self.run_test(Square(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_dynamic(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test(
            ArangeModel(),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )
        self.run_test(
            torch.jit.script(ArangeModel()),
            x,
            additional_test_inputs=[y],
            input_names=["input_1"],
            output_names=["output_1", "output_2", "output_3"],
            dynamic_axes={"input_1": [0], "output_1": [0]},
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_out(self):
        class ArangeOutModel(torch.nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_start_out(self):
        class ArangeStartOutModel(torch.nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8)
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace(self):
        class LinspaceModel(torch.nn.Module):
            def forward(self, start, end, steps):
                return torch.linspace(start, end, steps)

        x = torch.tensor(3, dtype=torch.float)
        y = torch.tensor(10, dtype=torch.float)
        z = torch.tensor(5, dtype=torch.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace_negative_start(self):
        class LinspaceModel(torch.nn.Module):
            def forward(self, start, end, steps):
                return torch.linspace(start, end, steps)

        x = torch.tensor(-1, dtype=torch.float)
        y = torch.tensor(1, dtype=torch.float)
        z = torch.tensor(6, dtype=torch.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_out(self):
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, 1.5, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats(self):
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStepNeg(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(end, start.size(0), -1.5)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStepNeg(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStepNeg(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStart(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStart(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStart(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_override(self):
        class ArangeModelEnd(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end, dtype=torch.int64)

        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), (y))

        class ArangeModelStep(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeModelStep(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_out(self):
        class ArangeOutModel(torch.nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_out(self):
        class ArangeStartOutModel(torch.nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(
            ArangeStartOutModel(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_no_type(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end), torch.arange(0, end)

        x = torch.tensor(6.2, dtype=torch.float)
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_size(self):
        class SizeModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.arange(input.size(0)),
                    torch.arange(input.size(-1)),
                    torch.ones(input.shape),
                )

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2]})
        self.run_test(SizeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()  # x.stride() not scriptable
    def test_as_strided(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return x.as_strided(
                    (3, 3, 3), (1, 4, 2), storage_offset=2
                ), x.as_strided(chunk_size, chunk_stride)

        x = torch.randn(5, 8, 7)
        self.run_test(Model(), x)

    @skipScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_tensor_index_advanced_indexing_ellipsis(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[..., torch.tensor([2, 1]), torch.tensor([0, 3])]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :,
                    torch.tensor([[0, 2], [1, 1]]),
                    :,
                    torch.tensor([2, 1]),
                    torch.tensor([0, 3]),
                ]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])
                ]

        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :,
                    torch.tensor([0, 2]),
                    torch.tensor([1]),
                    2:4,
                    torch.tensor([[1], [4]]),
                ]

        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[
                    :, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None
                ]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, ind, update):
                x[ind] = update
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor([1], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_singular(self):
        class IndexPutBoolModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = True
                return mask

        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutBoolModel(), (mask, indices))

        class IndexPutFloatModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = torch.tensor(5.5)
                return mask

        mask = torch.rand(100, dtype=torch.float)
        indices = (torch.rand(50) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutFloatModel(), (mask, indices))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_accumulate(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, ind, update):
                return x.index_put((ind,), update, accumulate=True)

        x = torch.randn(3, 4)
        ind = torch.tensor([2], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_slice_index(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                x[1:2, 1:3, torch.tensor([1])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(1, 2, 1)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), torch.tensor([1, 2])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.randn(2, 5)
        self.run_test(IndexPutModel2(), (x, update))

        class IndexPutModel3(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), 1:2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1, 1)
        self.run_test(IndexPutModel3(), (x, update))

        class IndexPutModel4(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel4(), (x, update))

        class IndexPutModel5(torch.nn.Module):
            def forward(self, x, update):
                x[1:3, torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel5(), (x, update))

        class IndexPutModel6(torch.nn.Module):
            def forward(self, x, update):
                x[1:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel6(), (x, update))

        class IndexPutModel7(torch.nn.Module):
            def forward(self, x, update):
                x[1:, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel7(), (x, update))

        class IndexPutModel8(torch.nn.Module):
            def forward(self, x, update):
                x[:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(3 * 5).to(torch.float).view(3, 5)
        self.run_test(IndexPutModel8(), (x, update))

        class IndexPutModel9(torch.nn.Module):
            def forward(self, poses):
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = torch.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes

        x = torch.zeros([2, 17, 3], dtype=torch.int64)
        self.run_test(IndexPutModel9(), (x,))

        class IndexPutModel10(torch.nn.Module):
            def forward(self, x, ind, update):
                x[ind, 1:3] = update.view(1, 1, 1, 5).expand(2, 2, 2, 5)
                return x

        x = torch.randn(3, 4, 5)
        ind = torch.tensor([[0, 2], [1, 1]])
        update = torch.randn(5)
        self.run_test(IndexPutModel10(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_index_put_ellipsis(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                x[..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                x[2, ..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    @unittest.skip(
        "regression in 1.18: https://github.com/microsoft/onnxruntime/issues/20855"
    )
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_loop(self):
        @torch.jit.script
        def ngram_attention_bias(
            sequence_length: int, ngram: int, device: torch.device, dtype: torch.dtype
        ):
            bias = torch.ones(
                (ngram, sequence_length), device=device, dtype=dtype
            ) * float("-inf")
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2
                    bias[stream_idx, i] = 5
                    bias = bias * 5
                    bias[0, 0] = 5

            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5
                    bias[0, i] = 5
            return bias

        class ScriptModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ngram = 2
                self.max_target_positions = 512

            def forward(self, hidden_states):
                seq_length, batch_size = hidden_states.shape[:2]
                predict_causal_mask = ngram_attention_bias(
                    self.max_target_positions,
                    self.ngram,
                    hidden_states.device,
                    hidden_states.dtype,
                )
                predict_causal_mask = predict_causal_mask[:, :seq_length]
                return predict_causal_mask

        x = torch.randn(6, 2)
        y = torch.randn(4, 1)
        self.run_test(
            ScriptModel(),
            x,
            input_names=["x"],
            dynamic_axes={"x": {0: "seq_length", 1: "batch_size"}},
            additional_test_inputs=[y],
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                x[1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(2, 4)
        self.run_test(CopyModel(), (x, update))

        # mixed slice and select
        class CopyModel2(torch.nn.Module):
            def forward(self, x, data):
                x[1:3, 0] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel2(), (x, update))

        class CopyModel3(torch.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel3(), (x, update))

        class CopyModel4(torch.nn.Module):
            def forward(self, x, ind, data):
                x[ind] = data
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor(2)
        data = torch.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

        class CopyModel5(torch.nn.Module):
            def forward(self, x, mask):
                if mask is not None:
                    x.copy_(mask)
                    return x

        x = torch.randn(3, 4)
        mask = torch.randn(3, 1)
        self.run_test(CopyModel5(), (x, mask))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()  # Model not scriptable (output with shape doesn't match the broadcast shape)
    def test_copy_tracing(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(1, 2)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, update):
                x[..., 1] = update
                return x

        x = torch.randn(2, 3, 4)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

        x = torch.randn(2, 3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis_script(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, update):
                # Insert reshape node to ensure no shape/type info for
                # x in scripting, without onnx shape inference.
                x = x.reshape(4, 3, 5, 6)
                x[2, ..., 1:3] = update
                return x

        x = torch.randn(3, 4, 5, 6)

        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint(self):
        class RandInt(torch.nn.Module):
            def forward(self, x):
                randint = torch.randint(1, 10, x.shape)
                x = 0 * randint + x
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_value(self):
        class RandInt(torch.nn.Module):
            def forward(self, x):
                # This randint call always returns 3
                return torch.randint(3, 4, x.shape) + x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_like(self):
        class RandInt(torch.nn.Module):
            def forward(self, x):
                # This randint call always returns 3
                return torch.randint_like(x, 3, 4) + x

        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randn(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.randn(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand(self):
        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_dtype(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.randn(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand_dtype(self):
        class Rand(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.rand(2, 3, 4, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randn_dynamic_size(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_rand_dynamic_size(self):
        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_like(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)
        self.run_test(torch.jit.script(RandNLike()), x)

    def test_rand_like(self):
        class RandLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)
        self.run_test(torch.jit.script(RandLike()), x)

    def test_randn_like_dtype(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.randn_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

    def test_rand_like_dtype(self):
        class RandLike(torch.nn.Module):
            def forward(self, x):
                # The resulting node's dtype should be double.
                return (
                    x.to(torch.float32)
                    * torch.rand_like(x, dtype=torch.double)
                    * torch.tensor(0, dtype=torch.float32)
                )

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_bernoulli(self):
        class Bernoulli(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.bernoulli(x).size(0))

        x = torch.empty(3, 3).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

        x = torch.empty(2, 3, 3, dtype=torch.double).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

    def test_bernoulli_p(self):
        class Bernoulli_float(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.bernoulli(x, 0.2).size(0))

        class Bernoulli_tensor(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).bernoulli_(x).size(0))

        x = torch.rand(3, 3)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

        x = torch.rand(2, 3, 3, dtype=torc

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 812 class(es): TestONNXRuntime, Fuse, Fuse, Fuse, Fuse, ConvTBC, Reshape, MyModel, MyModel, MyModel, MyModel, MyModel, MyModel, Data, MyModel, MyModel, MyModel, DictModelOutput, MyModel, MyModel

### Functions
This file defines 1722 function(s): _init_test_generalized_rcnn_transform, _init_test_rpn, _construct_tensor_for_quantization_test, _parameterized_class_attrs_and_values, _parametrize_rnn_args, test_fuse_conv_bn1d, __init__, forward, test_fuse_conv_bn2d, __init__, forward, test_fuse_conv_bn3d, __init__, forward, test_fuse_conv_in_block, __init__, forward, test_conv_tbc, __init__, reset_parameters, conv_tbc, forward, test_reshape_constant_fold, __init__, forward, run_word_language_model, get_image, get_test_images, test_paste_mask_in_image, test_heatmaps_to_keypoints


## Key Components

The file contains 36695 words across 13932 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 492969 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

# Documentation: `test/quantization/pt2e/test_quantize_pt2e.py`

## File Metadata

- **Path**: `test/quantization/pt2e/test_quantize_pt2e.py`
- **Size**: 116,833 bytes (114.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

import torch
from torch import Tensor
from torch.ao.quantization import observer, ObserverOrFakeQuantize, QConfigMapping
from torch.ao.quantization.qconfig import (
    default_per_channel_symmetric_qnnpack_qconfig,
    float_qparams_weight_only_qconfig,
    per_channel_weight_observer_range_neg_127_to_127,
    QConfig,
    weight_observer_range_neg_127_to_127,
)
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.ao.quantization.quantizer.composable_quantizer import (  # noqa: F811
    ComposableQuantizer,
)
from torch.ao.quantization.quantizer.embedding_quantizer import (  # noqa: F811
    EmbeddingQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
)
from torch.export import export
from torch.fx import Node
from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    PT2EQuantizationTestCase,
    skipIfNoQNNPACK,
    TestHelperModules,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfHpu,
    TemporaryFileName,
    TEST_CUDA,
    TEST_HPU,
)


@skipIfNoQNNPACK
class TestQuantizePT2E(PT2EQuantizationTestCase):
    def test_simple_quantizer(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
        }
        node_list = [
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        ]
        self._test_quantizer(
            TestHelperModules.ConvWithBNRelu(relu=False, bn=False),
            example_inputs,
            BackendAQuantizer(),
            node_occurrence,
            node_list,
        )

    def test_wo_annotate_conv_output_quantizer(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = QuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                weight_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                bias_qspec = QuantizationSpec(
                    dtype=torch.float32,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = torch.nn.Conv2d(2, 2, 1)
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        # Ensure the conv has no observer inserted at output
        node_occurrence = {
            # two for input of conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 1,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_max_pool2d_quantizer(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = QuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_observer,
                )
                weight_qspec = QuantizationSpec(
                    dtype=torch.int8,
                    quant_min=-128,
                    quant_max=127,
                    qscheme=torch.per_tensor_affine,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.default_weight_observer,
                )
                bias_qspec = QuantizationSpec(
                    dtype=torch.float32,
                    is_dynamic=False,
                    observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            _annotated=True,
                        )
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.max_pool2d.default
                    ):
                        maxpool_node = node
                        input_act = maxpool_node.args[0]
                        assert isinstance(input_act, Node)
                        maxpool_node.meta["quantization_annotation"] = (
                            QuantizationAnnotation(
                                input_qspec_map={
                                    input_act: act_qspec,
                                },
                                output_qspec=SharedQuantizationSpec(
                                    (input_act, maxpool_node)
                                ),
                                _annotated=True,
                            )
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ConvMaxPool2d()
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {
            # two for input of conv
            # one for input of maxpool
            # one for output of maxpool
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 3,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 4,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.max_pool2d.default),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_derived_qspec(self):
        # TODO: use OP_TO_ANNOTATOR
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )

                        def derive_qparams_fn(
                            obs_or_fqs: list[ObserverOrFakeQuantize],
                        ) -> tuple[Tensor, Tensor]:
                            assert len(obs_or_fqs) == 2, (
                                f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
                            )
                            act_obs_or_fq = obs_or_fqs[0]
                            weight_obs_or_fq = obs_or_fqs[1]
                            act_scale, act_zp = act_obs_or_fq.calculate_qparams()
                            (
                                weight_scale,
                                weight_zp,
                            ) = weight_obs_or_fq.calculate_qparams()
                            return torch.tensor([act_scale * weight_scale]).to(
                                torch.float32
                            ), torch.tensor([0]).to(torch.int32)

                        bias_qspec = DerivedQuantizationSpec(
                            derived_from=[(input_act, node), (weight, node)],
                            derive_qparams_fn=derive_qparams_fn,
                            dtype=torch.int32,
                            quant_min=-(2**31),
                            quant_max=2**31 - 1,
                            qscheme=torch.per_tensor_symmetric,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {
            # input, weight, bias, output for the conv
            # note: quantize op for weight and bias are const propagated
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 4,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_derived_qspec_per_channel(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_channel_affine,
                            is_dynamic=False,
                            ch_axis=0,
                            observer_or_fake_quant_ctr=observer.default_per_channel_weight_observer,
                        )

                        def derive_qparams_fn(
                            obs_or_fqs: list[ObserverOrFakeQuantize],
                        ) -> tuple[Tensor, Tensor]:
                            assert len(obs_or_fqs) == 1, (
                                f"Expecting one weight obs/fq, got: {len(obs_or_fqs)}"
                            )
                            weight_obs_or_fq = obs_or_fqs[0]
                            (
                                weight_scale,
                                weight_zp,
                            ) = weight_obs_or_fq.calculate_qparams()
                            return weight_scale, torch.zeros_like(weight_scale)

                        bias_qspec = DerivedQuantizationSpec(
                            derived_from=[(weight, node)],
                            derive_qparams_fn=derive_qparams_fn,
                            dtype=torch.int32,
                            quant_min=-(2**31),
                            quant_max=2**31 - 1,
                            qscheme=torch.per_channel_symmetric,
                            ch_axis=0,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        m = self._quantize(m, BackendAQuantizer(), example_inputs)

        node_occurrence = {
            # input, output for the conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
            # weight and bias for conv
            # note: quantize op for weight and bias are const propagated
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_channel.default
            ): 0,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ): 2,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            ),
            ns.call_function(torch.ops.aten.conv2d.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_fixed_qparams_qspec_ptq(self):
        self._test_fixed_qparams_qspec(is_qat=False)

    # TODO: refactor and move this to test_quantize_pt2_qat.py
    def test_fixed_qparams_qspec_qat(self):
        self._test_fixed_qparams_qspec(is_qat=True)

    def _test_fixed_qparams_qspec(self, is_qat: bool):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.sigmoid.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        act_qspec = FixedQParamsQuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            scale=1.0 / 256.0,
                            zero_point=0,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        m = self._quantize(m, BackendAQuantizer(), example_inputs, is_qat)
        fixed_scale = 1.0 / 256.0
        fixed_zero_point = 0
        for n in m.graph.nodes:
            if n.op == "call_function":
                if (
                    n.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    scale_0 = n.args[1]
                    zero_point_0 = n.args[2]
                if (
                    n.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ):
                    scale_1 = n.args[1]
                    zero_point_1 = n.args[2]
        self.assertEqual(scale_0, fixed_scale)
        self.assertEqual(zero_point_0, fixed_zero_point)
        self.assertEqual(scale_1, fixed_scale)
        self.assertEqual(zero_point_1, fixed_zero_point)
        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 2,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 2,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.sigmoid.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_fixed_qparams_qspec_observer_dedup(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                act_qspec = FixedQParamsQuantizationSpec(
                    dtype=torch.uint8,
                    quant_min=0,
                    quant_max=255,
                    qscheme=torch.per_tensor_affine,
                    scale=1.0 / 256.0,
                    zero_point=0,
                )
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.sigmoid.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.add.Tensor
                    ):
                        input_act0 = node.args[0]
                        assert isinstance(input_act, Node)
                        input_act1 = node.args[1]
                        assert isinstance(input_act, Node)
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act0: act_qspec,
                                input_act1: act_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.sigmoid(x) + y

            def example_inputs(self):
                return (
                    torch.randn(1, 3, 5, 5),
                    torch.randn(1, 3, 5, 5),
                )

        m = M().eval()
        example_inputs = m.example_inputs()
        m = self._quantize(m, BackendAQuantizer(), example_inputs, is_qat=False)

        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 4,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 4,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.sigmoid.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.add.Tensor),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_shared_qspec(self):
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[first_input_node] = act_qspec
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
                            (first_input_node, cat_node)
                        )
                        for input_node in input_nodes[1:]:
                            input_qspec_map[input_node] = (
                                share_qparams_with_input_act0_qspec
                            )

                        cat_node.meta["quantization_annotation"] = (
                            QuantizationAnnotation(
                                input_qspec_map=input_qspec_map,
                                output_qspec=share_qparams_with_input_act0_qspec,
                                _annotated=True,
                            )
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.Conv2dWithCat().eval()
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))

        # program capture
        m = export(m, example_inputs, strict=True).module()
        m = prepare_pt2e(m, BackendAQuantizer())
        # make sure the two observers for input are shared
        conv_output_obs = []
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.conv2d.default:
                conv_output_obs.append(getattr(m, next(iter(n.users)).target))
            if n.op == "call_function" and n.target == torch.ops.aten.cat.default:
                inputs = n.args[0]
                input0 = inputs[0]
                input1 = inputs[1]
                assert input0.op == "call_module"
                assert input1.op == "call_module"
                obs_ins0 = getattr(m, input0.target)
                obs_ins1 = getattr(m, input1.target)
                assert obs_ins0 == obs_ins1
        assert len(conv_output_obs) == 2, (
            "expecting two observer that follows conv2d ops"
        )
        # checking that the output observers for the two convs are shared as well
        assert conv_output_obs[0] == conv_output_obs[1]

        m(*example_inputs)
        m = convert_pt2e(m)

        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 5,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 7,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.cat.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def _test_transitive_sharing_with_cat_helper(self, quantizer):
        m = TestHelperModules.Conv2dWithTwoCat().eval()
        example_inputs = (
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 3, 5, 5),
            torch.randn(1, 6, 3, 3),
            torch.randn(1, 6, 3, 3),
        )

        # program capture
        m = export(m, example_inputs, strict=True).module()
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        # make sure the two input observers and output are shared
        conv_output_obs = []
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.conv2d.default:
                conv_output_obs.append(getattr(m, next(iter(n.users)).target))
            if n.op == "call_function" and n.target == torch.ops.aten.cat.default:
                inputs = n.args[0]
                input0 = inputs[0]
                input1 = inputs[1]
                assert input0.op == "call_module"
                assert input1.op == "call_module"
                obs_ins0 = getattr(m, input0.target)
                obs_ins1 = getattr(m, input1.target)
                assert obs_ins0 == obs_ins1

                output_obs = next(iter(n.users))
                assert output_obs.op == "call_module"
                obs_ins2 = getattr(m, output_obs.target)
                assert obs_ins0 == obs_ins2, "input observer does not match output"

        assert len(conv_output_obs) == 2, (
            "expecting two observer that follows conv2d ops"
        )
        # checking that the output observers for the two convs are shared as well
        assert conv_output_obs[0] == conv_output_obs[1]

        m(*example_inputs)
        m = convert_pt2e(m)

        node_occurrence = {
            # two for input of the first conv, one for output for the first conv
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ): 7,
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ): 9,
        }
        node_list = [
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.cat.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
            ns.call_function(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ),
            ns.call_function(torch.ops.aten.cat.default),
            ns.call_function(
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ),
        ]
        self.checkGraphModuleNodes(
            m, expected_node_list=node_list, expected_node_occurrence=node_occurrence
        )

    def test_shared_qspec_transitivity(self):
        """This tests the transitivity of SharedQuantizationSpec, that is
        if A is shared with B, B is shared with C, then C should be shared with A as well

        x1 -> conv1 -> cat1 -----> cat2
        x2 -> conv2 -/            /
                       x3 -> add /
                       x4  /

        both cat has shared input and output, and because of cat and (cat1 -> cat2) is the same Tensor
        so there is an implicit sharing here, all tensors connect to cat1 and cat2 are in the same
        sharing group after transitive sharing
        """

        # TODO: refactor this to a common util
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[first_input_node] = act_qspec
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
                            (first_input_node, cat_node)
                        )
                        for input_node in input_nodes[1:]:
                            input_qspec_map[input_node] = (
                                share_qparams_with_input_act0_qspec
                            )

                        cat_node.meta["quantization_annotation"] = (
                            QuantizationAnnotation(
                                input_qspec_map=input_qspec_map,
                                output_qspec=share_qparams_with_input_act0_qspec,
                                _annotated=True,
                            )
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        self._test_transitive_sharing_with_cat_helper(BackendAQuantizer())

    def test_shared_qspec_transitivity_case_2(self):
        """This tests the transitivity of SharedQuantizationSpec, that is
        if A is shared with B, B is shared with C, then C should be shared with A as well

        x1 -> conv1 -> cat1 -----> cat2
        x2 -> conv2 -/            /
                       x3 -> add /
                       x4  /

        both cat has shared input and output, and because of cat and (cat1 -> cat2) is the same Tensor
        so there is an implicit sharing here, all tensors connect to cat1 and cat2 are in the same
        sharing group after transitive sharing

        the difference is that for this one, all edges and nodes are shared with the second input edge of cat
        instead of the first input edge of cat as in previous example
        """

        # TODO: refactor this to a common util
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.conv2d.default
                    ):
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        weight_qspec = QuantizationSpec(
                            dtype=torch.int8,
                            quant_min=-128,
                            quant_max=127,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_weight_observer,
                        )
                        bias_qspec = QuantizationSpec(
                            dtype=torch.float32,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.PlaceholderObserver,
                        )
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={
                                input_act: act_qspec,
                                weight: weight_qspec,
                                bias: bias_qspec,
                            },
                            output_qspec=act_qspec,
                            _annotated=True,
                        )
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        second_input_node = input_nodes[1]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[second_input_node] = act_qspec
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec(
                            (second_input_node, cat_node)
                        )
                        input_qspec_map[first_input_node] = (
                            share_qparams_with_input_act1_qspec
                        )

                        cat_node.meta["quantization_annotation"] = (
                            QuantizationAnnotation(
                                input_qspec_map=input_qspec_map,
                                output_qspec=share_qparams_with_input_act1_qspec,
                                _annotated=True,
                            )
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        self._test_transitive_sharing_with_cat_helper(BackendAQuantizer())

    def test_allow_implicit_sharing(self):
        """This tests the allow_transitive_sharing flag of QuantizationAnnotation, that is
        if a node is configured with allow_implicit_sharing=False, we will not have implicit sharing
        for node and (node, consumer) even they refer to the same Tensor

        x1 -> add1 -----> add3
        x2 -/              /
               x3 -> add2 /
               x4 -/

        all add has shared input and output, and second input is using shared quantization spec pointing
        to first input, but we set allow_implicit_sharing to False for all add nodes so input and output of add1,
        add2 and add3 will each belong to one sharing group, so we'll have:

        x1 -> obs1 -> add1 -> obs1 -> obs3--> add3 -> obs3
        x2 -> obs1 -/                         /
               x3 -> obs2 -> add2 -> obs2 -> obs3
               x4 -> obs2 -/
        """

        # TODO: refactor this to a common util
        class BackendAQuantizer(Quantizer):
            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                for node in model.graph.nodes:
                    if node.target is torch.ops.aten.add.Tensor:
                        add_node = node
                        first_input_node = add_node.args[0]
                        second_input_node = add_node.args[1]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(
                            dtype=torch.uint8,
                            quant_min=0,
                            quant_max=255,
                            qscheme=torch.per_tensor_affine,
                            is_dynamic=False,
                            observer_or_fake_quant_ctr=observer.default_observer,
                        )
                        input_qspec_map[second_input_node] = act_qspec
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec(
                            (second_input_node, add_node)
                        )
                        input_qspec_map[first_input_node] = (
                            share_qparams_with_input_act1_qspec
                        )

                        add_node.meta["quantization_annotation"] = (
                            QuantizationAnnotation(
                                input_qspec_map=input_qspec_map,
                                output_qspec=share_qparams_with_input_act1_qspec,
                                allow_implicit_sharing=False,
                                _annotated=True,
                            )
                        )

            def validate(self, model: torch.fx.GraphModule) -> None:
                pass

        m = TestHelperModules.ThreeAdd().eval()
        example_inputs =
```



## High-Level Overview


This Python file contains 47 class(es) and 149 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestQuantizePT2E`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `M`, `BackendAQuantizer`, `BackendAQuantizer`, `M`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `BackendAQuantizer`, `DtypeActQuantizer`, `M`, `M`, `BackendAQuantizer`, `M`, `M`

**Functions defined**: `test_simple_quantizer`, `annotate`, `validate`, `test_wo_annotate_conv_output_quantizer`, `annotate`, `validate`, `test_max_pool2d_quantizer`, `annotate`, `validate`, `test_derived_qspec`, `annotate`, `derive_qparams_fn`, `validate`, `test_derived_qspec_per_channel`, `annotate`, `derive_qparams_fn`, `validate`, `test_fixed_qparams_qspec_ptq`, `test_fixed_qparams_qspec_qat`, `_test_fixed_qparams_qspec`

**Key imports**: torch, Tensor, observer, ObserverOrFakeQuantize, QConfigMapping, export, Node, custom_op, ObserverBase, time, MappingType, PerGroup, PerToken, operator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/pt2e`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.quantization`: observer, ObserverOrFakeQuantize, QConfigMapping
- `torch.export`: export
- `torch.fx`: Node
- `torch.library`: custom_op
- `torch.ao.quantization.observer`: ObserverBase
- `time`
- `operator`


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/quantization/pt2e/test_quantize_pt2e.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/pt2e`):

- [`test_graph_utils.py_docs.md`](./test_graph_utils.py_docs.md)
- [`test_numeric_debugger.py_docs.md`](./test_numeric_debugger.py_docs.md)
- [`test_quantize_pt2e_qat.py_docs.md`](./test_quantize_pt2e_qat.py_docs.md)
- [`test_representation.py_docs.md`](./test_representation.py_docs.md)
- [`test_xnnpack_quantizer.py_docs.md`](./test_xnnpack_quantizer.py_docs.md)
- [`test_metadata_porting.py_docs.md`](./test_metadata_porting.py_docs.md)
- [`test_x86inductor_quantizer.py_docs.md`](./test_x86inductor_quantizer.py_docs.md)
- [`test_duplicate_dq.py_docs.md`](./test_duplicate_dq.py_docs.md)


## Cross-References

- **File Documentation**: `test_quantize_pt2e.py_docs.md`
- **Keyword Index**: `test_quantize_pt2e.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

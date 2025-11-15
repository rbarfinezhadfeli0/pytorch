# Documentation: `docs/test/quantization/fx/test_quantize_fx.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/fx/test_quantize_fx.py_docs.md`
- **Size**: 54,509 bytes (53.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/fx/test_quantize_fx.py`

## File Metadata

- **Path**: `test/quantization/fx/test_quantize_fx.py`
- **Size**: 404,004 bytes (394.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

from collections import OrderedDict
import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.reference as nnqr
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.multiprocessing as mp
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY

# graph mode quantization based on fx
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
    convert_to_reference_fx,
    _convert_to_reference_decomposed_fx,
    prepare_qat_fx,
    fuse_fx,
)


from torch.ao.quantization.fx.quantize_handler import DefaultNodeQuantizeHandler

from torch.ao.quantization.fx.match_utils import (
    _is_match,
    MatchAllNode,
)

from torch.ao.quantization import (
    QuantType,
)
from torch.ao.quantization.quant_type import _get_quant_type_to_str

from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    QuantWrapper,
    default_qconfig,
    default_dynamic_qconfig,
    default_per_channel_qconfig,
    default_qat_qconfig,
    default_reuse_input_qconfig,
    default_symmetric_qnnpack_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    per_channel_dynamic_qconfig,
    float16_dynamic_qconfig,
    float16_static_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    get_default_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    fuse_modules,
    fuse_modules_qat,
    prepare,
    prepare_qat,
    convert,
    quantize_dynamic,
    default_placeholder_observer,
    default_weight_observer,
    PerChannelMinMaxObserver,
    FixedQParamsFakeQuantize,
    FixedQParamsObserver,
    FusedMovingAvgObsFakeQuantize,
    FakeQuantize,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    ReuseInputObserver,
    QConfig,
    default_embedding_qat_qconfig,
)

from torch.ao.quantization.backend_config import (
    get_fbgemm_backend_config,
    get_qnnpack_backend_config,
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType
)
from torch.ao.quantization.backend_config.native import (
    get_test_only_legacy_native_backend_config,
)

from torch.ao.quantization.qconfig_mapping import (
    _get_symmetric_qnnpack_qconfig_mapping,
    _get_symmetric_qnnpack_qat_qconfig_mapping,
    _GLOBAL_DICT_KEY,
    _MODULE_NAME_DICT_KEY,
    _MODULE_NAME_OBJECT_TYPE_ORDER_DICT_KEY,
    _MODULE_NAME_REGEX_DICT_KEY,
    _OBJECT_TYPE_DICT_KEY,
    QConfigMapping,
)

from torch.ao.quantization.fx.qconfig_mapping_utils import (
    _get_object_type_qconfig,
    _get_module_name_qconfig,
    _get_module_name_regex_qconfig,
    _maybe_adjust_qconfig_for_module_name_object_type_order,
)

from torch.ao.quantization.fx.pattern_utils import (
    _DEFAULT_FUSION_PATTERNS,
    _DEFAULT_QUANTIZATION_PATTERNS,
    _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP,
    _DEFAULT_OUTPUT_OBSERVER_MAP,
    _register_fusion_pattern,
    _register_quant_pattern,
    get_default_output_activation_post_process_map
)

from torch.ao.quantization.fx.custom_config import (
    STANDALONE_MODULE_NAME_DICT_KEY,
    STANDALONE_MODULE_CLASS_DICT_KEY,
    FLOAT_TO_OBSERVED_DICT_KEY,
    OBSERVED_TO_QUANTIZED_DICT_KEY,
    NON_TRACEABLE_MODULE_NAME_DICT_KEY,
    NON_TRACEABLE_MODULE_CLASS_DICT_KEY,
    INPUT_QUANTIZED_INDEXES_DICT_KEY,
    OUTPUT_QUANTIZED_INDEXES_DICT_KEY,
    PRESERVED_ATTRIBUTES_DICT_KEY,
    FuseCustomConfig,
    ConvertCustomConfig,
    PrepareCustomConfig,
    StandaloneModuleConfigEntry,
)
import torch.ao.quantization.fx.lstm_utils

from torch.ao.quantization.fx.utils import (
    _reroute_tuple_getitem_pattern,
    NodeInfo,
)

from torch.ao.quantization.fake_quantize import (
    default_fixed_qparams_range_0to1_fake_quant,
    default_fixed_qparams_range_neg1to1_fake_quant,
)

from torch.ao.quantization.observer import (
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    MinMaxObserver,
    _is_activation_post_process,
)

# test utils
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_quantization import (
    LinearReluLinearModel,
    LinearReluModel,
    LinearBnLeakyReluModel,
    LinearTanhModel,
    ConvBnAddReluModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    skip_if_no_torchvision,
    train_one_epoch,
    run_ddp,
    test_only_eval_fn,
    test_only_train_fn,
    ModelForConvTransposeBNFusion,
    get_supported_device_types,
    skipIfNoONEDNN,
)

from torch.testing._internal.common_quantization import (
    LinearModelWithSubmodule,
    ResNetBase,
    RNNDynamicModel,
    RNNCellDynamicModel,
)

from torch.testing._internal.common_quantized import (
    supported_qengines,
    override_qengines,
    override_quantized_engine,
)

from torch.testing._internal.common_utils import (
    TemporaryFileName,
    IS_ARM64,
    skipIfTorchDynamo,
)

from torch.testing._internal.common_quantization import NodeSpec as ns

from torch.testing import FileCheck

import copy
import itertools
import operator
import unittest
import io
from typing import Optional
from collections.abc import Callable

class BinaryOp(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        self.is_scalar = is_scalar
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op

    def forward(self, x, y):
        x = self.conv1(x)
        y = 3 if self.is_scalar else self.conv2(y)
        # x = x + y
        x = self.op(x, y)
        # x = y + x
        x = self.op(y, x)
        return x

class BinaryOpNonQuantizedInput(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.is_scalar = is_scalar
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op

    def forward(self, x, y):
        y = 3 if self.is_scalar else y
        x = self.op(x, y)
        return x

class BinaryOpRelu(torch.nn.Module):
    def __init__(self, binary_op, ibinary_op, is_inplace, relu_callable,
                 is_scalar):
        """ ibinary_op means inplace binary op
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1).float()
        self.conv2 = torch.nn.Conv2d(1, 1, 1).float()
        self.op = ibinary_op if ibinary_op and is_inplace else binary_op
        self.relu_callable = relu_callable
        self.is_scalar = is_scalar
        if relu_callable is torch.nn.ReLU:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = relu_callable

    def forward(self, x, y):
        x = self.conv1(x)
        y = 3 if self.is_scalar else self.conv2(y)
        x = self.op(x, y)
        x = self.relu(x)
        x = self.op(y, x)
        x = self.relu(x)
        return x

@torch.fx.wrap
def _user_func_with_complex_return_type(x):
    return list(torch.split(x, 1, 1))

class TestFuseFx(QuantizationTestCase):
    def test_fuse_conv_bn_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.conv2d = nn.Conv2d(1, 1, 1)
                self.conv3d = nn.Conv3d(1, 1, 1)
                self.bn1d = nn.BatchNorm1d(1)
                self.bn2d = nn.BatchNorm2d(1)
                self.bn3d = nn.BatchNorm3d(1)
                self.conv1d2 = nn.Conv1d(1, 1, 1)
                self.conv2d2 = nn.Conv2d(1, 1, 1)
                self.conv3d2 = nn.Conv3d(1, 1, 1)
                self.bn1d2 = nn.BatchNorm1d(1)
                self.bn2d2 = nn.BatchNorm2d(1)
                self.bn3d2 = nn.BatchNorm3d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1d(x)
                x = self.bn1d(x)
                x = self.conv2d(x)
                x = self.bn2d(x)
                x = self.conv3d(x)
                x = self.bn3d(x)
                x = self.conv1d2(x)
                x = self.bn1d2(x)
                x = self.relu(x)
                x = self.conv2d2(x)
                x = self.bn2d2(x)
                x = self.relu(x)
                x = self.conv3d2(x)
                x = self.bn3d2(x)
                x = self.relu(x)
                return x

        # test train mode
        m = M().train()
        # currently we don't check if the module are configured with qconfig before fusion
        # TODO: if we decide to do that in the future, this test needs to
        # be updated
        # train mode fuse_fx is called in prepare_qat_fx
        m = prepare_qat_fx(m, {}, example_inputs=(torch.randn(1, 1, 1, 1),))
        expected_nodes = [
            ns.call_module(nni.ConvBn1d),
            ns.call_module(nni.ConvBn2d),
            ns.call_module(nni.ConvBn3d),
            ns.call_module(nni.ConvBnReLU1d),
            ns.call_module(nni.ConvBnReLU2d),
            ns.call_module(nni.ConvBnReLU3d),
        ]
        expected_occurrence = {
            ns.call_module(nn.ReLU): 0
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

        # test eval mode
        m = M().eval()
        # fuse_fx is a top level api and only supports eval mode
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nn.Conv1d),
            ns.call_module(nn.Conv2d),
            ns.call_module(nn.Conv3d),
            ns.call_module(nni.ConvReLU1d),
            ns.call_module(nni.ConvReLU2d),
            ns.call_module(nni.ConvReLU3d),
        ]
        # ConvBnRelu1d is not fused
        expected_occurrence = {
            ns.call_module(nn.ReLU): 0
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_fuse_linear_bn_eval(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.bn1d = nn.BatchNorm1d(1)

            def forward(self, x):
                x = self.linear(x)
                x = self.bn1d(x)
                return x

        # test eval mode
        m = M().eval()
        # fuse_fx is a top level api and only supports eval mode
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nn.Linear),
        ]
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    @skipIfNoONEDNN
    def test_fuse_linear_bn_leaky_relu_onednn(self):
        # linear - bn - leaky_relu is fused for onednn backend only
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        expected_nodes = [
            ns.call_module(nni.LinearLeakyReLU),
        ]
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
            ns.call_module(nn.LeakyReLU): 0,
        }

        for with_bn in [True, False]:
            # test eval mode
            m = LinearBnLeakyReluModel(with_bn).eval()
            # fuse_fx is a top level api and only supports eval mode
            m = fuse_fx(m,
                        backend_config=get_onednn_backend_config())
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    def test_linear_bn_leaky_relu_not_fused_by_default(self):
        # Make sure linear - bn - leaky_relu is not fused by default
        for with_bn in [True, False]:
            # test eval mode
            m = LinearBnLeakyReluModel(with_bn).eval()
            # fuse_fx is a top level api and only supports eval mode
            m = fuse_fx(m)
            expected_nodes = [
                ns.call_module(nn.Linear),
                ns.call_module(nn.LeakyReLU),
            ]
            expected_occurrence = {
                ns.call_module(nni.LinearLeakyReLU): 0,
            }
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    @skipIfNoONEDNN
    def test_fuse_linear_tanh_for_onednn_backend(self):
        # linear - tanh is fused for onednn backend only
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        expected_nodes = [
            ns.call_module(nni.LinearTanh),
        ]
        expected_occurrence = {
            ns.call_module(nn.Linear): 0,
            ns.call_module(nn.Tanh): 0,
        }

        # test eval mode
        m = LinearTanhModel().eval()
        # fuse_fx is a top level api and only supports eval mode
        m = fuse_fx(m,
                    backend_config=get_onednn_backend_config())
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_linear_tanh_not_fused_by_default(self):
        # Make sure linear - tanh is not fused by default
        # test eval mode
        m = LinearTanhModel().eval()
        # fuse_fx is a top level api and only supports eval mode
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nn.Linear),
            ns.call_module(nn.Tanh),
        ]
        expected_occurrence = {
            ns.call_module(nni.LinearTanh): 0,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)

    def test_fuse_conv_bn_add_relu_onednn(self):
        # conv - bn - add - relu is fused for onednn backend only
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        options = itertools.product(
            [True, False],  # with_bn
            [True, False],  # with_relu
            [True, False],  # conv in the left
            [True, False],  # with_two_conv
            [True, False],  # use_torch_add
        )
        for with_bn, with_relu, left_conv, two_conv, use_torch_add in options:
            expected_nodes = [
                ns.call_module(nni.ConvAddReLU2d if with_relu else nni.ConvAdd2d),
            ]
            expected_occurrence = {
                ns.call_module(nni.ConvAddReLU2d if with_relu else nni.ConvAdd2d): 1,
                ns.call_module(nn.BatchNorm2d): 0,
            }

            # test eval mode
            m = ConvBnAddReluModel(
                with_bn=with_bn,
                with_relu=with_relu,
                left_conv=left_conv,
                two_conv=two_conv,
                use_torch_add=use_torch_add).eval()

            m = fuse_fx(m,
                        backend_config=get_onednn_backend_config())
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    def test_fuse_conv_bn_add_relu_by_default(self):
        options = itertools.product(
            [True, False],  # with_bn
            [True, False],  # with_relu
            [True, False],  # conv in the left
            [True, False],  # with_two_conv
            [True, False],  # use_torch_add
        )
        for with_bn, with_relu, left_conv, two_conv, use_torch_add in options:
            # test eval mode
            expected_nodes = [
                ns.call_module(nn.Conv2d),
            ]
            expected_occurrence = {
                ns.call_module(nni.ConvAdd2d): 0,
            }
            m = ConvBnAddReluModel(
                with_bn=with_bn,
                with_relu=with_relu,
                left_conv=left_conv,
                two_conv=two_conv,
                use_torch_add=use_torch_add).eval()
            m = fuse_fx(m)
            self.checkGraphModuleNodes(
                m,
                expected_node_list=expected_nodes,
                expected_node_occurrence=expected_occurrence)

    @skipIfNoONEDNN
    def test_fuse_conv_bn_add_relu_lowering(self):
        """ Test fusion and lowering of Conv2d - (bn -) ReLU
            by FX. For onednn backedn only.
        """
        from torch.ao.quantization.backend_config import get_onednn_backend_config
        qconfig_mapping = get_default_qconfig_mapping('onednn')
        with override_quantized_engine('onednn'):
            options = itertools.product(
                [True, False],  # with_bn
                [True, False],  # with_relu
                [True, False],  # conv in the left
                [True, False],  # two_conv
                [True, False],  # use_torch_add
            )
            for with_bn, with_relu, left_conv, two_conv, use_torch_add in options:
                node_occurrence = {
                    ns.call_function(torch.quantize_per_tensor): 1 if two_conv else 2,
                    ns.call_method("dequantize"): 1,
                    ns.call_module(nniq.ConvAddReLU2d if with_relu else nniq.ConvAdd2d): 1,
                    ns.call_module(nn.Conv2d): 0,
                    ns.call_module(nn.ReLU): 0,
                }
                node_occurrence_ref = {
                    ns.call_function(torch.quantize_per_tensor): 3,
                    ns.call_method("dequantize"): 3,
                }

                # test eval mode
                m = ConvBnAddReluModel(
                    with_bn=with_bn,
                    with_relu=with_relu,
                    left_conv=left_conv,
                    two_conv=two_conv,
                    use_torch_add=use_torch_add).eval()
                example_x = m.get_example_inputs()
                m = prepare_fx(m, qconfig_mapping,
                               example_inputs=example_x,
                               backend_config=get_onednn_backend_config())
                m_copy = copy.deepcopy(m)
                m = convert_fx(m, backend_config=get_onednn_backend_config())
                m_ref = convert_to_reference_fx(m_copy)
                self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)
                self.checkGraphModuleNodes(m_ref, expected_node_occurrence=node_occurrence_ref)
                m(*example_x)

    def test_fuse_convtranspose_bn_eval(self):

        m = ModelForConvTransposeBNFusion().eval()
        m = fuse_fx(m)

        expected_nodes = [
            ns.call_module(nn.ConvTranspose1d),
            ns.call_module(nn.ConvTranspose2d),
            ns.call_module(nn.ConvTranspose3d),
        ]
        expected_occurrence = {
            ns.call_module(nn.BatchNorm1d): 0,
            ns.call_module(nn.BatchNorm2d): 0,
            ns.call_module(nn.BatchNorm3d): 0,
        }
        self.checkGraphModuleNodes(
            m,
            expected_node_list=expected_nodes,
            expected_node_occurrence=expected_occurrence)


    def test_fuse_module_relu(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1d = nn.Conv1d(1, 1, 1)
                self.conv2d = nn.Conv2d(1, 1, 1)
                self.conv3d = nn.Conv3d(1, 1, 1)
                self.bn1d = nn.BatchNorm1d(1)
                self.bn2d = nn.BatchNorm2d(1)
                self.bn3d = nn.BatchNorm3d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1d(x)
                x = self.relu(x)
                x = self.conv2d(x)
                x = self.relu(x)
                x = self.conv3d(x)
                x = self.relu(x)
                x = self.bn1d(x)
                x = self.relu(x)
                x = self.bn2d(x)
                x = self.relu(x)
                x = self.bn3d(x)
                x = self.relu(x)
                return x

        m = M().eval()
        m = fuse_fx(m)
        expected_nodes = [
            ns.call_module(nni.ConvReLU1d),
            ns.call_module(nni.ConvReLU2d),
            ns.call_module(nni.ConvReLU3d),
            ns.call_module(nni.BNReLU2d),
            ns.call_module(nni.BNReLU3d),
        ]
        self.checkGraphModuleNodes(m, expected_node_list=expected_nodes)

    @skipIfNoFBGEMM
    def test_qconfig_fused_module(self):
        """ TODO: add test for all fused modules
        """
        qconfig_dict = {
            "": None,
            "object_type": [(nn.Linear, default_qconfig),
                            (nn.ReLU, default_qconfig),
                            (F.relu, default_qconfig)]
        }

        linearRelu_node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_method('dequantize')
        ]

        linearReluLinear_node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        tests = [(LinearReluModel, linearRelu_node_list),
                 (LinearReluLinearModel, linearReluLinear_node_list)]

        for M, node_list in tests:
            m = M().eval()
            example_inputs = (torch.rand(5, 5),)
            prepared = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)

            prepared(*example_inputs)
            quantized = convert_fx(prepared)

            self.checkGraphModuleNodes(quantized, expected_node_list=node_list)

    def test_problematic_fuse_example(self):
        class LinearRelu(nn.Sequential):
            def __init__(self) -> None:
                super().__init__(
                    nn.Linear(5, 5),
                    nn.ReLU(),
                )

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin_relu = LinearRelu()
                self.linear = nn.Linear(5, 5)

            def forward(self, x):
                x = self.lin_relu(x)
                x = self.linear(x)
                return x

        model = M().eval()
        # these qconfigs somehow fail equality where default_qconfig does not
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, get_default_qconfig('fbgemm')),
                (torch.nn.ReLU, get_default_qconfig('fbgemm')),
            ],
        }
        m = prepare_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 5),))

        self.checkGraphModuleNodes(m, expected_node=ns.call_module(torch.ao.nn.intrinsic.modules.fused.LinearReLU))

    @unittest.skip("Temporarily skipping the test case, will enable after the simple"
                   "pattern format is supported")
    def test_fuse_addtional_fuser_method(self):
        class MyConvReLU(torch.nn.Module):
            pass

        def my_conv_relu_fuser(conv, relu):
            return MyConvReLU()

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        m = M().eval()
        m = fuse_fx(m, fuse_custom_config={
            "additional_fuser_method_mapping": {
                (torch.nn.Conv2d, torch.nn.ReLU): my_conv_relu_fuser
            }
        })
        self.checkGraphModuleNodes(m, expected_node=ns.call_module(MyConvReLU))

    def test_fuse_custom_pattern(self):
        class M(torch.nn.Module):
            def __init__(self, use_torch_add=True):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3)
                if use_torch_add:
                    self.add = torch.add
                else:
                    self.add = operator.add

            def forward(self, x):
                y = x
                y = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.add(y, x)
                x = self.relu(x)
                return x

        for use_torch_add in [True, False]:
            m = M(use_torch_add).eval()

            def fuse_conv_bn_relu(is_qat, relu, add_pattern):
                _, _, bn_pattern = add_pattern
                bn, conv = bn_pattern
                return conv

            conv_bn_res_relu_config1 = BackendPatternConfig() \
                ._set_pattern_complex_format((nn.ReLU, (torch.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))) \
                .set_fuser_method(fuse_conv_bn_relu)
            conv_bn_res_relu_config2 = BackendPatternConfig() \
                ._set_pattern_complex_format((nn.ReLU, (operator.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))) \
                .set_fuser_method(fuse_conv_bn_relu)
            backend_config = BackendConfig() \
                .set_backend_pattern_config(conv_bn_res_relu_config1) \
                .set_backend_pattern_config(conv_bn_res_relu_config2)
            m = fuse_fx(m, backend_config=backend_config)
            self.assertEqual(type(m.conv), torch.nn.Conv2d)
            # check bn and relu are gone since we replaced the whole pattern to conv
            self.assertFalse(hasattr(m, "bn"))
            self.assertFalse(hasattr(m, "relu"))

    def test_fusion_pattern_with_multiple_inputs(self):
        """ This test tests two keys in backend_config: root_node_getter and
        extra_inputs_getter,
        root_node_getter is used to identify a "root" module in the node pattern,
        the node that we'll keep after fusion.
        extra_inputs_getter will return a list of node that needs to be added to the
        fused node as extra inputs.
        """
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3)

            def forward(self, x):
                y = x
                y = self.maxpool(x)
                x = self.conv(x)
                x = self.bn(x)
                x = torch.add(x, y)
                x = self.relu(x)
                return x

        m = M().eval()

        def fuse_conv_bn_relu(is_qat, relu, add_pattern):
            _, bn_pattern, _ = add_pattern
            bn, conv = bn_pattern
            return conv

        def conv_bn_res_relu_root_node_getter(pattern):
            relu, add_pattern = pattern
            _, bn_pattern, _ = add_pattern
            bn, conv = bn_pattern
            return conv

        def conv_bn_res_relu_extra_inputs_getter(pattern):
            """ get inputs pattern for extra inputs, inputs for root node
            are assumed to be copied over from root node to the fused node
            """
            relu, add_pattern = pattern
            _, bn_pattern, extra_input = add_pattern
            bn, conv = bn_pattern
            return [extra_input]

        conv_bn_res_relu_config = BackendPatternConfig() \
            ._set_pattern_complex_format((nn.ReLU, (torch.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))) \
            .set_fuser_method(fuse_conv_bn_relu) \
            ._set_root_node_getter(conv_bn_res_relu_root_node_getter) \
            ._set_extra_inputs_getter(conv_bn_res_relu_extra_inputs_getter)
        backend_config = BackendConfig().set_backend_pattern_config(conv_bn_res_relu_config)
        m = fuse_fx(m, backend_config=backend_config)
        self.assertEqual(type(m.conv), torch.nn.Conv2d)
        # check bn and relu are gone since we replaced the whole pattern to conv
        self.assertFalse(hasattr(m, "bn"))
        self.assertFalse(hasattr(m, "relu"))

        # check conv module has two inputs
        named_modules = dict(m.named_modules())
        for node in m.graph.nodes:
            if node.op == "call_module" and type(named_modules[node.target]) is torch.nn.Conv2d:
                self.assertTrue(len(node.args) == 2, msg="Expecting the fused op to have two arguments")

    def test_fusion_pattern_with_matchallnode(self):
        """This test tests that the node matched by MatchAllNode will be regared as an input
        instead of a module to be fused. For instance, we have two patterns:
            (nn.ReLU, (torch.add, MatchAllNode, nn.Conv2d))
            (nn.ReLU, nn.Conv2d)
        And we wanna fuse the following model
            Conv2d -> ReLU +
            Conv2d ------ Add -> ReLU
        ReLU in the first row is matched as MatchAllNode in the residual pattern. But it won't be
        fused as part of that pattnern. It needs to be properly fused with the upstream Conv2d.
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                y = self.conv1(x)
                y = self.relu1(y)

                x = self.conv2(x)
                x = torch.add(x, y)
                x = self.relu2(x)
                return x

        m = M().eval()

        def fuse_conv_relu(is_qat, conv, relu):
            return conv

        def fuse_conv_res_relu(is_qat, relu, add_pattern):
            _, conv, _ = add_pattern
            return conv

        def conv_res_relu_root_node_getter(pattern):
            relu, (_, conv, _) = pattern
            return conv

        def conv_res_relu_extra_inputs_getter(pattern):
            relu, (_, _, extra_input) = pattern
            return [extra_input]

        conv_relu_config = BackendPatternConfig((nn.Conv2d, nn.ReLU)) \
            .set_fuser_method(fuse_conv_relu)
        conv_res_relu_config = BackendPatternConfig() \
            ._set_pattern_complex_format((nn.ReLU, (torch.add, nn.Conv2d, MatchAllNode))) \
            .set_fuser_method(fuse_conv_res_relu) \
            ._set_root_node_getter(conv_res_relu_root_node_getter) \
            ._set_extra_inputs_getter(conv_res_relu_extra_inputs_getter)
        backend_config = BackendConfig() \
            .set_backend_pattern_config(conv_relu_config) \
            .set_backend_pattern_config(conv_res_relu_config)
        m = fuse_fx(m, backend_config=backend_config)
        self.assertEqual(type(m.conv1), torch.nn.Conv2d)
        self.assertEqual(type(m.conv2), torch.nn.Conv2d)
        # check relu are gone since we replaced both patterns to conv
        self.assertFalse(hasattr(m, "relu1"))
        self.assertFalse(hasattr(m, "relu2"))


@skipIfNoFBGEMM
class TestQuantizeFx(QuantizationTestCase):
    def test_pattern_match(self):
        """ test MatchAllNode with
            conv - bn - add - relu pattern
        """
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x, y):
                x = self.conv(x)
                x = self.bn(x)
                x = x + y
                x = self.relu(x)
                return x

        pattern = (nn.ReLU, (operator.add, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))
        m = torch.fx.symbolic_trace(M())
        modules = dict(m.named_modules())
        for n in m.graph.nodes:
            if n.op == 'call_module' and type(modules[n.target]) is nn.ReLU:
                self.assertTrue(_is_match(modules, n, pattern))

    def test_pattern_match_constant(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x, _ = torch.ops.aten.max_pool2d_with_indices.default(x)
                return x

        pattern = (operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)
        m = torch.fx.symbolic_trace(M())
        # eliminate the code that get the second output of maxpool, so that the pattern
        # can be matched
        m.graph.eliminate_dead_code()
        modules = dict(m.named_modules())
        for n in m.graph.nodes:
            if n.op == "call_function" and n.target == operator.getitem:
                self.assertTrue(_is_match(modules, n, pattern))

    def test_fused_module_qat_swap(self):
        class Tmp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tmp = torch.nn.Linear(5, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.tmp(x)
                return self.relu(x)


        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods1 = torch.nn.Sequential(Tmp(), torch.nn.Linear(5, 5))
                self.mods2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                a = self.mods1(x)
                x = torch.add(x, 5)
                x = self.mods2(x)
                x = torch.add(x, 5)
                return a, x


        model = M().train()
        qconfig_dict = {
            "": None,
            "object_type": [
                (torch.nn.Linear, default_qat_qconfig),
                (torch.nn.ReLU, default_qat_qconfig),
            ],
        }
        prepared = prepare_qat_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 5),))
        self.assertTrue(isinstance(getattr(prepared.mods1, "0").tmp, torch.ao.nn.intrinsic.qat.LinearReLU))

    def _get_conv_linear_test_cases(self, is_reference):
        """ Returns a list of test cases, with format:
        is_dynamic, ModuleClass, module_constructor_inputs,
        inputs, quantized_node, weight_prepack_op
        """
        class FunctionalConv1d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = 1
                self.padding = 0
                self.dilation = 1
                self.groups = 1

            def forward(self, x):
                return F.conv1d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)


        class Conv1d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv1d(*args)

            def forward(self, x):
                return self.conv(x)

        conv1d_input = torch.rand(1, 3, 224)
        conv1d_weight = torch.rand(3, 3, 3)
        conv1d_module_args = (3, 3, 3)

        class FunctionalConv2d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        class Conv2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv2d(*args)

            def forward(self, x):
                return self.conv(x)

        conv2d_input = torch.rand(1, 3, 224, 224)
        conv2d_weight = torch.rand(3, 3, 3, 3)
        conv2d_module_args = (3, 3, 3)

        class FunctionalConv3d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1, 1)
                self.padding = (0, 0, 0)
                self.dilation = (1, 1, 1)
                self.groups = 1

            def forward(self, x):
                return F.conv3d(
                    x,
                    self.weight,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

        class Conv3d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.conv = torch.nn.Conv3d(*args)

            def forward(self, x):
                return self.conv(x)

        conv3d_input = torch.rand(1, 3, 32, 224, 224)
        conv3d_weight = torch.rand(3, 3, 3, 3, 3)
        conv3d_module_args = (3, 3, 3)

        class Linear(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                return F.linear(x, self.weight)

        linear_input = torch.rand(8, 5)
        linear_weight = torch.rand(10, 5)

        class LinearModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                return self.linear(x)

        linear_module_input = torch.rand(8, 5)

        # is_dynamic, ModuleClass, module_constructor_inputs,
        # inputs, quantized_node, weight_prepack_node
        tests = [
            (
                False,
                FunctionalConv1d,
                (conv1d_weight,),
                (conv1d_input,),
                ns.call_function(torch.nn.functional.conv1d if is_reference else torch.ops.quantized.conv1d) ,
                ns.call_function(torch.ops.quantized.conv1d_prepack),
            ),
            (
                False,
                FunctionalConv2d,
                (conv2d_weight,),
                (conv2d_input,),
                ns.call_function(torch.nn.functional.conv2d if is_reference else torch.ops.quantized.conv2d),
                ns.call_function(torch.ops.quantized.conv2d_prepack),
            ),
            (
                False,
                FunctionalConv3d,
                (conv3d_weight,),
                (conv3d_input,),
                ns.call_function(torch.nn.functional.conv3d if is_reference else torch.ops.quantized.conv3d),
                ns.call_function(torch.ops.quantized.conv3d_prepack),
            ),
            (
                False,
                Conv1d,
                conv1d_module_args,
                (conv1d_input,),
                ns.call_module(nnqr.Conv1d if is_reference else nnq.Conv1d),
                None
            ),
            (
                False,
                Conv2d,
                conv2d_module_args,
                (conv2d_input,),
                ns.call_module(nnqr.Conv2d if is_reference else nnq.Conv2d),
                None
            ),
            (
                False,
                Conv3d,
                conv3d_module_args,
                (conv3d_input,),
                ns.call_module(nnqr.Conv3d if is_reference else nnq.Conv3d),
                None
            ),
            (
                True,
                Linear,
                (linear_weight,),
                (linear_input,),
                None if is_reference else ns.call_function(torch.ops.quantized.linear_dynamic),
                ns.call_function(torch.ops.quantized.linear_prepack),
            ),
            (
                False,
                Linear,
                (linear_weight,),
                (linear_input,),
                ns.call_function(torch.nn.functional.linear if is_reference else torch.ops.quantized.linear),
                ns.call_function(torch.ops.quantized.linear_prepack),
            ),
            (
                True,
                LinearModule,
                (),
                (linear_module_input,),
                ns.call_module(nnqr.Linear) if is_reference else ns.call_module(nnqd.Linear),
                None,
            ),
            (
                False,
                LinearModule,
                (),
                (linear_module_input,),
                ns.call_module(nnqr.Linear if is_reference else nnq.Linear),
                None,
            ),
        ]
        return tests

    @skipIfNoFBGEMM
    def test_conv_linear_not_reference(self):
        """ Test quantizing conv and linear
        """
        tests = self._get_conv_linear_test_cases(is_reference=False)
        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = {}
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=False)

    @skipIfNoFBGEMM
    def test_conv_linear_reference(self):
        """ Test quantizing functional conv and linear with reference option
        """
        tests = self._get_conv_linear_test_cases(is_reference=True)

        def _get_keys(prefix, is_dynamic):
            all_keys = [prefix + "." + k for k in ["weight_qscheme", "weight_dtype"]]
            if not is_dynamic:
                all_keys.extend([prefix + "." + k for k in ["weight_scale", "weight_zero_point"]])
            return all_keys

        for (is_dynamic, ModuleClass, module_constructor_inputs,
             inputs, quantized_node, weight_prepack_node) in tests:
            quant_type = QuantType.DYNAMIC if is_dynamic else QuantType.STATIC
            node_occurrence = {}
            if weight_prepack_node:
                node_occurrence[weight_prepack_node] = 0
            result_dict = self.checkGraphModeFxOp(
                ModuleClass(*module_constructor_inputs),
                inputs, quant_type,
                expected_node=quantized_node,
                expected_node_occurrence=node_occurrence,
                is_reference=True)
            qr = result_dict["quantized_reference"]

            def checkWeightQParams(model):
                for module_name in ("linear", "conv"):
                    if hasattr(model, module_name):
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_qscheme"))
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_scale"))
                        self.assertTrue(hasattr(qr.get_submodule(module_name), "weight_zero_point"))
                        self.assertTrue("Reference" in qr.get_submodule(module_name)._get_name())

            def checkSerDeser(model, is_dynamic):
                for module_name in ("linear", "conv"):
                    if hasattr(model, module_name):
                        # make sure serialization works
                        state_dict = copy.deepcopy(model.state_dict())
                        all_keys = _get_keys(module_name, is_dynamic)
                        for key in all_keys:
                            self.assertTrue(key in state_dict)
                        # check load_state_dict restores states
                        module = getattr(model, module_name)
                        prev_scale = module.weight_scale
                        module.weight_scale = None
                        model.load_state_dict(state_dict)
                        module = getattr(model, module_name)
                        self.assertTrue(torch.equal(prev_scale, module.weight_scale))


            checkWeightQParams(qr)
            qr = copy.deepcopy(qr)
            # make sure the qparams are preserved after copy
            checkWeightQParams(qr)

            checkSerDeser(qr, is_dynamic)

    def _get_conv_transpose_test_cases(self, use_relu, is_reference):
        """ Returns a list of test cases, with format:
        is_dynamic, ModuleClass, module_constructor_inputs,
        inputs, quantized_node, weight_prepack_op
        """
        class FunctionalConvTranspose1d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = 1
                self.padding = 0
                self.output_padding = 0
                self.dilation = 1
                self.groups = 1

            def forward(self, x):
                y = F.conv_transpose1d(
                    x,
                    self.weight,
                    None,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.groups,
                    self.dilation
                )
                if use_relu:
                    y = F.relu(y)
                return y

        class ConvTranspose1d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose1d(*args)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.deconv(x)
                if use_relu:
                    y = self.relu(y)
                return y

        conv_transpose1d_input = torch.rand(1, 3, 224)
        conv_transpose1d_weight = torch.rand(3, 3, 3)
        conv_transpose1d_module_args = (3, 3, 3)

        class FunctionalConvTranspose2d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.output_padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x):
                y = F.conv_transpose2d(
                    x,
                    self.weight,
                    None,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.groups,
                    self.dilation
                )
                if use_relu:
                    y = F.relu(y)
                return y

        class ConvTranspose2d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(*args)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.deconv(x)
                if use_relu:
                    y = self.relu(y)
                return y

        conv_transpose2d_input = torch.rand(1, 3, 224, 224)
        conv_transpose2d_weight = torch.rand(3, 3, 3, 3)
        conv_transpose2d_module_args = (3, 3, 3)

        class FunctionalConvTranspose3d(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)
                self.stride = (1, 1, 1)
                self.padding = (0, 0, 0)
                self.output_padding = (0, 0, 0)
                self.dilation = (1, 1, 1)
                self.groups = 1

            def forward(self, x):
                y = F.conv_transpose3d(
                    x,
                    self.weight,
                    None,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.groups,
                    self.dilation
                )
                if use_relu:
                    y = F.relu(y)
                return y

        class ConvTranspose3d(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose3d(*args)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.deconv(x)
                if use_relu:
                    y = self.relu(y)
                return y

        conv_transpose3d_input = torch.rand(1, 3, 32, 224, 224)
        conv_transpose3d_weight = torch.rand(3, 3, 3, 3, 3)
        conv_transpose3d_module_args = (3, 3, 3)

        # is_dynamic, ModuleClass, module_constructor_inputs,
        # inputs, quantized_node, weight_prepack_node
        tests = [
            (
                False,
                FunctionalConvTranspose1d,
                (conv_transpose1d_weight,),
                (conv_transpose1d_input,),
                ns.call_function(
                    torch.nn.functional.conv_transpose1d if is_reference else torch
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/quantization/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/quantization/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/quantization/fx/test_quantize_fx.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/quantization/fx`):

- [`test_equalize_fx.py_kw.md_docs.md`](./test_equalize_fx.py_kw.md_docs.md)
- [`test_equalize_fx.py_docs.md_docs.md`](./test_equalize_fx.py_docs.md_docs.md)
- [`test_numeric_suite_fx.py_kw.md_docs.md`](./test_numeric_suite_fx.py_kw.md_docs.md)
- [`test_subgraph_rewriter.py_kw.md_docs.md`](./test_subgraph_rewriter.py_kw.md_docs.md)
- [`test_subgraph_rewriter.py_docs.md_docs.md`](./test_subgraph_rewriter.py_docs.md_docs.md)
- [`test_numeric_suite_fx.py_docs.md_docs.md`](./test_numeric_suite_fx.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`test_quantize_fx.py_kw.md_docs.md`](./test_quantize_fx.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_quantize_fx.py_docs.md_docs.md`
- **Keyword Index**: `test_quantize_fx.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

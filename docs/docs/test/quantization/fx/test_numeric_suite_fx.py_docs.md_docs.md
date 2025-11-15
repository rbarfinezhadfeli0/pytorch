# Documentation: `docs/test/quantization/fx/test_numeric_suite_fx.py_docs.md`

## File Metadata

- **Path**: `docs/test/quantization/fx/test_numeric_suite_fx.py_docs.md`
- **Size**: 53,974 bytes (52.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/quantization/fx/test_numeric_suite_fx.py`

## File Metadata

- **Path**: `test/quantization/fx/test_numeric_suite_fx.py`
- **Size**: 115,944 bytes (113.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

import copy
import math
import operator
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    default_dynamic_qconfig,
    QConfigMapping,
    get_default_qconfig_mapping,
)
import torch.ao.nn.quantized as nnq
toq = torch.ops.quantized
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_quantization import (
    ConvBnModel,
    ConvBnReLUModel,
    ConvModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    withQNNPACKBackend,
    SingleLayerLinearDynamicModel,
    SingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    SparseNNModel,
    skip_if_no_torchvision,
    TwoLayerLinearModel
)
from torch.testing._internal.common_utils import raise_on_run_directly, skipIfTorchDynamo
from torch.ao.quantization.quantization_mappings import (
    get_default_static_quant_module_mappings,
    get_default_dynamic_quant_module_mappings,
    get_default_float_to_quantized_operator_mappings,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.ao.quantization.fx.pattern_utils import get_default_quant_patterns
import torch.ao.quantization.fx.quantize_handler as qh
from torch.ao.ns.fx.pattern_utils import (
    get_type_a_related_to_b,
)
from torch.ao.ns.fx.graph_matcher import (
    get_matching_subgraph_pairs,
    GraphMatchingException,
)
from torch.ao.ns.fx.utils import (
    compute_sqnr,
    compute_normalized_l2_error,
    compute_cosine_similarity,
)
from torch.ao.ns.fx.mappings import (
    get_node_type_to_io_type_map,
    get_unmatchable_types_map,
    get_base_name_to_sets_of_related_ops,
    get_base_name_for_op,
    add_op_to_sets_of_related_ops,
)
from torch.ao.ns.fx.weight_utils import (
    get_op_to_type_to_weight_extraction_fn,
)
from torch.ao.ns._numeric_suite_fx import (
    extract_weights,
    _extract_weights_impl,
    add_loggers,
    _add_loggers_impl,
    OutputLogger,
    add_shadow_loggers,
    _add_shadow_loggers_impl,
    extract_logger_info,
    extract_shadow_logger_info,
    extend_logger_results_with_comparison,
    prepare_n_shadows_model,
    convert_n_shadows_model,
    extract_results_n_shadows_model,
    OutputComparisonLogger,
    print_comparisons_n_shadows_model,
    loggers_set_enabled,
    loggers_set_save_activations,
    _prepare_n_shadows_add_loggers_model,
    _n_shadows_compare_weights,
)
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from torch.ao.quantization.backend_config import get_native_backend_config
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers


# Note: these models are not for use outside of this file. While it's good
# to reuse code, we also need to be able to iterate on tests
# quickly when debugging. If a test model has a large number of callsites
# across various different files, speed of debugging on individual test cases
# decreases.
class LinearReluFunctional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 4))
        self.b1 = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = F.linear(x, self.w1, self.b1)
        x = F.relu(x)
        return x


class LinearFunctional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 4))
        self.b1 = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        x = F.linear(x, self.w1, self.b1)
        return x


class LinearReluLinearFunctional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(4, 4))
        self.b = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        x = F.linear(x, self.w, self.b)
        x = F.relu(x)
        x = F.linear(x, self.w, self.b)
        return x


class AddMulFunctional(nn.Module):
    def forward(self, x, y):
        x = x + 1.0
        x = x * 1.0
        x = 1.0 + x
        x = 1.0 * x
        x = x + y
        x = x * y
        return x


class AllConvAndLinearFusionModules(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # conv1d
        self.conv1d_0 = nn.Conv1d(1, 1, 1)
        # conv1d - relu
        self.conv1d_1 = nn.Conv1d(1, 1, 1)
        self.relu_0 = nn.ReLU()
        # conv1d - bn (qat only)
        self.conv1d_2 = nn.Conv1d(1, 1, 1)
        self.bn1d_0 = nn.BatchNorm1d(1)
        # conv1d - bn - relu (qat only)
        self.conv1d_3 = nn.Conv1d(1, 1, 1)
        self.bn1d_1 = nn.BatchNorm1d(1)
        self.relu_4 = nn.ReLU()
        # conv2d
        self.conv2d_0 = nn.Conv2d(1, 1, 1)
        # conv2d - relu
        self.conv2d_1 = nn.Conv2d(1, 1, 1)
        self.relu_1 = nn.ReLU()
        # conv2d - bn (qat only)
        self.conv2d_2 = nn.Conv2d(1, 1, 1)
        self.bn2d_0 = nn.BatchNorm2d(1)
        # conv2d - bn - relu (qat only)
        self.conv2d_3 = nn.Conv2d(1, 1, 1)
        self.bn2d_1 = nn.BatchNorm2d(1)
        self.relu_5 = nn.ReLU()
        # conv3d
        self.conv3d_0 = nn.Conv3d(1, 1, 1)
        # conv3d - relu
        self.conv3d_1 = nn.Conv3d(1, 1, 1)
        self.relu_2 = nn.ReLU()
        # conv3d - bn (qat only)
        self.conv3d_2 = nn.Conv3d(1, 1, 1)
        self.bn3d_0 = nn.BatchNorm3d(1)
        # conv3d - bn - relu (qat only)
        self.conv3d_3 = nn.Conv3d(1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(1)
        self.relu_6 = nn.ReLU()
        # linear
        self.linear_0 = nn.Linear(1, 1)
        # linear - relu
        self.linear_1 = nn.Linear(1, 1)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # conv1d
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.relu_0(x)
        x = self.conv1d_2(x)
        x = self.bn1d_0(x)
        x = self.conv1d_3(x)
        x = self.bn1d_1(x)
        x = self.relu_4(x)
        # conv2d
        x = x.reshape(1, 1, 1, 1)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.bn2d_0(x)
        x = self.conv2d_3(x)
        x = self.bn2d_1(x)
        x = self.relu_5(x)
        # conv3d
        x = x.reshape(1, 1, 1, 1, 1)
        x = self.conv3d_0(x)
        x = self.conv3d_1(x)
        x = self.relu_2(x)
        x = self.conv3d_2(x)
        x = self.bn3d_0(x)
        x = self.conv3d_3(x)
        x = self.bn3d_1(x)
        x = self.relu_6(x)
        # linear
        x = x.reshape(1, 1)
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.relu_3(x)
        return x


class AllConvFunctional(torch.nn.Module):
    def __init__(self, weight1d, weight2d, weight3d, bias1d, bias2d, bias3d):
        super().__init__()
        self.weight1d = torch.nn.Parameter(weight1d)
        self.weight2d = torch.nn.Parameter(weight2d)
        self.weight3d = torch.nn.Parameter(weight3d)
        self.bias1d = torch.nn.Parameter(bias1d)
        self.bias2d = torch.nn.Parameter(bias2d)
        self.bias3d = torch.nn.Parameter(bias3d)
        self.stride1d = 1
        self.padding1d = 0
        self.dilation1d = 1
        self.stride2d = (1, 1)
        self.padding2d = (0, 0)
        self.dilation2d = (1, 1)
        self.groups = 1
        self.stride3d = (1, 1, 1)
        self.padding3d = (0, 0, 0)
        self.dilation3d = (1, 1, 1)

    def forward(self, x):
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.relu(x)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.relu(x)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.relu(x)
        return x

@torch.fx.wrap
def _wrapped_hardswish(x):
    return F.hardswish(x)

@torch.fx.wrap
def _wrapped_hardswish_fp16(x):
    x = x.dequantize()
    x = F.hardswish(x)
    x = x.to(torch.float16)
    return x

@torch.fx.wrap
def _wrapped_sigmoid(x):
    return F.sigmoid(x)

@torch.fx.wrap
def _wrapped_linear(x, w, b):
    return F.linear(x, w, b)

def get_all_quant_patterns():
    """ we are in the process to migrate the frontend of fx graph mode quant
    to use backend_config_dict, so some of the patterns are moved to backend_config_dict
    this function will include these patterns so that we can still have all the patterns
    """
    # TODO: we can remove this call, and get all patterns from backend_config_dict in
    # the future when the frontend refactor is done in fx graph mode quantization
    all_quant_patterns = get_default_quant_patterns()
    # some of the patterns are moved to (native) backend_config_dict so we need to
    # add them back here
    for pattern, quantize_handler in _get_pattern_to_quantize_handlers(get_native_backend_config()).items():
        all_quant_patterns[pattern] = quantize_handler
    return all_quant_patterns

class TestFXGraphMatcher(QuantizationTestCase):

    @skipIfNoFBGEMM
    def test_simple_mod(self):
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'

        expected_types = {
            conv_name_0: ((nn.Conv2d, torch.ao.quantization.MinMaxObserver), (nnq.Conv2d, nnq.Conv2d)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_fun(self):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(torch.empty(1, 4))
                self.b = nn.Parameter(torch.zeros(1))
                torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

            def forward(self, x):
                return F.linear(x, self.w, self.b)

        m = M().eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        expected_types = {
            linear_name_0:
                ((F.linear, torch.ao.quantization.MinMaxObserver), (toq.linear, toq.linear))
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_fusion(self):
        m = LinearReluFunctional().eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(4, 4),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        expected_types = {
            linear_name_0:
                ((F.linear, torch.ao.quantization.MinMaxObserver), (toq.linear_relu, toq.linear_relu)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_mod_multi(self):
        m = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 1, 1),
            ),
            nn.Conv2d(1, 1, 1),
        ).eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    def test_simple_tensor_ops(self):
        class M(nn.Module):
            def forward(self, x, y):
                z = x + y
                return z

        m = M().eval()
        example_inputs = (torch.randn(1), torch.randn(1))
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    def test_matching_failure_node_count(self):
        # verify that matching graphs with matching node types but
        # different counts of matchable nodes fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp1 = prepare_fx(m1, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp2 = prepare_fx(m2, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @skipIfNoFBGEMM
    def test_matching_failure_node_type(self):
        # verify that matching graphs with non-matching node types fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Linear(1, 1)).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp1 = prepare_fx(m1, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        example_inputs = (torch.randn(1, 1),)
        mp2 = prepare_fx(m2, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @skipIfNoFBGEMM
    def test_nodes_before_cat(self):
        # verify that nodes before cat get matched
        class M(nn.Module):
            def forward(self, x0):
                x1 = torch.add(x0, 1.0)
                y1 = torch.add(x0, 1.0)
                x2 = torch.cat([x1, y1])
                return x2

        m = M().eval()
        example_inputs = (torch.randn(1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        cat_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.cat) + '_0'
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'

        expected_types = {
            cat_name_0: ((torch.cat, torch.cat), (torch.cat, torch.cat)),
            add_name_0: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_dict_return_type(self):
        # verify that we can traverse up nodes which return dictionaries
        class M(nn.Module):
            def forward(self, x0):
                x1 = torch.add(x0, 1.0)
                y1 = torch.add(x0, 1.0)
                z1 = torch.add(x0, 1.0)
                a1 = {'x1': x1, 'y1': (y1,), 'z1': [{'key': (z1,)}]}
                return a1

        m = M().eval()
        example_inputs = (torch.randn(1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'
        add_name_2 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_2'

        expected_types = {
            add_name_0: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_2: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_nodes_with_equal_types_get_matched(self):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = torch.mul(x, x)
                x = torch.sigmoid(x)
                x = F.relu(x)
                return x

        m = M().eval()
        # prevent conv2 from getting quantized, so we can test
        # modules with equal types
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping().set_module_name("conv2", None)
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'
        conv_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_1'
        mul_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.mul) + '_0'
        relu_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.relu) + '_0'
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'

        # all of these should be matched
        expected_types = {
            conv_name_1:
                ((nn.Conv2d, torch.ao.quantization.HistogramObserver), (nnq.Conv2d, nnq.Conv2d)),
            conv_name_0:
                ((nn.Conv2d, torch.ao.quantization.HistogramObserver), (nn.Conv2d, nn.Conv2d)),
            mul_name_0: ((torch.mul, torch.ao.quantization.HistogramObserver), (toq.mul, toq.mul)),
            relu_name_0: ((F.relu, torch.ao.quantization.FixedQParamsObserver), (F.relu, F.relu)),
            sigmoid_name_0:
                ((torch.sigmoid, torch.ao.quantization.FixedQParamsObserver), (torch.sigmoid, torch.sigmoid)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    def test_methods(self):
        """
        Verify that graph matching works on methods
        """
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        m1 = M().eval()
        m2 = M().eval()
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1),)
        m1p = prepare_fx(m1, qconfig_mapping, example_inputs=example_inputs)
        m2p = prepare_fx(m2, qconfig_mapping, example_inputs=example_inputs)
        results = get_matching_subgraph_pairs(m1p, m2p)
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'
        expected_types = {
            sigmoid_name_0:
                (('sigmoid', torch.ao.quantization.FixedQParamsObserver), ('sigmoid', torch.ao.quantization.FixedQParamsObserver)),
        }
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1p, m2p)

    def test_op_relationship_mapping(self):
        """
        Tests that the mapping of op relationships is complete.
        """
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        type_a_related_to_b = \
            get_type_a_related_to_b(base_name_to_sets_of_related_ops)

        # 1. check static quant module mappings
        static_quant_mod_mappings = get_default_static_quant_module_mappings()
        for fp32_type, int8_type in static_quant_mod_mappings.items():
            # skip quants and dequants, for the purposes of Numerical Suite
            types_to_skip = (
                torch.ao.quantization.QuantStub,
                torch.ao.quantization.DeQuantStub,
                nnq.FloatFunctional,
                # the ConvTranspose3d swap is not implemented in FX Graph
                # mode quantization yet
                nn.ConvTranspose3d,
                # the GroupNorm swap is not implemented in FX Graph
                # mode quantization yet
                nn.GroupNorm,
                # nnq.ReLU6 is no longer swapped, because nn.ReLU6 can
                # take quantized inputs
                nn.ReLU6,
            )
            if fp32_type in types_to_skip:
                continue

            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 2. check static quant op mappings
        static_quant_fun_mappings = get_default_float_to_quantized_operator_mappings()
        for fp32_type, int8_type in static_quant_fun_mappings.items():
            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 3. check dynamic quant mappings
        dynamic_quant_mappings = get_default_dynamic_quant_module_mappings()
        for fp32_type, int8_type in dynamic_quant_mappings.items():
            # TODO(future PR): enable correct weight extraction for these
            # and remove from this list.
            types_to_skip = (
                nn.GRUCell,
                nn.GRU,
                nn.LSTMCell,
                nn.RNNCell,
            )
            if fp32_type in types_to_skip:
                continue
            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 4. go through the ops mapped to each QuantizeHandler type, and verify
        # correctness.
        def _op_in_base_sets_of_related_ops(op):
            for ops in base_name_to_sets_of_related_ops.values():
                if op in ops:
                    return True
            return False

        unmatchable_types_map = get_unmatchable_types_map()
        FUNS_UNMATCHABLE = unmatchable_types_map['funs_unmatchable']
        MODS_UNMATCHABLE = unmatchable_types_map['mods_unmatchable']
        METHS_UNMATCHABLE = unmatchable_types_map['meths_unmatchable']

        def _op_is_unmatchable(op):
            return (
                op in FUNS_UNMATCHABLE or
                op in MODS_UNMATCHABLE or
                op in METHS_UNMATCHABLE
            )

        default_quant_patterns = get_all_quant_patterns()
        for pattern, qhandler_cls in default_quant_patterns.items():
            base_op = None
            if isinstance(pattern, tuple):
                base_op = pattern[-1]
            elif isinstance(pattern, str):
                base_op = pattern
            else:
                base_op = pattern

            qhandler_cls_all_ops_quantizeable = [
                qh.CatQuantizeHandler,
                qh.ConvReluQuantizeHandler,
                qh.LinearReLUQuantizeHandler,
                qh.BatchNormQuantizeHandler,
                qh.EmbeddingQuantizeHandler,
                qh.RNNDynamicQuantizeHandler,
            ]

            qhandler_cls_quant_op_same_signature = [
                qh.FixedQParamsOpQuantizeHandler,
                qh.CopyNodeQuantizeHandler,
                qh.GeneralTensorShapeOpQuantizeHandler,
            ]

            if qhandler_cls == qh.BinaryOpQuantizeHandler:
                # these ops do not have quantized equivalents
                ops_to_skip = [
                    torch.bmm,
                    torch.div,
                    torch.sub,
                    operator.truediv,
                    operator.sub
                ]
                if base_op in ops_to_skip:
                    continue
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls == qh.RNNDynamicQuantizeHandler:
                # TODO(future PR): add support for all classes in
                # RNNDynamicQuantizeHandler
                pass
            elif qhandler_cls == qh.DefaultNodeQuantizeHandler:
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls in qhandler_cls_quant_op_same_signature:
                # these ops use the same op signature for fp32 and quantized
                # tensors
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op) or
                    _op_is_unmatchable(base_op),
                    f"{base_op} not in sets of related ops or unmatchable")
            elif qhandler_cls in qhandler_cls_all_ops_quantizeable:
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            else:
                # torch.sum does not have quantized equivalents
                if base_op in [
                        torch.sum,
                        nn.GRUCell,
                        nn.GRU,
                        nn.LSTMCell,
                        nn.RNNCell,
                ]:
                    continue
                if isinstance(base_op, tuple):
                    # skip fusion patterns
                    continue
                # didn't match explicit quantize handler class, we can check if the
                # operator is in the related op set directly
                if not (_op_in_base_sets_of_related_ops(base_op) or _op_is_unmatchable(base_op)):
                    raise AssertionError(
                        f"handling for {qhandler_cls} for op {base_op} not implemented")

    @skipIfNoFBGEMM
    def test_user_defined_function(self):
        """
        Verify that graph matching works on user defined functions
        """
        class M1(nn.Module):
            def forward(self, x):
                x = F.hardswish(x)
                return x

        class M2(nn.Module):
            def forward(self, x):
                x = _wrapped_hardswish(x)
                return x

        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m1 = prepare_fx(M1().eval(), qconfig_mapping, example_inputs=example_inputs)
        m2 = prepare_fx(M2().eval(), qconfig_mapping, example_inputs=example_inputs)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_hardswish, F.hardswish)

        results = get_matching_subgraph_pairs(
            m1, m2,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)

        hardswish_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.hardswish) + '_0'

        expected_types = {
            hardswish_name_0:
                ((F.hardswish, torch.ao.quantization.HistogramObserver), (_wrapped_hardswish, _wrapped_hardswish)),
        }
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1, m2)

    @skipIfNoFBGEMM
    def test_results_order(self):
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Linear(1, 1),
        ).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)
        self.assertTrue(len(results) == 2)
        results_iter = iter(results.items())
        _, (subgraph_a_0, subgraph_b_0) = next(results_iter)
        self.assertTrue(subgraph_a_0.start_node.name == '_0' and
                        subgraph_b_0.start_node.name == '_0')
        _, (subgraph_a_1, subgraph_b_1) = next(results_iter)
        self.assertTrue(subgraph_a_1.start_node.name == '_1' and
                        subgraph_b_1.start_node.name == '_1')


class TestFXGraphMatcherModels(QuantizationTestCase):

    @skipIfTorchDynamo("too slow")
    @skipIfNoFBGEMM
    @skip_if_no_torchvision
    def test_mobilenet_v2(self):
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).eval().float()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_fx(copy.deepcopy(m), {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # assume success if no exceptions
        results_m_mp = get_matching_subgraph_pairs(torch.fx.symbolic_trace(m), mp)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results_mp_mq = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    @skip_if_no_torchvision
    def test_mobilenet_v2_qat(self):
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).float()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_qat_fx(
            copy.deepcopy(m),
            {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')},
            example_inputs=example_inputs)
        # assume success if no exceptions
        results_m_mp = get_matching_subgraph_pairs(torch.fx.symbolic_trace(m), mp)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results_mp_mq = get_matching_subgraph_pairs(mp, mq)


class FXNumericSuiteQuantizationTestCase(QuantizationTestCase):
    def _test_extract_weights(
        self, m, example_inputs, results_len=0, qconfig_dict=None, prepare_fn=prepare_fx
    ):
        m = torch.fx.symbolic_trace(m)
        if qconfig_dict is None:
            qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        # test both the public API as well as the internal GraphModule API
        for extract_weights_fun in (extract_weights, _extract_weights_impl):
            # test both m vs mp and mp vs mq
            for m1, m2 in ((m, mp), (mp, mq)):
                results = extract_weights_fun('a', m1, 'b', m2)
                self.assertTrue(
                    len(results) == results_len,
                    f"expected len {results_len}, got len {len(results)}")
                self.assert_ns_compare_dict_valid(results)
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_sqnr, 'sqnr')
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_normalized_l2_error, 'l2_error')
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_cosine_similarity,
                    'cosine_similarity')

    def _test_match_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=0,
        should_log_inputs=False,
        qconfig_dict=None,
        skip_scripting=False,
        prepare_fn=prepare_fx,
    ):
        if qconfig_dict is None:
            qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping()
        if prepare_fn == prepare_fx:
            m.eval()
        else:
            m.train()
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=data)
        mp(*data)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        m_ns, mp_ns2 = add_loggers(
            'a', m, 'b', copy.deepcopy(mp), OutputLogger,
            should_log_inputs=should_log_inputs)
        mp_ns, mq_ns = add_loggers(
            'a', mp, 'b', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        if prepared_expected_node_occurrence:
            self.checkGraphModuleNodes(
                m_ns, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_ns2, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_ns, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mq_ns, expected_node_occurrence=prepared_expected_node_occurrence)

        if not skip_scripting:
            m_ns = torch.jit.script(m_ns)
            mp_ns = torch.jit.script(mp_ns)
            mq_ns = torch.jit.script(mq_ns)

        # calibrate
        m_ns(*data)
        mp_ns2(*data)
        mp_ns(*data)
        mq_ns(*data)

        # check activation result correctness
        results = []
        for m1, m2 in ((m_ns, mp_ns2), (mp_ns, mq_ns)):
            act_compare_dict = extract_logger_info(
                m1, m2, OutputLogger, 'b')
            self.assertTrue(
                len(act_compare_dict) == results_len,
                f"expected len {results_len}, got len {len(act_compare_dict)}")
            self.assert_ns_compare_dict_valid(act_compare_dict)
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_normalized_l2_error, 'l2_error')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_cosine_similarity,
                'cosine_similarity')
            results.append(act_compare_dict)
        return results

    def _test_match_shadow_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=None,
        should_log_inputs=False, qconfig_dict=None, skip_scripting=False,
        prepare_fn=prepare_fx, compare_fp32_vs_fp32_prepared=True,
    ):
        if qconfig_dict is None:
            qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping()
        if prepare_fn == prepare_fx:
            m.eval()
        else:
            m.train()
        print("qconfig_dict:", qconfig_dict)
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=data)
        print("prepared:", mp)
        mp(*data)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        print("quantized:", mq)

        if compare_fp32_vs_fp32_prepared:
            m_shadows_mp = add_shadow_loggers(
                'a', copy.deepcopy(m), 'b', copy.deepcopy(mp),
                OutputLogger, should_log_inputs=should_log_inputs)
        mp_shadows_mq = add_shadow_loggers(
            'a', mp, 'b', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        if prepared_expected_node_occurrence:
            if compare_fp32_vs_fp32_prepared:
                self.checkGraphModuleNodes(
                    m_shadows_mp, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_shadows_mq, expected_node_occurrence=prepared_expected_node_occurrence)

        if not skip_scripting:
            if compare_fp32_vs_fp32_prepared:
                m_shadows_mp = torch.jit.script(m_shadows_mp)
            mp_shadows_mq = torch.jit.script(mp_shadows_mq)

        # calibrate
        if compare_fp32_vs_fp32_prepared:
            m_shadows_mp(*data)
        mp_shadows_mq(*data)

        # check activation result correctness
        results = []
        models = (m_shadows_mp, mp_shadows_mq) if \
            compare_fp32_vs_fp32_prepared else (mp_shadows_mq,)
        for model in models:
            act_compare_dict = extract_shadow_logger_info(
                model, OutputLogger, 'b')
            if results_len is not None:
                self.assertTrue(
                    len(act_compare_dict) == results_len,
                    f"expected len {results_len}, got len {len(act_compare_dict)}")
            self.assert_ns_compare_dict_valid(act_compare_dict)
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_normalized_l2_error, 'l2_error')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_cosine_similarity,
                'cosine_similarity')
            results.append(act_compare_dict)
        return results


class TestFXNumericSuiteCoreAPIs(FXNumericSuiteQuantizationTestCase):

    @skipIfNoFBGEMM
    def test_extract_weights_mod_ptq(self):
        m = AllConvAndLinearFusionModules().eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(m, example_inputs, results_len=14)

    @skipIfNoFBGEMM
    def test_extract_weights_mod_qat(self):
        m = AllConvAndLinearFusionModules().train()
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(
            m, example_inputs, results_len=14, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_linear_fun_ptq(self):
        m = LinearReluLinearFunctional().eval()
        example_inputs = (torch.randn(1, 4),)
        self._test_extract_weights(m, example_inputs, results_len=2)

    @skipIfNoFBGEMM
    def test_extract_weights_linear_fun_qat(self):
        m = LinearReluLinearFunctional().train()
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 4),)
        self._test_extract_weights(
            m, example_inputs, results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_conv_fun_ptq(self):
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(m, example_inputs, results_len=6)

    @skipIfNoFBGEMM
    def test_extract_weights_conv_fun_qat(self):
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).train()
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(
            m, example_inputs, results_len=6, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_dynamic(self):
        # TODO(future PR): add Linear-ReLU, after #55393 is fixed.
        m = nn.Sequential(nn.Linear(1, 1)).eval()
        qconfig_dict = {
            'object_type': [
                (nn.Linear, default_dynamic_qconfig),
            ],
        }
        example_inputs = (torch.randn(1, 1),)
        self._test_extract_weights(m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_extract_weights_fqn(self):
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        results = extract_weights('a', mp, 'b', mq)
        fqn_a_0 = results['_0_0']['weight']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['weight']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        fqn_a_1 = results['_1']['weight']['a'][0]['fqn']
        fqn_b_1 = results['_1']['weight']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    def _test_match_activations_mod_impl(self, prepare_fn=prepare_fx):
        m = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(2, 1, 2, 2),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_fn)

    @skipIfNoFBGEMM
    def test_match_activations_mod_ptq(self):
        self._test_match_activations_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_mod_qat(self):
        self._test_match_activations_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_match_activations_fun_impl(self, prepare_fn=prepare_fx):
        m = LinearReluLinearFunctional().eval()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(4, 4),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_match_activations_fun_ptq(self):
        self._test_match_activations_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_fun_qat(self):
        self._test_match_activations_fun_impl(prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_match_activations_meth_ptq(self):
        """
        Verify that add_loggers works on methods
        """
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        m = M().eval()
        res = self._test_match_activations(
            m, (torch.randn(4, 4),),
            results_len=1)

    @skipIfNoFBGEMM
    def test_match_activations_fqn(self):
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        mp_ns, mq_ns = add_loggers('a', mp, 'b', mq, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        mp_ns(datum)
        mq_ns(datum)

        results = extract_logger_info(mp_ns, mq_ns, OutputLogger, 'b')
        fqn_a_0 = results['_0_0']['node_output']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        fqn_a_1 = results['_1']['node_output']['a'][0]['fqn']
        fqn_b_1 = results['_1']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    def _test_add_shadow_loggers_mod_impl(self, prepare_fn=prepare_fx):
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        res = self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),), results_len=2,
            prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_ptq(self):
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_qat(self):
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_add_shadow_loggers_fun_impl(self, prepare_fn=prepare_fx):
        m = LinearReluLinearFunctional()
        qconfig_dict = None
        if prepare_fn == prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),), results_len=2, prepare_fn=prepare_fn,
            qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_ptq(self):
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_qat(self):
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_meth_ptq(self):
        """
        Verify that add_loggers works on methods
        """
        class M(nn.Module):
            def forward(self, x):
                x = x.sigmoid()
                return x

        m = M().eval()
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            # For now, sigmoid is not supported for shadowing because the dtype
            # inference for it is not implemented yet. So, this is just testing
            # that shadowing models with method calls does not crash.
            results_len=0)

    @skipIfNoFBGEMM
    def test_shadow_activations_fqn(self):
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        mp_shadows_mq = add_shadow_loggers('a', mp, 'b', mq, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        mp_shadows_mq(datum)

        results = extract_shadow_logger_info(mp_shadows_mq, OutputLogger, 'b')
        fqn_a_0 = results['_0_0']['node_output']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        fqn_a_1 = results['_1']['node_output']['a'][0]['fqn']
        fqn_b_1 = results['_1']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    @skipIfNoFBGEMM
    def test_loggin
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
python docs/test/quantization/fx/test_numeric_suite_fx.py_docs.md
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
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`test_quantize_fx.py_docs.md_docs.md`](./test_quantize_fx.py_docs.md_docs.md)
- [`test_quantize_fx.py_kw.md_docs.md`](./test_quantize_fx.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_numeric_suite_fx.py_docs.md_docs.md`
- **Keyword Index**: `test_numeric_suite_fx.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

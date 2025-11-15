# Documentation: `torch/_inductor/fx_passes/quantization.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/quantization.py`
- **Size**: 143,110 bytes (139.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import copy
import functools
import itertools
import math
import operator
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.fx.node import map_arg

from .. import config
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    KeywordArg,
    ListOf,
    Match,
    stable_topological_sort,
)
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern


aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized

# Only for per tensor quant since permute may changes the channel idx
_PER_TENSOR_QUANTIZE_OPS = [
    quantized_decomposed.quantize_per_tensor.default,
    quantized_decomposed.quantize_per_tensor.tensor,
]

_VIEW_OPS = [
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
    aten.reshape.default,
]

"""
The quantization.py file primarily incorporates passes related to quantization fusion
in inductor, includes:
1. Dequant Promotion;
2. Conv/GEMM weight prepack with oneDNN Library;
3. Conv/GEMM quantization fusion with output quant node (if have);
4. Other pointwise operators' quantization fusion like: qmaxpool2d, qcat and more;

It also involves int8-mixed-fp32 and int8-mixed-bf16 quantization. The main difference
of patterns for int8-mixed-bf16, comparing with int8-mixed-fp32, is
1. There is to(dtype=torch.bfloat16) node at the inputs of activation and weight for Conv/GEMM.
2. There is to(dtype=torch.float32) node at the outputs of Conv/GEMM before inputs to next quant node.
Refer to: https://github.com/pytorch/pytorch/issues/111640 for detail design of int8-mixed-bf16
quantization.
"""


def _get_pattern_output_dtype(match: Match):
    """
    Get the pattern's output dtype from node's meta
    Assume only 1 output node in this matched pattern.
    """
    pattern_output_nodes = match.output_nodes()
    assert len(pattern_output_nodes) == 1
    output_node = pattern_output_nodes[0]
    assert isinstance(output_node, torch.fx.Node)
    output_dtype = output_node.meta["val"].dtype
    assert output_dtype in [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ]
    return output_dtype


def _may_generate_pattern_with_dtype_convert(
    pattern, dtype=Arg(), with_dtype_convert=True, users=1
):
    if with_dtype_convert:
        return CallFunction(
            prims.convert_element_type.default,
            pattern,
            dtype,
            _users=users,
        )
    else:
        return pattern


def _may_generate_pattern_with_reshape(pattern, reshape_size=Arg(), with_reshape=True):
    if with_reshape:
        return CallFunction(
            torch.ops.aten.reshape.default,
            pattern,
            reshape_size,
        )
    else:
        return pattern


def _generate_linear_t_pattern(
    _dequant_per_channel_pattern,
    dtype,
):
    assert dtype in [torch.float32, torch.bfloat16]
    t_pattern = CallFunction(
        aten.permute.default,
        _may_generate_pattern_with_dtype_convert(
            _dequant_per_channel_pattern,
            KeywordArg("autocast_wgt_dtype"),
            dtype == torch.bfloat16,
        ),
        KeywordArg("permute_axes"),
    )
    return t_pattern


def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
    # only insert to_dtype if is_bf16 is True
    computation_call = _may_generate_pattern_with_dtype_convert(
        call_fn, dtype=KeywordArg("to_float"), with_dtype_convert=is_bf16, users=users
    )
    return unary_fusion(computation_call)


def get_dequantize_per_tensor_activation_pattern(is_tensor_overload=False):
    dequantize_per_tensor_activation_pattern = CallFunction(
        quantized_decomposed.dequantize_per_tensor.tensor
        if is_tensor_overload
        else quantized_decomposed.dequantize_per_tensor.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("x_quant_min"),
        KeywordArg("x_quant_max"),
        KeywordArg("x_dq_dtype"),
    )
    return dequantize_per_tensor_activation_pattern


dequantize_per_channel_weight_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)

dequantize_per_channel_to_bf16_weight_pattern = (
    _may_generate_pattern_with_dtype_convert(
        dequantize_per_channel_weight_pattern,
        KeywordArg("autocast_wgt_dtype"),
    )
)

dequantize_per_channel_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

dequantize_per_channel_to_bf16_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_to_bf16_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)


def get_qconv_pt2e_pattern(users=1):
    return CallFunction(
        torch.ops.onednn.qconv_pointwise.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("postop_name"),
        KeywordArg("postop_args"),
        KeywordArg("postop_algorithm"),
        _users=users,
    )


def get_qconv2d_binary_pt2e_pattern(users=1):
    return CallFunction(
        torch.ops.onednn.qconv2d_pointwise.binary,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("accum"),
        KeywordArg("b"),
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("accum_scale"),
        KeywordArg("accum_zero_point"),
        KeywordArg("binary_op_name"),
        KeywordArg("alpha"),
        KeywordArg("unary_op_name"),
        KeywordArg("unary_op_args"),
        KeywordArg("unary_op_algorithm"),
        _users=users,
    )


def get_qlinear_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.default
    )
    return CallFunction(
        qlinear_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("b"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("postop_name"),
        KeywordArg("postop_args"),
        KeywordArg("postop_algorithm"),
        _users=users,
    )


def get_qlinear_binary_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.binary_tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.binary
    )
    return CallFunction(
        qlinear_op,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("packed_weight"),
        KeywordArg("w_scale"),
        KeywordArg("w_zp"),
        KeywordArg("x_2"),
        KeywordArg("b"),
        KeywordArg("output_scale"),
        KeywordArg("output_zero_point"),
        KeywordArg("output_dtype"),
        KeywordArg("x2_scale"),
        KeywordArg("x2_zp"),
        KeywordArg("binary_op_name"),
        KeywordArg("alpha"),
        KeywordArg("unary_op_name"),
        KeywordArg("unary_op_args"),
        KeywordArg("unary_op_algorithm"),
        _users=users,
    )


dequantize_accum_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    KeywordArg("accum"),
    KeywordArg("accum_scale"),
    KeywordArg("accum_zp"),
    Arg(),
    Arg(),
    KeywordArg("accum_dq_dtype"),
)


def generate_pattern_with_binary(
    binary_post_op,
    computation_call,
    extra_input_pattern,
    dtype_convert=False,
    swap_inputs=False,
):
    binary_pattern = (
        CallFunction(
            binary_post_op,
            extra_input_pattern,
            computation_call,
        )
        if swap_inputs
        else CallFunction(
            binary_post_op,
            computation_call,
            extra_input_pattern,
        )
    )
    return _may_generate_pattern_with_dtype_convert(
        binary_pattern,
        KeywordArg("convert_dtype_after_inplace_add"),
        dtype_convert,
    )


def generate_pattern_with_unary(computation_call, unary_post_op):
    if unary_post_op is not None:
        return CallFunction(
            unary_post_op,
            computation_call,
        )
    return computation_call


def generate_pattern_with_output_quant(computation_call, with_dtype_convert=False):
    quantized_op_output_pattern_pt2e = CallFunction(
        quantized_decomposed.quantize_per_tensor.default,
        _may_generate_pattern_with_dtype_convert(
            computation_call,
            Arg(),
            with_dtype_convert,
        ),
        KeywordArg("o_inv_scale"),
        KeywordArg("o_zp"),
        KeywordArg("o_qmin"),
        KeywordArg("o_qmax"),
        KeywordArg("o_dtype"),
    )
    return quantized_op_output_pattern_pt2e


def _check_node_kwarg_arg_value(check_node, kwarg_name, args_index, expected_value):
    if kwarg_name in check_node.kwargs:
        actual_value = check_node.kwargs[kwarg_name]
        return actual_value == expected_value
    else:
        assert len(check_node.args) >= (args_index + 1)
        actual_value = check_node.args[args_index]
        return actual_value == expected_value


def _is_valid_quantized_conv_optimization_pattern():
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype in [torch.float32, torch.bfloat16]:
            # Only keep matched pattern with same output_dtype
            qconv_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qconv_pointwise
            )[0]
            return _check_node_kwarg_arg_value(
                qconv_node_after_weight_prepack, "output_dtype", 13, output_dtype
            )
        return True

    return fn


def _is_valid_qconv_post_op_fusion_pattern(has_binary_post_op=False):
    return (
        _is_valid_qconv_binary_optimization_pattern()
        if has_binary_post_op
        else _is_valid_quantized_conv_optimization_pattern()
    )


def _is_valid_qconv_lowering_pattern():
    def fn(match):
        if len(match.nodes) != 1:
            return False
        return match.nodes[0].target in (
            torch.ops.onednn.qconv_pointwise.default,
            torch.ops.onednn.qconv_pointwise.tensor,
            torch.ops.onednn.qconv2d_pointwise.binary,
            torch.ops.onednn.qconv2d_pointwise.binary_tensor,
        )

    return fn


def _register_quantized_conv_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_lowering_pattern(),
        pass_number=pass_number,
    )
    def qconv(match: Match, *args, **kwargs):
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        # Conv Params
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype in [torch.int8, torch.uint8, torch.float32, torch.bfloat16]
        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]
        output_dtype = kwargs["output_dtype"]
        # post op
        postop_name = kwargs["postop_name"]
        postop_args = kwargs["postop_args"]
        postop_algorithm = kwargs["postop_algorithm"]

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            postop_name,
            postop_args,
            postop_algorithm,
        )
        counters["inductor"]["qconv_unary_lower_count"] += 1
        counters["inductor"]["qconv_unary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qconv


def _is_valid_quantized_linear_optimization_pattern():
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)
        if output_dtype in [torch.float32, torch.bfloat16]:
            # Only keep matched pattern with same output_dtype
            qlinear_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qlinear_pointwise
            )[0]
            return _check_node_kwarg_arg_value(
                qlinear_node_after_weight_prepack, "output_dtype", 9, output_dtype
            )
        return True

    return fn


def _is_valid_qlinear_post_op_fusion_pattern(has_binary_post_op=False):
    return (
        _is_valid_qlinear_binary_optimization_pattern()
        if has_binary_post_op
        else _is_valid_quantized_linear_optimization_pattern()
    )


def _is_valid_qlinear_lowering_pattern():
    def fn(match):
        if len(match.nodes) != 1:
            return False
        return match.nodes[0].target in (
            torch.ops.onednn.qlinear_pointwise.default,
            torch.ops.onednn.qlinear_pointwise.tensor,
            torch.ops.onednn.qlinear_pointwise.binary,
            torch.ops.onednn.qlinear_pointwise.binary_tensor,
        )

    return fn


def _register_quantized_linear_unary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_lowering_pattern(),
        pass_number=pass_number,
    )
    def qlinear(match: Match, *args, **kwargs):
        output_dtype = _get_pattern_output_dtype(match)
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )

        # bias
        b = kwargs.get("b")

        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]

        # post op
        postop_name = kwargs["postop_name"]
        postop_args = kwargs["postop_args"]
        postop_algorithm = kwargs["postop_algorithm"]

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            postop_name,
            postop_args,
            postop_algorithm,
        )
        counters["inductor"]["qlinear_unary_lower_count"] += 1
        counters["inductor"]["qlinear_unary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qlinear


def _register_quantized_linear_binary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_lowering_pattern(),
        pass_number=pass_number,
    )
    def qlinear_binary(match: Match, *args, **kwargs):
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype is not None
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        x2 = kwargs["x_2"]
        x2_scale = kwargs["x2_scale"]
        x2_zp = kwargs["x2_zp"]
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        # bias
        b = kwargs.get("b")
        # Output QParams
        o_inv_scale = kwargs["output_scale"]
        o_zero_point = kwargs["output_zero_point"]

        x2.realize()
        from .mkldnn_fusion import _can_be_inplace

        binary_op_name = kwargs["binary_op_name"]
        alpha = kwargs["alpha"]
        unary_op_name = kwargs["unary_op_name"]
        unary_op_args = kwargs["unary_op_args"]
        unary_op_algorithm = kwargs["unary_op_algorithm"]

        if binary_op_name == "sum" and not _can_be_inplace(x2):
            # When we enable the GEMM Template, the output of QLinear
            # will be reshaped from 2D back to 3D if the input is 3D.
            # This causes _can_be_inplace(x2) to return False if x2 happens
            # to be the output of QLinear in this scenario.
            # Change the post op from sum to binary add for this case.
            # Refer to test case:
            #   test_mkldnn_pattern_matcher.py::test_qlinear_dequant_promotion_cpu_input_dim_exceeds_2
            binary_op_name = "add"

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            x2,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            x2_scale,
            x2_zp,
            binary_op_name,
            alpha,
            unary_op_name,
            unary_op_args,
            unary_op_algorithm,
        )
        counters["inductor"]["qlinear_binary_lower_count"] += 1
        counters["inductor"]["qlinear_binary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qlinear_binary


def _is_valid_qconv_binary_optimization_pattern():
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qconv_pointwise
    )


def _is_valid_qlinear_binary_optimization_pattern():
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qlinear_pointwise,
        # we don't insert q-dq for extra input due to accuracy issues
        extra_input_from_dequant=False,
    )


def _is_valid_quantized_op_binary_optimization_pattern(
    qop, extra_input_from_dequant=True
):
    # Check if it's a valid Binary Pattern for qconv2d and qlinear:
    # * qop_pointwise should only has one users
    # * If extra_input_from_dequant is True, extra input of binary node should come from dequant pattern
    # * the two inputs of binary node should have attribute "meta" and should be tensors
    # * the two inputs of binary node should have the same shape
    # * All users of the extra input in this pattern should be
    #   ancestor nodes of the compute node, except for the binary node
    #   connected to the compute node.
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)
        compute_node = filter_nodes(match.nodes, qop)[0]
        # qop_pointwise should only have one user
        if len(compute_node.users) != 1:
            return False
        binary_node_inputs = next(iter(compute_node.users)).args
        assert len(binary_node_inputs) == 2, "Expects binary node with 2 inputs"
        if output_dtype in [torch.float32, torch.bfloat16]:
            extra_input_of_binary_node = None
            for arg in binary_node_inputs:
                if arg != compute_node:
                    extra_input_of_binary_node = arg
                    break
            assert extra_input_of_binary_node is not None
            # Extra input of binary node comes from dequant pattern
            if extra_input_from_dequant and (
                (not isinstance(extra_input_of_binary_node, torch.fx.Node))
                or (
                    extra_input_of_binary_node.target
                    != quantized_decomposed.dequantize_per_tensor.default
                )
            ):
                return False

        # the two inputs of binary node should have attribute "meta" and should be tensors
        if not (
            hasattr(binary_node_inputs[0], "meta")
            and isinstance(binary_node_inputs[0].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ) or not (
            hasattr(binary_node_inputs[1], "meta")
            and isinstance(binary_node_inputs[1].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ):
            return False
        # the two inputs of binary node should have the same shape
        if (
            binary_node_inputs[0].meta["val"].size()  # type: ignore[union-attr]
            != binary_node_inputs[1].meta["val"].size()  # type: ignore[union-attr]
        ):
            return False

        # All users of the extra input in this pattern should be
        # ancestor nodes of the compute node, except for the binary node
        # connected to the compute node.

        from .mkldnn_fusion import _get_remaining_users

        extra_input_of_pattern = (
            match.kwargs["other"]
            if "other" in match.kwargs
            else (
                match.kwargs["accum"]
                if (output_dtype in [torch.uint8, torch.int8])
                or (not extra_input_from_dequant)
                else match.kwargs["accum_after_dequant"]
            )
        )
        if (
            len(_get_remaining_users(extra_input_of_pattern, compute_node)) > 1
            or extra_input_of_pattern == compute_node.args[0]
        ):
            return False
        return True

    return fn


def _register_quantized_conv_binary_lowering(
    pattern,
    pass_number,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_lowering_pattern(),
        pass_number=pass_number,
    )
    def qconv_binary(match: Match, *args, **kwargs):
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype is not None
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        accum = kwargs["accum"]
        accum_scale = kwargs["accum_scale"]
        accum_zp = kwargs["accum_zero_point"]
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )
        # Output QParams
        output_scale = kwargs["output_scale"]
        output_zero_point = kwargs["output_zero_point"]

        # post ops
        binary_op_name = kwargs["binary_op_name"]
        alpha = kwargs["alpha"]
        unary_op_name = kwargs["unary_op_name"]
        unary_op_args = kwargs["unary_op_args"]
        unary_op_algorithm = kwargs["unary_op_algorithm"]

        accum.realize()
        from .mkldnn_fusion import _can_be_inplace

        assert _can_be_inplace(accum), (
            "QConv Binary Inplace Fusion requires accum is not an alias or mutation."
        )

        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            accum,
            b,
            stride,
            padding,
            dilation,
            groups,
            output_scale,
            output_zero_point,
            output_dtype,
            accum_scale,
            accum_zp,
            binary_op_name,
            alpha,
            unary_op_name,
            unary_op_args,
            unary_op_algorithm,
        )
        counters["inductor"]["qconv2d_binary_lower_count"] += 1
        counters["inductor"]["qconv2d_binary_lower_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qconv_binary


def _register_quantization_unary_lowering():
    # QConv2d
    for users in [1, 2]:
        qconv_pattern = get_qconv_pt2e_pattern(users)
        _register_quantized_conv_lowering(
            qconv_pattern,
            2,  # pass_number
            torch.ops.onednn.qconv_pointwise.default,  # computation_op
        )

    # QLinear
    for x_scale_zp_are_tensors in (False, True):
        qlinear_pattern = get_qlinear_pt2e_pattern(x_scale_zp_are_tensors)
        computation_op = (
            torch.ops.onednn.qlinear_pointwise.tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.default
        )
        _register_quantized_linear_unary_lowering(
            qlinear_pattern,
            2,  # pass_number
            computation_op,
        )


def _register_quantization_binary_lowering():
    # QConv2d
    for users in (1, 2):
        qconv_pattern = get_qconv2d_binary_pt2e_pattern(users)
        _register_quantized_conv_binary_lowering(
            qconv_pattern,
            2,  # pass_number
            torch.ops.onednn.qconv2d_pointwise.binary,  # computation_op
        )

    # QLinear
    for x_scale_zp_are_tensors in (False, True):
        qlinear_pattern = get_qlinear_binary_pt2e_pattern(x_scale_zp_are_tensors)
        computation_op = (
            torch.ops.onednn.qlinear_pointwise.binary_tensor
            if x_scale_zp_are_tensors
            else torch.ops.onednn.qlinear_pointwise.binary
        )
        _register_quantized_linear_binary_lowering(
            qlinear_pattern,
            2,  # pass_number
            computation_op,
        )


def _is_valid_quantized_maxpool2d_optimization_pattern():
    def fn(match):
        # Only match the pattern which max_pool2d_with_indices returns value
        # instead of indices.
        get_item_node = filter_nodes(match.nodes, operator.getitem)[0]
        return get_item_node.args[1] == 0

    return fn


def _register_quantized_maxpool2d_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_quantized_maxpool2d_optimization_pattern(),
    )
    def qmaxpool2d(match: Match, *args, **kwargs):
        x = kwargs["x"]
        kernel_size = kwargs["kernel_size"]
        stride = kwargs.get("stride")
        padding = kwargs.get("padding", 0)
        dilation = kwargs.get("dilation", 1)
        ceil_mode = kwargs.get("ceil_mode", False)

        if padding == 0:
            padding = [0, 0]
        if dilation == 1:
            dilation = [1, 1]
        if not stride:
            stride = kernel_size
        kernel_size = pad_listlike(kernel_size, 2)
        stride = pad_listlike(stride, 2)
        padding = pad_listlike(padding, 2)
        dilation = pad_listlike(dilation, 2)

        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        assert len(dilation) == 2

        computation_args = (
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        computation_args, _ = require_channels_last(computation_op, *computation_args)
        counters["inductor"]["qmaxpool2d_matcher_count"] += 1
        counters["inductor"]["qmaxpool2d_matcher_nodes"] += len(match.nodes)
        return L[computation_op](*computation_args)

    return qmaxpool2d


def _register_quantization_maxpool2d():
    # Currently, the default parameters are not in FX Graph generated by Dynamo export.
    # So, if user defines nn.MaxPool2d with different assignment of default parameter,
    # it will generate graph with different number of input nodes and hence
    # different pattern to be matched.
    # Refer to the issue: https://github.com/pytorch/pytorch/issues/105901
    max_pool2d_args_list = [
        [
            KeywordArg("stride"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
            KeywordArg("ceil_mode"),
        ],
    ]
    for max_pool2d_args in max_pool2d_args_list:
        dequantize_maxpool2d_pattern = CallFunction(
            aten.max_pool2d_with_indices.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
        )
        dequantize_lowmem_maxpool2d_pattern = CallFunction(
            prims._low_memory_max_pool_with_offsets.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
            KeywordArg("offset_dtype"),
        )
        dequantize_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_maxpool2d_pattern,
            Arg(),
        )
        dequantize_lowmem_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_lowmem_maxpool2d_pattern,
            Arg(),
        )
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(dequantize_maxpool2d_get_item_pattern),
            quantized.max_pool2d.default,
        )
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(
                dequantize_lowmem_maxpool2d_get_item_pattern
            ),
            quantized.max_pool2d.default,
        )


def _is_input_output_same_scale_zp(check_node):
    def fn(match):
        # Ensure all the inputs and output has same scale and zero point
        # Step 1: Check inputs/output zero point
        # Get dequant nodes at input
        dequant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.dequantize_per_tensor.default
        )
        zero_points = [node.args[2] for node in dequant_nodes]
        # Get quant nodes at output
        quant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.quantize_per_tensor.default
        )
        assert len(quant_nodes) == 1, "expect only 1 add node at output quant pattern"
        zero_points.append(quant_nodes[0].args[2])
        if not all(zero_point == zero_points[0] for zero_point in zero_points):
            return False

        # Step 2: Check inputs/output scale
        scales = [node.args[1] for node in dequant_nodes]
        scales.append(quant_nodes[0].args[1])
        if not all(math.isclose(scale, scales[0], rel_tol=1e-5) for scale in scales):  # type: ignore[arg-type]
            return False

        return True

    return fn


def _register_quantized_cat_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.cat.default),
    )
    def qcat(match: Match, inputs, dim, **kwargs):
        # inputs is with format: [[x1, x1_dq_dtype, x1_zp, x1_scale], ...]
        uint8_inputs = [input[0] for input in inputs]
        counters["inductor"]["qcat_matcher_count"] += 1
        counters["inductor"]["qcat_matcher_nodes"] += len(match.nodes)
        return L[computation_op](uint8_inputs, dim)

    return qcat


_raw_dequantize_per_tensor_activation_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
)


def _register_quantization_cat():
    dequantize_cat_pattern = CallFunction(
        aten.cat.default,
        ListOf(_raw_dequantize_per_tensor_activation_pattern),
        KeywordArg("dim"),
    )
    _register_quantized_cat_lowering(
        generate_pattern_with_output_quant(dequantize_cat_pattern),
        aten.cat,
    )


def _register_quantized_reshape_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.reshape.default),
    )
    def qreshape(match: Match, *args, **kwargs):
        qx = kwargs["x"]
        shape = kwargs["shape"]
        counters["inductor"]["qreshape_matcher_count"] += 1
        counters["inductor"]["qreshape_matcher_nodes"] += len(match.nodes)
        return L[computation_op](qx, shape)

    return qreshape


def _register_quantization_reshape():
    dequantize_reshape_pattern = CallFunction(
        torch.ops.aten.reshape.default,
        get_dequantize_per_tensor_activation_pattern(),
        KeywordArg("shape"),
    )
    _register_quantized_reshape_lowering(
        generate_pattern_with_output_quant(dequantize_reshape_pattern),
        aten.reshape,
    )


def _is_valid_concat_linear_int8_woq_optimization_pattern():
    def fn(match):
        if not config.cpp.enable_concat_linear:
            return False
        assert all(k in match.kwargs for k in ("x", "w1", "w2", "w3", "scales"))
        if not all(
            hasattr(match.kwargs[key], "meta")
            for key in ["x", "w1", "w2", "w3", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        w1 = match.kwargs["w1"].meta["val"]
        w2 = match.kwargs["w2"].meta["val"]
        w3 = match.kwargs["w3"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        if len(match.kwargs["scales"].meta["val"].size()) > 1:
            return False
        num_scales = match.kwargs["scales"].meta["val"].numel()
        w1_cols = match.kwargs["w1"].meta["val"].size()[0]
        w2_cols = match.kwargs["w2"].meta["val"].size()[0]
        w3_cols = match.kwargs["w3"].meta["val"].size()[0]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and w1.dtype == torch.int8
            and w2.dtype == torch.int8
            and w3.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            and x.device.type in ("cpu", "cuda")
            and x.device == w1.device
            and w1.device == w2.device
            and w2.device == w3.device
            and x.device == scales.device
            and num_scales == w1_cols + w2_cols + w3_cols
        )

    return fn


def _is_valid_woq_optimization_pattern():
    def fn(match):
        assert all(k in match.kwargs for k in ("x", "weight", "scales"))
        if not all(
            hasattr(match.kwargs[key], "meta") for key in ["x", "weight", "scales"]
        ):
            return False
        x = match.kwargs["x"].meta["val"]
        weight = match.kwargs["weight"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        return (
            # For now, we only support woq mm kernels
            # with x.type=bfloat16 and w.type=int8
            x.dtype == torch.bfloat16
            and weight.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            and x.device.type in ("cpu", "cuda")
            and x.device == weight.device
            and x.device == scales.device
        )

    return fn


def _register_concat_linear_int8_woq_lowering(
    pattern, computation_woq, computation_reshape
):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_concat_linear_int8_woq_optimization_pattern(),
        pass_number=4,
    )
    def woq_int8(match: Match, *args, **kwargs):
        x = kwargs["x"]
        w1 = kwargs["w1"]
        w2 = kwargs["w2"]
        w3 = kwargs["w3"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = (
            w1.meta["val"].size()[0]
            + w2.meta["val"].size()[0]
            + w3.meta["val"].size()[0]
        )
        origin_x_size = tuple(x.meta["val"].size())
        x_shape = [-1, origin_x_size[-1]]
        out_shape = list(origin_x_size[:-1] + (out_features,))
        mm_node_of_x = None
        for candidate in iter(x.users.keys()):
            if (
                candidate.target is aten.mm.default
                and list(candidate._input_nodes)[1].target is aten.cat.default
            ):
                mm_node_of_x = candidate
                break
        assert mm_node_of_x is not None, "unable to find mm node"
        _, cat_wgt_node = mm_node_of_x._input_nodes
        scaling_node = next(iter(mm_node_of_x.users.keys()))
        user_of_scaling_node = next(iter(scaling_node.users.keys()))
        # Some other pass is making some changes that entails
        # adding a node before it's used, but it can only be found when
        # lint is run. stable_topological_sort() is being run before lint,
        # so that error was not being being discovered.
        # We call stable_topological_sort here as a workaround.
        stable_topological_sort(match.graph)
        with match.graph.inserting_before(user_of_scaling_node):
            new_cat_node = match.graph.call_function(
                aten.cat.default,
                args=([w1, w2, w3], 0),
            )
            x_reshape_node = match.graph.call_function(
                computation_reshape, args=(x, x_shape)
            )
            new_woq_node = match.graph.call_function(
                computation_woq,
                args=(x_reshape_node, new_cat_node, scales),
            )
            new_woq_node.meta = copy.copy(x.meta)
            output_reshape_node = match.graph.call_function(
                computation_reshape, args=(new_woq_node, out_shape)
            )
            scaling_node.replace_all_uses_with(output_reshape_node)
            match.graph.erase_node(scaling_node)
            match.graph.erase_node(mm_node_of_x)
            match.graph.erase_node(cat_wgt_node)
            match.graph.lint()

    return woq_int8


def _register_woq_lowering(pattern, computation_woq, computation_reshape):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_woq_optimization_pattern(),
    )
    def woq_int8(match: Match, *args, **kwargs):
        x = kwargs["x"]
        weight = kwargs["weight"]
        scales = kwargs["scales"]
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        out_features = weight.get_size()[0]
        origin_x_size = x.get_size()
        x_shape = [-1, origin_x_size[-1]]
        out_shape = origin_x_size[:-1] + [
            out_features,
        ]
        func1 = L[computation_reshape](x, x_shape)
        func2 = L[computation_woq](func1, weight, scales)
        return L[computation_reshape](func2, out_shape)

    return woq_int8


def _register_woq_mm_int8_pattern1():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, with x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                CallFunction(aten.reshape.default, KeywordArg("x"), Arg()),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern2():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to mm, w/o x reshape
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                KeywordArg("x"),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern3():
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # case of dispatching to bmm
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.bmm.default,
            CallFunction(aten.expand.default, KeywordArg("x"), Arg()),
            CallFunction(
                aten.expand.default,
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern4():
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.mm.default,
            KeywordArg("x"),
            CallFunction(
                prims.convert_element_type.default,
                CallFunction(
                    aten.permute.default,
                    KeywordArg("weight"),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_int8_woq_concat_linear_pattern():
    def _create_wgt_node(wgt_node_name: str):
        return CallFunction(
            prims.convert_element_type.default,
            CallFunction(
                aten.permute.default,
                KeywordArg(wgt_node_name),
                Arg(),
            ),
            Arg(),
        )

    cat_wgt = CallFunction(
        aten.cat.default, [_create_wgt_node(wgt) for wgt in ["w1", "w2", "w3"]], 1
    )

    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(aten.mm.default, KeywordArg("x"), cat_wgt),
        KeywordArg("scales"),
    )
    _register_concat_linear_int8_woq_lowering(
        _woq_pattern, aten._weight_int8pack_mm.default, aten.reshape
    )


def _register_quantization_lowerings():
    _register_quantization_unary_lowering()
    _register_quantization_binary_lowering()
    _register_quantization_maxpool2d()
    _register_quantization_cat()
    _register_quantization_reshape()


def _register_woq_lowerings():
    _register_woq_mm_int8_pattern1()
    _register_woq_mm_int8_pattern2()
    _register_woq_mm_int8_pattern3()
    _register_woq_mm_int8_pattern4()


def _is_valid_dequant_promotion_pattern(dtype=torch.float32):
    def _inner(match):
        assert dtype in [torch.float32, torch.bfloat16]
        dequant_pattern_end_node = match.output_node()
        if dequant_pattern_end_node.target not in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
            prims.convert_element_type.default,
            aten.reshape.default,
        ]:
            return False

        if dequant_pattern_end_node.target is aten.reshape.default:
            dequant_node = (
                dequant_pattern_end_node.args[
                    0
                ]  # pattern: linear <- reshape <- dequant
                if dtype == torch.float32
                else dequant_pattern_end_node.args[0].args[
                    0
                ]  # pattern: linear <- reshape <- to_bf16 <- dequant
            )
        else:
            dequant_node = (
                dequant_pattern_end_node  # pattern: linear <- dequant
                if dtype == torch.float32
                else dequant_pattern_end_node.args[
                    0
                ]  # pattern: linear <- to_bf16 <- dequant
            )

        if (
            dequant_node.target
            in [
                quantized_decomposed.dequantize_per_tensor.default,
                quantized_decomposed.dequantize_per_tensor.tensor,
            ]
            and len(list(dequant_pattern_end_node.users)) > 1
        ):
            # If dequant pattern has more than 1 users, then do dequant promoted
            return True
        return False

    return _inner


def _register_dequant_promotion_pass(pattern, pass_number, dtype=torch.float32):
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_promotion_pattern(dtype),
        pass_number=pass_number,
    )
    def dequant_promotion(match: Match, *args, **kwargs):
        # Dequant_promotion will transform
        # graph 1:
        #            quant
        #      + - - - | - - - +
        #      |    dequant    |
        #      |    /     \    |
        #      |  node1  node2 |
        #      + - | - - - | - +
        #        quant   quant
        # into:
        # graph 2:
        #            quant
        #      + - - / - \ - - +
        #      |dequant dequant|
        #      |    |      |   |
        #      | node1 node2   |
        #      + - | - - - | - +
        #        quant   quant
        # In graph 1, the dequant node is shared by node1 and node2,
        # as a result, neither node1 nor node2 could form an int8
        # fusion pattern.
        # After this transformation, the graph 2 could hit the int8
        # fusion pattern: dequant-node-quant, respectively for
        # node1 and node2.
        assert dtype in [torch.float32, torch.bfloat16]

        def clone_to_new_node(graph, source_node, user_node):
            # Clone the source_node to a new node
            # Replace user_node's input from source_node to new_node
            assert source_node.op == "call_function", (
                "clone_to_new_node only support node.op call_function"
            )
            with graph.inserting_before(user_node):
                new_node = graph.call_function(
                    source_node.target,
                    args=source_node.args,
                    kwargs=source_node.kwargs,
                )
                new_node.meta = copy.copy(source_node.meta)
                user_node.replace_input_with(source_node, new_node)
            return new_node

        # Find the start node and end node of a dequant pattern
        # * End node should be the match.output_node()
        # * Start node should be the node of dequantize_per_tensor
        dequant_pattern_end_node = match.output_node()
        assert dequant_pattern_end_node.target in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
            prims.convert_element_type.default,
            aten.reshape.default,
        ]

        # For a dequant pattern, we should expect see the node list as:
        # * OPT(aten.reshape.default)
        # * OPT(prims.convert_element_type.default) (to_bf16)
        # * dequantize_per_tensor
        def _find_first_node_in_dequant_pattern(_node):
            if _node.target in [
                quantized_decomposed.dequantize_per_tensor.default,
                quantized_decomposed.dequantize_per_tensor.tensor,
            ]:
                # For a dequant pattern, we expect the start node is a dequantize_per_tensor node
                return _node
            else:
                assert len(_node.args) >= 1, (
                    "In in dequant pattern, each node should have more than 1 arg."
                )
                return _find_first_node_in_dequant_pattern(_node.args[0])

        dequant_pattern_start_node = _find_first_node_in_dequant_pattern(
            dequant_pattern_end_node
        )

        assert dequant_pattern_start_node.target in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
        ]

        # Clone the dequant pattern for each user node
        graph = match.graph
        user_node_list = list(dequant_pattern_end_node.users)
        for user_node in user_node_list[1:]:
            _source_node = dequant_pattern_end_node
            _user_node = user_node
            while _source_node != dequant_pattern_start_node.args[0]:
                _user_node = clone_to_new_node(graph, _source_node, _user_node)
                _source_node = _source_node.args[0]  # type: ignore[assignment]

        counters["inductor"]["dequant_promotion_matcher_count"] += 1
        counters["inductor"]["dequant_promotion_matcher_nodes"] += len(match.nodes)


def _is_valid_dequant_conv_pattern(dtype, with_dtype_convert):
    def _inner(match):
        # 
```



## High-Level Overview

"""The quantization.py file primarily incorporates passes related to quantization fusion

This Python file contains 1 class(es) and 118 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PostOpAttr`

**Functions defined**: `_get_pattern_output_dtype`, `_may_generate_pattern_with_dtype_convert`, `_may_generate_pattern_with_reshape`, `_generate_linear_t_pattern`, `_unary_fusion_pattern`, `get_dequantize_per_tensor_activation_pattern`, `get_qconv_pt2e_pattern`, `get_qconv2d_binary_pt2e_pattern`, `get_qlinear_pt2e_pattern`, `get_qlinear_binary_pt2e_pattern`, `generate_pattern_with_binary`, `generate_pattern_with_unary`, `generate_pattern_with_output_quant`, `_check_node_kwarg_arg_value`, `_is_valid_quantized_conv_optimization_pattern`, `fn`, `_is_valid_qconv_post_op_fusion_pattern`, `_is_valid_qconv_lowering_pattern`, `fn`, `_register_quantized_conv_lowering`

**Key imports**: copy, functools, itertools, math, operator, Any, torch, counters, has_free_symbols, map_arg


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `functools`
- `itertools`
- `math`
- `operator`
- `typing`: Any
- `torch`
- `torch._dynamo.utils`: counters
- `torch.fx.experimental.symbolic_shapes`: has_free_symbols
- `torch.fx.node`: map_arg
- `..`: config
- `..lowering`: lowerings as L, require_channels_last
- `..utils`: pad_listlike
- `.freezing_patterns`: register_freezing_graph_pattern
- `.post_grad`: register_lowering_pattern
- `.mkldnn_fusion`: _can_be_inplace


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`numeric_utils.py_docs.md`](./numeric_utils.py_docs.md)
- [`dedupe_symint_uses.py_docs.md`](./dedupe_symint_uses.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `quantization.py_docs.md`
- **Keyword Index**: `quantization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

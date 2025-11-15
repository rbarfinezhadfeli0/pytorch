# Documentation: symbolic_opset9.py

## File Metadata
- **Path**: `torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py`
- **Size**: 226531 bytes
- **Lines**: 6689
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""This file exports ONNX ops for opset 9.

Opset 9 is supported by ONNX release 1.4.1
release on 01/23/19
"""

from __future__ import annotations

import builtins
import functools
import math
import sys
import warnings
from typing import TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import (
    _type_utils,
    jit_utils,
    registration,
    symbolic_helper,
)
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.types import Number

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = [
    "abs",
    "acos",
    "add",
    "addcmul",
    "addmm",
    "alias",
    "amax",
    "amin",
    "aminmax",
    "arange",
    "argmax",
    "argmin",
    "as_strided",
    "as_tensor",
    "asin",
    "atan",
    "atan2",
    "baddbmm",
    "batch_norm",
    "bernoulli",
    "bitwise_not",
    "bitwise_or",
    "bmm",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cdist",
    "ceil",
    "clamp_max",
    "clamp_min",
    "clamp",
    "clone",
    "constant_pad_nd",
    "contiguous",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "conv1d",
    "conv2d",
    "conv3d",
    "convert_element_type",
    "convolution",
    "cos",
    "cosine_similarity",
    "cross",
    "cumsum",
    "detach",
    "dim",
    "div",
    "dot",
    "dropout",
    "elu",
    "embedding_bag",
    "embedding",
    "empty_like",
    "empty",
    "eq",
    "erf",
    "exp",
    "expand_as",
    "expand",
    "eye",
    "fill",
    "flatten",
    "floor_divide",
    "floor",
    "floordiv",
    "frobenius_norm",
    "full_like",
    "full",
    "gather",
    "ge",
    "gelu",
    "get_pool_ceil_padding",
    "glu",
    "group_norm",
    "gt",
    "hann_window",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_select",
    "index",
    "instance_norm",
    "is_floating_point",
    "is_pinned",
    "isnan",
    "item",
    "kl_div",
    "layer_norm",
    "le",
    "leaky_relu",
    "lerp",
    "lift",
    "linalg_cross",
    "linalg_matrix_norm",
    "linalg_norm",
    "linalg_vector_norm",
    "linear",
    "linspace",
    "log_sigmoid",
    "log_softmax",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logsumexp",
    "lstm_cell",
    "lstm",
    "lt",
    "masked_fill",
    "masked_fill_",
    "matmul",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "max",
    "maximum",
    "meshgrid",
    "min",
    "minimum",
    "mish",
    "mm",
    "movedim",
    "mse_loss",
    "mul",
    "multinomial",
    "mv",
    "narrow",
    "native_layer_norm",
    "ne",
    "neg",
    "new_empty",
    "new_full",
    "new_ones",
    "new_zeros",
    "nonzero_numpy",
    "nonzero",
    "norm",
    "numel",
    "numpy_T",
    "one_hot",
    "ones_like",
    "ones",
    "onnx_placeholder",
    "pad",
    "pairwise_distance",
    "permute",
    "pixel_shuffle",
    "pixel_unshuffle",
    "pow",
    "prelu",
    "prim_constant_chunk",
    "prim_constant_split",
    "prim_constant",
    "prim_data",
    "prim_device",
    "prim_dtype",
    "prim_if",
    "prim_layout",
    "prim_list_construct",
    "prim_list_unpack",
    "prim_loop",
    "prim_max",
    "prim_min",
    "prim_shape",
    "prim_tolist",
    "prim_tuple_construct",
    "prim_type",
    "prim_unchecked_cast",
    "prim_uninitialized",
    "rand_like",
    "rand",
    "randint_like",
    "randint",
    "randn_like",
    "randn",
    "reciprocal",
    "reflection_pad",
    "relu",
    "relu6",
    "remainder",
    "repeat_interleave",
    "repeat",
    "replication_pad",
    "reshape_as",
    "reshape",
    "roll",
    "rrelu",
    "rsqrt",
    "rsub",
    "scalar_tensor",
    "scatter_add",
    "scatter",
    "select",
    "selu",
    "sigmoid",
    "sign",
    "silu",
    "sin",
    "size",
    "slice",
    "softmax",
    "softplus",
    "softshrink",
    "sort",
    "split_with_sizes",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std_mean",
    "std",
    "sub",
    "t",
    "take",
    "tan",
    "tanh",
    "tanhshrink",
    "tensor",
    "threshold",
    "to",
    "topk",
    "transpose",
    "true_divide",
    "type_as",
    "unbind",
    "unfold",
    "unsafe_chunk",
    "unsafe_split_with_sizes",
    "unsafe_split",
    "unsqueeze",
    "unsupported_complex_operators",
    "noop_complex_operators",
    "unused",
    "var_mean",
    "var",
    "view_as",
    "view",
    "where",
    "wrap_logical_op_with_cast_to",
    "wrap_logical_op_with_negation",
    "zeros_like",
    "zeros",
    "zero",
]


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=9)


def _export(name: str):
    """Exports the function in the current global namespace."""

    def wrapper(func):
        globals()[name] = func
        __all__.append(name)
        return func

    return wrapper


def unused(g):
    """Represents "missing" optional inputs."""
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


@_onnx_symbolic("aten::_shape_as_tensor")
def _shape_as_tensor(g: jit_utils.GraphContext, input):
    return g.op("Shape", input)


@_onnx_symbolic("aten::_reshape_from_tensor")
def _reshape_from_tensor(g: jit_utils.GraphContext, input, shape):
    if isinstance(shape, list):
        shape = g.op("Concat", *shape, axis_i=0)
    return reshape(g, input, shape)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
def reshape(g: jit_utils.GraphContext, self, shape):
    return symbolic_helper._reshape_helper(g, self, shape)


@_onnx_symbolic("aten::reshape_as")
@symbolic_helper.quantized_args(True)
def reshape_as(g: jit_utils.GraphContext, self, other):
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


@_onnx_symbolic("aten::add")
def add(g: jit_utils.GraphContext, self, other, alpha=None):
    """
    This function takes the add function and returns the corresponding ONNX operator.

    This function is not meant to be called directly by the user.

    Args:
        g (GraphContext): The graph context.
        self (Tensor): The first operand.
        other (Tensor): The second operand.
        alpha (float, optional): The scaling factor for the second operand. Defaults to None.

    Returns:
        ONNX operator.
    """
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Add", 9, 11, "Add between list of tensors not supported", self
        )
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        other = g.op("Mul", other, alpha)
    return g.op("Add", self, other)


@_onnx_symbolic("aten::sub")
def sub(g: jit_utils.GraphContext, self, other, alpha=None):
    """
    Consumes sub function and returns the corresponding ONNX operator.

    This function is not meant to be called directly by the user.

    Args:
        g (GraphContext): The graph context.
        self (Tensor): The first operand.
        other (Tensor): The second operand.
        alpha (Optional[Tensor]): A scaling factor to apply to the second operand.
            If `alpha` is not provided, it defaults to 1.

    Returns:
        ONNX operator
    """
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        other = g.op("Mul", other, alpha)
    return g.op("Sub", self, other)


@_onnx_symbolic("aten::rsub")
def rsub(g: jit_utils.GraphContext, self, other, alpha=None):
    return sub(g, other, self, alpha=alpha)


@_onnx_symbolic("aten::mul")
def mul(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_bool(self) and symbolic_helper._is_bool(other):
        # ONNX Mul doesn't support Boolean, so use And as an equivalent operator.
        return g.op("And", self, other)
    else:
        return g.op("Mul", self, other)


@_onnx_symbolic("aten::div")
def div(g: jit_utils.GraphContext, self, other, *args):
    if len(args) == 0:
        return true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@_onnx_symbolic("aten::addcmul")
@symbolic_helper.parse_args("v", "v", "v", "f")
def addcmul(g: jit_utils.GraphContext, self, tensor1, tensor2, value=1.0):
    value_tens = g.op("Constant", value_t=torch.tensor([value]))
    return add(g, self, mul(g, mul(g, tensor1, tensor2), value_tens))


@symbolic_helper.parse_args("v", "v", "s")
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode is None:
        return true_divide(g, self, other)
    elif rounding_mode == "floor":
        return _floor_divide(g, self, other)
    elif rounding_mode == "trunc":
        return _trunc_divide(g, self, other)
    else:
        raise errors.SymbolicValueError(
            f'Unsupported rounding mode: "{rounding_mode}". Expected None, "floor" or "trunc"',
            self,
        )


def _trunc_divide(g: jit_utils.GraphContext, self, other):
    out = g.op("Div", self, other)
    # the correct operation is truncate, which is not supported in ONNX,
    # we cannot call floor since it will behave differently for negative numbers
    # (eg. -0.1 should become -0 )
    # - if scalar_type information are not available, assume that
    # we need to call floor (treat as float)
    out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.INT64)

    # Matching PyTorch's behavior:
    # - if self is fp the output's type is self's type
    # - if self is not fp and other is fp, the output is of type JitScalarType.FLOAT
    # - self is not fp and other is not fp, the output's type is self's output type
    # - the output type defaults to Float
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        if not symbolic_helper._is_fp(self) and symbolic_helper._is_fp(other):
            out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        else:
            out = g.op(
                "Cast",
                out,
                to_i=scalar_type.onnx_type(),
            )
    else:
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return out


def _floor_divide(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does truncation rounding
        div = g.op("Div", self, other)
        # Division is negative if: self < 0 != other < 0
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op(
            "Xor",
            symbolic_helper._lt_helper(g, self, zero),
            symbolic_helper._lt_helper(g, other, zero),
        )

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        mod = g.op("Sub", self, g.op("Mul", div, other))
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op("Mul", fixup_mask, one)
        return g.op("Sub", div, fixup)


@_onnx_symbolic("aten::floor_divide")
def floor_divide(g: jit_utils.GraphContext, self, other):
    # Deprecated behavior, floor_divide actually truncates
    return _trunc_divide(g, self, other)


@_onnx_symbolic("aten::floordiv")
def floordiv(g: jit_utils.GraphContext, self, other):
    return floor_divide(g, self, other)


@_onnx_symbolic("aten::true_divide")
def true_divide(g: jit_utils.GraphContext, self, other):
    """Division where both inputs are cast to floating types

    If both inputs are floating, performs div as usual
    If only one input is a floating type, the other input is cast to its type
    If neither input is a floating type, both inputs are cast to the default scalar type
    """

    # Case 1: either values are floating
    # Performs div as usual.
    # Implicit casting will be handled in scalar type analysis pass.
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        return g.op("Div", self, other)

    # Case 2: neither is floating
    # Casts both inputs to the default scalar type
    scalar_type = torch.get_default_dtype()
    onnx_scalar_type = _C_onnx.TensorProtoDataType.FLOAT
    assert scalar_type is torch.float or scalar_type is torch.double
    if torch.get_default_dtype() is torch.double:
        onnx_scalar_type = _C_onnx.TensorProtoDataType.DOUBLE

    self = g.op("Cast", self, to_i=onnx_scalar_type)
    other = g.op("Cast", other, to_i=onnx_scalar_type)
    return g.op("Div", self, other)


@_onnx_symbolic("aten::reciprocal")
def reciprocal(g: jit_utils.GraphContext, self):
    # torch.reciprocal implicitly casts to float, so we do the same.
    if not symbolic_helper._is_fp(self):
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return g.op("Reciprocal", self)


@_onnx_symbolic("aten::cat")
@symbolic_helper.parse_args("v", "i")
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    """Implement concatenation of pytorch tensors in ONNX along the specified `dim` dimension.

    Parameters:
        g (jit_utils.GraphContext): Graph context.
        tensor_list (List[torch.Tensor]): List of tensors to concatenate.
        dim (int): Dimension along which to concatenate the tensors.

    Returns:
        ONNX graph node representing the concatenated tensor.
    """
    tensors = symbolic_helper._unpack_list(tensor_list)
    # torch.cat ignores empty tensors such as `torch.Tensor([])`
    # These needs to be removed as input from ONNX's concat too, otherwise shape inference
    # will likely fail due to inputs with different ranks (0 for empty tensor, > 0 for anything else)
    nonempty_tensors = []
    for t in tensors:
        if symbolic_helper._is_constant(t) and not symbolic_helper._get_tensor_dim_size(
            t, 0
        ):
            continue
        nonempty_tensors.append(t)
    assert len(nonempty_tensors) > 0
    assert all(
        symbolic_helper._get_tensor_rank(nonempty_tensors[0]) is None
        or symbolic_helper._get_tensor_rank(t) is None
        or symbolic_helper._get_tensor_rank(t)
        == symbolic_helper._get_tensor_rank(nonempty_tensors[0])
        for t in nonempty_tensors
    )
    tensor_list.node().removeAllInputs()
    for t in nonempty_tensors:
        tensor_list.node().addInput(t)

    tensors = symbolic_helper._unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


@_onnx_symbolic("aten::stack")
@symbolic_helper.parse_args("v", "i")
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    unsqueezed = [
        symbolic_helper._unsqueeze_helper(g, t, [dim])
        for t in symbolic_helper._unpack_list(tensor_list)
    ]
    return g.op("Concat", *unsqueezed, axis_i=dim)


@_onnx_symbolic("aten::list")
def _list(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    C = g.op("Constant", value_t=torch.tensor([1]))
    return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::bmm")
def bmm(g: jit_utils.GraphContext, self, other):
    return g.op("MatMul", self, other)


@_onnx_symbolic("aten::matmul")
def matmul(g: jit_utils.GraphContext, self, other):
    return g.op("MatMul", self, other)


@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha):
    scalar_type = None
    self_scalar_type = symbolic_helper._try_get_scalar_type(self)
    mat1_scalar_type = symbolic_helper._try_get_scalar_type(mat1)
    mat2_scalar_type = symbolic_helper._try_get_scalar_type(mat2)
    if self_scalar_type is not None:
        scalar_type = self_scalar_type
    elif mat1_scalar_type is not None:
        scalar_type = mat1_scalar_type
    elif mat2_scalar_type is not None:
        scalar_type = mat2_scalar_type

    mat1_rank = symbolic_helper._get_tensor_rank(mat1)
    mat2_rank = symbolic_helper._get_tensor_rank(mat2)

    def is_not_none_nor(v, u):
        return v is not None and v != u

    if scalar_type is not None and (
        is_not_none_nor(mat1_rank, 2) or is_not_none_nor(mat2_rank, 2)
    ):
        res1 = g.op("MatMul", mat1, mat2)
        res2 = self

        alpha = symbolic_helper._scalar(alpha)
        beta = symbolic_helper._scalar(beta)

        if alpha != 1:
            alpha = g.op(
                "Constant", value_t=torch.tensor(alpha, dtype=scalar_type.dtype())
            )
            res1 = g.op("Mul", res1, alpha)
        if beta != 1:
            beta = g.op(
                "Constant",
                value_t=torch.tensor(
                    symbolic_helper._scalar(beta), dtype=scalar_type.dtype()
                ),
            )
            res2 = g.op("Mul", res2, beta)

        return g.op("Add", res1, res2)

    return g.op(
        "Gemm",
        mat1,
        mat2,
        self,
        beta_f=symbolic_helper._scalar(beta),
        alpha_f=symbolic_helper._scalar(alpha),
    )


@_onnx_symbolic("aten::neg")
def neg(g: jit_utils.GraphContext, self):
    return g.op("Neg", self)


@_onnx_symbolic("aten::sqrt")
def sqrt(g: jit_utils.GraphContext, self):
    if _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
        _type_utils.JitScalarType.INT16,
        _type_utils.JitScalarType.INT,
        _type_utils.JitScalarType.INT64,
    }:
        # torch converts all int inputs to sqrt to float
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    return g.op("Sqrt", self)


@_onnx_symbolic("aten::rsqrt")
def rsqrt(g: jit_utils.GraphContext, self):
    return g.op(
        "Div", symbolic_helper._if_scalar_type_as(torch.ones(1), self), sqrt(g, self)
    )


@_onnx_symbolic("aten::tanh")
# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qtanh.cpp
@symbolic_helper.quantized_args(True, scale=2.0 / 256.0, zero_point=128)
def tanh(g: jit_utils.GraphContext, self):
    return g.op("Tanh", self)


@_onnx_symbolic("aten::sin")
def sin(g: jit_utils.GraphContext, self):
    return g.op("Sin", self)


@_onnx_symbolic("aten::cos")
def cos(g: jit_utils.GraphContext, self):
    return g.op("Cos", self)


@_onnx_symbolic("aten::tan")
def tan(g: jit_utils.GraphContext, self):
    return g.op("Tan", self)


@_onnx_symbolic("aten::asin")
def asin(g: jit_utils.GraphContext, self):
    return g.op("Asin", self)


@_onnx_symbolic("aten::acos")
def acos(g: jit_utils.GraphContext, self):
    return g.op("Acos", self)


@_onnx_symbolic("aten::atan")
def atan(g: jit_utils.GraphContext, self):
    return g.op("Atan", self)


@_onnx_symbolic("aten::atan2")
def atan2(g: jit_utils.GraphContext, self, other):
    # self is y, and other is x on coordinate
    slope = g.op("Div", self, other)
    atan = g.op("Atan", slope)
    const_zero = g.op("Constant", value_t=torch.tensor(0))
    const_pi = g.op("Constant", value_t=torch.tensor(math.pi))

    condition_second_or_third_quadrant = g.op("Greater", self, const_zero)
    second_third_quadrant = g.op(
        "Where",
        condition_second_or_third_quadrant,
        g.op("Add", atan, const_pi),
        g.op("Sub", atan, const_pi),
    )

    condition_14_or_23_quadrant = g.op("Less", other, const_zero)
    result = g.op("Where", condition_14_or_23_quadrant, second_third_quadrant, atan)

    return result


@_onnx_symbolic("aten::sigmoid")
# Fixed scale and zero_point, discovered from aten/src/ATen/native/quantized/cpu/qsigmoid.cpp
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
def sigmoid(g: jit_utils.GraphContext, self):
    """Converts the corresponding PyTorch function into ONNX operators.

    It is not meant to be called directly by a user.

    Args:
        g (jit_utils.GraphContext): Graph context.
        self (Tensor): the input tensor.
    Returns:
        ONNX operator
    """
    return g.op("Sigmoid", self)


@_onnx_symbolic("aten::sign")
def sign(g: jit_utils.GraphContext, self):
    return g.op("Sign", self)


@symbolic_helper.quantized_args(True)
def _slice(g: jit_utils.GraphContext, input, axes, starts, ends):
    assert len(starts) == len(ends)
    if len(starts) == 1 and starts[0] == 0 and ends[0] == _constants.INT64_MAX:
        return input
    return g.op("Slice", input, axes_i=axes, starts_i=starts, ends_i=ends)


@_onnx_symbolic(
    "aten::sum", decorate=[symbolic_helper._apply_params("ReduceSum", "sum")]
)
@_onnx_symbolic(
    "aten::mean", decorate=[symbolic_helper._apply_params("ReduceMean", "mean")]
)
# torch.prod does not support multidimensional "dim"
@_onnx_symbolic(
    "aten::prod",
    decorate=[
        symbolic_helper._apply_params(
            "ReduceProd", "prod", allow_multi_dim_support=False
        )
    ],
)
def _reduce_with_dtype(onnx_op: str, name: str, allow_multi_dim_support: bool = True):
    return symbolic_helper._reduce_with_dtype_helper(
        onnx_op, name, allow_multi_dim_support
    )


@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
def cumsum(g: jit_utils.GraphContext, input, dim, dtype) -> None:
    symbolic_helper._onnx_opset_unsupported("cumsum", 9, 11, input)


@_onnx_symbolic("aten::_sample_dirichlet")
def _sample_dirichlet(g: jit_utils.GraphContext, self, generator):
    return symbolic_helper._onnx_unsupported("_sample_dirichlet", self)


@_onnx_symbolic("aten::_standard_gamma")
def _standard_gamma(g: jit_utils.GraphContext, self, generator):
    return symbolic_helper._onnx_unsupported("_standard_gamma", self)


@_onnx_symbolic("aten::t")
def t(g: jit_utils.GraphContext, self):
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is None or rank < 2:
        # The transpose of a 1d or 0d tensor is itself. ONNX does not define the behavior
        # clearly and onnxruntime fails on these cases. So we add an Identity node to
        # mirror the behavior of eager mode.
        return g.op("Identity", self)
    return g.op("Transpose", self, perm_i=(1, 0))


@_onnx_symbolic("aten::numpy_T")
@symbolic_helper.quantized_args(True)
def numpy_T(g: jit_utils.GraphContext, input):
    ndim = symbolic_helper._get_tensor_rank(input)
    assert ndim is not None
    perm = list(reversed(range(ndim)))
    return g.op("Transpose", input, perm_i=perm)


@_onnx_symbolic("aten::expand")
@symbolic_helper.quantized_args(True)
def expand(g: jit_utils.GraphContext, self, size, implicit):
    """Implement the expand function for a pytorch tensor in ONNX according to specified `size`"""
    size = symbolic_helper._maybe_get_const(size, "is")
    if not symbolic_helper._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    elif symbolic_helper._is_packed_list(size):
        # Expand with -1 dim value means dim is unchanged.
        # Since onnx::expand supports two-way broadcasting,
        # -1 dim value can be exported to onnx as 1
        size = symbolic_helper._reshape_helper(
            g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1]))
        )
    dtype = _type_utils.JitScalarType.INT64
    ones = ones_like(g, size, dtype)
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    return g.op("Expand", self, size)


@_onnx_symbolic("aten::broadcast_to")
@symbolic_helper.quantized_args(True)
def broadcast_to(g: jit_utils.GraphContext, self, size):
    size = symbolic_helper._maybe_get_const(size, "is")
    if not symbolic_helper._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    elif symbolic_helper._is_packed_list(size):
        # Expand with -1 dim value means dim is unchanged.
        # Since onnx::expand supports two-way broadcasting,
        # -1 dim value can be exported to onnx as 1
        size = symbolic_helper._reshape_helper(
            g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1]))
        )
    dtype = _type_utils.JitScalarType.INT64
    ones = ones_like(g, size, dtype)
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    return g.op("Expand", self, size)


@_onnx_symbolic("aten::expand_as")
@symbolic_helper.quantized_args(True, True)
def expand_as(g: jit_utils.GraphContext, self, other):
    self_t = symbolic_helper._maybe_get_const(self, "t")
    if isinstance(self_t, torch.Tensor):
        orig_type = self_t.dtype
        self_t = self_t.to(torch.double)
        dims = []
        for d in range(self_t.dim()):
            if torch.equal(self_t.mean(d).unsqueeze(d).expand_as(self_t), self_t):
                dims.append(d)
                self = g.op(
                    "Constant", value_t=self_t.mean(dims, keepdim=True).to(orig_type)
                )

    shape = g.op("Shape", other)
    return g.op("Expand", self, shape)


@_onnx_symbolic("aten::embedding")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "i", "b", "v")
def embedding(
    g: jit_utils.GraphContext,
    weight,
    indices,
    padding_idx,
    scale_grad_by_freq,
    sparse,
):
    if scale_grad_by_freq and GLOBALS.export_training:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of embedding with scale_grad_by_freq=True "
            "for training mode. ONNX does not support scaling the gradients.",
            weight,
        )
    if padding_idx >= 0 and GLOBALS.export_training:
        warnings.warn(
            "Warning: ONNX export of embedding with padding_idx >= 0 "
            "for training mode. "
            "ONNX does not support not updating the embedding vector at padding_idx during training.",
            stacklevel=2,
        )

    return g.op("Gather", weight, indices)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    if not symbolic_helper._is_none(per_sample_weights):
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with per_sample_weights"
        )

    return symbolic_helper._onnx_unsupported("embedding_bag", embedding_matrix)


@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
def size(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    if symbolic_helper._maybe_get_const(dim, "i") < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            dim = symbolic_helper._maybe_get_const(dim, "i") + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    return symbolic_helper._size_helper(g, self, dim)


@_onnx_symbolic("aten::transpose")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "i")
def transpose(g: jit_utils.GraphContext, self, dim0, dim1):
    if dim0 == dim1:  # micro-optimization
        return self

    # NB: Transpose in ONNX is actually a Permute
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None:
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return g.op("Transpose", self, perm_i=axes)
    else:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of transpose for tensor of unknown rank.",
            self,
        )


@_onnx_symbolic("aten::permute")
@symbolic_helper.parse_args("v", "is")
def permute(g: jit_utils.GraphContext, self, dims):
    if dims == list(range(len(dims))):
        return self
    return g.op("Transpose", self, perm_i=dims)


@_onnx_symbolic("aten::view")
@symbolic_helper.quantized_args(True)
def view(g: jit_utils.GraphContext, self, size):
    return reshape(g, self, size)


@_onnx_symbolic("aten::view_as")
def view_as(g: jit_utils.GraphContext, self, other):
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unsafe_chunk", 9, 11, "Dynamic number of outputs not supported", self
        )
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        return symbolic_helper._unimplemented(
            "unsafe_chunk", "unknown dimension size", self
        )
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split", 9, 11, "Dynamic number of outputs not supported", self
        )
    split_val = symbolic_helper._node_get(split_size_or_sizes.node(), "value")
    if split_val.dim() > 0:
        return split_with_sizes(g, self, split_size_or_sizes, dim, _outputs)
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")

    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "split", 9, 11, "Unknown dimension size not supported", self
            )
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    # pyrefly: ignore [bad-argument-type]
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::unsafe_split")
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None
):
    return split(g, self, split_size_or_sizes, dim, _outputs)


@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "is", "i", "i")
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split_with_sizes", 9, 11, "Dynamic number of outputs not supported", self
        )
    # pyrefly: ignore [bad-argument-type]
    return g.op("Split", self, split_i=split_sizes, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::unsafe_split_with_sizes")
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None
):
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unbind", 9, 11, "Dynamic number of outputs not supported", self
        )

    outputs = g.op("Split", self, split_i=[1] * _outputs, axis_i=dim, outputs=_outputs)
    outputs = [outputs] if _outputs == 1 else outputs
    squeezed_outputs = [
        symbolic_helper._squeeze_helper(g, out, [dim]) for out in outputs
    ]
    return squeezed_outputs


@_onnx_symbolic("aten::select")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
def select(g: jit_utils.GraphContext, self, dim, index):
    """Implement the select functionality for a pytorch tensor in ONNX.

    Selects elements from the input tensor along the specified `dim` dimension based on the `index` tensor.
    """
    index = symbolic_helper._maybe_get_scalar(index)
    if (not symbolic_helper._is_value(index)) and (index < 0):
        if index == -1:
            end_index = _constants.INT64_MAX
        else:
            end_index = index + 1
        slice_node = symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[index], ends=[end_index]
        )
        return symbolic_helper._squeeze_helper(g, slice_node, [dim])
    else:
        # FIXME(justinchuby): can index be an int and not a value?
        return g.op("Gather", self, index, axis_i=dim)


@_onnx_symbolic("aten::square")
def square(g: jit_utils.GraphContext, self):
    return g.op("Mul", self, self)


@_onnx_symbolic("aten::squeeze")
def squeeze(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    squeeze_dim = symbolic_helper._get_const(dim, "i", "dim")
    # Handle negative dims
    if squeeze_dim < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            warnings.warn(
                "ONNX export squeeze with negative axis "
                + str(squeeze_dim)
                + " might cause the onnx model to be incorrect. "
                + "Negative axis is not supported in ONNX. "
                + "Axis is converted to "
                + str(squeeze_dim + rank)
                + " based on input shape at export time. "
                + "Passing an tensor of different rank in execution will be incorrect.",
                stacklevel=2,
            )
            squeeze_dim += rank
        else:
            return symbolic_helper._unimplemented(
                "squeeze", "negative axis with unknown input rank", self
            )

    dim_size = symbolic_helper._get_tensor_dim_size(self, squeeze_dim)
    if dim_size is None:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(squeeze_dim)
            + " on an input "
            + "with unknown shape. Note that if the size of dimension "
            + str(squeeze_dim)
            + " of the input "
            + "is not 1, the ONNX model will return an error. Opset version 11 supports squeezing on "
            + "non-singleton dimensions, it is recommended to export this model using opset "
            + "version 11 or higher.",
            stacklevel=2,
        )
        return symbolic_helper._squeeze_helper(g, self, axes_i=[squeeze_dim])
    if dim_size > 1:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(squeeze_dim)
            + ". The size of "
            + "this dimension in the given input is "
            + str(dim_size)
            + ". The model will "
            + "be exported without the squeeze node. If the model is intended to be used with dynamic "
            + "input shapes, please use opset version 11 to "
            + "export the model.",
            stacklevel=2,
        )
        return self

    warnings.warn(
        "This model contains a squeeze operation on dimension "
        + str(squeeze_dim)
        + ". If the model is "
        + "intended to be used with dynamic input shapes, please use opset version 11 to export the model.",
        stacklevel=2,
    )
    return symbolic_helper._squeeze_helper(g, self, axes_i=[squeeze_dim])


@_onnx_symbolic("aten::prelu")
def prelu(g: jit_utils.GraphContext, self, weight):
    self_rank = symbolic_helper._get_tensor_rank(self)
    weight_sizes = symbolic_helper._get_tensor_sizes(weight)
    weight_rank = len(weight_sizes)
    if self_rank is not None:
        if self_rank > 2:
            # make weight unidirectional broadcastable
            weight = symbolic_helper._unsqueeze_helper(
                g, weight, list(range(1, self_rank - 1))
            )
        elif self_rank == 0 and weight_sizes == [1]:
            # self and weight are both scalar but weight has rank == 1, squeeze weight.
            weight = symbolic_helper._squeeze_helper(g, weight, [0])
            weight_rank = 0

    if self_rank is not None and weight_rank is not None:
        assert self_rank >= weight_rank, (
            f"rank(x) should be >= rank(slope) but got {self_rank} < {weight_rank}"
        )
    return g.op("PRelu", self, weight)


@_onnx_symbolic("aten::silu")
def silu(g: jit_utils.GraphContext, input):
    return g.op("Mul", input, g.op("Sigmoid", input))


@_onnx_symbolic("aten::mish")
def mish(g: jit_utils.GraphContext, input):
    return g.op("Mul", input, g.op("Tanh", g.op("Softplus", input)))


@_onnx_symbolic("aten::relu")
@symbolic_helper.quantized_args(True)
def relu(g: jit_utils.GraphContext, input):
    return symbolic_helper._op_with_optional_float_cast(
        g, "Relu", input, opset_before=14
    )


@_onnx_symbolic("aten::relu6")
@symbolic_helper.quantized_args(True)
def relu6(g: jit_utils.GraphContext, input):
    return clamp(g, input, 0, 6)


@_onnx_symbolic("aten::ceil")
def ceil(g: jit_utils.GraphContext, input):
    return g.op("Ceil", input)


@_onnx_symbolic("aten::floor")
def floor(g: jit_utils.GraphContext, input):
    return g.op("Floor", input)


@_onnx_symbolic("aten::len")
def _len(g: jit_utils.GraphContext, self):
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


@_onnx_symbolic("aten::threshold")
@symbolic_helper.parse_args("v", "t", "t")
def threshold(g: jit_utils.GraphContext, self, threshold, value):
    # See Note [Export inplace]
    if symbolic_helper._scalar(threshold) != 0:
        return symbolic_helper._unimplemented("threshold", "non-zero threshold", self)
    if symbolic_helper._scalar(value) != 0:
        return symbolic_helper._unimplemented("threshold", "non-zero value", self)
    return g.op("Relu", self)


@_onnx_symbolic("aten::leaky_relu")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "b")
def leaky_relu(
    g: jit_utils.GraphContext,
    input: _C.Value,
    negative_slope: float,
    inplace: bool = False,
):
    # See Note [Export inplace]
    return g.op("LeakyRelu", input, alpha_f=negative_slope)


@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
def glu(g: jit_utils.GraphContext, input, dim):
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        assert dim_size % 2 == 0

    first, second = g.op("Split", input, axis_i=dim, outputs=2)
    return g.op("Mul", first, g.op("Sigmoid", second))


@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
def softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    # Softmax does normalization at vector level.
    # PyTorch and ONNX use different strategies to split the input tensor into vectors.
    # Thus dim and axis have different meanings.
    # PyTorch slices the input tensor into vectors along the `dim`-th dimension.
    # ONNX reshapes the input into a 2-D tensor, and `axis` indicates where the input is coerced.
    # If input is a 2 x 3 tensor:
    # input = [[1.0, 1.0, 1.0],
    #          [1.0, 1,0, 1,0]]
    # with dim = 0, the result is:
    # result = [[0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5]]
    # with axis = 0, the result is:
    # result = [[0.167, 0.167, 0.167],
    #           [0.167, 0.167, 0.167]]
    # So only when dim and axis both equal to ndim - 1 (the last dimension),
    # their semantics are equivalent.
    # So use softmax when dim and axis both equal to ndim - 1,
    # otherwise transpose the input to put the vectors to be normalized to the last dimension.
    # When input rank is not known at export time we compute softmax using a subgraph
    # with other operators
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is not None:
        # TODO: remove this as onnx opset 11 spec allows negative axes
        if dim < 0:
            dim = input_dim + dim

        is_transpose_required = input_dim != dim + 1

        if is_transpose_required:
            axes = list(range(input_dim))
            axes[dim], axes[-1] = axes[-1], axes[dim]
            input = g.op("Transpose", input, perm_i=axes)
            dim = input_dim - 1

        softmax = g.op("Softmax", input, axis_i=dim)
        if dtype and dtype.node().kind() != "prim::Constant":
            parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
            softmax = g.op(
                "Cast",
                softmax,
                to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type(),
            )

        if is_transpose_required:
            softmax = g.op("Transpose", softmax, perm_i=axes)  # type: ignore[possibly-undefined]
        return softmax

    # Apply max normalization.
    input = g.op("Sub", input, g.op("ReduceMax", input, axes_i=[dim], keepdims_i=1))

    exp = g.op("Exp", input)
    sum = symbolic_helper._reducesum_helper(g, exp, axes_i=[dim])
    softmax = g.op("Div", exp, sum)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        softmax = g.op(
            "Cast", softmax, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    return softmax


@_onnx_symbolic("aten::softplus")
def softplus(g: jit_utils.GraphContext, self, beta, threshold):
    beta_const = symbolic_helper._maybe_get_const(beta, "f")
    if beta_const != 1:
        return g.op("Div", g.op("Softplus", g.op("Mul", self, beta)), beta)
    return g.op("Softplus", self)


@_onnx_symbolic("aten::get_pool_ceil_padding")
def get_pool_ceil_padding(input, kernel_size, stride, padding):
    # TODO(justinchuby): Looks like this op is deprecated in torch
    sizes = symbolic_helper._get_tensor_sizes(input)
    dim = sizes[-len(padding) :] if sizes is not None else None
    if dim is None or any(i is None for i in dim):
        return symbolic_helper._unimplemented(
            "get_pool_ceil_padding", "input size not accessible", input
        )
    ceiled_output_dim = [
        math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i])) + 1
        for i in range(len(padding))
    ]
    # ensure last pooling starts inside
    ceiled_output_dim = [
        (
            ceiled_output_dim[i] - 1
            if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
            else ceiled_output_dim[i]
        )
        for i in range(len(ceiled_output_dim))
    ]
    padding_ceil = [
        (
            0
            if (stride[i] == 1)
            else (
                kernel_size[i]
                - (
                    dim[i]
                    + 2 * padding[i]
                    - ((ceiled_output_dim[i] - 1) * stride[i] + 1)
                )
            )
        )
        for i in range(len(padding))
    ]
    # ensure padding is not > kernel_size
    padding_ceil = [
        (
            (
                int(padding_ceil[i])
                if padding_ceil[i] < kernel_size[i] - 1
                else int(kernel_size[i] - 1)
            )
            if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
            else int(padding_ceil[i])
        )
        for i in range(len(padding_ceil))
    ]
    return padding_ceil


@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d", torch.nn.modules.utils._single, 1, return_indices=False
        ),
        _export("max_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d", torch.nn.modules.utils._pair, 2, return_indices=False
        ),
        _export("max_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d", torch.nn.modules.utils._triple, 3, return_indices=False
        ),
        _export("max_pool3d"),
    ],
)
def _max_pool(name, tuple_fn, ndims, return_indices):
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if set(tuple_fn(dilation)) != {1}:
            return symbolic_helper._unimplemented(name, "dilation", input)
        if not stride:
            stride = kernel_size
        padding = tuple(tuple_fn(padding))
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            padding = padding + tuple(a + b for (a, b) in zip(padding_ceil, padding))
        else:
            padding = padding * 2
        kwargs = {
            "kernel_shape_i": tuple_fn(kernel_size),
            "pads_i": padding,
            "strides_i": tuple_fn(stride),
        }
        # easy but hacky way to get flattened indices values
        # to be used to convert the indices values to non-flattened.
        # In ONNX the indices are computed as a flatten 1-D tensor,
        # so the values in indices are in [0, N x C x D1 x ... x Dn).
        # To convert the indices to the same format used by Pytorch,
        # we first execute a maxpool with a kernel and stride of 1 on the same input.
        # This will result in a tensor of indices in which each index will have it's own value.
        # Using this tensor as a reference, we extract the first index of each axis and subtract
        # it from each index of this axis in the indices to convert.
        # This step will result in a tensor were each dimension has values of indices within
        # the dimension it is in.
        # For more information :
        # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
        if return_indices:
            r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
            _, flattened_indices = g.op(
                "MaxPool",
                input,
                outputs=2,
                kernel_shape_i=[1 for _ in range(ndims)],
                strides_i=[1 for _ in range(ndims)],
            )
            # convert indices to have non-flattened indices values
            s = symbolic_helper._slice_helper(
                g,
                flattened_indices,
                axes=[2 + i for i in range(ndims)],
                starts=list(tuple_fn(0)),
                ends=list(tuple_fn(1)),
            )
            indices = sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d_with_indices = _onnx_symbolic("aten::max_pool1d_with_indices")(
    _max_pool(
        "max_pool1d_with_indices",
        torch.nn.modules.utils._single,
        1,
        return_indices=True,
    )
)
max_pool2d_with_indices = _onnx_symbolic("aten::max_pool2d_with_indices")(
    _max_pool(
        "max_pool2d_with_indices",
        torch.nn.modules.utils._pair,
        2,
        return_indices=True,
    )
)
max_pool3d_with_indices = _onnx_symbolic("aten::max_pool3d_with_indices")(
    _max_pool(
        "max_pool3d_with_indices",
        torch.nn.modules.utils._triple,
        3,
        return_indices=True,
    )
)


@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[
        symbolic_helper._apply_params("avg_pool1d", torch.nn.modules.utils._single),
        _export("avg_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[
        symbolic_helper._apply_params("avg_pool2d", torch.nn.modules.utils._pair),
        _export("avg_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[
        symbolic_helper._apply_params("avg_pool3d", torch.nn.modules.utils._triple),
        _export("avg_pool3d"),
    ],
)
def _avg_pool(name, tuple_fn):
    @symbolic_helper.quantized_args(True)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: int | Sequence[int],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        if not stride:
            stride = kernel_size
        padding = symbolic_helper._avgpool_helper(
            tuple_fn, padding, kernel_size, stride, divisor_override, name
        )
        assert isinstance(padding, tuple)
        adjusted_padding = padding
        # Although onnx::AvgPool provides count_include_pad,
        # The corner case of Average Pooling with ceil_mode on
        # PyTorch allows sliding window go off bound, which leads to
        # this accommodation.
        # More detail on https://github.com/pytorch/pytorch/issues/57178
        if count_include_pad:
            input = symbolic_helper._op_with_optional_float_cast(
                g,
                "Pad",
                input,
                pads_i=((0,) * 2 + padding) * 2,
                mode_s="constant",
                value_f=0.0,
                opset_before=11,
            )
            adjusted_padding = (0,) * len(padding)
        if ceil_mode:
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            adjusted_padding = adjusted_padding + tuple(
                a + b for (a, b) in zip(padding_ceil, adjusted_padding)
            )
        else:
            adjusted_padding = adjusted_padding * 2
        output = g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=adjusted_padding,
        )
        return output

    return symbolic_fn


@_onnx_symbolic(
    "aten::adaptive_avg_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool1d", "AveragePool", torch.nn.modules.utils._single
        ),
        _export("adaptive_avg_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_avg_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool2d", "AveragePool", torch.nn.modules.utils._pair
        ),
        _export("adaptive_avg_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_avg_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool3d", "AveragePool", torch.nn.modules.utils._triple
        ),
        _export("adaptive_avg_pool3d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_max_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool1d",
            "MaxPool",
            torch.nn.modules.utils._single,
            max_pool1d_with_indices,
        ),
        _export("adaptive_max_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_max_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool2d",
            "MaxPool",
            torch.nn.modules.utils._pair,
            max_pool2d_with_indices,
        ),
        _export("adaptive_max_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::adaptive_max_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool3d",
            "MaxPool",
            torch.nn.modules.utils._triple,
            max_pool3d_with_indices,
        ),
        _export("adaptive_max_pool3d"),
    ],
)
def _adaptive_pool(name, type, tuple_fn, fn=None):
    @symbolic_helper.quantized_args(True, False)
    def symbolic_fn(g, input, output_size):
        # _adaptive_pool is supported for cases where output_size is 1 for all dimensions,
        # by executing a GlobalPool.
        # It is also supported for cases where the output size is a factor of the input size.
        # For these cases the stride and kernel size are uniform along all the indices of
        # the same dimension, which makes it possible to export it to ONNX.
        # for MaxPool, GlobalMaxPool does not return indices,
        # so we try using max_poolxd_with_indices, and if it is not possible
        # (input is not a complete tensor or output size not factor of input size)
        # then we call GlobalAveragePool and return None for the indices
        output_size_value = output_size
        try:
            output_size = symbolic_helper._parse_arg(output_size, "is")
        except Exception:
            # FIXME(justinchuby): Avoid catching Exception.
            # Catch a more specific exception instead.
            return symbolic_helper._onnx_unsupported(
                "adaptive pooling, since output_size is not constant.", input
            )
        if output_size == [1] * len(output_size) and type == "AveragePool":
            return g.op("GlobalAveragePool", input)
        sizes = symbolic_helper._get_tensor_sizes(input)
        try:
            dim = sizes[2:]
        except Exception:
            # FIXME(justinchuby): Avoid catching Exception.
            # Catch a more specific exception instead.
            dim = None
        if dim is None or any(i is None for i in dim):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "input size not accessible", input
            )
        # verify if output size % input size = 0 for all dim
        mod = [dim[i] % output_size[i] for i in range(len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "output size that are not factor of input size", output_size_value
            )
        k = [int(dim[i] / output_size[i]) for i in range(len(dim))]
        # call max_poolxd_with_indices to get indices in the output
        if type == "MaxPool":
            # pyrefly: ignore [not-callable]
            return fn(g, input, k, k, (0,) * len(dim), (1,) * len(dim), False)
        output = g.op(type, input, kernel_shape_i=tuple_fn(k), strides_i=tuple_fn(k))
        return output

    return symbolic_fn


def _prepare_onnx_paddings(dim: int, pad):
    """Generate paddings in ONNX order based on pad in pytorch.
    Args:
        dim: the dimension of the tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ...
    """
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # assume zero-dimensions in the beginning
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    return paddings


def _convert_padding_node(input):
    padding = symbolic_helper._maybe_get_const(input, "is")
    if symbolic_helper._is_value(padding) and symbolic_helper._is_packed_list(padding):
        input_list = symbolic_helper._unpack_list(padding)
        try:
            padding = [
                symbolic_helper._get_const(v, "i", "padding") for v in input_list
            ]
        except Exception:
            # FIXME(justinchuby): Avoid catching Exception.
            # Catch a more specific exception instead.
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "Pad", 9, 11, "The sizes of the padding must be constant", input
            )
    return padding


@_onnx_symbolic("aten::constant_pad_nd")
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value):
    mode = "constant"
    try:
        value = symbolic_helper._get_const(value, "f", "value")
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Pad", 9, 11, "The value for the padding must be constant", value
        )

    padding = _convert_padding_node(padding)
    # pyrefly: ignore [bad-argument-type]
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, value_f=value, opset_before=11
    )


def _pad_circular(g: jit_utils.GraphContext, input: _C.Value, pad: _C.Value):
    padding = _convert_padding_node(pad)
    assert len(padding) % 2 == 0
    ndim = len(padding) // 2

    cur = input
    for idx in range(ndim):
        pad_r = padding[-(2 * idx + 1)]
        pad_l = padding[-(2 * idx + 2)]
        tensors = []
        if pad_l > 0:
            left = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[-(pad_l)], ends=[_constants.INT64_MAX]
            )
            tensors.append(left)

        if pad_l < 0 or pad_r < 0:
            start = builtins.max(0, -pad_l)
            end = -(builtins.max(0, -pad_r))
            middle = symbolic_helper._slice_helper(
                g,
                cur,
                axes=[2 + idx],
                starts=[start],
                ends=[end],
            )
            tensors.append(middle)
        else:
            tensors.append(cur)

        if pad_r > 0:
            right = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[0], ends=[pad_r]
            )
            tensors.append(right)

        cur = g.op("Concat", *tensors, axis_i=(2 + idx))

    return cur


@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    mode = "reflect"
    padding = _convert_padding_node(padding)
    # pyrefly: ignore [bad-argument-type]
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
def replication_pad(g: jit_utils.GraphContext, input, padding):
    mode = "edge"
    padding = _convert_padding_node(padding)
    # pyrefly: ignore [bad-argument-type]
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


@_onnx_symbolic("aten::pad")
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
):
    mode = symbolic_helper._parse_arg(mode, "s")
    if mode == "replicate":
        return replication_pad(g, input, pad)
    elif mode == "reflect":
        return reflection_pad(g, input, pad)
    elif mode == "constant":
        return constant_pad_nd(g, input, pad, value)
    elif mode == "circular":
        return _pad_circular(g, input, pad)
    else:
        raise errors.SymbolicValueError(f"Unrecognized padding mode {mode}", input)


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest"),
        _export("upsample_nearest1d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest"),
        _export("upsample_nearest2d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest"),
        _export("upsample_nearest3d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[
        symbolic_helper._apply_params("upsample_linear1d", 3, "linear"),
        _export("upsample_linear1d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[
        symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear"),
        _export("upsample_bilinear2d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[
        symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear"),
        _export("upsample_trilinear3d"),
    ],
)
def _interpolate(name: str, dim: int, interpolate_mode: str):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        return g.op("Upsample", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


@_onnx_symbolic("aten::__interpolate")
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Upsample", input, scales, mode_s=mode)


@_onnx_symbolic("aten::bitwise_not")
def bitwise_not(g: jit_utils.GraphContext, input):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise Not "
            "for non-boolean input values",
            input,
        )
    return g.op("Not", input)


@_onnx_symbolic("aten::bitwise_or")
def bitwise_or(g, self, other):
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values. self: ",
            self,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values. other: ",
            other,
        )
    return g.op("Or", self, other)


def wrap_logical_op_with_cast_to(to_type):
    def decorator(fn):
        @functools.wraps(fn)
        def wrap_with_cast(g, input, other):
            to_cast_func = globals()[f"_cast_{to_type}"]
            return fn(g, to_cast_func(g, input, False), to_cast_func(g, other, False))

        return wrap_with_cast

    return decorator


def wrap_logical_op_with_negation(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrap_with_not(g, input, other):
        return g.op("Not", func(g, input, other))

    return wrap_with_not


@_onnx_symbolic("aten::__not_")
def __not_(g: jit_utils.GraphContext, self):
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise Not "
            "for non-boolean input values",
            self,
        )
    return g.op("Not", self)


@_onnx_symbolic("aten::eq")
@symbolic_helper.quantized_args(True, True)
def eq(g: jit_utils.GraphContext, self, other):
    if isinstance(self.type(), _C.DeviceObjType) and isinstance(
        other.type(), _C.DeviceObjType
    ):
        # ONNX doesn't have devices, so consider them all to be equal.
        # The no-op check for equality will get constant-folded.
        return g.op("Constant", value_t=torch.tensor(True, dtype=torch.bool))
    self_node = self.node()
    other_node = other.node()
    if self_node.kind() == other_node.kind() == "onnx::Constant":
        if self_node.kindOf("value") == other_node.kindOf("value") == "s":
            # Exporting strings to ONNX is not supported.
            # If both strings are constant, we can compare them directly.
            # The no-op check for equality will get constant-folded.
            return g.op(
                "Constant",
                value_t=torch.tensor(
                    self_node.s("value") == other_node.s("value"),
                    dtype=torch.bool,
                ),
            )

    return g.op("Equal", self, other)


@_onnx_symbolic("aten::ne")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
def ne(g: jit_utils.GraphContext, self, other):
    return eq(g, self, other)


@_onnx_symbolic("aten::gt")
@symbolic_helper.quantized_args(True, True)
def gt(g: jit_utils.GraphContext, input, other):
    return _gt_impl(g, input, other)


def _gt_impl(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op("Greater", input, other)


@_onnx_symbolic("aten::lt")
@symbolic_helper.quantized_args(True, True)
def lt(g: jit_utils.GraphContext, input, other):
    return _lt_impl(g, input, other)


def _lt_impl(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.INT32)
    return g.op("Less", input, other)


@_onnx_symbolic("aten::ge")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
def ge(g: jit_utils.GraphContext, input, other):
    return _lt_impl(g, input, other)


@_onnx_symbolic("aten::le")
@symbolic_helper.quantized_args(True, True)
@wrap_logical_op_with_negation
def le(g: jit_utils.GraphContext, input, other):
    return _gt_impl(g, input, other)


@_onnx_symbolic("aten::__and_")
def __and_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise AND "
            "for non-boolean input values",
            input,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise AND "
            "for non-boolean input values",
            other,
        )
    return g.op("And", input, other)


@_onnx_symbolic("aten::__or_")
def __or_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values",
            input,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values",
            other,
        )
    return g.op("Or", input, other)


@_onnx_symbolic("aten::__xor_")
def __xor_(g: jit_utils.GraphContext, input, other):
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise XOR "
            "for non-boolean input values",
            input,
        )
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise XOR "
            "for non-boolean input values",
            other,
        )
    return g.op("Xor", input, other)


@_onnx_symbolic("aten::logical_and")
@wrap_logical_op_with_cast_to("Bool")
def logical_and(g: jit_utils.GraphContext, input, other):
    return g.op("And", input, other)


@_onnx_symbolic("aten::logical_or")
@wrap_logical_op_with_cast_to("Bool")
def logical_or(g: jit_utils.GraphContext, input, other):
    return g.op("Or", input, other)


@_onnx_symbolic("aten::logical_xor")
@wrap_logical_op_with_cast_to("Bool")
def logical_xor(g: jit_utils.GraphContext, input, other):
    return g.op("Xor", input, other)


@_onnx_symbolic("aten::logical_not")
def logical_not(g: jit_utils.GraphContext, input):
    return g.op("Not", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.BOOL))


@_onnx_symbolic("aten::__rshift_")
def __rshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    self_scalar_type = _type_utils.JitScalarType.from_value(self)
    if (
        _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED)
        != self_scalar_type
    ):
        other = g.op(
            "Cast",
            other,
            to_i=self_scalar_type.onnx_type(),
        )

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=self_scalar_type.onnx_type(),
    )
    rshift = g.op("Div", self, two_pow)
    return rshift


@_onnx_symbolic("aten::__lshift_")
def __lshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    self_scalar_type = _type_utils.JitScalarType.from_value(self)
    if (
        _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED)
        != self_scalar_type
    ):
        other = g.op(
            "Cast",
            other,
            to_i=self_scalar_type.onnx_type(),
        )

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=self_scalar_type.onnx_type(),
    )
    lshift = g.op("Mul", self, two_pow)
    return lshift


@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
def where(g: jit_utils.GraphContext, condition, self=None, other=None, _outputs=None):
    # Assumes that torch.where's first argument takes only Bool and Byte tensors.
    if not symbolic_helper._is_bool(condition):
        condition = g.op("Cast", condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    if self is None:
        condition = nonzero(g, condition)
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    # pyrefly: ignore [bad-argument-type]
    return g.op("Where", condition, self, other)


@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    # PyTorch dim and ONNX axis have different meanings.
    # See Softmax comment for details.
    # TODO: remove this as onnx opset 11 spec allows negative axes
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )
    if dim < 0:
        dim = input_dim + dim
    is_transpose_required = input_dim != dim + 1
    # ONNX only supports log_softmax with dim = -1. Transpose must be added before and after log_softmax to support other cases.
    if is_transpose_required:
        axes = list(range(input_dim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        input = g.op("Transpose", input, perm_i=axes)
        dim = input_dim - 1
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return_op = g.op(
            "Cast", return_op, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    if is_transpose_required:
        return_op = g.op("Transpose", return_op, perm_i=axes)  # type: ignore[possibly-undefined]
    return return_op


@_onnx_symbolic("aten::_log_softmax")
@symbolic_helper.parse_args("v", "i", "i")
def _log_softmax(g: jit_utils.GraphContext, input, dim, half_to_float):
    if (
        half_to_float
        and _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.UNDEFINED
        )
        == _type_utils.JitScalarType.HALF
    ):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return log_softmax(g, input, dim)


@_onnx_symbolic("aten::_convolution")
@symbolic_helper.parse_args(
    "v", "v", "v", "is", "is", "is", "i", "is", "i", "i", "i", "i", "i"
)
def _convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32=None,
):
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        kernel_shape = None

    if kernel_shape is None or any(i is None for i in kernel_shape):
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of convolution for kernel of unknown shape.",
            input,
        )

    args = [input, weight]
    # ONNX only supports 1D bias
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) == 1
    ):
        args.append(bias)

    kwargs = {
        "kernel_shape_i": weight_size[2:],
        "strides_i": stride,
        # NB: ONNX supports asymmetric padding, whereas PyTorch supports only
        # symmetric padding
        "pads_i": padding + padding,
        "dilations_i": dilation,
        "group_i": groups,
    }

    if any(o != 0 for o in output_padding):
        # ONNX supports both output_shape and output_padding. they are equivalent expressive.
        # output_padding is more straightforward, so we use it here.
        # output_shape = stride * (input_shape - 1) + output_padding + kernel_shape - padding * 2
        assert transposed
        assert len(stride) == len(output_padding)
        kwargs["output_padding_i"] = output_padding

    n = g.op("ConvTranspose" if transposed else "Conv", *args, **kwargs)

    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) != 1
    ):
        return g.op("Add", n, bias)
    else:
        return n


@_onnx_symbolic("aten::_convolution_mode")
@symbolic_helper.parse_args(
    "v",
    "v",
    "v",
    "is",
    "s",
    "is",
    "i",
)
def _convolution_mode(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
):
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        kernel_shape = weight_size[2:]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        kernel_shape = None

    if kernel_shape is None or any(i is None for i in kernel_shape):
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of convolution for kernel of unknown shape.",
            input,
        )

    args = [input, weight]
    # ONNX only supports 1D bias
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) == 1
    ):
        args.append(bias)

    if padding == "valid":
        padding = "VALID"
    elif padding == "same":
        padding = "SAME_UPPER"
    kwargs = {
        "kernel_shape_i": weight_size[2:],
        "strides_i": stride,
        "auto_pad_s": padding,
        "dilations_i": dilation,
        "group_i": groups,
    }

    # pyrefly: ignore [bad-argument-type]
    n = g.op("Conv", *args, **kwargs)

    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) != 1
    ):
        return g.op("Add", n, bias)
    else:
        return n


@_onnx_symbolic("aten::convolution")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is", "i")
def convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::conv1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
def conv1d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    str_padding = symbolic_helper._parse_arg(padding, "s")
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


@_onnx_symbolic("aten::conv2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
def conv2d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    str_padding = symbolic_helper._parse_arg(padding, "s")
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


@_onnx_symbolic("aten::conv3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
def conv3d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    str_padding = symbolic_helper._parse_arg(padding, "s")
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


@_onnx_symbolic("aten::conv_transpose1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose1d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::conv_transpose2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose2d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::conv_transpose3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
def conv_transpose3d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


@_onnx_symbolic("aten::batch_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
):
    symbolic_helper.check_training_mode(training, "batch_norm")

    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            9,
            15,
            "All input tensors must have the same `dtype`."
            " Turn off Autocast or export using opset version 15.",
            input,
        )

    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    out = g.op(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        momentum_f=1 - momentum,
        outputs=1 if not training else 5,
    )
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var, saved_mean, saved_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        saved_mean.setDebugName("batch_norm_dead_output-" + saved_mean.debugName())
        saved_var.setDebugName("batch_norm_dead_output-" + saved_var.debugName())
        return res


@_onnx_symbolic("aten::native_layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f")
def native_layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
) -> tuple[_C.Value, _C.Value, _C.Value]:
    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = symbolic_helper._generate_wrapped_number(g, 2.0)
    eps_cst = symbolic_helper._generate_wrapped_number(g, eps)

    if g.opset < 18:
        mean = g.op("ReduceMean", input, axes_i=axes)
    else:
        mean = g.op(
            "ReduceMean",
            input,
            g.op("Constant", value_t=torch.tensor(axes, dtype=torch.long)),
        )

    numerator = sub(g, input, mean)

    # Cast it to eps dtype to avoid precision loss
    is_type_half = (
        _type_utils.JitScalarType.from_value(numerator)
        == _type_utils.JitScalarType.HALF
    )
    if is_type_half:
        eps_dtype = _type_utils.JitScalarType.from_value(eps_cst)
        numerator = g.op(
            "Cast", numerator, to_i=_type_utils.JitScalarType(eps_dtype).onnx_type()
        )

    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the layer_norm formula
    if g.opset < 18:
        # pyrefly: ignore [no-matching-overload]
        variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
    else:
        variance = g.op(
            "ReduceMean",
            # pyrefly: ignore [no-matching-overload]
            pow(g, numerator, two_cst),
            g.op("Constant", value_t=torch.tensor(axes, dtype=torch.long)),
        )

    denominator = sqrt(g, g.op("Add", variance, eps_cst))
    normalized = g.op("Div", numerator, denominator)

    # Cast back to input type as eps related ops are all done
    if is_type_half:
        input_dtype = _type_utils.JitScalarType.from_value(input)
        normalized = g.op(
            "Cast", normalized, to_i=_type_utils.JitScalarType(input_dtype).onnx_type()
        )

    if not (weight is None or symbolic_helper._is_none(weight)):
        normalized = mul(g, normalized, weight)
    if not (bias is None or symbolic_helper._is_none(bias)):
        normalized = add(g, normalized, bias)

    # rdenominator := 1 / sqrt(variance + eps)
    # According to aten::native_layer_norm, rdenominator should have the same dtype as input,
    # mean and normalized, so we need to Cast it back
    if is_type_half:
        denominator = g.op(
            "Cast",
            denominator,
            to_i=_type_utils.JitScalarType(input_dtype).onnx_type(),  # type: ignore[possibly-undefined]
        )
        rdenominator = g.op("Reciprocal", denominator)
    else:
        rdenominator = reciprocal(g, denominator)

    return normalized, mean, rdenominator


@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "b")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
) -> _C.Value:
    normalized, _, _ = native_layer_norm(g, input, normalized_shape, weight, bias, eps)
    return normalized


@_onnx_symbolic("aten::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "b", "f", "f", "b")
def instance_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    use_input_stats: bool,
    momentum: Number,
    eps: Number,
    cudnn_enabled: bool,
):
    symbolic_helper.check_training_mode(use_input_stats, "instance_norm")
    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    if weight is None or symbolic_helper._is_none(weight):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm for unknown channel size.",
                input,
            )
        weight_value = torch.tensor(
            [1.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or symbolic_helper._is_none(bias):
        if channel_size is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm for unknown channel size.",
                input,
            )
        bias_value = torch.tensor(
            [0.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        bias = g.op("Constant", value_t=bias_value)
    if (
        running_mean is None
        or symbolic_helper._is_none(running_mean)
        or running_var is None
        or symbolic_helper._is_none(running_var)
    ):
        return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)
    else:
        input_size = symbolic_helper._get_tensor_sizes(input)
        # If input shape is [N, C, H, W], reshape to [1, N * C, H, W] and call batch_norm.
        # For more information instance_norm():
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L542
        input_size_reshape = input_size.copy()
        n = input_size[0]
        if n is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm training for unknown "
                "batch size.",
                input,
            )
        c = input_size[1]
        input_size_reshape[0] = 1
        input_size_reshape[1] = n * c
        weight_ = repeat(
            g, weight, g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64))
        )
        bias_ = repeat(
            g, bias, g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64))
        )
        running_mean_ = repeat(
            g,
            running_mean,
            g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64)),
        )
        running_var_ = repeat(
            g,
            running_var,
            g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64)),
        )
        input_reshaped = g.op(
            "Reshape",
            input,
            g.op("Constant", value_t=torch.LongTensor(input_size_reshape)),
        )
        out = batch_norm(
            g,
            input_reshaped,
            weight_,
            bias_,
            running_mean_,
            running_var_,
            use_input_stats,
            momentum,
            eps,
            cudnn_enabled,
        )
        return view(g, out, g.op("Constant", value_t=torch.tensor(input_size)))


@_onnx_symbolic("aten::unfold")
@symbolic_helper.parse_args("v", "i", "i", "i")
def unfold(g: jit_utils.GraphContext, input, dimension, size, step):
    sizes = symbolic_helper._get_tensor_sizes(input)
    # FIXME(justinchuby): Get rid of the try catch here to improve readability
    try:
        sizedim = sizes[dimension]
    except Exception:
        # FIXME(justinchuby): Avoid catching Exception.
        # Catch a more specific exception instead.
        sizedim = None
    if sizedim is not None:
        low_indices = range(0, sizedim, step)
        hi_indices = range(size, sizedim + 1, step)
        stack = [
            symbolic_helper._slice_helper(
                g, input, axes=[dimension], starts=[low], ends=[hi]
            )
            for low, hi in zip(low_indices, hi_indices)
        ]
        ndim = len(sizes)
        perm = list(range(ndim))
        perm.append(perm.pop(dimension))
        unsqueeze = [
            symbolic_helper._unsqueeze_helper(
                g, g.op("Transpose", t, perm_i=perm), [dimension]
            )
            for t in stack
        ]
        return g.op("Concat", *unsqueeze, axis_i=dimension)
    else:
        return symbolic_helper._unimplemented(
            "Unfold", "input size not accessible", input
        )


@_onnx_symbolic("aten::elu")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "t", "t", "t")
def elu(g: jit_utils.GraphContext, input, alpha, scale, input_scale):
    if scale and scale != 1.0:
        return symbolic_helper._unimplemented(
            "scale", "does not support scale in Elu", scale
        )
    if input_scale and input_scale != 1.0:
        return symbolic_helper._unimplemented(
            "input_scale", "does not support input_scale in Elu", input_scale
        )
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=symbolic_helper._scalar(alpha))


@_onnx_symbolic("aten::selu")
@symbolic_helper.quantized_args(True)
def selu(g: jit_utils.GraphContext, input):
    return g.op("Selu", input)


@_onnx_symbolic("aten::index_select")
@symbolic_helper.parse_args("v", "i", "v")
def index_select(g: jit_utils.GraphContext, self, dim, index):
    # In case of a scalar index, index_select returns a tensor with the same rank as the input.
    # To match this behavior in ONNX, we make index a 1D tensor so that the following gather
    # also produces a tensor with the same rank as the input.
    return symbolic_helper._select_helper(g, self, dim, index)


@_onnx_symbolic("aten::index_put")
def index_put(g: jit_utils.GraphContext, self, indices_list_value, values, accumulate):
    if symbolic_helper._is_packed_list(indices_list_value):
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]

    accumulate = symbolic_helper._parse_arg(accumulate, "b")

    if len(indices_list) == 0:
        if accumulate:
            return add(g, self, values)
        return values
    symbolic_helper._onnx_opset_unsupported("index_put", 9, 11, self)


@_onnx_symbolic("aten::index_fill")
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, self)
    expanded_value = expand(g, value, expanded_index_shape, None)

    return scatter(g, self, dim, expanded_index, expanded_value)


@_onnx_symbolic("aten::index_copy")
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    _expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    return scatter(g, self, dim, expanded_index, source)


@_onnx_symbolic("aten::bucketize")
@symbolic_helper.parse_args("v", "v", "b", "b")
def bucketize(
    g: jit_utils.GraphContext, self, boundaries, out_int32=False, right=False
):
    out_type = _C_onnx.TensorProtoDataType.INT64
    if out_int32:
        out_type = _C_onnx.TensorProtoDataType.INT32
    # A tensor expanded_boundaries is created such that it
    # contains a copy of boundaries for each element of self.
    new_shape = g.op("Concat", g.op("Shape", boundaries), g.op("Shape", self), axis_i=0)
    # Unsqueeze step is performed to respect ONNX's numpy style broadcasting for comparison ops
    # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    tensor_rank = symbolic_helper._get_tensor_rank(self)
    assert tensor_rank is not None
    unsqueeze_axes = list(range(1, tensor_rank + 1))
    expanded_boundaries = expand(
        g,
        symbolic_helper._unsqueeze_helper(g, boundaries, unsqueeze_axes),
        new_shape,
        None,
    )
    # Compare each element of self to boundaries to get a tensor
    # with leading 1s and trailing 0s.
    # e.g., 4 > [1, 3, 4] = [1, 1, 0]
    # The index of the last 1 is the bucket where the element should go.
    if right:
        cond = ge(g, self, expanded_boundaries)
    else:
        cond = gt(g, self, expanded_boundaries)
    cond_out = g.op("Cast", cond, to_i=out_type)
    # Sum to get the number of 1s corresponding to each element,
    # which is the same as the bucket index.
    # e.g., sum(4 > [1, 3, 4]) = sum([1, 1, 0]) = 2
    return symbolic_helper._reducesum_helper(g, cond_out, axes_i=[0], keepdims_i=0)


@_onnx_symbolic("aten::type_as")
def type_as(g: jit_utils.GraphContext, self, other):
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    other_dtype = symbolic_helper._try_get_scalar_type(other)
    if self_dtype == other_dtype and self_dtype is not None:
        return self
    if other_dtype is not None:
        return g.op(
            "Cast",
            self,
            to_i=other_dtype.onnx_type(),
        )

    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of type_as for tensor "
        "of unknown dtype. Please check if the dtype of the "
        "parameter passed to the type_as function is correct.",
        other,
    )


@_onnx_symbolic("aten::cosine_similarity")
@symbolic_helper.parse_args("v", "v", "i", "f")
def cosine_similarity(g: jit_utils.GraphContext, x1, x2, dim, eps):
    cross = symbolic_helper._reducesum_helper(
        g, mul(g, x1, x2), axes_i=[dim], keepdims_i=0
    )
    x1_l2 = symbolic_helper._reducesum_helper(
        g, mul(g, x1, x1), axes_i=[dim], keepdims_i=0
    )
    x2_l2 = symbolic_helper._reducesum_helper(
        g, mul(g, x2, x2), axes_i=[dim], keepdims_i=0
    )
    div_tens = max(
        g, sqrt(g, mul(g, x1_l2, x2_l2)), g.op("Constant", value_t=torch.tensor([eps]))
    )
    return div(g, cross, div_tens)


@_onnx_symbolic("aten::pairwise_distance")
def pairwise_distance(g: jit_utils.GraphContext, input1, input2, p, eps, keepdim):
    if not symbolic_helper._is_value(eps):
        eps = g.op("Constant", value_t=torch.tensor([eps]))
    inv_p = div(
        g,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.float)),
        add(g, p, eps),
    )
    summation = symbolic_helper._reducesum_helper(
        g,
        # pyrefly: ignore [no-matching-overload]
        pow(g, sub(g, input1, input2), p),
        axes_i=[-1],
        keepdims_i=symbolic_helper._parse_arg(keepdim, "i"),
    )
    # pyrefly: ignore [no-matching-overload]
    return pow(g, summation, inv_p)


@_onnx_symbolic("aten::clone")
# ignore clone operators that are inserted by PyTorch autograd
def clone(g: jit_utils.GraphContext, input, unused_memory_format):
    return input


@_onnx_symbolic("aten::abs")
def abs(g: jit_utils.GraphContext, self):
    return g.op("Abs", self)


@_onnx_symbolic("aten::log")
def log(g: jit_utils.GraphContext, self):
    return g.op("Log", self)


@_onnx_symbolic("aten::log1p")
def log1p(g: jit_utils.GraphContext, self):
    return log(g, add(g, symbolic_helper._if_scalar_type_as(torch.ones(1), self), self))


@_onnx_symbolic("aten::log10")
def log10(g: jit_utils.GraphContext, self):
    _ln10 = 2.30258509299404568401
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor([_ln10])))


@_onnx_symbolic("aten::pow")
def pow(g: jit_utils.GraphContext, self, exponent):
    f_dtype = _type_utils.JitScalarType.from_value(self)
    if not symbolic_helper._is_fp(self):
        f_dtype = _type_utils.JitScalarType.FLOAT
        self = g.op("Cast", self, to_i=f_dtype.onnx_type())
    if not symbolic_helper._is_fp(exponent):
        exponent = g.op(
            "Cast",
            exponent,
            to_i=f_dtype.onnx_type(),
        )
    pow = g.op("Pow", self, exponent)
    return pow


@_onnx_symbolic("aten::clamp")
def clamp(g: jit_utils.GraphContext, self, min, max):
    # min or max may be None that we need to dispatch to
    # Clip separately, as ONNX does not have None syntax
    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        if symbolic_helper._is_constant(min) and symbolic_helper._is_constant(max):
            return symbolic_helper._op_with_optional_float_cast(
                g,
                "Clip",
                self,
                min_f=symbolic_helper._parse_a

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 339 function(s): _export, wrapper, unused, _shape_as_tensor, _reshape_from_tensor, reshape, reshape_as, add, sub, rsub, mul, div, addcmul, _div_rounding_mode, _trunc_divide, _floor_divide, floor_divide, floordiv, true_divide, reciprocal, cat, stack, _list, mm, bmm, matmul, addmm, is_not_none_nor, neg, sqrt


## Key Components

The file contains 19748 words across 6689 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 226531 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

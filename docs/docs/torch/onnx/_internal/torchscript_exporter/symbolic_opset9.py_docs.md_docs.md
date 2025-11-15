# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py_docs.md`
- **Size**: 53,435 bytes (52.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/symbolic_opset9.py`
- **Size**: 226,531 bytes (221.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
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
        # Although o
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/torchscript_exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset14.py_docs.md_docs.md`](./symbolic_opset14.py_docs.md_docs.md)
- [`symbolic_opset18.py_kw.md_docs.md`](./symbolic_opset18.py_kw.md_docs.md)
- [`_experimental.py_kw.md_docs.md`](./_experimental.py_kw.md_docs.md)
- [`onnx_proto_utils.py_docs.md_docs.md`](./onnx_proto_utils.py_docs.md_docs.md)
- [`symbolic_opset13.py_kw.md_docs.md`](./symbolic_opset13.py_kw.md_docs.md)
- [`symbolic_opset12.py_docs.md_docs.md`](./symbolic_opset12.py_docs.md_docs.md)
- [`symbolic_opset16.py_docs.md_docs.md`](./symbolic_opset16.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`symbolic_helper.py_kw.md_docs.md`](./symbolic_helper.py_kw.md_docs.md)
- [`symbolic_opset8.py_docs.md_docs.md`](./symbolic_opset8.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `symbolic_opset9.py_docs.md_docs.md`
- **Keyword Index**: `symbolic_opset9.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

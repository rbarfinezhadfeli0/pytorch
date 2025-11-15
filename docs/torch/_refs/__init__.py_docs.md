# Documentation: __init__.py

## File Metadata
- **Path**: `torch/_refs/__init__.py`
- **Size**: 223137 bytes
- **Lines**: 6872
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, cast, Optional, overload, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch import sym_float, sym_int
from torch._prims_common import (
    BoolLike,
    DeviceLikeType,
    Dim,
    DimsSequenceType,
    DimsType,
    dtype_to_type,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    FloatWithoutSymFloat,
    IntLike,
    is_contiguous_for_memory_format_or_false,
    is_contiguous_or_false,
    is_weakly_lesser_type,
    Number,
    NumberType,
    RealNumberType,
    REDUCTION_OUTPUT_TYPE_KIND,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    TensorOrNumberLikeType,
    TensorSequenceType,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)


# Experimental module containing prototype Python references for existing
#   PyTorch operations.

__all__ = [
    #
    # Elementwise Unary References
    #
    "abs",
    "acos",
    "acosh",
    "asinh",
    "asin",
    "atan",
    "atanh",
    "bitwise_not",
    # "cbrt",  # No corresponding torch operation
    "ceil",
    "conj_physical",
    "cos",
    "cosh",
    "count_nonzero",
    "deg2rad",
    "digamma",
    "erf",
    "erfinv",
    "erfc",
    "exp",
    "expm1",
    "exponential",
    "exp2",
    "fill",
    "fill_",
    "floor",
    "frac",
    "geometric",
    "index_add",
    "index_copy",
    "index_copy_",
    "index_select",
    "index_fill",
    "index_fill_",
    "isfinite",
    "isinf",
    "isposinf",
    "isneginf",
    "isnan",
    "isreal",
    "i0",
    "lerp",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "log_normal",
    "log_softmax",
    "mvlgamma",
    "norm",
    "normal",
    "nan_to_num",
    "neg",
    "positive",
    "rad2deg",
    "reciprocal",
    "round",  # TODO: model kwargs
    "sigmoid",
    "sgn",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "softmax",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trace",
    "trunc",
    #
    # Elementwise Binary References
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "clamp_min",
    "clamp_max",
    "copysign",
    "div",
    "eq",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "ge",
    "gt",
    "heaviside",
    "hypot",
    "igamma",
    "igammac",
    "imag",
    "isclose",
    "lcm",
    # 'ldexp',
    "le",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logsumexp",
    "lt",
    # 'max', # implement with reductions
    "maximum",
    # 'min', # implement with reductions
    "minimum",
    "mul",
    "ne",
    "nextafter",
    # 'polar',  # abs, cos, sin
    "pow",
    "real",
    "rpow",
    "remainder",
    "rsub",
    "rtruediv",
    "rfloordiv",
    "sub",
    "true_divide",
    "trunc_divide",
    "xlogy",
    #
    # Elementwise Ternary References
    #
    "addcdiv",
    "addcmul",
    "clamp",
    #
    # Conditional references
    #
    "masked_fill",
    "masked_fill_",
    "where",
    #
    # Data conversion and movement references
    #
    "clone",
    "copy_to",  # TODO: add OpInfo (or implement .to)
    "item",
    "to",
    #
    # Reduction ops
    #
    "all",
    "amax",
    "amin",
    "any",
    "cumsum",
    "cumprod",
    "mean",
    "dot",
    "vdot",
    "std",
    "std_mean",
    "sum",
    "sum_to_size",
    "prod",
    "var",
    "var_mean",
    #
    # Linear algebra ops
    #
    "addr",
    #
    # View & Shape Ops
    #
    "alias",
    "alias_copy",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "block_diag",
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "cat",
    "chunk",
    "column_stack",
    "conj",
    "constant_pad_nd",
    "contiguous",
    "diag_embed",
    "diag",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "dsplit",
    "dstack",
    "expand",
    "expand_as",
    "expand_copy",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "meshgrid",
    "movedim",
    "narrow",
    "narrow_copy",
    "native_group_norm",
    "native_layer_norm",
    "permute",
    "permute_copy",
    "ravel",
    "repeat",
    "reshape",
    "reshape_as",
    "roll",
    "rot90",
    "rsqrt",
    "split_with_sizes",
    "stack",
    "swap_axes",  # alias for transpose
    "squeeze",
    "squeeze_copy",
    "t",
    "t_copy",
    "T",
    "take_along_dim",
    "tensor_split",
    "transpose",
    "transpose_copy",
    "unbind_copy",
    "unfold",
    "unfold_copy",
    "unsqueeze",
    "unsqueeze_copy",
    "view",
    "view_as",
    "view_copy",
    "vsplit",
    "vstack",
    "view_as_complex",
    "unflatten",
    "unbind",
    "triu",
    "tril",
    "triu_indices",
    "tril_indices",
    #
    # Tensor Creation
    #
    "arange",
    "cauchy",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "ones",
    "ones_like",
    "randn",
    "scalar_tensor",
    "zero",
    "zeros",
    "zeros_like",
    #
    # Test-related functions
    #
    "allclose",
    "equal",
    #
    # Statistical operations
    #
    "bucketize",
    #
    # Misc
    #
    "is_complex",
    "renorm",
    "stft",
    "istft",
]

Tensor = torch.Tensor
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]
aten = torch._ops.ops.aten

# Note that the docstrings for the public methods from this file are in
# torch/_torch_docs.py


def is_noncontiguous_supported(device):
    return device is None or device.type != "hpu"


def handle_noncontiguous_outputs(input_tlist, output):
    device = None
    from torch._subclasses.fake_tensor import FakeTensor

    for t in input_tlist:
        if isinstance(t, FakeTensor):
            device = t.fake_device
            break

    if not is_noncontiguous_supported(device):
        output = output.contiguous()

    return output


def _broadcast_shapes(*_shapes):
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        is_nested_int,
        size_hint,
    )

    backed_so = torch.fx.experimental._config.backed_size_oblivious

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x
        for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    for shape in shapes:
        if not isinstance(shape, Sequence):
            raise RuntimeError(
                "Input shapes should be of type ints, a tuple of ints, or a list of ints, got ",
                shape,
            )

    # Computes common shape
    common_shape: list[Union[int, torch.SymInt]] = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            # NB: handle nested ints specially to avoid invalid guarding on Ne(j0, 1).
            if is_nested_int(shape[idx]):
                # Broadcasting is allowed for (j0, 1) or (j0, j0);
                # not (j0, j1), (j0, 5), etc.
                if is_nested_int(common_shape[idx]) and guard_or_false(
                    shape[idx] == common_shape[idx]
                ):
                    continue
            else:
                # When backed size oblivious is used, we specialize for broadcasting
                # if its the only way to compile the example input.
                # i.e: s0:1, s1:1 ==>
                #           assert s0==s1, no specialization on ==1 or !=1.
                #            The non-broadcast path is picked
                #      s0:1, s1:4 ==>
                #           specialize(s0) to be 1.
                #      s0:4, s1:1 ==>
                #           specialize(s1) to be 1.
                if backed_so:
                    a = size_hint(shape[idx], allow_none=True)
                    b = size_hint(common_shape[idx], allow_none=True)
                    if a == 1 and b != 1:
                        torch._check(shape[idx] == 1)
                    if b == 1 and a != 1:
                        torch._check(common_shape[idx] == 1)
                if guard_or_false(shape[idx] == common_shape[idx]):
                    continue

            if guard_or_false(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]

            if not is_nested_int(shape[idx]) and guard_or_false(shape[idx] == 1):
                # broadcast case .
                continue
            else:
                # If broadcasting is undecided we pick non-broadcast path and add runtime assertion.
                torch._check(
                    common_shape[idx] == shape[idx],
                    lambda: f"Attempting to broadcast a dimension of length {shape[idx]} at {idx}! "
                    f"Mismatching argument at index {arg_idx} had {shape}; but expected shape "
                    f"should be broadcastable to {common_shape}",
                )

    return common_shape


def _maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    # Computes common shape
    common_shape = _broadcast_shapes(
        *(t.shape if isinstance(t, TensorLike) else None for t in args)
    )

    def should_expand(a: ShapeType, b: ShapeType) -> bool:
        from torch.fx.experimental.symbolic_shapes import (
            guard_or_false,
            sym_and,
            sym_or,
        )

        if len(a) != len(b):
            return True

        for x, y in zip(a, b):
            if guard_or_false(x != y):
                # We know they are not the same.
                return True

            # They are the same or we do not know if they are the same or not.
            # 1==1 no-broadcast
            # u0==1 and 1==u0 cases. We broadcast!
            if guard_or_false(sym_and(x == 1, y == 1)):
                pass
            elif guard_or_false(sym_or(x == 1, y == 1)):
                # assume broadcasting.
                return True

            # u0==u1 assume the same, no broadcasting!
            torch._check(
                x == y,
                lambda: "sizes assumed to be the same due to unbacked broadcasting semantics",
            )

        return False

    def __maybe_broadcast(x, shape):
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            if preserve_cpu_scalar_tensors and utils.is_cpu_scalar_tensor(x):
                return x

            if should_expand(x.shape, common_shape):
                return x.expand(common_shape)

            return x
        else:
            raise RuntimeError(
                "Unexpected type when broadcasting: " + str(type(x)) + "!"
            )

    return tuple(__maybe_broadcast(x, common_shape) for x in args)


# Utilities should come BEFORE this import
from torch._decomp import register_decomposition


#
# Elementwise unary references
#

infer_aten_op = object()


# TODO: add type promotion support
def _make_elementwise_unary_reference(
    type_promotion_kind,
    *,
    aten_op=infer_aten_op,
    extra_meta=None,
    exact_dtype=False,
) -> Callable:
    def inner(prim: Callable):
        nonlocal aten_op

        @wraps(prim)
        @out_wrapper(exact_dtype=exact_dtype)
        @elementwise_unary_scalar_wrapper
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a",),
            type_promotion_kind=type_promotion_kind,
        )
        def _ref(a: TensorLikeType) -> TensorLikeType:
            if extra_meta is not None:
                extra_meta(a)

            output = prim(a)
            return handle_noncontiguous_outputs([a], output)

        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, prim.__name__)
        if aten_op is not None:
            register_decomposition(aten_op)(_ref)

        return _ref

    return inner


def _make_alias(fn, name):
    """
    This function defines an alias of another function and sets its __name__ argument.
    It also sets its __module__ argument to the module of the caller.
    Note that when naively doing `alias = fn`, we have that `alias.__name__ == "fn"`, and
    `alias.__module__ == fn.__module__`.
    """

    def _fn(*args, **kwargs):
        return fn(*args, **kwargs)

    _fn.__name__ = name
    _fn.__module__ = inspect.currentframe().f_back.f_globals["__name__"]  # type: ignore[union-attr]
    return _fn


def _make_inplace(fn):
    """
    Given a function with out variant (i.e. using `out_wrapper()), it returns its in-place variant
    See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-do-in-place-operations-work-in-pytorch
    """

    # nb. We use the name of the first argument used in the unary references
    @wraps(fn)
    def _fn(a, *args, **kwargs):
        return fn(a, *args, out=a, **kwargs)

    inplace_name = f"{fn.__name__}_"
    _fn.__name__ = inplace_name
    _fn = register_decomposition(getattr(aten, inplace_name))(_fn)  # type: ignore[assignment]

    # We access the __all__ attribute of the module where fn is defined
    # There may be a cleaner way of doing this...
    from inspect import getmodule

    _all = getmodule(fn).__all__  # type: ignore[union-attr]
    if inplace_name not in _all:
        _all.append(inplace_name)
    return _fn


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    exact_dtype=True,
)
def abs(a):
    return prims.abs(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def acos(a):
    return prims.acos(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def acosh(a):
    return prims.acosh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def asin(a):
    return prims.asin(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def asinh(a):
    return prims.asinh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def atan(a):
    return prims.atan(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def atanh(a):
    return prims.atanh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def bitwise_not(a):
    return prims.bitwise_not(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    exact_dtype=True,
)
def ceil(a):
    return prims.ceil(a)


@register_decomposition(aten.is_complex)
def is_complex(input: TensorLikeType):
    return utils.is_complex_dtype(input.dtype)


@register_decomposition(aten.conj_physical)
@out_wrapper()
def conj_physical(input: TensorLikeType):
    if not utils.is_complex_dtype(input.dtype):
        return input
    return prims.conj_physical(input)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def cos(a):
    return prims.cos(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def cosh(a):
    return prims.cosh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def digamma(a):
    return prims.digamma(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erf(a):
    return prims.erf(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erfinv(a):
    return prims.erf_inv(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erfc(a):
    return prims.erfc(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def exp(a):
    return prims.exp(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def expm1(a):
    return prims.expm1(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def exp2(a):
    return prims.exp2(a)


# Fill has its own implementation because it has a value parameter
# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def fill(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    assert isinstance(value, Number)

    python_type = utils.dtype_to_type(a.dtype)
    if not utils.is_weakly_lesser_type(type(value), python_type):
        msg = f"value argument of type {type(value)} cannot be safely cast to type {python_type}!"
        raise ValueError(msg)

    return prims.fill(a, value)


def fill_(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    r = prims.fill(a, value)
    prims.copy_to(a, r)
    return a


@register_decomposition(aten.zero)
@out_wrapper()
def zero(input: TensorLikeType) -> TensorLikeType:
    return torch.zeros_like(input)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    exact_dtype=True,
)
def floor(a):
    return prims.floor(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    exact_dtype=True,
)
def frac(x: TensorLikeType) -> TensorLikeType:
    trunc_x = torch.mul(torch.floor(torch.abs(x)), torch.sign(x))
    return torch.sub(x, trunc_x)


# imag does not use _make_elementwise_unary_reference because it does not support out
def imag(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    torch._check(
        utils.is_complex_dtype(a.dtype), lambda: "imag only supports complex tensors."
    )
    return prims.imag(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)
def isfinite(a: TensorLikeType) -> TensorLikeType:
    if utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype):
        return prims.isfinite(a)

    return ones_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isinf(a: TensorLikeType) -> TensorLikeType:
    if utils.is_complex_dtype(a.dtype):
        return torch.logical_or(isinf(torch.real(a)), isinf(torch.imag(a)))
    if utils.is_float_dtype(a.dtype):
        return torch.abs(a) == float("inf")
    return torch.zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    exact_dtype=True,
)
def isposinf(a: TensorLikeType) -> TensorLikeType:
    torch._check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isposinf, got dtype {a.dtype}",
    )
    if utils.is_float_dtype(a.dtype):
        return a == float("inf")
    return torch.zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    exact_dtype=True,
)
def isneginf(a: TensorLikeType) -> TensorLikeType:
    torch._check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isneginf, got dtype {a.dtype}",
    )
    if utils.is_float_dtype(a.dtype):
        return a == float("-inf")
    return torch.zeros_like(a, dtype=torch.bool)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def isnan(a: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, a)


# alias
mvlgamma = _make_alias(torch.special.multigammaln, "mvlgamma")  # type: ignore[has-type]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)
def isreal(a: TensorLikeType) -> TensorLikeType:
    if utils.is_complex_dtype(a.dtype):
        return torch.imag(a) == 0
    return torch.ones_like(a, dtype=torch.bool)


# TODO: if this is special maybe it should be defined there and imported here?
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=aten.i0
)
def i0(a):
    return prims.bessel_i0(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def lgamma(a):
    return prims.lgamma(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log(a):
    return prims.log(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log1p(a):
    return prims.log1p(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log2(a):
    return prims.log2(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log10(a):
    return prims.log10(a)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    result_dtype = dtype or a.dtype
    computation_dtype = utils.get_computation_dtype(result_dtype)
    a_ = _maybe_convert_to_dtype(a, computation_dtype)
    return _maybe_convert_to_dtype(a_ - logsumexp(a_, dim, keepdim=True), result_dtype)  # type: ignore[return-value]


@register_decomposition(aten.logsumexp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logsumexp(
    self: TensorLikeType, dim: DimsType, keepdim: bool = False
) -> TensorLikeType:
    if not isinstance(dim, Iterable):
        dim = (dim,)
    if self.numel() == 0:
        # pyrefly: ignore [no-matching-overload]
        return torch.sum(torch.exp(self), dim, keepdim).log()
    # pyrefly: ignore [bad-argument-type]
    maxes = torch.amax(torch.real(self), dim, keepdim=True)
    maxes = torch.masked_fill(maxes, maxes.abs() == float("inf"), 0)
    # pyrefly: ignore [no-matching-overload]
    maxes_squeezed = maxes if keepdim else torch.squeeze(maxes, dim)
    # pyrefly: ignore [no-matching-overload]
    result = torch.sum(torch.exp(self - maxes), dim, keepdim)
    return result.log().add(maxes_squeezed)


@register_decomposition(aten.nan_to_num)
@out_wrapper()
def nan_to_num(
    a: TensorLikeType,
    nan: Optional[NumberType] = 0.0,
    posinf: Optional[NumberType] = None,
    neginf: Optional[NumberType] = None,
) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
        return a.clone()

    if nan is None:
        nan = 0.0

    if posinf is None:
        posinf = torch.finfo(a.dtype).max

    if neginf is None:
        neginf = torch.finfo(a.dtype).min

    result = torch.where(torch.isnan(a), nan, a)  # type: ignore[call-overload]
    result = torch.where(torch.isneginf(a), neginf, result)  # type: ignore[call-overload]
    result = torch.where(torch.isposinf(a), posinf, result)  # type: ignore[call-overload]
    return result


def _neg_meta(a: TensorLikeType):
    torch._check(
        a.dtype is not torch.bool,
        lambda: (
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` "
            "operator instead."
        ),
    )


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, extra_meta=_neg_meta
)
def neg(a):
    return prims.neg(a)


# positive does not use _make_elementwise_unary_reference because it does not support out
# CompositeImplicitAutograd - don't register decomp
def positive(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    if a.dtype is torch.bool:
        msg = "positive does not support bool tensors."
        raise RuntimeError(msg)
    return a


# real does not use _make_elementwise_unary_reference because it does not support out
def real(a: TensorLikeType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    if utils.is_complex_dtype(a.dtype):
        return prims.real(a)
    return a


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def reciprocal(a):
    return prims.reciprocal(a)


@register_decomposition(aten.round)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def round(a: TensorLikeType, *, decimals: int = 0) -> TensorLikeType:
    if decimals == 0:
        return prims.round(a)
    else:
        ten_pow = 10**decimals
        ten_neg_pow = 10 ** (-decimals)
        return prims.mul(prims.round(prims.mul(a, ten_pow)), ten_neg_pow)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def rsqrt(a):
    return prims.rsqrt(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sigmoid(a: TensorLikeType) -> TensorLikeType:
    return true_divide(1, add(1, exp(neg(a))))


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    exact_dtype=True,
)
def sgn(a):
    if utils.is_complex_dtype(a.dtype):
        a_abs = a.abs()
        return torch.where(a_abs == 0, 0, a / a_abs)
    else:
        return a.sign()


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    exact_dtype=True,
)
def sign(a):
    return prims.sign(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    exact_dtype=True,
)
def signbit(a):
    return prims.signbit(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sin(a):
    return prims.sin(a)


# Autograd note: This will give the right first derivative at zero (by chance),
# but not the right second derivative
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinc(a):
    a = math.pi * a
    return torch.where(a == 0, 1, torch.sin(a) / a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinh(a):
    return prims.sinh(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sqrt(a):
    return prims.sqrt(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
    aten_op=None,  # CompositeImplicitAutograd,
)
def square(a: TensorLikeType) -> TensorLikeType:
    return mul(a, a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tan(a):
    return prims.tan(a)


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tanh(a):
    return prims.tanh(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    exact_dtype=True,
)
def trunc(a):
    return prims.trunc(a)


# TODO: register this as a real ref/decomposition once TorchInductor supports complex!
def view_as_complex(self: TensorLikeType) -> TensorLikeType:
    input_dtype = self.dtype
    torch._check(
        utils.is_float_dtype(input_dtype),
        lambda: f"view_as_complex is only supported for floating point"
        f"tensors, but got a tensor of scalar type: {input_dtype}",
    )
    sizes = self.size()
    torch._check(
        len(sizes) != 0,
        lambda: "Input tensor must have one or more dimensions",
    )
    torch._check(
        sizes[-1] == 2,
        lambda: "Tensor must have a last dimension of size 2",
    )

    old_strides = self.stride()
    torch._check(
        old_strides[-1] == 1,
        lambda: "Tensor must have a last dimension with stride 1",
    )
    dims = old_strides[:-1]
    torch._check(
        builtins.all(stride % 2 == 0 for stride in dims),
        lambda: "Tensor must have a stride divisible by 2 for all but last dimension",
    )
    torch._check(
        self.storage_offset() % 2 == 0,
        lambda: "Tensor must have a storage_offset divisible by 2",
    )
    return prims.view_element_type(
        self, utils.corresponding_complex_dtype(input_dtype)
    ).squeeze(-1)


def _make_elementwise_binary_reference(
    type_promotion_kind,
    aten_op=infer_aten_op,
    name=None,
    has_out=True,
    supports_lhs_python_scalar=True,
    supports_rhs_python_scalar=True,
    supports_two_python_scalars=False,
    should_register_decomposition=True,
) -> Callable:
    def inner(prim: Callable):
        nonlocal aten_op, name
        if name is None:
            name = prim.__name__

        @wraps(prim)
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
        def _ref(
            a: Union[Tensor, NumberType],
            b: Union[Tensor, NumberType],
        ) -> Tensor:
            torch._check_value(
                supports_lhs_python_scalar or not isinstance(a, Number),
                lambda: f"{name}: Received a lhs Python scalar to an elementwise binary "
                "operation that does not accept lhs scalars!",
            )
            torch._check_value(
                supports_rhs_python_scalar or not isinstance(b, Number),
                lambda: f"{name}: Received a rhs Python scalar to an elementwise binary "
                "operation that does not accept rhs scalars!",
            )
            torch._check_value(
                supports_two_python_scalars
                or not (isinstance(a, Number) and isinstance(b, Number)),
                lambda: f"{name}: Receive two Number inputs to an elementwise binary operation!",
            )
            a, b = _maybe_broadcast(a, b)
            output = prim(a, b)
            return handle_noncontiguous_outputs([a, b], output)

        if has_out:
            _ref = out_wrapper()(_ref)  # type: ignore[assignment]

        _ref.__name__ = name
        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, name)
        if aten_op is not None and should_register_decomposition:
            register_decomposition(aten_op)(_ref)

        return _ref

    return inner


# Add has its own implementation because it has an alpha argument
@register_decomposition(aten.add)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def add(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    """
    Reference implementation of torch.add
    """

    a, b = _maybe_broadcast(a, b)

    if alpha is not None:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if python_type is not bool and not utils.is_weakly_lesser_type(
            type(alpha), python_type
        ):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        if isinstance(b, TensorLike):
            b = prims.mul(b, alpha)
        else:
            b = b * alpha

    output = prims.add(a, b)
    return handle_noncontiguous_outputs([a, b], output)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def atan2(a, b):
    return prims.atan2(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_and(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.bitwise_and(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_left_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.shift_left(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_or(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.bitwise_or(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_right_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.shift_right_arithmetic(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_xor(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.bitwise_xor(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
)
def copysign(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    if isinstance(b, Number) and isinstance(a, Tensor):
        # pyrefly: ignore [bad-argument-type]
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        msg = f"Expected divisor (b) to be on the same device ({a.device}) as dividend (a), but it is found on {b.device}!"
        raise RuntimeError(msg)
    # pyrefly: ignore [bad-argument-type]
    return where(signbit(b), neg(abs(a)), abs(a))


# complex =  _make_elementwise_binary_reference(prims.complex, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)


@register_decomposition(aten.div)
@out_wrapper()
def div(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    rounding_mode: Optional[str] = None,
):
    """
    Reference implementation of torch.div
    """
    if rounding_mode is None:
        return true_divide(a, b)
    elif rounding_mode == "trunc":
        return trunc_divide(a, b)
    elif rounding_mode == "floor":
        return floor_divide(a, b)
    else:
        msg = f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found {rounding_mode}."
        raise ValueError(msg)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def eq(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.eq(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
)
def pow(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> TensorLikeType:
    assert isinstance(a, TensorLikeType) or isinstance(b, TensorLikeType)

    if isinstance(b, Number):
        if b == 1.0:
            return a.clone()  # type: ignore[return-value,union-attr]
        elif b == 2.0:
            return a * a  # type: ignore[return-value]
        elif b == 0.5:
            return torch.sqrt(a)  # type: ignore[arg-type]
    elif isinstance(a, Number):
        if a == 1.0:
            return torch.fill(b, True)
        if a == 2.0 and (
            utils.is_float_dtype(b.dtype) or utils.is_complex_dtype(b.dtype)
        ):
            return torch.exp2(b)

    return prims.pow(a, b)


# Float power has its own implementation because it has unique type promotion.
# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def float_power(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> Tensor:
    if isinstance(a, Number) and isinstance(b, Number):
        raise ValueError(
            "Receive two Number inputs to an elementwise binary operation!"
        )

    # Handles type promotion
    dtype = utils.get_higher_dtype(a, b)
    assert dtype is not None
    if utils.is_complex_dtype(dtype):
        dtype = torch.complex128
    else:
        dtype = torch.float64

    # Float power has the following contiguous cast behavior to be
    # consistent with its C++ impl

    a = _maybe_convert_to_dtype(a, dtype)

    b = _maybe_convert_to_dtype(b, dtype)

    a, b = _maybe_broadcast(a, b)
    # pyrefly: ignore [bad-return]
    return pow(a, b)


# >>> a = torch.tensor(-0.2500, dtype=torch.float64)
# tensor(-0.250000000000000, dtype=torch.float64)
#
# >>> b = torch.tensor(-0.0010, dtype=torch.float64)
# tensor(-0.001000000000000, dtype=torch.float64)
#
# Note: In this case, casting float to double will expand the float mantissa with zeros,
# while creating a double generates a distinct mantissa.
# >>> torch.tensor(-0.001).to(dtype=torch.float64)
# tensor(-0.001000000047497, dtype=torch.float64)
#
# Floor Division
# The difference is caused because torch.remainder(a, b) = -0.001.
#
# >>> torch.floor(torch.true_divide(a, b))
# tensor(250., dtype=torch.float64)
#
# >>> torch.div(a, b, rounding_mode='floor')
# tensor(249., dtype=torch.float64)
#
# Definition: a // b = (a - remainder(a, b)) / b
# >>> torch.true_divide(torch.sub(a, torch.remainder(a, b)), b)
# tensor(249., dtype=torch.float64)
#
# For reference, see CPython's implementation:
# https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636


@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
    should_register_decomposition=False,
)
def floor_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    # Wrap scalars because some references only accept tensor arguments.
    if isinstance(a, Number) and isinstance(b, Number):
        # pyrefly: ignore [bad-argument-type]
        a = scalar_tensor(a)
        # pyrefly: ignore [bad-argument-type]
        b = scalar_tensor(b)
    elif isinstance(b, Number) and isinstance(a, Tensor):
        # pyrefly: ignore [bad-argument-type]
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(a, Number) and isinstance(b, Tensor):
        # pyrefly: ignore [bad-argument-type]
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        if a.device == torch.device("cpu"):
            msg = f"Expected divisor (b) to be on the same device ({a.device}) as dividend (a), but it is found on {b.device}!"
            raise RuntimeError(msg)
        else:
            b = prims.device_put(b, device=a.device)

    assert isinstance(a, Tensor) and isinstance(b, Tensor)
    dtype = a.dtype
    if utils.is_float_dtype(dtype):
        return _floor_divide_float(a, b)
    elif utils.is_integer_dtype(dtype):
        return _floor_divide_integer(a, b)
    else:
        torch._check(False, lambda: f"{dtype} not supported for floor_divide")


def _floor_divide_integer(a: Tensor, b: Tensor) -> Tensor:
    a, b = _maybe_broadcast(a, b)

    if not a.dtype.is_signed:
        return prims.div(a, b)

    # Convert truncation to flooring:
    offset = (torch.signbit(a) != torch.signbit(b)).logical_and(torch.fmod(a, b) != 0)
    return prims.div(a, b) - _maybe_convert_to_dtype(offset, a.dtype)


def _floor_divide_float(a: Tensor, b: Tensor) -> Tensor:
    mod = fmod(a, b)
    div = true_divide(sub(a, mod), b)

    # Ensure that the remainder has the same sign as denominator
    different_signed_inputs = bitwise_xor(lt(a, 0), lt(b, 0))
    non_zero_remainder = ne(mod, 0)
    mask = bitwise_and(non_zero_remainder, different_signed_inputs)
    div = where(mask, sub(div, 1), div)

    # Map quotient to nearest integer value
    floor_div = floor(div)
    mask = gt(sub(div, floor_div), 0.5)
    floor_div = where(mask, add(floor_div, 1), floor_div)

    basic_div = true_divide(a, b)
    zero_tensor = scalar_tensor(0, dtype=basic_div.dtype, device=basic_div.device)

    # If quotient is zero, copy signbit from true_divide quotient
    floor_div = where(ne(div, 0), floor_div, copysign(zero_tensor, basic_div))

    # If denominator is zero, then follow true_divide behavior
    return where(ne(b, 0), floor_div, basic_div)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def fmax(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.fmax(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def fmin(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.fmin(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=True,
)
def fmod(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.fmod(a, b)


@register_decomposition(aten.frexp)
@out_wrapper("mantissa", "exponent")
def frexp(self: TensorLikeType) -> tuple[TensorLikeType, TensorLikeType]:
    return torch.return_types.frexp(prims.frexp(self))


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def gcd(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.gcd(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def ge(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.ge(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def gt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.gt(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def heaviside(input: TensorLikeType, values: TensorLikeType) -> TensorLikeType:
    input_eq_zero = torch.eq(input, 0)
    input_lt_zero = torch.logical_or(torch.lt(input, 0), torch.isnan(input))
    zeros_and_ones = torch.where(input_lt_zero, 0, 1)
    output = torch.where(input_eq_zero, values, zeros_and_ones)
    return output


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def hypot(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.hypot(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def igamma(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.igamma(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def igammac(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.igammac(a, b)


def _check_close_args(
    name: str,
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float,
    atol: float,
) -> None:
    torch._check_value(
        a.dtype == b.dtype,
        lambda: f"{name}: Attempting to compare tensors of different dtypes {a.dtype} and {b.dtype}!",
    )
    torch._check(
        rtol >= 0,
        lambda: f"{name}: rtol must be greater than or equal to zero, but got {rtol}!",
    )
    torch._check(
        atol >= 0,
        lambda: f"{name}: atol must be greater than or equal to zero, but got {atol}!",
    )


# CompositeImplicitAutograd - don't register decomp
def isclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorLikeType:
    _check_close_args(name="torch.isclose", a=a, b=b, rtol=rtol, atol=atol)

    close = eq(a, b)
    if equal_nan and (utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype)):
        close = logical_or(close, logical_and(isnan(a), isnan(b)))

    # Note: In case of zero tolerances the closeness inequality degenerates to an equality check.
    # In this case, the short-circuit prevents false positives as detailed in the paragraph below.
    if atol == 0 and rtol == 0:
        return close

    # Note [closeness error computation]
    # atol and rtol are provided as doubles, so the computation
    # rtol * other will produce a float or complex tensor.
    # When the difference (self - other) is compared to it then the
    # tensor representing the difference will also be cast to float or complex.
    # However, since (self - other) in uint8 is very likely to produce a
    # negative value, this moves the cast forward so the difference is
    # always computed in a float or complex type.
    # If the values of the integer tensors cannot be exactly represented
    # by the default scalar type then this may cause an incorrect result.
    if not utils.is_float_dtype(a.dtype) and not utils.is_complex_dtype(a.dtype):
        a = prims.convert_element_type(a, torch.get_default_dtype())
        b = prims.convert_element_type(b, torch.get_default_dtype())

    allowed_error = add(atol, abs(mul(b, rtol)))
    actual_error = abs(sub(a, b))

    # Computes finite closeness
    result = logical_or(
        close, logical_and(isfinite(actual_error), le(actual_error, allowed_error))
    )

    return result


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def lcm(a: TensorLikeType, b: TensorLikeType):
    dtype = a.dtype
    # promoting to int32 to maintain 100% consistency with C++ and to
    # prevent overflow in case of int8 and int16
    promote_to_int = dtype in (torch.int8, torch.int16)
    if promote_to_int:
        a = prims.convert_element_type(a, torch.int32)
        b = prims.convert_element_type(b, torch.int32)

    g = torch.gcd(a, b)
    # Avoid division by zero in case gcd(0, 0) == 0
    g = torch.where(g == 0, 1, g)
    res = torch.abs(prims.div(a, g) * b)
    return res if not promote_to_int else prims.convert_element_type(res, dtype)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def le(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.le(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def logaddexp(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # Nb. this implementation does not distribute the gradients evenly when a == b
    mask = torch.real(a) >= torch.real(b)
    max_ = torch.where(mask, a, b)
    min_ = torch.where(mask, b, a)
    inf_mask = torch.logical_and(
        torch.logical_not(torch.isfinite(torch.real(a))), torch.real(a) == torch.real(b)
    )
    if utils.is_complex_dtype(a.dtype) or utils.is_complex_dtype(b.dtype):
        # are you wondering what this bunch of codes are for? edge cases!
        neg_min_mask = torch.real(min_) < 0
        inf_vals = torch.where(
            neg_min_mask, min_, torch.log(torch.exp(min_) + torch.exp(max_))
        )
        non_nan_vals = torch.where(
            inf_mask, inf_vals, max_ + torch.log1p(torch.exp(min_ - max_))
        )
        # the type for full_like does not include tensor yet
        nan_mask = torch.isnan(min_)
        return torch.where(nan_mask, complex(float("nan"), float("nan")), non_nan_vals)  # type: ignore[call-overload]
    else:
        return torch.where(inf_mask, a, max_ + torch.log1p(torch.exp(min_ - max_)))


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def logaddexp2(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    torch._check(
        not (utils.is_complex_dtype(a.dtype) or utils.is_complex_dtype(b.dtype)),
        lambda: "logaddexp2 doesn't support complex dtypes",
    )
    # Nb. this implementation does not distribute the gradients evenly when a == b
    mask = a >= b
    max_ = torch.where(mask, a, b)
    min_ = torch.where(mask, b, a)
    inf_mask = torch.logical_and(torch.isinf(a), a == b)
    inv_log_2 = 1.0 / math.log(2)
    result = max_ + torch.log1p(torch.exp2(min_ - max_)) * inv_log_2
    return torch.where(inf_mask, a, result)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_and(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return a & b


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def logical_not(a: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        return a == 0
    return ~a


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_or(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return bitwise_or(a, b)


# TODO: skip unnecessary conversion of long to float
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_xor(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    return a ^ b


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def lt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.lt(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def maximum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.maximum(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def minimum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.minimum(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
)
def mul(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.mul(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def ne(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def nextafter(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.nextafter(a, b)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def remainder(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.remainder(a, b)


# reverse sub
@register_decomposition(aten.rsub)
@out_wrapper()
def rsub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    alpha: NumberType = 1,
):
    if isinstance(a, Number):
        msg = "Received a Number for the first argument, but expected a Tensor"
        raise ValueError(msg)

    return torch.sub(b, a, alpha=alpha)


# TODO: consider refactoring this with add impl
# sub has its own implementation because it has an alpha argument
@register_decomposition(aten.sub)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def sub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: NumberType = 1,
):
    """
    Reference implementation of torch.sub
    """

    a, b = _maybe_broadcast(a, b)

    if isinstance(a, TensorLike) and isinstance(b, TensorLike):
        torch._check(
            not utils.is_boolean_dtype(a.dtype) and not utils.is_boolean_dtype(b.dtype),
            lambda: (
                "Subtraction, the `-` operator, with two bool tensors is not supported. "
                "Use the `^` or `logical_xor()` operator instead."
            ),
        )

    if alpha != 1:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        if isinstance(b, torch.Tensor):
            b = prims.mul(b, alpha)
        else:
            # Carefully not to use prims.mul if b is a scalar / symint.
            # prims.mul always returns a tensor,
            # which will mess with type promotion.
            b = b * alpha

    output = prims.sub(a, b)
    return handle_noncontiguous_outputs([a, b], output)


@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    name="true_divide",
    aten_op=None,  # CompositeImplicitAutograd
    supports_two_python_scalars=True,
)
def true_divide(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.div(a, b)


@register_decomposition(aten.xlogy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlogy(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # Operations like eq and log do not handle scalar values, so we convert them to scalar_tensors.
    if isinstance(b, TensorLike) and isinstance(a, Number):
        # pyrefly: ignore [bad-argument-type]
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)
    elif isinstance(a, TensorLike) and isinstance(b, Number):
        # pyrefly: ignore [bad-argument-type]
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)

    # mypy: expected "Tensor"
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log(b)))
    return torch.where(torch.isnan(b), float("nan"), rhs)


@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # CompositeImplicitAutograd
    supports_two_python_scalars=True,
)
def trunc_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    dtype = utils.get_dtype(a)
    if utils.is_integer_dtype(dtype):
        return prims.div(a, b)

    return trunc(prims.div(a, b))


#
# Elementwise Ternary References
#


@register_decomposition(aten.addcdiv)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def addcdiv(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcdiv
    """
    if value is not None:
        dtype = self.dtype  # no scalars allowed, see add
        python_type = utils.dtype_to_type(dtype)
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    return self + value * tensor1 / tensor2


@register_decomposition(aten.addcmul)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addcmul(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcmul
    """
    if value is not None:
        dtype = self.dtype  # no scalars allowed, see add
        python_type = utils.dtype_to_type(dtype)
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    return self + value * tensor1 * tensor2


@register_decomposition(aten.clamp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "min", "max"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def clamp(
    a: TensorLikeType,
    min: Optional[TensorOrNumberLikeType] = None,
    max: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    # NOTE: grad behavior with implementation `where` is not consistent on `nan`
    if min is None and max is None:
        msg = "clamp called but both min and max are none!"
        raise ValueError(msg)
    if min is not None:
        a_isnan = torch.isnan(a)
        condition = torch.bitwise_or(torch.ge(a, min), a_isnan)  # type: ignore[arg-type]
        # we should also propagate `nan` coming from boundaries. However, that's
        # not necessary since `ge` would already `False` when either operands has
        # a `nan`. So this line below is redundant
        #   `condition = bitwise_and(condition, bitwise_not(isnan(min)))`
        a = torch.where(condition, a, min)  # type: ignore[arg-type]
    if max is not None:
        a_isnan = torch.isnan(a)
        # same as above, no need to adjust `nan` from `max`
        condition = torch.bitwise_or(torch.le(a, max), a_isnan)  # type: ignore[arg-type]
        a = torch.where(condition, a, max)  # type: ignore[arg-type]

    return a


@register_decomposition(aten.clamp_min)
@out_wrapper()
def clamp_min(
    self: TensorLikeType,
    min: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    return torch.clamp(self, min=min)  # type: ignore[arg-type]


@register_decomposition(aten.clamp_max)
@out_wrapper()
def clamp_max(
    self: TensorLikeType,
    max: Optional[TensorOrNumberLikeType] = None,
) -> TensorLikeType:
    return torch.clamp(self, max=max)  # type: ignore[arg-type]


#
# Conditional references
#


# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement where.default
@register_decomposition(aten.where.self)
@register_decomposition(aten.where.ScalarSelf)
@register_decomposition(aten.where.ScalarOther)
@register_decomposition(aten.where.Scalar)
@register_decomposition(aten.where.self_out)
@out_wrapper(exact_dtype=True)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def where(
    pred: Tensor,
    a: Optional[TensorOrNumberLikeType] = None,
    b: Optional[TensorOrNumberLikeType] = None,
):
    """ """

    if a is None or b is None:
        raise NotImplementedError

    utils.check_same_device(pred, a, b, allow_cpu_scalar_tensors=True)
    torch._check(
        pred.dtype is torch.bool,
        lambda: f"expected predicate to be bool, got {pred.dtype}",
    )

    pred, a, b = _maybe_broadcast(pred, a, b)
    return prims.where(pred, a, b)


#
# Data Movement References
#
@register_decomposition(aten.clone)
@out_wrapper()
def clone(
    a: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    result = prims.clone(a, memory_format=memory_format)
    return result


def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=True):
    if not allow_cross_device and a.device != b.device:
        msg = f"Attempting to copy from device {b.device} to device {a.device}, but cross-device copies are not allowed!"
        raise RuntimeError(msg)

    return prims.copy_to(a, b)


@register_decomposition(aten.item)
def item(a: TensorLikeType) -> NumberType:
    if a.numel() != 1:
        msg = f"Can't convert a tensor with {a.numel()} elements to a number!"
        raise ValueError(msg)

    # NOTE: explicit conversion is necessary for bool!
    # See https://github.com/pytorch/pytorch/issues/78071
    number_type = utils.dtype_to_type(a.dtype)
    return number_type(prims.item(a))


# fast path when `to` returns an alias to input. This mimics the same function in aten
def _to_will_alias(
    a: TensorLikeType,
    device: Optional[DeviceLikeType] = None,
    dtype: Optional[torch.dtype] = None,
    copy: Optional[bool] = None,
    layout: Optional[torch.layout] = None,
    memory_format: Optional[torch.memory_format] = None,
    pin_memory: Optional[bool] = False,
    non_blocking: bool = False,  # not using non_blocking
) -> bool:
    return (
        not copy
        and (device is None or a.device == device)
        and (dtype is None or a.dtype == dtype)
        and (layout is None or a.layout == layout)
        # is_pinned issue #84925
        # and (pin_memory is None or pin_memory == a.is_pinned())
        and (
            memory_format is None
            or memory_format == torch.preserve_format
            or utils.is_contiguous_for_memory_format(a, memory_format=memory_format)
        )
    )


@singledispatch
def _to_dispatch(*args, **kwargs):
    raise NotImplementedError


@_to_dispatch.register
def _to_device(
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> dict[str, Any]:
    kwargs = {
        "device": device,
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


@_to_dispatch.register
def _to_device_str(
    device: str,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> dict[str, Any]:
    kwargs = {
        "device": torch.device(device),
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


@_to_dispatch.register
def _to_dtype(
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> dict[str, Any]:
    kwargs = {
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


@_to_dispatch.register
def _to_other(
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> dict[str, Any]:
    device = other.device
    dtype = other.dtype
    layout = other.layout
    # is_pinned issue #84925
    # pin_memory = other.is_pinned()
    kwargs = {
        "device": device,
        "dtype": dtype,
        "layout": layout,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


# remove to_kwargs that is already present in `a`
def _canonicalize_to_arguments(a: Tensor, to_kwargs: dict):
    options_to_check = ["dtype", "device", "layout", "memory_format"]
    # "device" option could be passed a str instead torch.device
    if "device" in to_kwargs and isinstance(to_kwargs["device"], str):
        to_kwargs["device"] = torch.device(to_kwargs["device"])

    for kw in options_to_check:
        if kw in to_kwargs:
            if (
                (kw == "memory_format" and to_kwargs[kw] is torch.preserve_format)
                or (
                    kw == "device"
                    and to_kwargs[kw].type == a.device.type
                    and (
                        not to_kwargs[kw].index or to_kwargs[kw].index == a.device.index
                    )
                )
                or (
                    getattr(a, kw, None) == to_kwargs[kw]
                )  # this also handles {"memory_format": None}
            ):
                to_kwargs.pop(kw)


def to(a: TensorLikeType, *args, **kwargs) -> TensorLikeType:
    # handled dispatch via positional arguments
    if len(args) != 0:
        kwargs = _to_dispatch(*args, **kwargs)

    # TODO: is_pinned is not currently supported in refs or fake_tensor
    # https://github.com/pytorch/pytorch/issues/84925
    assert "pin_memory" not in kwargs
    _canonicalize_to_arguments(a, kwargs)

    if _to_will_alias(a, **kwargs):
        return a

    copy = kwargs.pop("copy") if "copy" in kwargs else False
    non_blocking = kwargs.pop("non_blocking") if "non_blocking" in kwargs else False

    # short-circuit to `prims.convert_element_type` when `to` is just a dtype change
    if (
        (copy or (kwargs.get("dtype", a.dtype) != a.dtype))
        and (not non_blocking)
        and ("memory_format" not in kwargs)
        and ("device" not in kwargs)
        and ("layout" not in kwargs)
        # is_pinned issue #84925
        # and ("pin_memory" not in kwargs)
    ):
        return prims.convert_element_type(a, kwargs.get("dtype", a.dtype))

    result = torch.empty_like(a, **kwargs)
    # TODO: non_blocking should be handled by `copy_to`
    copy_to(result, a)
    return result


#
# Reduction references
#


def _reduction(
    a: TensorLikeType,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # to handle min/argmin that accept single dim only
    dims: Optional[DimsType] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # should be specified for ops that support it
    out: Optional[Tensor] = None,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
) -> TensorLikeType:  # it is usually SAME, but I want
    # ref writers to actually think about what to put here
    assert isinstance(a, TensorLike)
    if a.ndim > 64:
        raise RuntimeError(
            f"Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!"
        )

    if out is not None:
        assert isinstance(out, TensorLike)
        if dtype is not None:
            # TODO - this is true for eager mode currently, but it's wrong behavior for complex norms
            if dtype != out.dtype:
                raise RuntimeError(
                    "dtype argument and out dtype must match in reduction"
                )
    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, Dim)
    if isinstance(dims, Dim):
        dims = (dims,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dims)
    if not has_identity:
        from torch.fx.experimental.symbolic_shapes import sym_and

        valid_shape = a.ndim == 0 or sym_and(*(a.shape[i] > 0 for i in dims))
        torch._check(
            valid_shape,
            lambda: "reducing over zero-size dimension for reduction operation without identity",
        )

    computation_dtype, result_dtype = utils.reduction_dtypes(
        a, output_dtype_kind, dtype
    )
    a = _maybe_convert_to_dtype(a, computation_dtype)  # type: ignore[method-assign]
    result = prim(a, dims)
    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)

    if out is not None:
        assert result_dtype is not None
        if dtype is not None and result_dtype != out.dtype:
            raise RuntimeError(
                "Expected the dtype of reduction result and out to match"
            )
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]

    if result.dtype != result_dtype and result_dtype is not None:
        result = prims.convert_element_type(result, result_dtype)

    return result


def _make_copy_from_view(fn, return_none_on_out_variant=False):
    """
    Given a view function (e.g. torch.diagonal) generates its copy variant (e.g. torch.diagonal_copy)
    """
    aten_fn = getattr(aten, fn.__name__)
    annotations = getattr(fn, "__annotations__", {})
    # view ops should not change dtypes, this ensures that the decomp path has
    # the same error checks as eager.
    fn = out_wrapper(exact_dtype=True)(aten_fn)

    @wraps(fn)
    def _fn(*args, out=None, **kwargs):
        result = fn(*args, out=out, **kwargs)
        if return_none_on_out_variant and out is not None:
            return None
        if out is not None:
            return result

        return pytree.tree_map(
            lambda x: x.clone(memory_format=torch.contiguous_format),
            result,
        )

    copy_name = f"{fn.__name__}_copy"
    _fn.__name__ = copy_name
    _fn.__annotations__.update(annotations)
    register_decomposition(getattr(aten, copy_name))(_fn)
    return _fn


@register_decomposition(aten.all)
@out_wrapper()
def all(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    result = torch.logical_not(torch.any(torch.logical_not(a), dim, keepdim=keepdim))

    if a.dtype == torch.uint8:
        result = result.to(dtype=torch.uint8)

    return result


@register_decomposition(aten.any)
@out_wrapper()
def any(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    a_ = _maybe_convert_to_dtype(a, torch.bool)
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        result = a_.clone()
    else:
        result = a_.sum(dim=dim, keepdim=keepdim).ne(False)

    # Preserves uint8 -- probably a legacy mask thing
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    return result


@register_decomposition([aten.sum.dim_IntList, aten.sum.IntList_out])
def sum(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[list[int]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if out is not None:
            dtype = out.dtype
        elif utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def sum_to_size(
    a: Tensor,
    *shape,
) -> Tensor:
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    torch._check(
        utils.is_expandable_to(shape, a.shape),
        lambda: f'sum_to_size: size "{shape}" is not expandable to size "{a.shape}"',
    )
    # In ATen scalar tensors are sent through sum and the result is returned as
    # type promoted
    if utils.is_same_shape(shape, a.shape) and len(shape) > 0:
        return prims.view_of(a)
    leading_dims = a.ndim - len(shape)
    reduce_dims = tuple(range(leading_dims)) + tuple(
        i
        for i in range(leading_dims, len(shape))
        if shape[i - leading_dims] == 1 and a.shape[i] != 1
    )
    return torch.sum(a, dim=reduce_dims, keepdim=True, dtype=None)


@register_decomposition(aten.prod)
def prod(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[list[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    if dtype is None:
        if out is not None:
            dtype = out.dtype
        elif utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.prod,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(aten.amin)
def amin(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(aten.amax)
def amax(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def _dim_var_dispatch(dim=None, unbiased=None):
    # There's the following overload of torch.var:
    # var(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
    # We need to explicitly convert bool dims to unbiased arg
    if unbiased is None and isinstance(dim, bool):
        unbiased = dim
        dim = None
    return dim, unbiased


@register_decomposition(aten.var)
@out_wrapper()
def var(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
) -> TensorLikeType:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None

    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


@register_decomposition(aten.std)
@out_wrapper()
def std(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[list[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
) -> TensorLikeType:
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)

    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )
    a = _maybe_convert_to_dtype(a, opmath_dtype)
    a_var = torch.var(a, dim, correction=correction, keepdim=keepdim)
    a_std = torch.sqrt(a_var)
    assert dtype is not None
    return _maybe_convert_to_dtype(a_std, dtype)


@register_decomposition(aten.mean)
def mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out=None,
) -> TensorLikeType:
    # reduces over all dimensions if dim=() is passed
    if dim == () or dim == []:
        dim = None
    orig_dtype = dtype
    if dtype is None:
        dtype = a.dtype
    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=None,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )
    torch._check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: (
            f"mean(): could not infer output dtype. "
            f"{'Input' if orig_dtype is None else 'Optional'} dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        ),
    )
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    dims = utils.reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    result = true_divide(result, nelem)
    result_dtype = a.dtype if dtype is None else dtype
    result = _maybe_convert_to_dtype(result, result_dtype)  # type: ignore[method-assign]
    if out is not None:
        assert isinstance(out, TensorLike)
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
    return result


@register_decomposition(aten.std_mean)
@out_wrapper("out0", "out1")
def std_mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    *,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    correction: Optional[NumberType] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    correction = utils.set_correction(unbiased, correction)
    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )
    original_dtype = a.dtype
    a = _maybe_convert_to_dtype(a, opmath_dtype)
    a_var, a_mean = torch.var_mean(a, dim, correction=correction, keepdim=keepdim)
    a_std = torch.sqrt(a_var)
    assert dtype is not None
    return (
        _maybe_convert_to_dtype(a_std, dtype),
        _maybe_convert_to_dtype(a_mean, original_dtype),
    )


@register_decomposition(aten.var_mean)
@out_wrapper("out0", "out1")
def var_mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
):
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    v = var(a, dim, unbiased, keepdim, correction=correction)
    m = mean(a, dim, keepdim)
    return v, m


@register_decomposition(aten.addr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "vec1", "vec2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addr(
    self: TensorLikeType,
    vec1: TensorLikeType,
    vec2: TensorLikeType,
    *,
    beta: NumberType = 1,
    alpha: NumberType = 1,
) -> TensorLikeType:
    torch._check(
        vec1.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec1, but got {vec1.ndim}-D",
    )
    torch._check(
        vec2.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec2, but got {vec2.ndim}-D",
    )
    for arg, arg_name in ((alpha, "alpha"), (beta, "beta")):
        if isinstance(arg, bool):
            torch._check(
                utils.is_boolean_dtype(self.dtype)
                and utils.is_boolean_dtype(vec1.dtype)
                and utils.is_boolean_dtype(vec2.dtype),
                lambda: f"Boolean {arg_name} only supported for Boolean results.",
            )
    self = self.expand(vec1.shape[0], vec2.shape[0])
    if utils.is_boolean_dtype(self.dtype):
        # Integers are accepted for booleans
        torch._check(
            is_weakly_lesser_type(type(beta), int),
            lambda: f"expected bool/int beta but got {type(beta)}",
        )
        torch._check(
            is_weakly_lesser_type(type(alpha), int),
            lambda: f"expected bool/int alpha but got {type(beta)}",
        )
        if not beta:
            return torch.outer(vec1, vec2) if alpha else torch.full_like(self, False)
        else:
            return torch.logical_or(
                self,
                torch.outer(vec1, vec2) if alpha else torch.full_like(self, False),
            )
    else:
        torch._check(
            is_weakly_lesser_type(type(beta), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(beta)} to {self.dtype}",
        )
        torch._check(
            is_weakly_lesser_type(type(alpha), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(alpha)} to {self.dtype}",
        )
        if beta == 0:
            # This means NaNs from self are dropped if beta is zero
            return alpha * torch.outer(vec1, vec2)
        else:
            return beta * self + alpha * torch.outer(vec1, vec2)


# CompositeImplicitAutograd - don't register decomp
def atleast_1d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_1d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    res = tuple(a if a.ndim >= 1 else unsqueeze(a, 0) for a in args_)
    return res if len(res) > 1 else res[0]


# Helper function with assert to avoid MyPy error
# of incompatible type passed to unsqueeze
def _unsqueeze_atleast(
    at_least_fn: Callable, dim: int, arg: TensorLikeType
) -> TensorLikeType:
    arg_ = at_least_fn(arg)
    assert isinstance(arg_, TensorLike)
    return unsqueeze(arg_, dim)


# CompositeImplicitAutograd - don't register decomp
def atleast_2d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_2d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    unsqueeze_atleast_1d = partial(_unsqueeze_atleast, atleast_1d, 0)
    res = tuple(a if a.ndim >= 2 else unsqueeze_atleast_1d(a) for a in args_)
    return res if len(res) > 1 else res[0]


# CompositeImplicitAutograd - don't register decomp
def atleast_3d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_3d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    unsqueeze_atleast_2d = partial(_unsqueeze_atleast, atleast_2d, -1)
    res = tuple(a if a.ndim >= 3 else unsqueeze_atleast_2d(a) for a in args_)
    return res if len(res) > 1 else res[0]


def as_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: Optional[int] = None,
) -> TensorLikeType:
    storage_offset_int = (
        storage_offset if storage_offset is not None else a.storage_offset()
    )
    return prims.as_strided(a, size, stride, storage_offset_int)


@register_decomposition(aten.as_strided_scatter)
@out_wrapper()
def as_strided_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: Optional[int] = None,
) -> TensorLikeType:
    storage_offset_int = 0 if storage_offset is None else storage_offset
    return prims.as_strided_scatter(input, src, size, stride, storage_offset_int)


def broadcast_shapes(*shapes) -> ShapeType:
    return torch.Size(_broadcast_shapes(*shapes))


@aten.broadcast_tensors.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.broadcast_tensors.default.py_impl(DispatchKey.Meta)
def broadcast_tensors(*tensors) -> list[TensorLikeType]:
    if len(tensors) == 1 and not isinstance(tensors[0], Tensor):
        tensors = tensors[0]
    return list(_maybe_broadcast(*tensors, preserve_cpu_scalar_tensors=False))


# CompositeImplicitAutograd - don't register decomp
def broadcast_to(a: TensorLikeType, size: ShapeType) -> TensorLikeType:
    start = len(size) - len(a.shape)
    dims = tuple(range(start, len(a.shape) + start))
    return prims.broadcast_in_dim(a, size, dims)


@register_decomposition(aten.cat)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("tensors",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def cat(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    def cat_compute_output_memory_format(inputs):
        format = None
        for t in inputs:
            f = utils.suggest_memory_format(t)
            if f == torch.contiguous_format:
                return f
            if format is not None and format != f:
                return torch.contiguous_format
            format = f
        assert format is not None
        return format

    if len(tensors) == 0:
        msg = "cat expects at least one tensor, but received zero!"
        raise ValueError(msg)

    for tensor in tensors:
        assert isinstance(tensor, TensorLike)

    utils.check_same_device(*tensors, allow_cpu_scalar_tensors=False)

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    # This is a bit tricky.  Naively, you would expect to just pick one
    # arbitrary tensor and check that all tensors match this tensor.  However,
    # there is legacy behavior which says that if you have a 1-D empty tensor
    # (0,), this is permissible.  So you can't assume that all the tensors
    # have same dimensionality, and you can't assume that the first tensor is
    # the correct stencil.
    #
    # We'll implement this in a few passes.  First, we will try to infer the
    # ndim of the cat output.  If this ndim != 1, then we know that all ndim =
    # 1 inputs must be empty, or are errors.  If this ndim == 1, then life
    # is easy (the legacy special case coincides with regular handling).
    #
    # NB: The regular implementation of cat just filters out empty inputs,
    # but we do it slightly different here for better handling for unbacked
    # SymInts

    example = None
    # pyrefly: ignore [bad-assignment]
    for i, t in enumerate(tensors):
        if example is None:
            if t.ndim != 1:
                example = t
        else:
            if t.ndim != 1:
                torch._check(
                    t.ndim == example.ndim,
                    lambda: "Number of dimensions of tensors must match.  "
                    f"Expected {example.ndim}-D tensors, but got {t.ndim}-D for "
                    f"tensor number {i} in the list",
                )

    if example is None:
        # example is None if everything is 1-D.  If so, just arbitrarily pick
        # the first one
        example = tensors[0]

    shape = example.shape
    filtered = []
    for tensor_idx, tensor in enumerate(tensors):
        if len(shape) != len(tensor.shape):
            assert tensor.ndim == 1  # we've already checked this above
            # Don't suggest the legacy behavior in the error message
            torch._check(
                # NB: it is not enough to simply assert that tensor.shape[0] == 0;
                # this MUST be true even under guard size oblivious.
                # Effectively, we must actually know that the shape is zero,
                # passing an unbacked SymInt which we will defer a runtime
                # assert on won't cut it.  This is a policy decision (size
                # oblivious semantics say that u0 tensors never are inferred
                # to be zero size, even if they must be that for the cat to go
                # through), and is load bearing for our Inductor lowerings
                # (which assume that size oblivious tests are OK to determine
                # if a shape is permissibly zero.)
                guard_or_false(tensor.shape[0] == 0),
                lambda: f"Number of dimensions of tensors must match.  "
                f"Expected {example.ndim}-D tensors, but got 1-D for "
                f"tensor number {tensor_idx} in the list",
            )
        else:
            # Remove inputs that are 1-D, zero size
            if tensor.ndim == 1 and guard_or_false(tensor.shape[0] == 0):
                continue
            # Don't bother checking size match, prims.cat will handle it
            filtered.append(tensor)

    memory_format = cat_compute_output_memory_format(tensors)

    if len(filtered) == 0:
        t = tensors[0]

        # TODO: fix this to work with meta tensors
        try:
            # BUG? This looks like it wants to call builtins.any() but is
            # actually calling .any() (in this file). Changing to builtins.any()
            # causes tests to fail:
            # PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=4 python test/test_ops.py -k \
            #   TestFakeTensorCUDA.test_fake_crossref_backward_amp_cat_cuda_float32
            requires_grad = bool(any(x.requires_grad for x in tensors))  # type: ignore[arg-type]
        except Exception:
            requires_grad = False  # type: ignore[assignment]

        return empty(
            (0,),
            dtype=t.dtype,
            device=t.device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    dim = utils.canonicalize_dim(filtered[0].ndim, dim)
    utils.validate_idx(filtered[0].ndim, dim)

    return prims.cat(filtered, dim).clone(memory_format=memory_format)


# CompositeImplicitAutograd - don't register decomp
@out_wrapper()
def column_stack(tensors: TensorSequenceType) -> TensorLikeType:
    aligned_tensors = tuple(
        x if x.ndim > 1 else x.reshape((x.numel(), 1)) for x in tensors
    )
    return cat(aligned_tensors, 1)


def conj(input: TensorLikeType) -> TensorLikeType:
    if not utils.is_complex_dtype(input.dtype):
        return input
    if input.is_sparse:
        return torch.conj_physical(input)
    return prims.conj(input)


# This replicates at::constant_pad_nd, defined in ATen/native/PadNd.cpp
@register_decomposition(aten.constant_pad_nd)
@out_wrapper()
def constant_pad_nd(
    input: TensorLikeType, pad: list[int], value: NumberType = 0
) -> TensorLikeType:
    torch._check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    input_sizes = input.shape
    l_inp = len(input_sizes)

    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    torch._check(
        l_inp >= l_pad,
        lambda: "Length of pad should be no more than twice the number of "
        f"dimensions of the input. Pad length is {len(pad)} while the input has "
        f"{l_inp} dimensions.",
    )

    c_input = input
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] < 0:
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.shape[i] + pad[pad_idx])

        if pad[pad_idx + 1] < 0:
            c_input = c_input.narrow(i, 0, c_input.shape[i] + pad[pad_idx + 1])

    # If all the pads are negative we can return the result.
    # Avoid early exiting if all pads = 0 to prevent specialization on export.
    # During export, raw if statements are specialized on the input, meaning
    # that we lose a branch depending on the example input used to export.
    # Here, this is either the case where all pads = 0, or the case where at
    # least one pad > 0 and the rest are >= 0.
    # Avoiding the early exit when all pads = 0 ensures we can export
    # constant_pad_nd for cases when all pads >= 0.
    # Note: if any pads are negative, this code specializes due to the if statements above.
    if builtins.all(p < 0 for p in pad):
        return c_input.clone()

    new_shape = list(input_sizes[:l_diff])

    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        torch._check(
            new_dim >= 0,
            lambda: f"The input size {input_sizes[l_diff + i]}, plus negative padding "
            f"{pad[pad_idx]} and {pad[pad_idx + 1]} resulted in a negative output size, "
            f"which is invalid. Check dimension {l_diff + i} of your input.",
        )
        new_shape.append(new_dim)

    memory_format = utils.suggest_memory_format(input)
    output = torch.empty(
        new_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
        memory_format=memory_format,
    )

    if value == 0 and input.dtype == torch.bool:
        value = False
    # torch.fill isn't typed to allow complex values
    output = torch.fill(output, value)  # type: ignore[arg-type]

    c_output = output
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        if pad[pad_idx] >= 0:
            c_output = c_output.narrow(
                i, pad[pad_idx], c_output.shape[i] - pad[pad_idx]
            )
        if pad[pad_idx + 1] >= 0:
            c_output = c_output.narrow(i, 0, c_output.shape[i] - pad[pad_idx + 1])

    prims.copy_to(c_output, c_input)
    return output


def contiguous(
    a: Tensor, *, memory_format: torch.memory_format = torch.contiguous_format
) -> Tensor:
    torch._check(
        memory_format != torch.preserve_format,
        lambda: "preserve memory format is unsupported by the contiguous operator",
    )

    # TODO: make logic consistent with aten contiguous
    if is_contiguous_for_memory_format_or_false(a, memory_format=memory_format):
        return a

    return torch.clone(a, memory_format=memory_format)


@out_wrapper()
def dstack(tensors: TensorSequenceType) -> TensorLikeType:
    torch._check(len(tensors) > 0, lambda: "dstack expects a non-empty TensorList")
    aligned_tensors = atleast_3d(*tensors)
    return cat(aligned_tensors, 2)


@register_decomposition(aten.expand)
def expand(a: Tensor, *shape, implicit: bool = False) -> Tensor:
    from torch.fx.experimental.symbolic_shapes import guard_or_false, size_hint, sym_or

    backed_so = torch.fx.experimental._config.backed_size_oblivious

    # NOTE: cannot use utils.extract_shape_from_varargs here
    # because that also validates the shape, but the shape
    # given to expand may be "invalid"
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])

    torch._check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]

        # expand(in -> out) has 3 different semantics:
        # 1) out == -1 -> size = in, stride unchanged
        # 2) in == 1 -> size = out, stride = 0
        # 3) in == out -> size = in, stride unchanged
        #
        # the code below is written for unbacked semantics s.t. we assume unbacked symbols don't
        # represent -1 unless explicitly specified, and the user is opting for case 2) or 3).
        # the sym_or allows either case, but in the decomposition's current state, broadcast_in_dim()
        # will either assume case 3) (via validate_shape() marking the expanded shape size-like), or will
        # raise a data-dependent error trying to figure out if the stride is 0, requiring the user to manually
        # select between the semantics of cases 2) and 3).
        if guard_or_false(requested_length == -1):
            shape_[offset_idx] = x
        else:
            # When backed size oblivious is used, we specialize for broadcasting
            # if its the only way to compile the example input.
            # i.e: x:1, requested_length:1 ==>
            #           assert x==requested_length, no specialization on ==1 or !=1.
            #            The non-broadcast path is picked
            #      x:1, requested_length:4 ==>
            #           specialize(x) to be 1.
            if backed_so:
                x_hint = size_hint(x, allow_none=True)
                requested_hint = size_hint(requested_length, allow_none=True)
                if x_hint == 1 and requested_hint != 1:
                    torch._check(x == 1)

            torch._check(
                sym_or(x == 1, requested_length == x),
                lambda: f"expand: attempting to expand a dimension of length {x} -> {requested_length}!",
            )
            torch._check(requested_length >= 0)
            shape_[offset_idx] = requested_length

    # At this point shape must be valid
    utils.validate_shape(shape_)

    return prims.broadcast_in_dim(
        a, shape_, tuple(range(offset, len(a.shape) + offset))
    )


# CompositeImplicitAutograd - don't register decomp
def expand_as(a: Tensor, b: Tensor) -> Tensor:
    return a.expand(b.shape)


def chunk(a: TensorLikeType, chunks: int, dim: int = 0) -> tuple[TensorLikeType, ...]:
    if chunks <= 0:
        msg = f"Expected at least one chunk, but got {chunks}!"
        raise ValueError(msg)

    dim = utils.canonicalize_dim(a.ndim, dim)
    length = a.shape[dim]
    chunk_size = math.ceil(length / chunks)
    full_chunks = math.floor(length / chunk_size)
    tail_chunk_size = length % chunk_size

    result = [narrow(a, dim, i * chunk_size, chunk_size) for i in range(full_chunks)]

    if tail_chunk_size != 0:
        result.append(narrow(a, dim, full_chunks * chunk_size, tail_chunk_size))

    return tuple(result)


# Note: flatten, unlike other shape operators, returns the input tensor on a no-op (unless
# a 0D tensor is flattened, in which case it's returned in 1D)
# CompositeImplicitAutograd - don't register decomp
def flatten(a: TensorLikeType, start_dim: int = 0, end_dim: int = -1) -> TensorLikeType:
    start_dim = utils.canonicalize_dim(a.ndim, start_dim)
    end_dim = utils.canonicalize_dim(a.ndim, end_dim)

    # Short-circuits on no-op
    if start_dim == end_dim and a.ndim != 0:
        return a

    # Tries to take a view
    # TODO: we could look at directing collapse_view to skip its meta function here (unsafe_collapse_view)
    # Unbacked semantics: if validity of in-place flattening is undecided we copy.
    new_shape, _new_strides = prims._collapse_view_helper(
        a, start_dim, end_dim, must_be_valid=None
    )
    if new_shape is not None:
        return prims.collapse_view(a, start_dim, end_dim)

    # Makes a copy if it can't make a view
    return prims.collapse(a, start_dim, end_dim)


@register_decomposition(aten.flip)
@out_wrapper()
def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    if not isinstance(dims, tuple) and not isinstance(dims, list):
        raise ValueError("dims has to be a sequence of ints")
    dims = utils.canonicalize_dims(a.ndim, dims)  # type: ignore[assignment]
    utils.validate_no_repeating_dims(dims)
    return prims.rev(a, dims)


# CompositeImplicitAutograd - don't register decomp
def fliplr(a: TensorLikeType) -> TensorLikeType:
    if a.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")

    return flip(a, (1,))


# CompositeImplicitAutograd - don't register decomp
def flipud(a: TensorLikeType

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 308 function(s): is_noncontiguous_supported, handle_noncontiguous_outputs, _broadcast_shapes, _maybe_broadcast, should_expand, __maybe_broadcast, _make_elementwise_unary_reference, inner, _ref, _make_alias, _fn, _make_inplace, _fn, abs, acos, acosh, asin, asinh, atan, atanh, bitwise_not, ceil, is_complex, conj_physical, cos, cosh, digamma, erf, erfinv, erfc


## Key Components

The file contains 22444 words across 6872 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 223137 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

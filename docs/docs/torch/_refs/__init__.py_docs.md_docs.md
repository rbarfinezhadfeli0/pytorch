# Documentation: `docs/torch/_refs/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_refs/__init__.py_docs.md`
- **Size**: 52,977 bytes (51.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/_refs/__init__.py`

## File Metadata

- **Path**: `torch/_refs/__init__.py`
- **Size**: 223,137 bytes (217.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
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


@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAY
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_refs`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_refs`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_refs`):

- [`fft.py_kw.md_docs.md`](./fft.py_kw.md_docs.md)
- [`_conversions.py_kw.md_docs.md`](./_conversions.py_kw.md_docs.md)
- [`fft.py_docs.md_docs.md`](./fft.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`_conversions.py_docs.md_docs.md`](./_conversions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

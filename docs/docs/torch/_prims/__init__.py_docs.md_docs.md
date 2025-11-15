# Documentation: `docs/torch/_prims/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_prims/__init__.py_docs.md`
- **Size**: 53,231 bytes (51.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/_prims/__init__.py`

## File Metadata

- **Path**: `torch/_prims/__init__.py`
- **Size**: 82,701 bytes (80.76 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import operator
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial, reduce
from typing import Optional, Union

import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor
from torch._C import _get_default_device
from torch._higher_order_ops.effects import new_token_tensor
from torch._library.utils import is_functional_schema
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
    Dim,
    DimsSequenceType,
    DimsType,
    IntLike,
    Number,
    NumberType,
    RETURN_TYPE,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    type_to_dtype,
)
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


prim = torch.library.Library("prims", "DEF")
prim_impl = torch.library.Library("prims", "IMPL", "CompositeExplicitAutograd")
prim_backend_select_impl = torch.library.Library("prims", "IMPL", "BackendSelect")
prim_autograd_impl = torch.library.Library("prims", "IMPL", "Autograd")
prim_meta_impl = torch.library.Library("prims", "IMPL", "Meta")

# Experimental module containing prototype "primitive" operations.

__all__ = [
    #
    # Common datastructures and helpers
    #
    "RETURN_TYPE",
    #
    # Elementwise unary prims
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bessel_j0",
    "bessel_j1",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj_physical",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "erfcx",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "imag",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "ndtri",
    "neg",
    "real",
    "reciprocal",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spherical_bessel_j0",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    #
    # Elementwise binary prims
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    # 'complex',  # needs custom meta
    "div",
    "eq",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "ge",
    "gt",
    "hypot",
    "igamma",
    "igammac",
    "le",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "nextafter",
    "pow",
    "remainder",
    "rsqrt",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",  # not implemented
    "sub",
    "zeta",
    #
    # View prims
    #
    "as_strided",
    "broadcast_in_dim",
    "collapse_view",
    "conj",
    "expand_dims",
    "slice",
    "split_dim",
    "squeeze",
    "transpose",
    "view_of",
    "view_element_type",
    #
    # Functionalized view mutations
    #
    "as_strided_scatter",
    #
    # Shape prims
    #
    "collapse",
    "cat",
    "reshape",
    "rev",
    #
    # Conditional prims
    #
    "where",
    #
    # Data conversion and movement prims
    #
    "clone",
    "convert_element_type",
    "device_put",
    "item",
    "maximum_value",
    "minimum_value",
    "copy_strided",
    #
    # Inplace prims
    #
    "copy_to",
    "resize",
    # "_set",  # Commented out, see note below
    #
    # Reduction prims
    #
    "amax",
    "amin",
    "prod",
    "sum",
    "xor_sum",
    "var",
    #
    # Tensor Creation Prims
    #
    "empty_strided",
    "empty_permuted",
    "scalar_tensor",
    "iota",
    #
    # Linear algebra (linalg) Prims
    #
    "svd",
    #
    # Randomness Prims
    #
    "normal",
    "_uniform_helper",
    #
    # FFT prims
    #
    "fft_r2c",
    "fft_c2c",
    "fft_c2r",
    #
    # prims for making/sinking tokens
    #
    "_make_token",
    "_sink_tokens",
]


def TensorMeta(
    tensorlike: Optional[Union[NumberType, torch.Tensor]] = None,
    *,
    shape: Optional[ShapeType] = None,
    strides: Optional[StrideType] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[torch.device, str]] = None,
):
    if isinstance(tensorlike, Number):
        assert not shape and (shape is None or isinstance(shape, Sequence))
        assert not strides and (strides is None or isinstance(strides, Sequence))
        inferred_shape: tuple[int, ...] = ()
        inferred_strides: tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))
        inferred_device = torch.device("cpu")
        # TODO: This looks wrong, a number that is wrapped into a tensor
        # needs to behave differently than a scalar tensor for type
        # promotion purposes
    elif tensorlike is not None:
        assert isinstance(tensorlike, torch.Tensor)
        inferred_shape = tuple(tensorlike.shape)
        inferred_strides = tuple(tensorlike.stride())
        inferred_dtype = tensorlike.dtype
        inferred_device = tensorlike.device
    else:
        # If no tensorlike "example" is given then all metadata
        # must be provided explicitly
        assert shape is not None
        assert strides is not None
        assert dtype is not None
        assert device is not None

    shape = inferred_shape if shape is None else tuple(shape)  # type: ignore[possibly-undefined]
    strides = inferred_strides if strides is None else tuple(strides)  # type: ignore[possibly-undefined]
    dtype = inferred_dtype if dtype is None else dtype  # type: ignore[possibly-undefined]
    device = inferred_device if device is None else device  # type: ignore[possibly-undefined]

    if isinstance(device, str):
        device = torch.device(device)

    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def _make_prim(
    *,
    schema: str,
    return_type: Union[RETURN_TYPE, tuple[RETURN_TYPE, ...]],
    meta: Callable,
    impl_aten: Callable,
    doc: str,
    tags: Optional[Sequence[torch.Tag]] = None,
    use_old_custom_ops_api: bool = False,
    register_conj_neg_fallthrough: bool = False,
):
    """
    Creates a primitive operation.

    """

    def _prim_impl(*args, **kwargs):
        # always run the meta function because aten implementation will
        # typically accept more inputs (e.g., it will do promotion and
        # broadcasting) which we want to reject
        meta(*args, **kwargs)
        return impl_aten(*args, **kwargs)

    # Right now prims don't support autograd (we can and should add an
    # argument that provides an implementation for backward here.)  Because we
    # don't have derivative formulas, we must setup a custom autograd function
    # that raises an error if backwards is invoked
    def _autograd_impl(*args, **kwargs):
        return backwards_not_supported(_prim)(*args, **kwargs)

    def _backend_select_impl(*args, **kwargs):
        if kwargs.get("device") and kwargs["device"].type == "meta":
            return meta(*args, **kwargs)
        if any(isinstance(x, torch.device) and x.type == "meta" for x in args):
            return meta(*args, **kwargs)
        else:
            return _prim_impl(*args, **kwargs)

    name = schema.split("(", maxsplit=1)[0]
    schema = schema[len(name) :]

    # register non-functional ops with old custom ops API
    cpp_schema = torch._C.parse_schema(name + schema)
    if use_old_custom_ops_api or not is_functional_schema(cpp_schema):
        prim.define(name + schema, tags=torch.Tag.pt2_compliant_tag)
        prim_impl.impl(name, _prim_impl)
        prim_autograd_impl.impl(name, _autograd_impl)
        prim_meta_impl.impl(name, meta)
    else:
        mutates_args = [
            arg.name
            for arg in cpp_schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        prim_def = torch.library.custom_op(
            "prims::" + name,
            _prim_impl,
            mutates_args=tuple(mutates_args),
            schema=schema,
        )
        prim_def.register_fake(meta)

        # all view ops get conj/neg fallthroughs
        if return_type == RETURN_TYPE.VIEW or register_conj_neg_fallthrough:
            prim_def._lib.impl(name, torch.library.fallthrough_kernel, "Conjugate")
            prim_def._lib.impl(name, torch.library.fallthrough_kernel, "Negative")

    _prim_packet = getattr(torch._ops.ops.prims, name)
    _prim = _prim_packet.default
    if tags:
        _prim._tags = tags
    elif aten_packet := getattr(torch.ops.aten, name, None):
        overload_tags = [
            getattr(aten_packet, overload).tags for overload in aten_packet.overloads()
        ]
        tags_intersection = set(overload_tags[0])
        tags_intersection.intersection_update(*overload_tags[1:])

        # dont inadvertently add to prim ops
        tags_intersection.discard(torch.Tag.core)
        # causes errors with python ref executor tests, none of the
        # data dependent pytorch ops actually decompose to prims
        tags_intersection.discard(torch.Tag.data_dependent_output)

        # iter over first tags for determinism
        _prim._tags = tuple(t for t in overload_tags[0] if t in tags_intersection)

    from torch._subclasses.fake_tensor import contains_tensor_types

    if (
        not any(contains_tensor_types(a.type) for a in _prim._schema.arguments)
        or str(
            _prim
            # See https://github.com/pytorch/pytorch/issues/103532
        )
        == "prims.device_put.default"
    ):
        prim_backend_select_impl.impl(name, _backend_select_impl)

    for p in (_prim_packet, _prim):
        p.__doc__ = doc
        p.return_type = return_type  # type: ignore[attr-defined]

        p.schema = schema
        p.prim_impl = _prim_impl
        p.prim_meta_impl = meta
        p.impl_aten = impl_aten

    return _prim


class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)


# TODO: implement dtype validation here, too, or on the corresponding refs
def _prim_elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,
    args_with_fixed_dtypes: Optional[tuple[TensorLikeType, ...]] = None,
) -> FakeTensor:
    """
    Meta function for elementwise operations that produce outputs in the same dtype
    as their inputs.

    Stride logic is currently incorrect.
    """

    assert len(args) > 0

    utils.check_same_dtype(*args)

    args_ = list(args)
    if args_with_fixed_dtypes is not None:
        args_ = list(args_with_fixed_dtypes) + args_

    utils.check_same_device(*args_, allow_cpu_scalar_tensors=True)
    utils.check_same_shape(*args_, allow_cpu_scalar_tensors=True)

    l2p_perm, _ = utils.compute_elementwise_output_logical_to_physical_perm(*args_)
    shape = utils.extract_shape(*args_, allow_cpu_scalar_tensors=True)

    # Acquires the dtype
    dtype = None
    scalar_type = None
    for arg in args:
        if isinstance(arg, TensorLike):
            if not utils.is_cpu_scalar_tensor(arg):
                dtype = arg.dtype
                break
            else:
                dtype = arg.dtype
        elif isinstance(arg, Number):
            scalar_type = type(arg)

    if dtype is None and scalar_type is not None:
        dtype = utils.type_to_dtype(scalar_type)

    # Acquires the device (if it exists) or number
    device = None
    number = None
    # pyrefly: ignore [bad-assignment]
    for arg in args_:
        if isinstance(arg, TensorLike):
            if utils.is_cpu_scalar_tensor(arg):
                if device is None:
                    device = arg.device
                # keep going, in case there is a cuda tensor later
            else:
                device = arg.device
                break

        elif isinstance(arg, Number):
            if number is None:
                number = arg

    # NOTE: type promotion behavior here is mostly hidden from tests because
    # references will typically handle the type promotion properly even if this doesn't
    # (but getting it wrong will cause too many casts to be inserted in traces!)
    if device is not None:
        assert dtype is not None
        if type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
            dtype = torch.bool
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
            if utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype):
                dtype = torch.get_default_dtype()
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
            if utils.is_complex_dtype(dtype):
                dtype = utils.corresponding_real_dtype(dtype)

        assert shape is not None
        return torch.empty_permuted(shape, l2p_perm, device=device, dtype=dtype)  # type: ignore[return-value]

    # Number case
    # TODO: fix number type promotion (bool, complex->float)

    # For now for symint/float, just implementing the common / simple cases of (int,float,symint,symfloat)
    seen_float = False
    if isinstance(number, (torch.SymInt, torch.SymFloat)):
        for a in args:
            assert isinstance(a, (int, float, torch.SymInt, torch.SymFloat)), "NYI"
            seen_float = seen_float or isinstance(a, (float, torch.SymFloat))
        if seen_float:
            number = sym_float(number)

    return TensorMeta(number)  # type: ignore[arg-type]


def _complex_only_elementwise_meta(*args, **kwargs):
    torch._check(
        utils.is_complex_dtype(args[0].dtype), lambda: "Only complex dtype is supported"
    )
    return _prim_elementwise_meta(*args, **kwargs)


def _make_elementwise_unary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise unary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _make_elementwise_binary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise binary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self, Tensor other) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _not_impl(*args, **kwargs):
    raise NotImplementedError


#
# Elementwise unary operations
#


abs = _make_elementwise_unary_prim(
    "abs",
    impl_aten=torch.abs,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    "acos",
    impl_aten=torch.acos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

acosh = _make_elementwise_unary_prim(
    "acosh",
    impl_aten=torch.acosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asin = _make_elementwise_unary_prim(
    "asin",
    impl_aten=torch.asin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asinh = _make_elementwise_unary_prim(
    "asinh",
    impl_aten=torch.asinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan = _make_elementwise_unary_prim(
    "atan",
    impl_aten=torch.atan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atanh = _make_elementwise_unary_prim(
    "atanh",
    impl_aten=torch.atanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_prim(
    "cos",
    impl_aten=torch.cos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cosh = _make_elementwise_unary_prim(
    "cosh",
    impl_aten=torch.cosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j0 = _make_elementwise_unary_prim(
    "bessel_j0",
    impl_aten=torch.special.bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j1 = _make_elementwise_unary_prim(
    "bessel_j1",
    impl_aten=torch.special.bessel_j1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0 = _make_elementwise_unary_prim(
    "bessel_i0",
    impl_aten=torch.i0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0e = _make_elementwise_unary_prim(
    "bessel_i0e",
    impl_aten=torch.special.i0e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1 = _make_elementwise_unary_prim(
    "bessel_i1",
    impl_aten=torch.special.i1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1e = _make_elementwise_unary_prim(
    "bessel_i1e",
    impl_aten=torch.special.i1e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_not = _make_elementwise_unary_prim(
    "bitwise_not",
    impl_aten=torch.bitwise_not,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _cbrt_aten(a: torch.Tensor) -> Tensor:
    torch._check(
        not a.is_complex(),
        lambda: "cbrt: Complex inputs not supported. Consider calling torch.pow(a, 1.0/3.0)",
    )
    # Returns the real cubic root of the number.
    # Note that if a < 0, pow(a, (1. / 3.)) returns th complex number
    # exp(1/3 * log(a)) = exp(1/3 * (log(abs(a)) + pi*i)) = cbrt(abs(a)) * e^{pi/3*i}
    # which is a complex number.
    # For more info see the section Note in
    # https://en.cppreference.com/w/cpp/numeric/math/cbrt
    return torch.copysign(torch.pow(a.abs(), 1 / 3), a)


cbrt = _make_elementwise_unary_prim(
    "cbrt",
    impl_aten=_cbrt_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ceil = _make_elementwise_unary_prim(
    "ceil",
    impl_aten=torch.ceil,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _conj_physical_meta(input: TensorLikeType) -> TensorLikeType:
    if not input.dtype.is_complex:
        raise RuntimeError("prims.conj_physical is only defined for complex dtypes")

    strides = utils.compute_elementwise_output_strides(input)
    return TensorMeta(input, strides=strides)


conj_physical = _make_prim(
    schema="conj_physical(Tensor self) -> Tensor",
    meta=_conj_physical_meta,
    impl_aten=torch._conj_physical,
    doc="Returns the physical conjugation of a complex tensor",
    return_type=RETURN_TYPE.NEW,
)


def _clone_meta(
    input: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    if memory_format != torch.preserve_format:
        return torch.empty(
            input.shape,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
            memory_format=memory_format,
        )
    else:
        # Match eager behavior by preserving strides for non_overlapping_and_dense tensors
        # If not, eager clone creates contiguous strides
        computed_stride = None
        if utils.is_non_overlapping_and_dense(input):
            computed_stride = input.stride()
        else:
            computed_stride = utils.compute_elementwise_output_strides(input)

        return torch.empty_strided(
            input.shape,
            computed_stride,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )


clone = _make_prim(
    schema="clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    meta=_clone_meta,
    impl_aten=torch.clone,
    doc="Returns the copy of a tensor",
    return_type=RETURN_TYPE.NEW,
    register_conj_neg_fallthrough=True,
)

digamma = _make_elementwise_unary_prim(
    "digamma",
    impl_aten=torch.digamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf = _make_elementwise_unary_prim(
    "erf",
    impl_aten=torch.erf,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf_inv = _make_elementwise_unary_prim(
    "erf_inv",
    impl_aten=torch.special.erfinv,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfc = _make_elementwise_unary_prim(
    "erfc",
    impl_aten=torch.special.erfc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfcx = _make_elementwise_unary_prim(
    "erfcx",
    impl_aten=torch.special.erfcx,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp = _make_elementwise_unary_prim(
    "exp",
    impl_aten=torch.exp,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

expm1 = _make_elementwise_unary_prim(
    "expm1",
    impl_aten=torch.special.expm1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp2 = _make_elementwise_unary_prim(
    "exp2",
    impl_aten=torch.special.exp2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _fill_meta(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    return _prim_elementwise_meta(
        a, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


# NOTE: fill uses _make_prim directly because it has a value parameter
fill = _make_prim(
    schema="fill(Tensor self, Scalar value) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_fill_meta,
    impl_aten=torch.fill,
    doc="",
)

floor = _make_elementwise_unary_prim(
    "floor",
    impl_aten=torch.floor,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

imag = _make_prim(
    schema="imag(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.imag,
    doc="",
)

isfinite = _make_elementwise_unary_prim(
    "isfinite",
    impl_aten=torch.isfinite,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    "lgamma",
    impl_aten=torch.lgamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log = _make_elementwise_unary_prim(
    "log",
    impl_aten=torch.log,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log1p = _make_elementwise_unary_prim(
    "log1p",
    impl_aten=torch.log1p,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log2 = _make_elementwise_unary_prim(
    "log2",
    impl_aten=torch.log2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log10 = _make_elementwise_unary_prim(
    "log10",
    impl_aten=torch.log10,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

real = _make_prim(
    schema="real(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.real,
    doc="",
)

reciprocal = _make_elementwise_unary_prim(
    "reciprocal",
    impl_aten=torch.reciprocal,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ndtri = _make_elementwise_unary_prim(
    "ndtri",
    impl_aten=torch.special.ndtri,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

neg = _make_elementwise_unary_prim(
    "neg",
    impl_aten=torch.neg,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

round = _make_elementwise_unary_prim(
    "round",
    impl_aten=torch.round,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

rsqrt = _make_elementwise_unary_prim(
    "rsqrt",
    impl_aten=torch.rsqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sign = _make_elementwise_unary_prim(
    "sign",
    impl_aten=torch.sign,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

signbit = _make_elementwise_unary_prim(
    "signbit",
    impl_aten=torch.signbit,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sin = _make_elementwise_unary_prim(
    "sin",
    impl_aten=torch.sin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sinh = _make_elementwise_unary_prim(
    "sinh",
    impl_aten=torch.sinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

spherical_bessel_j0 = _make_elementwise_unary_prim(
    "spherical_bessel_j0",
    impl_aten=torch.special.spherical_bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sqrt = _make_elementwise_unary_prim(
    "sqrt",
    impl_aten=torch.sqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tan = _make_elementwise_unary_prim(
    "tan",
    impl_aten=torch.tan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tanh = _make_elementwise_unary_prim(
    "tanh",
    impl_aten=torch.tanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

trunc = _make_elementwise_unary_prim(
    "trunc",
    impl_aten=torch.trunc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

#
# Elementwise binary operations
#

add = _make_elementwise_binary_prim(
    name="add",
    impl_aten=torch.add,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan2 = _make_elementwise_binary_prim(
    name="atan2",
    impl_aten=torch.atan2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_and = _make_elementwise_binary_prim(
    "bitwise_and",
    impl_aten=torch.bitwise_and,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_or = _make_elementwise_binary_prim(
    "bitwise_or",
    impl_aten=torch.bitwise_or,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_xor = _make_elementwise_binary_prim(
    "bitwise_xor",
    impl_aten=torch.bitwise_xor,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: complex needs a special meta to account for its float -> complex behavior
# complex = _make_elementwise_binary_prim(
#   impl_aten=torch.complex,
#   doc="",
# )


# div prim performs truncation division on integer inputs
#   and true division for floating and complex inputs
def _div_aten(a, b):
    is_integral = isinstance(a, (bool, int, torch.SymInt)) or (
        isinstance(a, torch.Tensor) and utils.is_integer_dtype(a.dtype)
    )

    if is_integral:
        # pyrefly: ignore [bad-argument-type]
        return torch.div(a, b, rounding_mode="trunc")
    else:
        # pyrefly: ignore [bad-argument-type]
        return torch.true_divide(a, b)


div = _make_elementwise_binary_prim(
    "div",
    impl_aten=_div_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

eq = _make_elementwise_binary_prim(
    "eq",
    impl_aten=torch.eq,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

fmax = _make_elementwise_binary_prim(
    "fmax",
    impl_aten=torch.fmax,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmin = _make_elementwise_binary_prim(
    "fmin",
    impl_aten=torch.fmin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmod = _make_elementwise_binary_prim(
    "fmod",
    impl_aten=torch.fmod,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


gcd = _make_elementwise_binary_prim(
    "gcd",
    impl_aten=torch.gcd,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


ge = _make_elementwise_binary_prim(
    "ge",
    impl_aten=torch.ge,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

gt = _make_elementwise_binary_prim(
    "gt",
    impl_aten=torch.gt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

hypot = _make_elementwise_binary_prim(
    "hypot",
    impl_aten=torch.hypot,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igamma = _make_elementwise_binary_prim(
    "igamma",
    impl_aten=torch.special.gammainc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igammac = _make_elementwise_binary_prim(
    "igammac",
    impl_aten=torch.special.gammaincc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

le = _make_elementwise_binary_prim(
    "le",
    impl_aten=torch.le,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lt = _make_elementwise_binary_prim(
    "lt",
    impl_aten=torch.lt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)


# Note: the following impls are because torch.maximum and torch.minimum do not support scalar inputs
def _maximum_aten(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.maximum(a, b)  # type: ignore[arg-type]


maximum = _make_elementwise_binary_prim(
    "maximum",
    impl_aten=_maximum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _minimum_aten(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.minimum(a, b)  # type: ignore[arg-type]


minimum = _make_elementwise_binary_prim(
    "minimum",
    impl_aten=_minimum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

mul = _make_elementwise_binary_prim(
    "mul",
    impl_aten=torch.mul,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ne = _make_elementwise_binary_prim(
    "ne",
    impl_aten=torch.ne,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

nextafter = _make_elementwise_binary_prim(
    "nextafter",
    impl_aten=torch.nextafter,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

pow = _make_elementwise_binary_prim(
    "pow",
    impl_aten=torch.pow,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

remainder = _make_elementwise_binary_prim(
    "remainder",
    impl_aten=torch.remainder,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


shift_left = _make_elementwise_binary_prim(
    "shift_left",
    impl_aten=torch.bitwise_left_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_arithmetic = _make_elementwise_binary_prim(
    "shift_right_arithmetic",
    impl_aten=torch.bitwise_right_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_logical = _not_impl

sub = _make_elementwise_binary_prim(
    "sub",
    impl_aten=torch.sub,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

zeta = _make_elementwise_binary_prim(
    "zeta",
    impl_aten=torch.special.zeta,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


#
# View operations
def _as_strided_meta(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    assert len(size) == len(stride)
    assert storage_offset >= 0
    utils.validate_strides(stride)
    utils.validate_shape(size)

    if reduce(operator.mul, size) == 0:
        # NOTE: This special case is to avoid having to acquire the storage below
        # as_strided to shapes with no elements are trivially valid, so it's OK
        pass
    elif isinstance(a, torch.Tensor):
        utils.check_in_bounds_for_storage(
            a._typed_storage(), size, stride, storage_offset
        )

    return torch.as_strided(a, size, stride, storage_offset)


def _as_strided_aten(
    a: Tensor, size: ShapeType, stride: StrideType, storage_offset: int
) -> Tensor:
    return torch.as_strided(a, size, stride, storage_offset)


_as_strided_doc = """
    Creates a view of the tensor with the given shape (size), strides (stride) and
    storage offset (storage_offset).
"""

as_strided = _make_prim(
    schema="as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)",
    meta=_as_strided_meta,
    impl_aten=_as_strided_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_as_strided_doc,
)


def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]
):
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_or,
    )

    # Type checks
    assert isinstance(a, TensorLike)
    assert isinstance(shape, Sequence)
    assert isinstance(broadcast_dimensions, Sequence)

    # every dimension must be accounted for
    assert a.ndim == len(broadcast_dimensions)

    # broadcast shape must have weakly more dimensions
    assert len(shape) >= a.ndim

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        assert isinstance(x, Dim)
        assert x > acc
        assert x < len(shape)

        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        torch._check(
            sym_or(a.shape[idx] == 1, shape[new_idx] == a.shape[idx]),
            lambda: f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}",
        )

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            # Assigns a stride of zero to dimensions
            # which were actually broadcast
            if guard_or_false(a.shape[original_idx] == 1):
                if guard_or_false(a.shape[original_idx] == shape[idx]):
                    new_strides.append(a.stride()[original_idx])
                else:
                    new_strides.append(0)
            else:
                torch._check(
                    a.shape[original_idx] == shape[idx],
                    lambda: f"non-broadcasting semantics require {a.shape[original_idx]} == {shape[idx]}",
                )
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            if guard_or_true(shape[idx] != 1):
                # consistent with previous use of guard_size_oblivious
                new_strides.append(0)
            elif original_idx == a.ndim:
                new_strides.append(1)
            else:
                new_strides.append(a.stride()[original_idx] * a.size()[original_idx])

    return a.as_strided(shape, new_strides, a.storage_offset())


def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


_broadcast_in_dim_doc = """
  Creates a view of a with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in a to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

broadcast_in_dim = _make_prim(
    schema="broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)",
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


def _validate_collapse_args(a: Tensor, start: int, end: int) -> None:
    # Special-case for zero dimensional tensors
    ndim = max(1, a.dim())
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)

    # Verifies end is strictly greater than start
    # (Collapse requires a non-empty interval)
    torch._check_value(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )


def _collapsed_shape(shape: ShapeType, start: int, end: int) -> tuple[int, ...]:
    """
    Returns the shape of a with dims in [start, end) merged into a single dimension.
    """
    # Special-case for zero dimensional tensors
    shape = (1,) if len(shape) == 0 else tuple(shape)

    dim_length = 1
    for s in shape[start : end + 1]:
        dim_length = dim_length * s

    return shape[0:start] + (dim_length,) + shape[end + 1 :]


# If the collapse is invalid or cannot be determined (because of unbacked data)
# then `must_be_valid` determines the behavior:
#   None: return None, None.
#   str: Do a torch._check() to ensure the collapse is valid and if it isn't
#   then fail with the provided string.
def _collapse_view_helper(
    a: TensorLikeType, start: int, end: int, must_be_valid: Optional[str]
) -> tuple[Optional[ShapeType], Optional[StrideType]]:
    assert isinstance(a, TensorLike)

    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_and,
        sym_or,
    )

    _validate_collapse_args(a, start, end)

    # Special-case for zero dimensional tensors
    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape  # type: ignore[assignment]
        strides = a.stride()  # type: ignore[assignment]

    if a.ndim == 0 or (end == start):
        return shape, strides

    valid_op = True
    if guard_or_false(a.numel() != 0):
        for idx in range(end - 1, start - 1, -1):
            valid_op = sym_and(
                valid_op,
                sym_or(
                    shape[idx] == 1,
                    shape[idx + 1] == 1,
                    strides[idx] == strides[idx + 1] * shape[idx + 1],
                ),
            )  # type: ignore[assignment]

            # early exit if we already know its invalid.
            if guard_or_false(valid_op is False):
                break

    # for unbacked this become a runtime assertion.
    valid_op = sym_or(valid_op, a.numel() == 0)

    if must_be_valid:
        torch._check(valid_op, lambda: must_be_valid)
    else:
        if not guard_or_false(valid_op):
            return None, None

    # compute stride
    stride = strides[end]
    for idx in range(end - 1, start - 1, -1):
        if shape[idx] != 1:
            # TODO with unbacked we should really exclude when shape[idx] == 1
            # something like
            # min(stride[end], torch.ite(shape[x]!=1,stride[idx], inf), ...)
            stride = min(stride, strides[idx])

    # compute length
    length = shape[end]
    if guard_or_true(length != 0):
        for idx in range(end - 1, start - 1, -1):
            if guard_or_false(shape[idx] == 0):
                length = 0
                stride = 0
                break
            length = length * shape[idx]
    else:
        stride = 0

    new_shape = shape[:start] + (length,) + shape[end + 1 :]
    new_strides = strides[:start] + (stride,) + strides[end + 1 :]

    # NOTE: when the input has no elements it's restrided as if it were contiguous
    # except for unbacked.
    if guard_or_false(a.numel() == 0):
        new_strides = utils.make_contiguous_strides_for(new_shape)

    return new_shape, new_strides


def _collapse_view_meta(a: TensorLikeType, start: int, end: int) -> TensorLikeType:
    new_shape, new_strides = _collapse_view_helper(
        a, start, end, "Attempting to view a collapsed tensor, but no such view exists!"
    )
    assert new_strides is not None
    assert new_shape is not None
    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    new_shape = _collapsed_shape(a.shape, start, end)
    return a.view(new_shape)


_collapse_view_doc = """
  Creates a view of a with the dimensions between
  start (inclusive) and end (exclusive) merged into a
  single dimension.

  If it's not possible to take such a view then an error
  is thrown. See collapse instead.

  The dimensions can be merged if and only if
  they are all "nested" with each other. That is, they all
  have the property that

  stride[i] = stride[i+1] * shape[i+1]

  for all i in [start, end - 1).
  """

collapse_view = _make_prim(
    schema="collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)",
    meta=_collapse_view_meta,
    impl_aten=_collapse_view_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_collapse_view_doc,
)


def _conj_meta(a: TensorLikeType) -> TensorLikeType:
    if not a.dtype.is_complex:
        raise RuntimeError("Expected complex dtype in prims.conj")
    out = a.as_strided(a.shape, a.stride(), a.storage_offset())
    torch._C._set_conj(out, not a.is_conj())
    return out


_conj_doc = """
Returns a conjugated view of the original tensor
"""

conj = _make_prim(
    schema="conj(Tensor(a) a) -> Tensor(a)",
    meta=_conj_meta,
    impl_aten=torch.conj,
    return_type=RETURN_TYPE.VIEW,
    doc=_conj_doc,
)


def expand_dims(
    a: TensorLikeType, dimensions: DimsSequenceType, ndim=None
) -> TensorLikeType:
    """
    Creates a view of a with a.ndim + len(dimensions) dimensions, with new
    dimensions of length one at the dimensions specified by dimensions.
    """
    if ndim is not None:
        # TODO: this is only here to support the unsqueeze ref
        dims = sorted(utils.canonicalize_dims(ndim, dimensions))  # type: ignore[arg-type]
    else:
        dims = sorted(utils.canonicalize_dims(a.ndim, dimensions))  # type: ignore[arg-type]
    if len(set(dims)) != len(dims):
        msg = f"Received duplicate dimensions to expand in {str(dimensions)}"
        raise ValueError(msg)

    new_shape = list(a.shape)
    for idx in dims:
        new_shape.insert(idx, 1)

    broadcast_dimensions = [
        idx for idx in range(len(new_shape)) if idx not in dimensions
    ]
    return broadcast_in_dim(a, new_shape, broadcast_dimensions)


def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    utils.validate_idx(a.ndim, dim)
    utils.validate_dim_length(outer_length)

    # Verifies the dim can be split with the specified lhs_length
    inner_length = a.shape[dim] // outer_length

    if (a.shape[dim] % outer_length) != 0:
        msg = (
            f"Attempting to split dimension of length {a.shape[dim]}, "
            f"but outer length of {outer_length} divides it with a remainder!"
        )
        raise ValueError(msg)

    new_shape: list[int] = []
    new_strides: list[int] = []
    for idx in range(a.ndim):
        if idx == dim:
            new_shape.extend((outer_length, inner_length))
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])

    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    inner_length = a.shape[dim] // outer_length
    new_shape = a.shape[0:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]

    return a.view(new_shape)


_split_dim_doc = """
  Creates a view of a with the given dimension (of length l) split
  into two dimensions, with the outer of the two having
  length outer_length and the inner of the two having computed
  length inner_length such outer_length * inner_length = l.
  """

# TODO: consider renaming split_dim_view
split_dim = _make_prim(
    schema="split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)",
    meta=_split_dim_meta,
    impl_aten=_split_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_split_dim_doc,
)


# Note: allows dimensions to be specified redundantly
def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    assert isinstance(a, TensorLike)

    for idx in dimensions:
        utils.validate_idx(a.ndim, idx)
        assert a.shape[idx] == 1

    new_shape = []
    new_strides = []
    for idx in range(len(a.shape)):
        if idx in dimensions:
            continue

        new_shape.append(a.shape[idx])
        new_strides.append(a.stride()[idx])

    return a.as_strided(new_shape, new_strides, a.storage_offset())


_squeeze_doc = """
  Creates a view of the tensor with the specified dimensions removed.

  The removed dimensions must each have length one.
  """

squeeze = _make_prim(
    schema="squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)",
    meta=_squeeze_meta,
    impl_aten=torch.squeeze,
    return_type=RETURN_TYPE.VIEW,
    doc=_squeeze_doc,
)


def _transpose_meta(a: TensorLikeType, permutation: DimsSequenceType) -> TensorLikeType:
    if a.ndim != len(permutation):
        msg = f"Attempting to permute a tensor of rank {a.ndim}, but received a permutation of length {len(permutation)}!"
        raise ValueError(msg)

    if not utils.is_valid_permutation(a.ndim, permutation):
        msg = f"Received an invalid permutation, {permutation}!"
        raise ValueError(msg)

    new_shape = [0] * a.ndim
    new_strides = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]
        new_strides[idx] = a.stride()[dim]

    return a.as_strided(tuple(new_shape), tuple(new_strides), a.storage_offset())


def _transpose_aten(a: Tensor, permutation: DimsSequenceType) -> Tensor:
    return torch.permute(a, permutation)


_transpose_doc = """
    Creates a view of the tensor with its dimensions permuted.

    The length of the permutation must be the rank of the tensor,
    and each element of the permutation specifies the new order
    for the corresponding dimension.
    """

transpose = _make_prim(
    schema="transpose(Tensor(a) a, int[] permutation) -> Tensor(a)",
    meta=_transpose_meta,
    impl_aten=_transpose_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_transpose_doc,
)


def _view_of_meta(a: TensorLikeType) -> TensorLikeType:
    return a.as_strided(a.shape, a.stride(), a.storage_offset())


def _view_of_aten(a: Tensor) -> Tensor:
    return a.view(a.shape)


_view_of_doc = """
    Creates a view of the tensor.
    """

view_of = _make_prim(
    schema="view_of(Tensor(a) a) -> Tensor(a)",
    meta=_view_of_meta,
    impl_aten=_view_of_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_of_doc,
)


def _view_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    return a.view(dtype)


def _view_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    return a.view(dtype)


_view_element_type_doc = """
    Creates a view of the tensor with a different dtype.
    """

view_element_type = _make_prim(
    schema="view_of_dtype(Tensor(a) a, ScalarType dtype) -> Tensor(a)",
    meta=_view_element_type_meta,
    impl_aten=_view_element_type_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_element_type_doc,
)

#
# Functionalized view mutations
#


def _as_strided_scatter_meta(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int,
) -> TensorLikeType:
    utils.validate_shape(size)
    utils.validate_strides(stride)

    required_size = utils.compute_required_storage_length(size, stride, storage_offset)
    torch._check(
        input.numel() >= required_size,
        lambda: (
            f"as_strided_scatter: sizes {size}, strides {stride}, storage offset {storage_offset} "
            f" and itemsize {input.element_size()} requiring a storage size of "
            f"{required_size * input.element_size()} are out of bounds "
            f"for storage of size {input.numel() * input.element_size()}"
        ),
    )
    torch._check(
        utils.is_same_shape(src.shape, size),
        lambda: f"expected src to have a size equal to the slice of self. src size = {src.shape}, slice size = {size}",
    )

    return utils.clone_preserve_strides(input)


_as_strided_scatter_doc = """
    Creates a new tensor equivalent to ``out = input.clone()`` after mutation by
    ``out.as_strided(size, stride, storage_offset).copy_(src)``.
"""

as_strided_scatter = _make_prim(
    schema="as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor",
    meta=_as_strided_scatter_meta,
    impl_aten=torch.as_strided_scatter,
    return_type=RETURN_TYPE.NEW,
    doc=_as_strided_scatter_doc,
)


#
# Shape operations
#


def _collapse_meta(a: Tensor, start: int, end: int) -> Tensor:
    # Special-case for zero dim
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_prims`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_prims`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/_prims`):

- [`debug_prims.py_kw.md_docs.md`](./debug_prims.py_kw.md_docs.md)
- [`context.py_kw.md_docs.md`](./context.py_kw.md_docs.md)
- [`rng_prims.py_docs.md_docs.md`](./rng_prims.py_docs.md_docs.md)
- [`context.py_docs.md_docs.md`](./context.py_docs.md_docs.md)
- [`executor.py_kw.md_docs.md`](./executor.py_kw.md_docs.md)
- [`executor.py_docs.md_docs.md`](./executor.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`rng_prims.py_kw.md_docs.md`](./rng_prims.py_kw.md_docs.md)
- [`debug_prims.py_docs.md_docs.md`](./debug_prims.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

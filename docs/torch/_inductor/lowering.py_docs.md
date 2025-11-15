# Documentation: lowering.py

## File Metadata
- **Path**: `torch/_inductor/lowering.py`
- **Size**: 248019 bytes
- **Lines**: 7578
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec
from unittest.mock import patch

import sympy

import torch
import torch.ao.quantization.fx._decomposed
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._higher_order_ops.associative_scan import associative_scan_op
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_mutation
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.utils import get_layout_constraint_tag
from torch._prims_common import (  # pyrefly: ignore  # deprecated; pyrefly: ignore [deprecated]
    canonicalize_dim,
    canonicalize_dims,
    check,
    dtype_to_type,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    get_computation_dtype,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    Number,
)
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    has_free_unbacked_symbols,
    resolve_unbacked_bindings,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    CeilDiv,
    FloorDiv,
    Identity,
    Mod,
    ModularIndexing,
)

from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators  # NOQA: F401
from .decomposition import decompositions, get_decompositions
from .ir import (
    BaseView,
    DtypeView,
    ExpandView,
    IndexingConstant,
    IRNode,
    is_triton,
    MutableBox,
    OnlineSoftmaxReduction,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    ShapeAsConstantBuffer,
    SqueezeView,
    TensorBox,
    validate_ir,
    View,
)
from .utils import (
    ceildiv,
    decode_device,
    is_dynamic,
    is_gpu,
    is_pointwise_use,
    is_view,
    needs_fallback_due_to_atomic_add_limitations,
    pad_listlike,
    register_op_dtype_propagation_rules,
    register_op_requires_libdevice_fp64,
    sympy_product,
    use_scatter_fallback,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from .ops_handler import ReductionType


_T = TypeVar("_T")
_P = ParamSpec("_P")

# TODO(jansel): we should implement decomps or lowerings for these
# https://github.com/pytorch/torchdynamo/issues/327
FALLBACK_ALLOW_LIST = OrderedSet(
    [
        "torchvision::roi_align",
        "aten::index_add",
    ]
)

log = logging.getLogger(__name__)
lowerings: dict[Union[Callable[..., Any], str], Callable[..., Any]] = {}
# Use maybe_layout_constraints to access this dict, we lazily register tag-based layout constraints
_maybe_layout_constraints: dict[
    torch._ops.OpOverload, Optional[Callable[..., Any]]
] = {}
fallbacks = OrderedSet[torch._ops.OpOverload]()
aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
needs_realized_inputs = OrderedSet[torch._ops.OpOverload]()
foreach_ops = OrderedSet[torch._ops.OpOverload](
    [torch._higher_order_ops._foreach_map]  # type: ignore[list-item]
)
# TODO(rec): torch._higher_order_ops._foreach_map is not an OpOverload
# so why is it in foreach_ops?
inplace_foreach_ops = OrderedSet[torch._ops.OpOverload]()
inplaceable_foreach_ops: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}
quantized_decomposed = torch.ops.quantized_decomposed


def cur_node_has_non_foreach_users():
    for node in V.graph.current_node.users:
        for user in node.users:
            if not (user.op == "call_function" and (user.target in foreach_ops)):
                return True

    return False


# group by device, whether any of the inputs are dynamic
# note arg_pairs may or may not be a pair
# foreach_map for example just passes output buffers here
def group_foreach_args(arg_pairs: Iterable[Union[tuple[Any, Any], Any]]):
    out = defaultdict(list)
    unpack_args = False
    for i, args in enumerate(arg_pairs):
        if not isinstance(args, Iterable):
            unpack_args = True
            args = (args,)
        use_foreach = (
            not is_dynamic(*args) or config.combo_kernel_foreach_dynamic_shapes
        )
        device = None
        for t in args:
            if isinstance(t, TensorBox):
                device = t.data.get_device()
                break
        assert device is not None, "foreach op should have at least one tensor arg"
        if unpack_args:
            # pyrefly: ignore [bad-unpacking]
            (args,) = args
        out[(device, use_foreach)].append((i, args))
    return out


def maybe_layout_constraints(fn: Callable[..., Any]) -> Optional[Callable[..., Any]]:
    """Get layout constraints. Returns None if there are no layout constraints."""
    if not isinstance(fn, torch._ops.OpOverload):
        # Only OpOverloads have layout constraints.
        return None

    if maybe_layout_tag := get_layout_constraint_tag(fn, with_default=False):
        return tag_to_layout_constraint(maybe_layout_tag)

    if fn in _maybe_layout_constraints:
        return _maybe_layout_constraints[fn]
    return None


def tag_to_layout_constraint(tag):
    if tag == torch._C.Tag.needs_exact_strides:
        return constrain_to_fake_tensors
    if tag == torch._C.Tag.needs_contiguous_strides:  # type: ignore[attr-defined]
        return require_contiguous_strides
    if tag == torch._C.Tag.needs_fixed_stride_order:
        return constrain_to_fx_strides
    if tag == torch._C.Tag.flexible_layout:
        return None
    raise AssertionError(f"Unknown layout constraint tag: {tag}")


def assert_nyi(cond, msg):
    if not cond:
        raise NotImplementedError(f"inductor does not support {msg}")


def add_needs_realized_inputs(fn):
    if isinstance(fn, (list, set, tuple, OrderedSet)):  # noqa: set_linter
        return [add_needs_realized_inputs(x) for x in fn]
    needs_realized_inputs.add(fn)
    if isinstance(fn, torch._ops.OpOverloadPacket):
        needs_realized_inputs.update(
            getattr(fn, overload) for overload in fn.overloads()
        )


def add_layout_constraint(fn, constraint):
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.overloads():
            _maybe_layout_constraints[getattr(fn, overload)] = constraint
    else:
        _maybe_layout_constraints[fn] = constraint


add_needs_realized_inputs(
    [
        aten.as_strided,
        aten.as_strided_copy,
        aten.avg_pool2d,
        aten.avg_pool2d_backward,
        aten.bmm,
        aten.convolution,
        aten.convolution_backward,
        aten.max_pool2d_with_indices,
        aten.max_pool3d_with_indices,
        aten.max_pool2d_with_indices_backward,
        aten.mm,
        aten.upsample_nearest2d,
        aten._upsample_nearest_exact2d,
        aten._int_mm,
    ]
)

# TODO(jansel): ezyang says we won't need this in the future, try removing it
# based on https://github.com/pytorch/pytorch/blob/9e3eb329df8f701/c10/core/ScalarType.h#L28
DTYPE_ID_LOOKUP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex32,
    11: torch.bool,
    15: torch.bfloat16,
    # TODO(jansel): add quantized types?
    #  _(c10::qint8, QInt8) /* 12 */
    # _(c10::quint8, QUInt8) /* 13 */
    # _(c10::qint32, QInt32) /* 14 */
    # _(c10::quint4x2, QUInt4x2) /* 16 */
    # _(c10::quint2x4, QUInt2x4) /* 17 */
}


def decode_dtype(dtype: int):
    if not isinstance(dtype, int):
        return dtype
    assert dtype in DTYPE_ID_LOOKUP, f"id {dtype} missing from DTYPE_ID_LOOKUP"
    # pyrefly: ignore [bad-assignment]
    dtype = DTYPE_ID_LOOKUP[dtype]
    return dtype


def is_integer_type(x):
    if isinstance(x, TensorBox):
        return is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    elif isinstance(x, sympy.Expr):
        return x.is_integer is True  # type: ignore[attr-defined]
    else:
        return isinstance(x, int)


def is_boolean_type(x):
    if isinstance(x, TensorBox):
        return is_boolean_dtype(x.get_dtype())
    else:
        return isinstance(x, bool)


def get_promoted_dtype(*args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):
    def construct_input(inp):
        if isinstance(inp, (Number, sympy.Basic)):
            return inp
        else:
            dim = len(inp.get_size())
            # construct a tmp tensor to feed into torch.result_type
            return torch.zeros([1] * dim, dtype=inp.get_dtype())

    inps = [construct_input(arg) for arg in args]
    _, dtype = elementwise_dtypes(*inps, type_promotion_kind=type_promotion_kind)
    return dtype


def get_overloads(aten_fn):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)

    return aten_fn


def in_namespace(op, namespace):
    if isinstance(op, torch._ops.OpOverloadPacket):
        return namespace in op._qualified_op_name
    elif isinstance(op, torch._ops.OpOverload):
        return namespace in op.name()
    return False


def maybe_copy_cpu_scalar(x: TensorBox, device: torch.device) -> TensorBox:
    """
    Copy cpu scalar if doesn't not match with given `device`
    """
    if not isinstance(x.data, ir.ReinterpretView) or has_free_unbacked_symbols(
        x.get_size()
    ):
        return x
    size = [V.graph.sizevars.size_hint_or_throw(s) for s in x.get_size()]
    cur_device = x.get_device()
    if (
        cur_device is not None
        and cur_device.type == "cpu"
        and cur_device != device
        and (len(size) == 0 or (len(size) == 1 and size[0] == 1))
    ):
        return TensorBox(ir.StorageBox(ir.DeviceCopy.create(x, cur_device, False)))
    return x


def transform_args(
    args: list[Any],
    kwargs: dict[str, Any],
    broadcast: bool,
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND],
    convert_input_to_bool: bool,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Transforms arguments for broadcasting and type promotion
    """

    args_indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    kwargs_indices = [k for k, v in kwargs.items() if isinstance(v, TensorBox)]
    # check that there's something to transform
    if not args_indices and not kwargs_indices:
        return args, kwargs

    if type_promotion_kind or convert_input_to_bool:
        if convert_input_to_bool:
            dtype = torch.bool
        else:
            # FIXME this is a crude approximation for promoting args
            promoting_args = [
                a
                for a in args
                if isinstance(a, (Number, sympy.Basic)) or hasattr(a, "dtype")
            ]
            # only consider tensor kwargs for promotion, for now
            promoting_args.extend(a for a in kwargs.values() if hasattr(a, "dtype"))
            dtype = get_promoted_dtype(
                *promoting_args,
                type_promotion_kind=type_promotion_kind,  # type: ignore[arg-type]
            )

        device = (
            args[args_indices[0]] if args_indices else kwargs[kwargs_indices[0]]
        ).get_device()

        for i in args_indices:
            args[i] = maybe_copy_cpu_scalar(args[i], device)

        for k in kwargs_indices:
            kwargs[k] = maybe_copy_cpu_scalar(kwargs[k], device)

        # sometimes args are an immutable list so we can't mutate them
        def promote(arg):
            if isinstance(arg, TensorBox):
                return to_dtype(arg, dtype)
            elif isinstance(arg, ir.Constant):
                return ir.Constant(value=arg.value, dtype=dtype, device=device)
            else:
                return arg

        args = [promote(a) for a in args]
        kwargs = {k: promote(v) for k, v in kwargs.items()}

    if broadcast:
        broadcasted = broadcast_tensors(
            *list(
                itertools.chain(
                    (args[i] for i in args_indices),
                    (kwargs[k] for k in kwargs_indices),
                )
            )
        )
        size = list(broadcasted[0].get_size())

        for i, x in zip(args_indices, broadcasted[: len(args_indices)]):
            args[i] = x
        for k, x in zip(kwargs_indices, broadcasted[len(args_indices) :]):
            kwargs[k] = x

        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], size)
        for k in kwargs:
            if isinstance(kwargs[k], ir.Constant):
                kwargs[k] = ExpandView.create(kwargs[k], size)

    return args, kwargs


def _register_foreach_lowering(aten_fn, decomp_fn):
    """
    Add a foreach lowering to lowerings dict.

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        assert len(args) <= 2
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out

    aten_fns = get_overloads(aten_fn)
    foreach_ops.update(aten_fns)
    lowerings.update(dict.fromkeys(aten_fns, wrapped))
    return wrapped


def _register_lowering(
    aten_fn,
    decomp_fn,
    broadcast,
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND],
    convert_input_to_bool,
    lowering_dict,
):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
        convert_input_to_bool: some logical ops require inputs are converted to bool
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args: list[Any] = list(args)
        kwargs: dict[str, Any] = dict(kwargs)
        unpacked = False
        # TODO maybe we need to use pytrees here
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(args[0])

        if not all(
            (fn in fallbacks or in_namespace(fn, "_c10d_functional")) for fn in aten_fn
        ):
            # explicitly assert for "out=" ops for better error messages
            assert not any(x == "out" for x in kwargs), "out= ops aren't yet supported"

        args, kwargs = transform_args(
            args, kwargs, broadcast, type_promotion_kind, convert_input_to_bool
        )

        if unpacked:
            args = [args]

        out = decomp_fn(*args, **kwargs)
        validate_ir(out)

        return out

    aten_fn = get_overloads(aten_fn)

    lowering_dict.update(dict.fromkeys(aten_fn, wrapped))
    return wrapped


def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind: Optional[
        ELEMENTWISE_TYPE_PROMOTION_KIND
    ] = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    lowering_dict=lowerings,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
        lowering_dict=lowering_dict,
    )


def broadcast_symbolic_shapes(a, b):
    """
    Broadcasting logic based on symbolic shapes.

    We give the shapes 0 and 1 concrete values, while all other shapes
    are symbolic sympy formulas.
    """
    output = []
    for x, y in itertools.zip_longest(reversed(a), reversed(b), fillvalue=sympy.S.One):
        if V.graph.sizevars.is_size_one_or_false(y):
            output.append(x)
        elif V.graph.sizevars.is_size_one_or_false(x):
            output.append(y)
        else:
            V.graph.sizevars.check_equals(x, y)
            if len(sympy.expand(y).free_symbols) < len(sympy.expand(x).free_symbols):
                output.append(y)  # prefer shorter formula
            else:
                output.append(x)
    return tuple(reversed(output))


def promote_constants(inputs, override_return_dtype=None, type_promotion_kind=None):
    assert override_return_dtype is None or type_promotion_kind is None, (
        "only one of override_return_dtype or type_promotion_kind may be given"
    )

    if override_return_dtype is None and type_promotion_kind is None:
        type_promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

    if not any(isinstance(x, (sympy.Basic, int, float)) for x in inputs):
        return inputs
    if all(isinstance(x, (int, float, sympy.Basic)) for x in inputs):
        dtype = override_return_dtype or get_promoted_dtype(
            *inputs,
            # pyrefly: ignore [bad-argument-type]
            type_promotion_kind=type_promotion_kind,
        )

        def const_func(x):
            if isinstance(x, sympy.Basic):
                return ir.IndexingConstant(
                    index=x, dtype=dtype, device=decode_device(None)
                )
            else:
                return ir.Constant(value=x, dtype=dtype, device=decode_device(None))

        return [const_func(x) for x in inputs]
    ex = next(x for x in inputs if isinstance(x, (TensorBox, ExpandView, ir.Constant)))
    out = []
    for x in inputs:
        if isinstance(x, (int, float)):
            out.append(
                ExpandView.create(
                    ir.Constant(
                        value=x, dtype=ex.get_dtype(), device=ex.get_device_or_error()
                    ),
                    list(ex.get_size()),
                )
            )
        elif isinstance(x, sympy.Basic):
            out.append(
                ExpandView.create(
                    IndexingConstant(
                        index=x, dtype=ex.get_dtype(), device=ex.get_device_or_error()
                    ),
                    list(ex.get_size()),
                )
            )
        else:
            out.append(x)

    return out


def make_pointwise(
    fn,
    override_return_dtype=None,
    override_device=None,
    override_fn_when_input_bool=None,
    allow_alpha=False,
    triton_fallback=None,
):
    def inner(*inputs: TensorBox, alpha=None):
        if triton_fallback is not None and any(
            isinstance(inp, IRNode) and is_triton(inp) for inp in inputs
        ):
            assert not allow_alpha  # not implemented
            return triton_fallback(*inputs)

        inputs = promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                # pyrefly: ignore [bad-assignment]
                inputs = list(inputs)
                # pyrefly: ignore [unsupported-operation]
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            assert alpha is None
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()

        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        # in tracing, we will annotate pointwise nodes that correspond to the output of
        # a pointwise node that would have been run in eager. intermediary pointwise nodes
        # during decompositions are not annotated.
        low_pr_fp = (torch.bfloat16, torch.float16)
        emulate_precision_casts = (
            V.graph is not None
            and getattr(V.graph, "current_node", None) is not None
            and V.graph.current_node.meta is not None
            and V.graph.current_node.meta.get("low_precision_pointwise_barrier", False)
        )
        emulate_output_cast = emulate_precision_casts and dtype in low_pr_fp

        def inner_fn(index):
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            else:
                inputs_loaded = []
                for inp_index, load in enumerate(loaders):
                    out = load(index)
                    inp_dtype = inputs[inp_index].get_dtype()
                    if emulate_precision_casts and inp_dtype in low_pr_fp:
                        downcast = ops.to_dtype(out, inp_dtype, use_compute_types=False)
                        out = ops.to_dtype(downcast, inp_dtype)
                    inputs_loaded.append(out)

                out = fn(*inputs_loaded)
                if emulate_output_cast:
                    # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
                    # then upcasting again, to emulate casts that eager would do.
                    downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                    return ops.to_dtype(downcast, dtype)
                return out

        if not override_device:
            device = None
            for i in inputs:
                # pyrefly: ignore [missing-attribute]
                if is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        # pyrefly: ignore [unbound-name]
        device = override_device or device

        return Pointwise.create(
            device=device,  # type: ignore[arg-type]
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner


def make_foreach_pointwise(pw_fn, allow_alpha=False):
    def inner(*inputs: list[list[TensorBox]], alpha=1):
        realize_outputs = (
            len(V.graph.current_node.users) == 0
            or V.graph.current_node.target in inplace_foreach_ops
            or cur_node_has_non_foreach_users()
        )

        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        assert a_list_input is not None, (
            "at least one input must be a list to a foreach op"
        )

        # broadcast scalar inputs to match length of list inputs
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                broadcast_inputs.append(input)

        groups = group_foreach_args(zip(*broadcast_inputs))

        outputs = [None] * len(a_list_input)
        for (device, use_foreach), group in groups.items():
            operation_list: list[str] = []
            for (
                output_ind,
                args,
            ) in group:
                if allow_alpha:
                    output = pw_fn(*args, alpha=alpha)
                else:
                    output = pw_fn(*args)

                outputs[output_ind] = output

                if (
                    # pyrefly: ignore [unbound-name]
                    V.graph.has_feature(device, BackendFeature.FOREACH)
                    and use_foreach
                    and realize_outputs
                ):
                    output.realize()
                    operation_list.append(output.get_operation_name())

            if operation_list:
                # pyrefly: ignore [unbound-name]
                V.graph.register_operation_list(operation_list)

        assert all(x is not None for x in outputs)
        return outputs

    return inner


def to_dtype(
    x: Union[TensorBox, ShapeAsConstantBuffer], dtype: torch.dtype, copy: bool = False
):
    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return clone(x) if copy else x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype, src_dtype=src_dtype)

    return make_pointwise(_to_dtype, override_return_dtype=dtype)(x)


@register_lowering(torch._higher_order_ops._foreach_map, type_promotion_kind=None)
def _foreach_map(subgraph, *args, **kwargs):
    """
    This lowers an invocation of foreach_map
    The way this works is that an arbitrary N-arg func is provided by the user, looped over by the
    polyfill with the same semantics as a foreach op (a loop applying an n-ary function to n args)
    and then traced into a subgraph by dynamo.
    This code allows us to inline the subgraph into the main graph lowering using the PontwiseSubgraphLowering.
    The graph outputs represent the vertically fused sequence of ops, and then register_operation_list
    below registers the buffers as horizontally fuseable in the scheduler.
    """
    from .subgraph_lowering import PointwiseSubgraphLowering

    inputs = args

    gm = subgraph.graph_module
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*inputs)

    sub_outputs = pw_subgraph.graph_outputs
    # group outputs by device and register as foreach
    assert sub_outputs  # mypy lol
    groups = group_foreach_args(sub_outputs)

    outputs = [None] * len(sub_outputs)
    for (device, use_foreach), group in groups.items():
        operation_list: list[str] = []
        for (
            output_ind,
            output,
        ) in group:
            outputs[output_ind] = output

            if V.graph.has_feature(device, BackendFeature.FOREACH) and use_foreach:
                output.realize()
                operation_list.append(output.get_operation_name())

        if operation_list:
            V.graph.register_operation_list(operation_list)

    assert all(x is not None for x in outputs)
    return outputs


@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def _convert_element_type(x: TensorBox, dtype: torch.dtype):
    if dtype.is_complex or x.get_dtype().is_complex:
        if x.get_size():
            # Decompose since aa aten fallback is more friendly for c++ codegen.
            # This decomposition doesn't work for empty tensor, which needs more investigation.
            dst = empty_like(x, dtype=dtype)
            ir.InplaceCopyFallback.create(dst, x)
            return dst
        else:
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)
    return to_dtype(x, dtype, copy=True)


def to_dtype_bitcast(x: TensorBox, dtype: torch.dtype, *, copy=False):
    x_dtype = x.get_dtype()
    if x_dtype == dtype:
        return clone(x) if copy else x

    def _get_primitive_bitwidth(dtype):
        if dtype.is_floating_point:
            return torch.finfo(dtype).bits
        else:
            return torch.iinfo(dtype).bits

    src_bits = _get_primitive_bitwidth(x_dtype)
    dst_bits = _get_primitive_bitwidth(dtype)
    if src_bits != dst_bits:
        # fallback to aten eager implementation for differing bitwidths
        return fallback_handler(aten.view.dtype)(x, dtype)
    else:
        return TensorBox(DtypeView.create(x, dtype))


@register_lowering(aten.view.dtype, type_promotion_kind=None)
def _view_dtype(x: TensorBox, dtype: torch.dtype):
    if dtype.is_complex or x.get_dtype().is_complex:
        return TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view.dtype, x, dtype)
        )
    return to_dtype_bitcast(x, dtype)


def to_device(x: TensorBox, device: torch.device, *, copy=False, non_blocking=False):
    device = decode_device(device)
    if x.get_device() == device:
        return clone(x) if copy else x
    return TensorBox.create(ir.DeviceCopy.create(x, device, non_blocking))


@register_lowering(prims.device_put, type_promotion_kind=None)
def _device_put(x: TensorBox, device: torch.device, non_blocking=False):
    return to_device(x, device, copy=True, non_blocking=non_blocking)


def register_pointwise(
    aten_fn,
    name=None,
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    override_return_dtype=None,
    override_fn_when_input_bool=None,
    allow_alpha=False,
    triton_fallback=None,
):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__
    fn = ops_wrapper(name)

    register_op_dtype_propagation_rules(
        name, type_promotion_kind, override_return_dtype
    )

    if override_fn_when_input_bool is not None:
        override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)

    fn = make_pointwise(
        fn,
        override_return_dtype=override_return_dtype,
        override_fn_when_input_bool=override_fn_when_input_bool,
        allow_alpha=allow_alpha,
        triton_fallback=triton_fallback,
    )
    fn = register_lowering(
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )(fn)

    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
            convert_input_to_bool=convert_input_to_bool,
        )(fn)
    return fn


def register_frexp():
    """A pointwise function that maps ops.frexp to inputs"""
    name = "frexp"
    frexp = ops_wrapper("frexp")

    def frexp0(*args, **kwargs):
        return frexp(*args, **kwargs)[0]  # type: ignore[index]

    def frexp1(*args, **kwargs):
        return frexp(*args, **kwargs)[1]  # type: ignore[index]

    pw_fns = [
        make_pointwise(frexp0),
        make_pointwise(frexp1, override_return_dtype=torch.int32),
    ]

    def fn(*args, **kwargs):
        return pw_fns[0](*args, **kwargs), pw_fns[1](*args, **kwargs)

    fn = register_lowering(
        aten.frexp,
    )(fn)

    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
        )(fn)
    return fn


register_frexp()


def register_foreach_pointwise(
    aten_fn,
    pointwise_lowering_fn,
    allow_alpha=False,
):
    fn = make_foreach_pointwise(pointwise_lowering_fn, allow_alpha=allow_alpha)
    fn = _register_foreach_lowering(aten_fn, fn)
    return fn


@register_lowering(aten.where, broadcast=False, type_promotion_kind=None)
def where(cond, a, b):
    def fn(*args):
        return ops.where(*args)

    if isinstance(a, (float, int)):
        a = constant_like(a)(b)
    if isinstance(b, (float, int)):
        b = constant_like(b)(a)

    args = [cond, a, b]
    dtype = get_promoted_dtype(
        args[1], args[2], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    return make_pointwise(fn, override_return_dtype=dtype)(
        args[0], to_dtype(args[1], dtype), to_dtype(args[2], dtype)
    )


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promotion_kind=None)
def broadcast_tensors(*inputs):
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        return broadcast_tensors(*inputs[0])
    target: list[sympy.Expr] = functools.reduce(
        broadcast_symbolic_shapes, [x.get_size() for x in inputs], []
    )
    outputs = []
    for x in inputs:
        sizes = x.get_size()

        if len(sizes) != len(target) or any(
            V.graph.sizevars.is_size_one_or_false(a)
            != V.graph.sizevars.is_size_one_or_false(b)
            for a, b in zip(sizes, target)
        ):
            x = expand(x, target)
        outputs.append(x)
    return outputs


@register_lowering([aten.alias, aten.detach, aten.detach_, aten.lift, prims.view_of])
def nop(x):
    return x  # AOT autograd handles this for us


if hasattr(aten, "lift_fresh"):
    register_lowering(aten.lift_fresh)(nop)


@register_lowering(aten.squeeze, type_promotion_kind=None)
def squeeze(x, dim=None):
    assert isinstance(x, TensorBox)
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))

    dim = (
        V.graph.sizevars.guard_int(dim)
        if isinstance(dim, (int, sympy.Expr))
        else tuple(V.graph.sizevars.guard_int(d) for d in dim)
    )
    dim = canonicalize_dims(len(x.get_size()), dim)  # type: ignore[call-overload]
    dims = OrderedSet((dim,) if not isinstance(dim, tuple) else dim)

    new_shape = []
    for d, s in enumerate(x.get_size()):
        if not (d in dims and V.graph.sizevars.guard_or_false(sympy.Eq(s, 1))):
            new_shape.append(s)

    # squeeze does nothing if the size isn't 1
    return view(x, new_shape) if new_shape != x.get_size() else x


@register_lowering(aten.squeeze_copy, type_promotion_kind=None)
def squeeze_copy(x, dim=None):
    return clone(squeeze(x, dim))


@register_lowering([aten.squeeze_])
def squeeze_(x, dim=None):
    val = squeeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    x.data = val.data
    return x


@register_lowering(aten.isinf)
def isinf(x):
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper("isinf")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


@register_lowering(aten.isnan)
def isnan(x):
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper("isnan")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


@register_lowering(aten.ceil)
def ceil(x):
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("ceil")
    return make_pointwise(fn)(x)


@register_lowering(aten.floor)
def floor(x):
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("floor")
    return make_pointwise(fn)(x)


@register_lowering(aten.round.default)
def round(x):
    if is_integer_type(x):
        return clone(x)
    else:
        fn = ops_wrapper("round")
        return make_pointwise(fn)(x)


@register_lowering(aten.trunc)
def trunc(x):
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper("trunc")
    return make_pointwise(fn)(x)


@register_lowering(aten.expand, type_promotion_kind=None)
def expand(x, sizes):
    (x,) = promote_constants([x])
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    if tuple(x.get_size()) == tuple(sizes):
        return x

    if not free_unbacked_symbols(x.get_size()):
        x_size_product = V.graph.sizevars.size_hint_or_throw(
            sympy_product(x.get_size())
        )
        # TODO: It would be better to realize the input if any of its sizes
        # are unbacked, because typically the size will be non-zero.  However,
        # this cannot be done directly as below as we'll choke on the size_hint
        # here
        if x_size_product > 0 and not free_unbacked_symbols(sizes):
            # maybe realize input before broadcasting it
            x.mark_reuse(
                V.graph.sizevars.size_hint_or_throw(sympy_product(sizes))
                // x_size_product
            )
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))


@register_lowering(prims.broadcast_in_dim, type_promotion_kind=None)
def broadcast_in_dim(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = unsqueeze(v, idx)

    return expand(v, shape)


@register_lowering(aten.expand_as, type_promotion_kind=None)
def expand_as(x, y):
    return expand(x, y.get_size())


@register_lowering(aten.repeat)
def repeat(x, repeats):
    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.S.One] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    assert len(repeats) == len(x.get_size())

    new_size = list(x.get_size())

    zero_tensor = False
    for i in range(len(repeats)):
        if repeats[i] == 0:
            zero_tensor = True
        new_size[i] = new_size[i] * repeats[i]

    if zero_tensor:
        return empty(new_size, dtype=x.get_dtype(), device=x.get_device())
    if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
        return clone(expand(x, new_size))

    x_loader: Callable[[Any], Any]

    def inner_fn(index):
        assert len(index) == len(repeats)
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] != 1:
                if old_size[i] == 1:
                    index[i] = sympy.S.Zero
                else:
                    index[i] = ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)

    if not free_unbacked_symbols(old_size) and not free_unbacked_symbols(new_size):
        old_size_product = V.graph.sizevars.size_hint_or_throw(sympy_product(old_size))
        if old_size_product > 0:
            # maybe realize the input but skip for unbacked symints since it'll
            # choke on the size hint.
            x.mark_reuse(
                V.graph.sizevars.size_hint_or_throw(sympy_product(new_size))
                // old_size_product
            )

    x_loader = x.make_loader()
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(new_size),
    )


@register_lowering(aten._unsafe_view, type_promotion_kind=None)
@register_lowering(aten.view, type_promotion_kind=None)
@register_lowering(aten.reshape, type_promotion_kind=None)
def view(x: TensorBox, sizes: Sequence[sympy.Expr]) -> TensorBox:
    return TensorBox(View.create(x.data, sizes))


@register_lowering(aten.permute, type_promotion_kind=None)
def permute(x, dims):
    assert isinstance(x, TensorBox)
    assert isinstance(dims, (list, tuple))
    return TensorBox(PermuteView.create(x.data, tuple(dims)))


@register_lowering(aten.slice, type_promotion_kind=None)
def slice_(x, dim=0, start=0, end=2**63, step=1, clamp=True):
    """
    Lowers a slice call, creating ExternKernels for the output size & storage offset symbols,
    if the indices are unbacked and appropriate semantics aren't known.
    If they are known (indices are static/backed/unbacked with info), a SliceView is created.
    """

    from torch.fx.experimental.symbolic_shapes import (
        CallMethodKey,
        resolve_unbacked_bindings,
    )

    assert isinstance(x, TensorBox)
    dim = _validate_dim(x, dim, 0)
    size = x.get_size()[dim]
    step = sympy.expand(step)
    assert isinstance(step, sympy.Expr) or step > 0, step

    # maybe apply slice optimization
    try:
        if (
            start == 0
            and V.graph.sizevars.statically_known_leq(size, end)
            and step == 1
        ):
            return x
    except TypeError:
        pass

    # try to avoid dynamic (unbacked) slice
    def compute_slice_index(index, size, default=None):
        if index is None:
            return default

        fn = lambda x: V.graph.sizevars.guard_or_false(x)  # noqa: E731
        index = sympy.expand(index)
        size = sympy.expand(size)
        if fn(sympy.Ge(index, 0)) and fn(sympy.Le(index, size)):
            return index
        elif fn(sympy.Lt(index, 0)) and fn(sympy.Ge(index, -size)):
            return index + size
        elif fn(sympy.Gt(index, size)):
            return size
        elif fn(sympy.Lt(index, -size)):
            return 0
        return None

    start_index, end_index = None, None
    ambiguous_slice = clamp
    if ambiguous_slice:
        start_index = compute_slice_index(start, size, 0)
        end_index = compute_slice_index(end, size, size)
        if start_index is not None and end_index is not None:
            start, end = start_index, end_index
            ambiguous_slice = False

    # ambiguous_slice=False means we know what semantics this slice call follows,
    # and don't need to generate an extern kernel to represent the output size.
    # This is assumed True for clamp=False
    # (meant to follow standard indexing semantics: 0 <= index < size)
    if not ambiguous_slice:
        return TensorBox(
            ir.SliceView.create(x.data, dim, start, end, step, clamp=clamp)
        )  # go to SliceView/ReinterpretView

    # unbacked territory: create DynamicSlice ExternKernel
    # clamp is True, unbacked start / end
    assert clamp
    unbacked_bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env, V.graph.current_node.meta["unbacked_bindings"]
    )
    assert unbacked_bindings is not None
    assert len(unbacked_bindings) <= 2, unbacked_bindings
    sym_size, sym_storage = None, None
    for sym, keypath in unbacked_bindings.items():
        if keypath == (CallMethodKey("size"), pytree.SequenceKey(dim)):
            sym_size = sym
        elif keypath == (CallMethodKey("storage_offset"),):
            sym_storage = sym

    assert start_index is None or end_index is None
    b_size = ir.DynamicSliceSize(
        sym_size,
        start,
        end,
        step,
        x.get_size()[dim],
    )
    b_size.name = V.graph.register_buffer(b_size)
    V.graph.register_operation(b_size)
    new_size = sym_size

    if x.maybe_get_layout() is None:
        # realize tensor before accessing layout
        x.realize()

    if start_index is not None:
        # we shouldn't have allocated storage offset symbol if start index was determinable
        assert sym_storage is None
        new_storage_offset = x.get_layout().offset + start_index * x.get_stride()[dim]
    else:
        b_storage = ir.DynamicSelectStorageOffset(
            sym_storage,
            start,
            x.get_layout().offset,
            x.get_stride()[dim],
            x.get_size()[dim],
            clamp=True,
        )
        b_storage.name = V.graph.register_buffer(b_storage)
        V.graph.register_operation(b_storage)
        new_storage_offset = sym_storage

    new_sizes = list(x.get_size())
    new_strides = list(x.get_stride())
    new_sizes[dim] = new_size
    new_strides[dim] *= step
    return as_strided(x, new_sizes, new_strides, new_storage_offset)


@register_lowering(aten.as_strided, type_promotion_kind=None)
def as_strided(x, size, stride, storage_offset=None):
    new_device = None
    new_dtype = None
    if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
        # Note: Merging views
        # When we use as_strided, we can rewrite the size/stride/offset
        # of the incoming buffer x. If x is a view, we would overwrite
        # its metadata. Except for dtype, which we need to propagate.

        # Technically device is not needed because it is not possible
        # to have a cross-device view today.
        new_device = x.get_device()
        new_dtype = x.dtype
        x = x.data.unwrap_view()
    x.realize()
    if not ir.is_storage_and_layout(x):
        raise NotImplementedError(f"unrealized as_strided({x}, ...)")
    storage, old_layout = ir.as_storage_and_layout(x)
    new_layout = ir.FixedLayout(
        new_device if new_device else old_layout.device,
        new_dtype if new_dtype else old_layout.dtype,
        [sympy.expand(s) for s in size],
        [sympy.expand(s) for s in stride],
        sympy.expand(storage_offset or 0),
    )
    return TensorBox(ir.ReinterpretView(data=storage, layout=new_layout))


@register_lowering(aten.as_strided_, type_promotion_kind=None)
def as_strided_(x, size, stride, storage_offset=None):
    assert isinstance(x, TensorBox)
    x.data = as_strided(x, size, stride, storage_offset).data
    return x


@register_lowering(aten.as_strided_copy, type_promotion_kind=None)
def as_strided_copy(x, size, stride, storage_offset=None):
    result = as_strided(x, size, stride, storage_offset)
    return clone(result)


def pointwise_cat(inputs, dim=0):
    # (inclusive, exclusive)
    inputs_ranges: list[tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))  # type: ignore[arg-type]
        prev_end = inputs_ranges[-1][-1]  # type: ignore[assignment]

    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        idx_dim = ops.index_expr(idx[dim], torch.int64)

        masks = []
        masked_loads = []
        for i in range(len(inputs)):
            start = (
                ops.constant(0, torch.int64)
                if i == 0
                else ops.index_expr(inputs_ranges[i][0], torch.int64)
            )
            end = ops.index_expr(inputs_ranges[i][1], torch.int64)

            start_cond = ops.ge(idx_dim, start)
            end_cond = ops.lt(idx_dim, end)
            if i == 0:
                mask = end_cond
            elif i == len(inputs) - 1:
                mask = start_cond
            else:
                mask = ops.and_(start_cond, end_cond)

            masks.append(mask)
            idx_load = list(idx)

            # if we're concatting [4], [2]
            # when we index the second tensor for 5 we want to index 5 - 4
            # Use Identity to prevent expansion of index * stride to keep expression
            # in same int bitwidth as shape
            idx_load[dim] = Identity(idx_load[dim] - inputs_ranges[i][0])

            masked_loads.append(
                ops.masked(
                    mask,
                    lambda: inputs_loaders[i](idx_load),
                    0.0,  # this value should be unused
                ),
            )

        next_val = masked_loads[-1]
        for i in range((len(inputs)) - 2, -1, -1):
            next_val = ops.where(
                masks[i],
                masked_loads[i],
                next_val,
            )
        return next_val

    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]

    return Pointwise.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
    )


@register_lowering(quantized_decomposed.quantize_per_channel, type_promotion_kind=None)
def quantized_decomposed_quantize_per_channel(
    input: TensorBox,
    scales: TensorBox,
    zero_points: TensorBox,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    assert len(scales.get_size()) == 1, "expect scales 1 dim"
    assert len(zero_points.get_size()) == 1, "expect zero_points 1 dim"

    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    assert input.get_dtype() == torch.float32, (
        f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
    )
    assert axis < len(input.get_size()), (
        f"Expecting axis to be < {len(input.get_size())}"
    )

    input_loader = input.make_loader()
    scales_loader = scales.make_loader()
    zero_points_loader = zero_points.make_loader()

    def inner_fn(idx):
        channel_idx = (idx[axis],)

        input = input_loader(idx)
        scale = scales_loader(channel_idx)
        zero_point = zero_points_loader(channel_idx)
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)

        if scales.dtype != torch.float32:
            scale = ops.to_dtype(scale, torch.float32)
        if zero_points.dtype != torch.int32:
            zero_point = ops.to_dtype(zero_point, torch.int32)
        inv_scale = ops.reciprocal(scale)
        val = ops.round(input * inv_scale) + zero_point
        clamped = ops.maximum(qmin, ops.minimum(qmax, val))
        return ops.to_dtype(clamped, dtype)

    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


def _assert_async(cond, msg):
    cond.realize()
    cond = to_dtype(cond, torch.bool)

    def inner_fn(index):
        with ir.ComputedBuffer.force_realize():
            return ops.device_assert_async(cond.make_loader()(index), msg)

    assertion_op = Pointwise.create(
        device=cond.get_device(),
        dtype=cond.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(cond.get_size()),
    )
    assertion_op.realize()
    return assertion_op


@register_lowering(aten._assert_async.msg)
def lower_assert_async(cond, msg):
    return _assert_async(cond, msg)


@register_lowering(aten._functional_assert_async.msg)
def lower_assert_functional_async(cond, msg):
    return _assert_async(cond, msg)


@register_lowering(
    quantized_decomposed.dequantize_per_channel, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_channel(
    input: TensorBox,
    scales: TensorBox,
    zero_points: TensorBox,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    assert len(scales.get_size()) == 1, "expect scales 1 dim"
    assert len(zero_points.get_size()) == 1, "expect zero_points 1 dim"
    assert input.get_dtype() == dtype, (
        f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
    )
    assert axis < len(input.get_size()), (
        f"Expecting axis to be < {len(input.get_size())}"
    )

    if out_dtype is None:
        out_dtype = torch.float32

    input_loader = input.make_loader()
    scales_loader = scales.make_loader()
    zero_points_loader = zero_points.make_loader()

    def inner_fn(idx):
        channel_idx = (idx[axis],)

        input = input_loader(idx)
        scale = scales_loader(channel_idx)
        zero_point = zero_points_loader(channel_idx)

        if scales.dtype != torch.float32:
            scale = ops.to_dtype(scale, torch.float32)
        if zero_points.dtype != torch.float32:
            zero_point = ops.to_dtype(zero_point, torch.float32)
        val = ops.sub(ops.to_dtype(input, torch.float32), zero_point) * scale
        val = ops.to_dtype(val, out_dtype)
        return val

    return Pointwise.create(
        device=input.get_device(),
        dtype=out_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.quantize_per_tensor.default, type_promotion_kind=None
)
def quantized_decomposed_quantize_per_tensor_default(
    input: TensorBox,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    assert input.get_dtype() == torch.float32, (
        f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
    )

    input_loader = input.make_loader()

    def inner_fn(idx, scale, zero_point):
        input = input_loader(idx)
        inv_scale, zero_point = _create_constants(
            1.0 / scale, zero_point, dtype=torch.float32
        )
        val = ops.round(input * inv_scale) + zero_point
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)
        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
        return ops.to_dtype(clamped, dtype)

    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=functools.partial(
            inner_fn, scale=float(scale), zero_point=int(zero_point)
        ),
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.dequantize_per_tensor.default, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_tensor_default(
    input: TensorBox,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    assert input.get_dtype() == dtype, (
        f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
    )

    if out_dtype is None:
        out_dtype = torch.float32

    input_loader = input.make_loader()

    def inner_fn(idx, scale, zero_point):
        input = input_loader(idx)
        scale, zero_point = _create_constants(scale, zero_point, dtype=torch.float32)
        val = ops.sub(ops.to_dtype(input, torch.float32), zero_point) * scale
        val = ops.to_dtype(val, out_dtype)
        return val

    return Pointwise.create(
        device=input.get_device(),
        dtype=out_dtype,
        inner_fn=functools.partial(
            inner_fn, scale=float(scale), zero_point=int(zero_point)
        ),
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.quantize_per_tensor.tensor, type_promotion_kind=None
)
def quantized_decomposed_quantize_per_tensor_tensor(
    input: TensorBox,
    scale: TensorBox,
    zero_point: TensorBox,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    assert input.get_dtype() == torch.float32, (
        f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
    )
    assert len(scale.get_size()) == 0 or (
        len(scale.get_size()) == 1 and scale.get_size()[0] == 1
    ), "expect scale as scalar tensor"
    assert len(zero_point.get_size()) == 0 or (
        len(zero_point.get_size()) == 1 and zero_point.get_size()[0] == 1
    ), "expect zero_point as scalar tensor"

    input_loader = input.make_loader()
    scale_loader = scale.make_loader()
    zero_point_loader = zero_point.make_loader()

    def inner_fn(idx):
        input = input_loader(idx)
        _scale = scale_loader((0,) if len(scale.get_size()) == 1 else ())
        _zero_point = zero_point_loader((0,) if len(scale.get_size()) == 1 else ())
        if scale.dtype != torch.float32:
            _scale = ops.to_dtype(_scale, torch.float32)
        if zero_point.dtype != torch.float32:
            _zero_point = ops.to_dtype(_zero_point, torch.float32)
        val = ops.round(input * ops.reciprocal(_scale)) + _zero_point
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)
        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
        return ops.to_dtype(clamped, dtype)

    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.dequantize_per_tensor.tensor, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_tensor_tensor(
    input: TensorBox,
    scale: TensorBox,
    zero_point: TensorBox,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    assert len(scale.get_size()) == 0 or (
        len(scale.get_size()) == 1 and scale.get_size()[0] == 1
    ), "expect scale as scalar tensor"
    assert len(zero_point.get_size()) == 0 or (
        len(zero_point.get_size()) == 1 and zero_point.get_size()[0] == 1
    ), "expect zero_point as scalar tensor"
    assert input.get_dtype() == dtype, (
        f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
    )

    if out_dtype is None:
        out_dtype = torch.float32

    input_loader = input.make_loader()
    scale_loader = scale.make_loader()
    zero_point_loader = zero_point.make_loader()

    def inner_fn(idx):
        input = input_loader(idx)
        _scale = scale_loader((0,) if len(scale.get_size()) == 1 else ())
        _zero_point = zero_point_loader((0,) if len(scale.get_size()) == 1 else ())
        if scale.dtype != torch.float32:
            _scale = ops.to_dtype(_scale, torch.float32)
        if zero_point.dtype != torch.float32:
            _zero_point = ops.to_dtype(_zero_point, torch.float32)
        val = ops.sub(ops.to_dtype(input, torch.float32), _zero_point) * _scale
        val = ops.to_dtype(val, out_dtype)
        return val

    return Pointwise.create(
        device=input.get_device(),
        dtype=out_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(aten.cat)
def cat(inputs, dim=0):
    cpu_device = inputs[0].get_device().type == "cpu"
    if cpu_device and all(
        input.get_dtype() in [torch.int8, torch.uint8] for input in inputs
    ):
        # TODO <leslie> Remove this fallback when we support vectorization
        # code gen with uint8 data type directly.
        for input in inputs:
            input.realize()
        if all(len(input.get_size()) == 4 for input in inputs):
            inputs, _ = require_channels_last(aten.cat, *inputs)
        return fallback_handler(aten.cat.default)(inputs, dim)

    if len(inputs) == 1:
        return clone(inputs[0])

    dim = _validate_dim(inputs[0], dim, 0)
    dtype = get_promoted_dtype(
        *inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [to_dtype(inp, dtype) for inp in inputs]

    def unwrap_tensor(x: Union[TensorBox, ir.StorageBox]) -> ir.IRNode:
        if isinstance(x, TensorBox):
            if isinstance(x.data, ir.BaseView):
                return x.data.unwrap_view()
            else:
                return x.data

        if isinstance(x, ir.StorageBox):
            return x.data

        return x

    def is_reduction(t):
        return isinstance(t, ir.ComputedBuffer) and isinstance(t.data, ir.Reduction)

    def can_fuse_reduction(t):
        if isinstance(t, (TensorBox, ir.StorageBox)):
            return can_fuse_reduction(unwrap_tensor(t))
        return (
            is_reduction(t)
            or isinstance(t, ir.Pointwise)
            and any(
                can_fuse_reduction(V.graph.get_buffer(read))
                for read in t.get_read_names()
            )
        )

    # fusing reducutions into computed concat buffer can cause regressions.
    fusable_reduction = any(can_fuse_reduction(t) for t in inputs)

    def should_lower_cat_input(x) -> bool:
        # Unrealized inputs will not be storage and layouts, and we dont want to realize
        # them in case we want to fuse
        if ir.is_storage_and_layout(x):
            storage, _ = ir.as_storage_and_layout(x, freeze=False)
            return not ir.ConcatKernel.can_realize_into_without_copy(storage)

        if isinstance(x, (TensorBox, ir.StorageBox)):
            return should_lower_cat_input(unwrap_tensor(x))

        if isinstance(x, ir.Pointwise):
            return True

        return False

    if config.force_pointwise_cat:
        return pointwise_cat(inputs, dim)

    # TODO: We observed negative performance impact of pointwise_cat optimization on CPU so disabled it.
    #             We will revisit this later after enabling vectorization on index_expr.
    if cpu_device:
        return TensorBox(ir.ConcatKernel.create(inputs, dim))

    def op_count(x):
        if isinstance(x, (TensorBox, ir.StorageBox)):
            return op_count(unwrap_tensor(x))

        # this will correspond to a direct memory read
        if not isinstance(x, ir.Pointwise):
            return 0

        count = x.inner_fn_opcount().num_ops
        for read in x.get_read_names():
            count += op_count(V.graph.get_buffer(read))

        return count

    # as of inputs increase, possibility for register spilling also increases
    # past a certain threshold of inputs we only fuse if the if the input kernels
    # are simple
    # not sure if we want to expose to users via config since logic may change in future
    MAX_COMPLEX_POINTWISE_CAT = 8
    MAX_SIMPLE_OP_COUNT = 2

    def additional_pointwise_ops(op: torch._ops.OpOverload):
        return op in (aten.cat.default, aten.constant_pad_nd.default)

    if len(inputs) <= MAX_COMPLEX_POINTWISE_CAT or (
        (len(inputs) <= config.max_pointwise_cat_inputs)
        and all(op_count(t) <= MAX_SIMPLE_OP_COUNT for t in inputs)
    ):
        pointwise_uses = all(
            is_pointwise_use(use, additional_pointwise_ops)
            for use in V.current_node.users
        )
        # fuse in case we will be used in a pointwise node, and there are any inputs we
        # we can prevent materialization of.
        fuse_pointwise_use = (
            any(should_lower_cat_input(inp) for inp in inputs) and pointwise_uses
        )

        # horizontal fuse in case all inputs will require a copy kernel anyway.
        # only horizontally fuse pointwise kernels
        horizontal_fuse_cat = all(
            should_lower_cat_input(inp) for inp in inputs
        ) and not any(can_fuse_reduction(t) for t in inputs)
        if fuse_pointwise_use or (horizontal_fuse_cat and not fusable_reduction):
            return pointwise_cat(inputs, dim)

    return TensorBox(ir.ConcatKernel.create(inputs, dim))


@register_lowering(aten.diagonal, type_promotion_kind=None)
def diagonal(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    original_shape = input.get_size()
    num_dims = len(original_shape)
    dim1 = canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = canonicalize_dim(idx=dim2, rank=num_dims)

    check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    offset_negative = V.graph.sizevars.evaluate_expr(sympy.Lt(offset, 0))
    if offset_negative:
        diag_size = V.graph.sizevars.evaluate_max(
            V.graph.sizevars.evaluate_min(
                original_shape[dim1] + offset, original_shape[dim2]
            ),
            0,  # type: ignore[arg-type]
        )
    else:
        diag_size = V.graph.sizevars.evaluate_max(
            V.graph.sizevars.evaluate_min(
                original_shape[dim1], original_shape[dim2] - offset
            ),
            0,  # type: ignore[arg-type]
        )

    base_idx = (0, 0)
    if offset_negative:
        base_idx = (-offset, 0)
    else:
        base_idx = (0, offset)

    sizes = [s for i, s in enumerate(original_shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    def reindexer(idx):
        diag_idx = idx[-1]
        original_idx = [0] * len(original_shape)
        cur_dim = 0
        for d in range(num_dims):
            if d == dim1:
                original_idx[d] = diag_idx + base_idx[0]
            elif d == dim2:
                original_idx[d] = diag_idx + base_idx[1]
            else:
                original_idx[d] = idx[cur_dim]
                cur_dim += 1

        assert cur_dim == len(original_shape) - 2
        return original_idx

    return TensorBox(ir.GenericView.create(input, sizes, reindexer))


@register_lowering(aten.diagonal_copy, type_promotion_kind=None)
def diagonal_copy(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    return clone(diagonal(input, offset, dim1, dim2))


@register_lowering(aten.diagonal_scatter, type_promotion_kind=None)
def diagonal_scatter(input, src, offset: int = 0, dim1: int = 0, dim2: int = 1):
    output = clone(input)
    target = diagonal(output, offset, dim1, dim2)
    mutate_to(target, src)
    return output


@register_lowering(aten.select, type_promotion_kind=None)
def select(x, dim, idx):
    idx = sympy.expand(idx)
    size = sympy.expand(x.get_size()[dim])
    actual_index = None

    if V.graph.sizevars.guard_or_false(sympy.Lt(idx, 0)):
        actual_index = idx + size
    elif V.graph.sizevars.guard_or_false(sympy.Ge(idx, 0)):
        actual_index = idx

    if actual_index is not None:
        if has_free_unbacked_symbols(idx):
            # Inductor could generate incorrect views for tensors with unbacked symbols here;
            # Squeeze operations are translated to views, resulting in incorrect strides.
            # Additionally, we want to avoid accidental unbacked unsqueeze semantics. To resolve this,
            # we use as_strided instead.
            # Removing this branch will cause test_unbacked_select_index_with_check to fail.

            # before accessing size, stride, and offset we need to realize.
            x.realize()
            new_size = x.get_size()
            new_stride = x.get_stride()
            new_storage_offset = x.get_layout().offset + new_stride[dim] * actual_index

            del new_size[dim]
            del new_stride[dim]
            return as_strided(x, new_size, new_stride, new_storage_offset)
        else:
            # no need to clamp, this function handles negative indexing itself
            slice_result = slice_(x, dim, actual_index, actual_index + 1, clamp=False)
            return squeeze(slice_result, dim)

    # Unbacked Semantics:
    # When the index idx is unbacked (e.g., u0), we compute the index dynamically
    # during the lowering of the select operation using DynamicSelectStorageOffset.

    unbacked_bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env, V.graph.current_node.meta["unbacked_bindings"]
    )
    assert unbacked_bindings is not None
    assert len(unbacked_bindings) == 1, unbacked_bindings
    unbacked_offset_sym, _ = next(iter(unbacked_bindings.items()))

    # before accessing size, stride, and offset we need to realize.
    x.realize()
    new_size = x.get_size()
    new_stride = x.get_stride()
    new_storage_offset = unbacked_offset_sym
    buffer = ir.DynamicSelectStorageOffset(
        unbacked_offset_sym,
        idx,
        x.get_layout().offset,
        new_stride[dim],
        x.get_size()[dim],
        clamp=False,
    )
    buffer.name = V.graph.register_buffer(buffer)
    V.graph.register_operation(buffer)

    del new_size[dim]
    del new_stride[dim]
    return as_strided(x, new_size, new_stride, new_storage_offset)


@register_lowering(aten.split, type_promotion_kind=None)
def split(x, sizes, dim=0):
    dim = _validate_dim(x, dim, 0)
    sizes_ = sizes

    # If sizes is an integer (or a SymInt), we turn it into a list of sizes
    # by computing what the actual size of each chunk should be.
    if not isinstance(sizes, (list, tuple)):
        x_size = x.get_size()[dim]
        chunks = V.graph.sizevars.guard_int(FloorDiv(x_size + sizes - 1, sizes))
        sizes_ = [sizes] * chunks
        # The last chunk might have a smaller size than the rest.
        sizes_[-1] = x_size - (chunks - 1) * sizes

    # From this point, we assume that the sum of the sizes of all chunks
    # equals the size of the base tensor.
    result = []
    start = 0
    for size in sizes_:
        end = start + size
        # No need for clamping here, since we compute the exact
        # start and end values.
        result.append(slice_(x, dim, start, end, clamp=False))
        start = end
    return result


@register_lowering(aten.split_with_sizes, type_promotion_kind=None)
def split_with_sizes(x, sizes, dim=0):
    return split(x, sizes, dim)


@register_lowering(aten.unbind, type_promotion_kind=None)
def unbind(x, dim=0):
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.guard_int(x.get_size()[dim])
    result = [select(x, dim, i) for i in range(x_size)]
    return result


@register_lowering(aten.unfold, type_promotion_kind=None)
def unfold(x, dimension, size, step):
    sizes = x.get_size()
    ndim = len(sizes)
    dim = canonicalize_dim(ndim, dimension)

    if ndim == 0:
        return slice_(unsqueeze(x, 0), end=size, clamp=False)

    dim_size = sizes[dim]
    sizevars = V.graph.sizevars
    sizevars.check_leq(size, dim_size)
    sizevars.check_lt(0, step)  # type: ignore[arg-type]

    new_dim_size = FloorDiv(dim_size - size, step) + 1
    if sizevars.size_hint_or_throw(dim_size) > 0:
        x.mark_reuse(
            sizevars.size_hint_or_throw(CeilDiv(new_dim_size * size, dim_size))
        )

    out_size = [*sizes[:dim], new_dim_size, *sizes[dim + 1 :], size]

    def reindexer(idx):
        dim_idx = idx[-1] + idx[dim] * step
        return (*idx[:dim], dim_idx, *idx[dim + 1 : -1])

    return TensorBox(ir.GenericView.create(x, out_size, reindexer))


@register_lowering(aten.unsqueeze, type_promotion_kind=None)
def unsqueeze(x, dim):
    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.S.One)
    return view(x, new_shape)


@register_lowering(aten.unsqueeze_, type_promotion_kind=None)
def unsqueeze_(x, dim):
    val = unsqueeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    x.data = val.data
    return x


def _validate_dim(x, dim, offset=0):
    dim = V.graph.sizevars.shape_env.evaluate_expr(sympy.sympify(dim))
    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    assert 0 <= dim < ndim + offset
    return dim


@register_lowering(aten.glu)
def glu(x, dim=-1):
    dim = _validate_dim(x, dim, 0)
    # TODO: don't guard on static shape here
    new_len = V.graph.sizevars.guard_int(x.get_size()[dim]) // 2
    # no need to clamp, index is int based on input size
    a = slice_(x, dim, 0, new_len, clamp=False)
    b = slice_(x, dim, new_len, new_len * 2, clamp=False)
    return mul(a, sigmoid(b))


def fallback_handler(kernel, add_to_fallback_set=True):
    if add_to_fallback_set:
        fallbacks.add(kernel)

    def handler(*args, **kwargs):
        def wrap_tensors(x):
            return TensorBox.create(x) if isinstance(x, ir.IRNode) else x

        return pytree.tree_map(
            wrap_tensors, ir.FallbackKernel.create(kernel, *args, **kwargs)
        )

    # This lets us detect that a lowering is a fallback handler.
    handler._is_fallback_handler = True  # type: ignore[attr-defined]

    return handler


@functools.cache
def _warn_complex_not_supported():
    warnings.warn(
        "Torchinductor does not support code generation for complex operators. Performance may be worse than eager."
    )


# There are some types (CPU) which we accept as input but not as
# output.
def unsupported_input_tensor(t: torch.Tensor, node=None):
    "Do not support reading or writing to this tensor"
    if t.is_complex():
        # Complex views are supported with IR ComplexView
        _warn_complex_not_supported()
        return True

    if t.is_meta:
        return True

    if t.is_sparse:
        return True

    if t.dtype == torch.float8_e8m0fnu:
        if not node:
            return True

        # allow bitcast, views, memory movement, but not arithmetic
        # TODO: delete once triton adds native support
        return not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target
            in (
                aten.view.dtype,
                aten.cat.default,
                aten.clone.default,
                aten._scaled_mm.default,
            )
            or (isinstance(node.target, torch._ops.OpOverload) and is_view(node.target))
        )

    return False


def unsupported_output_tensor(t: torch.Tensor, node=None):
    "Do not support writing tensor but can read from it"
    supported_complex_views = (
        aten.view.dtype,
        torch.ops.prims.convert_element_type.default,
    )
    if node is not None and node.target in supported_complex_views and t.is_complex():
        return False
    if unsupported_input_tensor(t, node):
        return True
    return t.is_cpu and config.disable_cpp_codegen


def fallback_node_due_to_unsupported_type(node: torch.fx.Node, allow_cpu_inputs=True):
    # Custom fallback lowering
    if node.target is aten.view_as_complex.default:
        return False

    if node.op == "placeholder":
        return False

    # We should be able to remove this special case once `disable_cpp_codegen` is killed.
    if node.target is aten.lift_fresh_copy.default:
        return False

    def check_skip_condition(inp_out_node, is_output):
        if not isinstance(inp_out_node, torch.fx.Node):
            return False

        if "val" not in inp_out_node.meta:
            return False

        for meta in pytree.tree_leaves(inp_out_node.meta["val"]):
            if not isinstance(meta, torch._subclasses.FakeTensor):
                continue

            if is_output:
                if unsupported_output_tensor(meta, node):
                    return True
            else:
                if unsupported_input_tensor(meta, node):
                    return True

        return False

    # only skip codegen if there is a cpu output, not input
    for arg in pytree.arg_tree_leaves(*node.args, **node.kwargs):
        if check_skip_condition(arg, is_output=False):
            return True

    return check_skip_condition(node, is_output=True)


def make_fallback(op, layout_constraint=None, warn=True, override_decomp=False):
    assert op not in decompositions or override_decomp, (
        f"both a fallback and a decomp for same op: {op}"
    )
    if (
        warn
        and bool(os.getenv("CI"))
        and get_decompositions([op])
        # if fallback_random, we allow not decomposing random
        and not (
            config.fallback_random
            and op in torch._decomp.decompositions_for_rng.extra_random_decomps
        )
        and not override_decomp
    ):
        # Note: 'warn' is holdover from when this was a warning, but for ops that previously
        # set warn=False we do not want a CI error.
        # Ignore the 'suppress errors' configs in CI, as this particular warning happens on startup anyway and is not
        # likely to be triggered preferentially on one CI config over another.
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning(
                "A make_fallback error occurred in suppress_errors config,"
                " and suppress_errors is being disabled to surface it."
            )
        raise AssertionError(
            f"make_fallback({op}): a decomposition exists, we should switch to it."
            " To fix this error, either add a decomposition to core_aten_decompositions (preferred)"
            " or inductor_decompositions, and delete the corresponding `make_fallback` line."
            " Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.",
        )

    def register_fallback(op_overload):
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.overloads():
            op_overload = getattr(op, ol)
            register_fallback(op_overload)
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)
    else:
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")


def philox_rand_offset(shape):
    """
    TorchInductor offset calculation differs from PyTorch eager offset
    calculation for random ops (tl.rand vs torch.rand). In future, we should
    strive for same impl for tl.rand and torch.rand.
    """
    numel = 1
    for s in shape:
        numel = numel * s
    return tensor(numel, dtype=torch.int64)


@register_lowering(torch.ops.rngprims.philox_rand, type_promotion_kind=None)
def philox_rand(size, seed, offset, stride, device, dtype):
    # stride arg is optional and will be used in future for distributed random
    # ops. Currently, its unused.
    random_pos = ir.FixedLayout(
        device,
        dtype,
        size,
        ir.FlexibleLayout.contiguous_strides(size),
    ).make_indexer()
    seed_loader = seed.make_loader()
    offset_loader = offset.make_loader()

    def inner_fn(index):
        # Both seed and offset in the philox_rand op are tensors.
        # torch seed and offsets are of type int64, but tl.rand accepts int32
        seed_index_expr = ops.to_dtype(seed_loader([]), torch.int32)
        offset_index_expr = ops.to_dtype(offset_loader([]), torch.int32)
        # Get the offset'd position
        rand_index_expr = ops.add(
            ops.index_expr(random_pos(index), torch.int32), offset_index_expr
        )
        result = ops.rand(
            seed_index_expr,
            rand_index_expr,
        )
        return ops.to_dtype(result, dtype)

    random_values_node = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )

    offset_node = philox_rand_offset(size)
    return random_values_node, offset_node


@register_lowering(aten.native_dropout, type_promotion_kind=None)
def native_dropout(x, p, train):
    if config.fallback_random:
        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(aten.native_dropout.default, x, p, train),
        )
    else:
        raise AssertionError("should be handled in replace_random.py")


@register_lowering(aten.bernoulli_, type_promotion_kind=None)
def bernoulli_(x, *args):
    assert config.fallback_random or x.get_device() == torch.device("cpu"), (
        "this should be handled in decomps unless config.fallback_random or the device is CPU"
    )
    x.realize()
    op_overload = (
        aten.bernoulli_.float
        if len(args) == 0 or isinstance(args[0], float)
        else aten.bernoulli_.Tensor
    )
    ir.InplaceBernoulliFallback(op_overload, x, *args)
    return x


@register_lowering(aten.bernoulli.p, type_promotion_kind=None)
def bernoulli_p(x, *args):
    assert config.fallback_random or x.get_device() == torch.device("cpu"), (
        "this should be handled in decomps unless config.fallback_random or the device is CPU"
    )
    return bernoulli_(clone(x), *args)


# This shouldn't be called in general
@register_lowering(aten._foobar)
def _foobar(_):
    raise AssertionError


@functools.lru_cache(1)
def _warn_triton_random(salt):
    log.info("using triton random, expect difference from eager")


def warn_triton_random():
    # only warn once per graph
    _warn_triton_random(V.graph.creation_time)


fallback_rand_default = fallback_handler(aten.rand.default)
fallback_rand_generator = fallback_handler(aten.rand.generator)
fallback_randn_default = fallback_handler(aten.randn.default)
fallback_randn_generator = fallback_handler(aten.randn.generator)
make_fallback(aten.randint)

# TODO: mlazos reevaluate if we want to codegen something different
make_fallback(torch.ops.streams.record_event.default)
make_fallback(torch.ops.streams.wait_event.default)


@register_lowering(aten.rand)
def rand(*args, **kwargs):
    if kwargs.get("generator") is not None:
        return fallback_rand_generator(*args, **kwargs)
    elif config.fallback_random:
        kwargs.pop("generator", None)
        return fallback_rand_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")


@register_lowering(aten.randn)
def randn(*args, **kwargs):
    if kwargs.get("generator") is not None:
        return fallback_randn_generator(*args, **kwargs)
    elif config.fallback_random:
        kwargs.pop("generator", None)
        return fallback_randn_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")


@register_lowering(inductor_prims.force_stride_order, type_promotion_kind=None)
def inductor_force_stride_order(input_tensor, stride):
    stride_order = ir.get_stride_order(stride)
    return ir.ExternKernel.require_stride_order(input_tensor, stride_order)


@register_lowering(inductor_prims.seed, type_promotion_kind=None)
def inductor_seed(device: torch.device):
    raise AssertionError("should be handled in fuse_seed_creation_pass()")


@register_lowering(inductor_prims.seeds, type_promotion_kind=None)
def inductor_seeds(count, device):
    warn_triton_random()
    return TensorBox.create(ir.RandomSeeds(count, decode_device(device)))


@register_lowering(inductor_prims.lookup_seed, type_promotion_kind=None)
def inductor_lookup_seed(seeds, index):
    def inner_fn(_):
        return ops.load_seed(seeds.get_name(), index)

    return Pointwise.create(
        device=seeds.get_device(),
        dtype=seeds.get_dtype(),
        inner_fn=inner_fn,
        ranges=[],
    )


@register_lowering(inductor_prims.random, type_promotion_kind=None)
def inductor_random(size: list[int], seed: TensorBox, mode: str, *, offset: int = 0):
    assert not config.fallback_random
    assert mode in ("rand", "randn")
    size = [*size]
    dtype = torch.float32
    device = seed.get_device_or_error()
    random_pos = ir.FixedLayout(
        device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset
    ).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        return getattr(ops, mode)(
            seed_loader([]),
            ops.index_expr(random_pos(index), torch.int32),
        )

    result = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*size],
    )
    result.realize()
    return result


@register_lowering(inductor_prims.randint, type_promotion_kind=None)
def inductor_randint(
    low: int, high: int, size: list[int], seed: TensorBox, *, offset: int = 0
):
    assert not config.fallback_random
    size = [*size]
    dtype = torch.int64
    device = seed.get_device_or_error()
    random_pos = ir.FixedLayout(
        device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset
    ).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        return ops.randint64(
            seed_loader([]),
            ops.index_expr(random_pos(index), torch.int32),
            ops.index_expr(low, torch.int64),
            ops.index_expr(high, torch.int64),
        )

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*size],
    )


def _boundaries_helper(tb: TensorBox) -> tuple[str, sympy.Expr, sympy.Expr, sympy.Expr]:
    # Calculate the maximum offset for the boundaries tensor
    # For a strided tensor, this is sum((size[i] - 1) * stride[i]) + stride[-1]
    # This ensures the mask check in bucketize_binary_search works correctly
    # for both contiguous and non-contiguous tensors.
    size = tb.get_size()
    stride = tb.get_stride()
    max_offset = sum((s - 1) * st for s, st in zip(size, stride)) + stride[-1]
    return (
        tb.get_name(),
        size[-1],
        max_offset,
        stride[-1],
    )


def _sorter_helper(tb: TensorBox) -> tuple[str, sympy.Expr]:
    return tb.get_name(), tb.get_stride()[-1]


@register_lowering(aten.searchsorted.Tensor, type_promotion_kind=None)
def searchsorted(
    sorted_sequence: TensorBox,
    self: TensorBox,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[TensorBox] = None,
) -> Union[TensorBox, ShapeAsConstantBuffer]:
    validate_bucketize = lambda tb: V.graph.has_feature(  # noqa: E731
        tb, BackendFeature.BUCKETIZE
    )
    if (
        not validate_bucketize(sorted_sequence)
        or not validate_bucketize(self)
        or (sorter is not None and not validate_bucketize(sorter))
    ):
        return fallback_handler(aten.searchsorted.Tensor, add_to_fallback_set=False)(
            sorted_sequence,
            self,
            out_int32=out_int32,
            right=right,
            side=side,
            sorter=sorter,
        )

    # If side is present, override the value of right if needed.  This assumes that
    # validation of the two options being non-contradictory is already done by the
    # searchsorted meta-function.
    if side is not None and side == "right":
        right = True

    index_dtype = torch.int32 if out_int32 else torch.int64
    values_loader = self.make_loader()

    # The entire sorted_sequence tensor needs to be used by ops.bucketize, so we need to
    # realize it into global memory; or in other words, we can't guarantee that
    # sorted_sequence.get_name() (used below) will exist unless we call
    # sorted_sequence.realize().
    sorted_sequence.realize()

    if sorter is not None:
        sorter.realize()

    if len(sorted_sequence.get_size()) == 1:

        def inner_fn(idx):
            val = values_loader(idx)
            return ops.bucketize(
                val,
                _boundaries_helper(sorted_sequence),
                0,
                index_dtype,
                right,
                sorter=None if sorter is None else _sorter_helper(sorter),
                sorter_indices=None if sorter is None else 0,
            )

    else:

        def inner_fn(idx):
            val = values_loader(idx)

            # Get index to the beginning of the sorted sequence within a flattened
            # version of the array.
            def get_flattened_index(tb: TensorBox):
                strides = tb.get_stride()
                return ops.index_expr(
                    functools.reduce(
                        operator.add, (s * i for s, i in zip(strides[:-1], idx[:-1]))
                    ),
                    index_dtype,
                )

            return ops.bucketize(
                val,
                _boundaries_helper(sorted_sequence),
                get_flattened_index(sorted_sequence),
                index_dtype,
                right,
                sorter=None if sorter is None else _sorter_helper(sorter),
                sorter_indices=None if sorter is None else get_flattened_index(sorter),
            )

    device = self.get_device()
    result = Pointwise.create(
        device=device,
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=self.shape,
    )
    # see [NOTE: inductor bucketize realize]
    result.realize()

    return result


@register_lowering(
    aten.bucketize, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
)
def bucketize(
    input: TensorBox,
    boundaries: TensorBox,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    assert len(boundaries.get_size()) == 1

    if not (
        V.graph.has_feature(input, BackendFeature.BUCKETIZE)
        and V.graph.has_feature(boundaries, BackendFeature.BUCKETIZE)
    ):
        return fallback_handler(aten.bucketize.Tensor, add_to_fallback_set=False)(
            input, boundaries, out_int32=out_int32, right=right
        )

    # The entire boundaries tensor needs to be used by ops.bucketize, so we
    # need to realize it into global memory; or in other words, we can't
    # guarantee that boundaries.get_name() (used below) will exist unless
    # we call boundaries.realize().
    boundaries.realize()
    device = input.get_device()
    input_loader = input.make_loader()

    index_dtype = torch.int32 if out_int32 else torch.int64

    def inner_fn(index):
        val = input_loader(index)
        indices = ops.bucketize(
            val,
            _boundaries_helper(boundaries),
            0,
            index_dtype,
            right,
        )

        return indices

    result = Pointwise.create(
        device=device,
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )

    # [NOTE: inductor bucketize realize]
    # bucketize_binary_search is relatively expensive, so we don't want to re-compute
    # it unnecessarily. If we run bucketize() and then broadcast the result, we don't
    # want this to be fused into a large number of duplicate bucketize() computations
    # for each of the elements in the result.
    #
    # If no broadcasting occurs, fusions can still occur in scheduler.py
    result.realize()

    return result


def require_dense(_, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_stride1, (args, kwargs)
    )
    return args, kwargs


def require_contiguous(_, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_contiguous, (args, kwargs)
    )
    return args, kwargs


def require_contiguous_strides(_, *args, **kwargs):
    # TODO: combine this with require_contiguous after
    # https://github.com/pytorch/pytorch/pull/148235 lands.
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_contiguous_strides, (args, kwargs)
    )
    return args, kwargs


def require_channels_last(_, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_channels_last, (args, kwargs)
    )
    return args, kwargs


def constrain_to_fake_tensor(arg, fake_arg):
    if isinstance(fake_arg, FakeScriptObject):
        return arg
    if isinstance(arg, ir.IRNode):
        meta_stride_expr = [
            s.node.expr if isinstance(s, torch.SymInt) else s for s in fake_arg.stride()
        ]
        return ir.ExternKernel.require_exact_strides(arg, meta_stride_expr)
    if isinstance(arg, dict):
        return {key: constrain_to_fake_tensor(arg[key], fake_arg[key]) for key in arg}
    elif isinstance(arg, (tuple, list)):
        return type(arg)(
            constrain_to_fake_tensor(a, f_a) for (a, f_a) in zip(arg, fake_arg)
        )
    return arg


def constrain_to_fake_tensors(args, kwargs, fake_args, fake_kwargs):
    args = tuple(
        constrain_to_fake_tensor(arg, fake_arg)
        for arg, fake_arg in zip(args, fake_args)
    )
    kwargs = {k: constrain_to_fake_tensor(v, fake_kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


def constrain_to_fx_strides(fx_node, *args, **kwargs):
    def apply_constraint(arg, fx_arg):
        if isinstance(arg, ir.IRNode):
            stride_order = ir.get_stride_order(
                fx_arg.meta["val"].stride(), V.graph.sizevars.shape_env
            )
            return ir.ExternKernel.require_stride_order(arg, stride_order)
        if isinstance(arg, dict):
            return {key: apply_constraint(arg[key], fx_arg[key]) for key in arg}
        return arg

    args = tuple(
        apply_constraint(arg, fx_arg) for arg, fx_arg in zip(args, fx_node.args)
    )
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


def sdpa_constraint(fx_node, *args, **kwargs):
    # sdpa requires dense last dimension]

    def apply_constraint(idx, arg, fx_arg):
        if not isinstance(arg, ir.IRNode):
            return arg

        meta_val = fx_arg.meta["val"]
        meta_stride_expr = [
            s.node.expr if isinstance(s, torch.SymInt) else s for s in meta_val.stride()
        ]
        shape_env = V.graph.sizevars.shape_env
        stride_order = ir.get_stride_order(meta_val.stride(), shape_env)

        if stride_order and stride_order[-1] != 0:
            # contiguous stride order
            stride_order = list(reversed(range(len(arg.get_size()))))

        if (
            fx_node.target
            == aten._scaled_dot_product_efficient_attention_backward.default
            and idx in (0, 5)
        ):
            assert len(stride_order) == 4
            # The 0 and 5th arguments for aten._scaled_dot_product_efficient_attention_backward.default
            # are for out and gradient_out. They have to be in
            # (3, 1, 2, 0) stride order. Otherwise the kernel will crash.
            # Check https://github.com/pytorch/pytorch/issues/138772
            stride_order = (3, 1, 2, 0)

        if not meta_val.is_cuda:
            return ir.ExternKernel.require_stride_order(arg, stride_order)

        # This is the minimum alignment required by SDPA kernels for attention_bias.
        # This value can be found in pytorch/aten/src/ATen/native/transformers/attention.cpp preprocess_mask
        ALIGNMENT = 8

        # effn_attn_fwd does requires dense last dim, not just alignment
        effn_attn_fwd_bias = (
            fx_node.target
            == torch.ops.aten._scaled_dot_product_efficient_attention.default
            and idx == 3
        )

        assert isinstance(arg, TensorBox)
        if len(arg.get_size()) not in (3, 4):
            return arg

        is_aligned_tensor = ir.is_aligned_realized_tensor(arg, ALIGNMENT)
        if is_aligned_tensor:
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        if (
            isinstance(arg, IRNode)
            and arg.maybe_get_stride() is not None
            and is_aligned_tensor
        ):
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        if effn_attn_fwd_bias:
            out_size = list(arg.get_size())

            expanded_dims = []
            # We require a dense last dimension, but the other strides
            # can be expanded, which results in a smaller tensor
            maybe_stride = arg.maybe_get_stride()
            for i in range(len(arg.get_size()) - 1):
                if V.graph.sizevars.statically_known_equals(meta_stride_expr[i], 0) or (
                    maybe_stride is not None
                    and V.graph.sizevars.statically_known_equals(maybe_stride[i], 0)
                ):
                    expanded_dims.append(i)

            # Now, pad strides to alignment
            out_strides = [-1] * len(out_size)
            out_strides[-1] = 1
            stride = 1
            for i in range(len(out_size) - 2, -1, -1):
                if out_strides[i + 1] != 0:
                    stride = stride * out_size[i + 1]

                # the expanded dims still need to be aligned, if they are,
                # we can make them expanded by setting the stride equal to 0
                if i in expanded_dims:
                    if V.graph.sizevars.statically_known_equals(
                        out_strides[i + 1] % ALIGNMENT, 0
                    ):
                        out_strides[i] = 0
                        continue

                if not V.graph.sizevars.statically_known_equals(stride % ALIGNMENT, 0):
                    stride = ceildiv(stride, ALIGNMENT) * ALIGNMENT

                out_strides[i] = stride

            return ir.ExternKernel.require_exact_strides(arg, out_strides)

        if is_aligned_tensor:
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        if (
            isinstance(arg, IRNode)
            and arg.maybe_get_stride() is not None
            and is_aligned_tensor
        ):
            return ir.try_match_insignificant_strides(
                ir.ExternKernel.realize_input(arg), meta_stride_expr
            )

        def is_aligned(x):
            return V.graph.sizevars.guard_or_false(
                sympy.Eq(Mod(x.get_size()[-1], ALIGNMENT), 0)
            )

        if isinstance(arg.data, ir.BaseView):
            if not is_aligned(arg):
                if is_aligned(arg.unwrap_view()):
                    return ir.try_match_insignificant_strides(
                        ir.ExternKernel.realize_input(arg), meta_stride_expr
                    )

        return ir.ExternKernel.require_stride_order(arg, stride_order)

    args = tuple(
        apply_constraint(idx, arg, fx_arg)
        for idx, (arg, fx_arg) in enumerate(zip(args, fx_node.args))
    )
    kwargs = {k: apply_constraint(-1, v, fx_node.kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


# WIP
make_fallback(aten._adaptive_avg_pool3d)  # @isuruf
make_fallback(aten.adaptive_max_pool3d)  # @isuruf
make_fallback(aten._scaled_dot_product_attention_math_for_mps)  # @malfet


# 1) Easy
make_fallback(aten.uniform, warn=False)
make_fallback(aten.exponential.default, warn=False)  # (fails accuracy on test_torch.py)
make_fallback(aten._pdist_forward)  # Has decomp. Needs benchmarks
make_fallback(aten.soft_margin_loss_backward, warn=False)  # py_impl?
make_fallback(aten._fused_rms_norm, warn=False)  # (MPS-only and faster than decomp)
if torch.xpu.is_available():
    make_fallback(
        aten.embedding_dense_backward, warn=False
    )  # (XPU-only and faster than decomp)


# 1.5) Easy or Impossible
make_fallback(aten._cdist_forward)  # p=2 should be feasible
make_fallback(aten._cdist_backward)

# 2) Medium
make_fallback(aten._trilinear)


# 3) Difficult
# Scans
# See the discussion at
# https://dev-discuss.pytorch.org/t/pytorch-sparse-gnn-compiler-rfc/1644/19
make_fallback(aten.segment_reduce.default)
make_fallback(aten._segment_reduce_backward.default)

# Histogram (need to implement Histogram IR)
make_fallback(aten.histc)
make_fallback(aten.histogram.bin_ct)
make_fallback(aten._histogramdd_bin_edges.default)
make_fallback(aten._histogramdd_from_bin_cts.default)

# Need templated kernel
make_fallback(aten.addbmm)
make_fallback(aten._addmm_activation, warn=False)

make_fallback(aten._grouped_mm, require_dense)

# Need templated kernel. Probably impossible to write efficiently
make_fallback(aten.convolution_backward, constrain_to_fx_strides)
make_fallback(aten._cudnn_rnn, require_dense)
make_fallback(aten._cudnn_rnn_backward, require_contiguous)

# Haven't checked but sound difficult / impossible
make_fallback(aten._embedding_bag, require_contiguous)
make_fallback(aten._embedding_bag_forward_only, require_contiguous)
make_fallback(aten._embedding_bag_backward)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten._fused_moving_avg_obs_fq_helper)
make_fallback(aten._fused_moving_avg_obs_fq_helper_functional)


# 4) Backwards (try py_impl'ing them) when fwd is written as a decomp
make_fallback(aten.max_pool3d_with_indices_backward)
make_fallback(aten._adaptive_avg_pool2d_backward, require_dense)
make_fallback(aten._adaptive_avg_pool3d_backward)
make_fallback(aten.adaptive_max_pool2d_backward)
make_fallback(aten.adaptive_max_pool3d_backward)
make_fallback(aten.fractional_max_pool2d_backward)
make_fallback(aten.fractional_max_pool3d_backward)
make_fallback(aten.replication_pad1d_backward)
make_fallback(aten.replication_pad2d_backward)
make_fallback(aten.upsample_linear1d_backward)
make_fallback(aten.upsample_bicubic2d_backward, require_contiguous)
make_fallback(aten.upsample_trilinear3d_backward)
make_fallback(aten.grid_sampler_2d_backward)
make_fallback(aten._pdist_backward)


# 5) Impossible (missing triton/CPU features)

# Sorting / Sorting-like
make_fallback(aten.sort)
make_fallback(aten.sort.stable)
make_fallback(aten.kthvalue)
make_fallback(aten.topk)
make_fallback(aten.mode)
make_fallback(aten.median)
make_fallback(aten.nanmedian)
make_fallback(aten.randperm)
# see: http

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Functions
This file defines 390 function(s): cur_node_has_non_foreach_users, group_foreach_args, maybe_layout_constraints, tag_to_layout_constraint, assert_nyi, add_needs_realized_inputs, add_layout_constraint, decode_dtype, is_integer_type, is_boolean_type, get_promoted_dtype, construct_input, get_overloads, in_namespace, maybe_copy_cpu_scalar, transform_args, promote, _register_foreach_lowering, wrapped, _register_lowering, wrapped, register_lowering, broadcast_symbolic_shapes, promote_constants, const_func, make_pointwise, inner, inner_fn, make_foreach_pointwise, inner


## Key Components

The file contains 21133 words across 7578 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 248019 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

# Documentation: `docs/torch/_inductor/lowering.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/lowering.py_docs.md`
- **Size**: 53,987 bytes (52.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/lowering.py`

## File Metadata

- **Path**: `torch/_inductor/lowering.py`
- **Size**: 248,019 bytes (242.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
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
   
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`mkldnn_lowerings.py_kw.md_docs.md`](./mkldnn_lowerings.py_kw.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `lowering.py_docs.md_docs.md`
- **Keyword Index**: `lowering.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

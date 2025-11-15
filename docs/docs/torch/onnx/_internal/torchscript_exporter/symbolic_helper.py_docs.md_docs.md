# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_helper.py_docs.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_helper.py_docs.md`
- **Size**: 53,605 bytes (52.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `torch/onnx/_internal/torchscript_exporter/symbolic_helper.py`

## File Metadata

- **Path**: `torch/onnx/_internal/torchscript_exporter/symbolic_helper.py`
- **Size**: 85,878 bytes (83.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "_apply_params",
    "_arange_cast_helper",
    "_arange_helper",
    "_argmin_argmax_helper",
    "_as_list_type",
    "_avgpool_helper",
    "_batchnorm_helper",
    "_block_list_in_opset",
    "_embedding_bag_helper",
    "_flatten_helper",
    "_generate_wrapped_number",
    "_get_const",
    "_get_dim_for_cross",
    "_get_interpolate_attributes",
    "_get_tensor_dim_size",
    "_get_tensor_rank",
    "_get_tensor_sizes",
    "_handle_reduce_dim_none",
    "_if_scalar_type_as",
    "_index_fill_reshape_helper",
    "_interpolate_get_scales_and_mode",
    "_interpolate_get_scales_if_available",
    "_interpolate_get_scales",
    "_interpolate_helper",
    "_interpolate_size_to_scales",
    "_interpolate_warning",
    "_is_bool",
    "_is_constant",
    "_is_fp",
    "_is_list",
    "_is_none",
    "_is_onnx_constant",
    "_is_packed_list",
    "_is_scalar_list",
    "_is_split_static",
    "_is_tensor_list",
    "_is_tensor",
    "_is_tuple_construct",
    "_is_value",
    "_linalg_vector_norm_helper",
    "_lt_helper",
    "_max_helper",
    "_maybe_cast_reduce_op_input",
    "_maybe_cast_to_type",
    "_maybe_get_const",
    "_maybe_get_scalar",
    "_min_helper",
    "_node_get",
    "_numel_helper",
    "_onnx_opset_unsupported_detailed",
    "_onnx_opset_unsupported",
    "_onnx_unsupported",
    "_op_with_optional_float_cast",
    "_optional_input_placeholder_tensor",
    "_overload_by_arg_count",
    "_parse_arg",
    "_reduce_op_symbolic_helper",
    "_reduce_with_dtype_helper",
    "_reducesum_helper",
    "_repeat_interleave_single_value_repeat_helper",
    "_repeat_interleave_split_helper",
    "_reshape_helper",
    "_scalar",
    "_scatter_helper",
    "_select_helper",
    "_size_helper",
    "_slice_helper",
    "_sort_helper",
    "_squeeze_helper",
    "_topk_helper",
    "_try_get_scalar_type",
    "_type_promote_from_values",
    "_unbind_helper",
    "_unimplemented",
    "_unpack_list",
    "_unpack_quantized_tensor",
    "_unpack_tuple",
    "_unsqueeze_helper",
    "_var_mean_helper",
    "args_have_same_dtype",
    "cast_pytorch_to_onnx",
    "check_training_mode",
    "dequantize_helper",
    "is_complex_value",
    "parse_args",
    "pytorch_name_to_type",
    "quantize_helper",
    "quantized_args",
    "requantize_bias_helper",
    "scalar_name_to_pytorch",
    "scalar_type_to_onnx",
    "scalar_type_to_pytorch_type",
]

import functools
import inspect
import math
import sys
import typing
import warnings
from typing import (
    Any,
    Concatenate as _Concatenate,
    Literal,
    NoReturn,
    TypeVar as _TypeVar,
)
from typing_extensions import ParamSpec as _ParamSpec

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, errors
from torch.onnx._internal.torchscript_exporter import _type_utils, jit_utils, utils
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS


if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.types import Number

_T = _TypeVar("_T")
_U = _TypeVar("_U")
_P = _ParamSpec("_P")

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

_ValueDescriptor = Literal[
    "v",
    "i",
    "is",
    "f",
    "fs",
    "b",
    "s",
    "t",
    "none",
]


def _parse_arg(
    value,
    desc: _ValueDescriptor,
    arg_name: str | None = None,
    node_name: str | None = None,
):
    if desc == "none":
        return value
    if desc == "v" or not _is_value(value):
        return value

    node = value.node()
    if node.mustBeNone():
        return None
    if node.kind() == "onnx::Constant":
        node_val = _node_get(node, "value")
        if desc == "i":
            return int(node_val)
        elif desc == "f":
            return float(node_val)
        elif desc == "b":
            return bool(node_val)
        elif desc == "s":
            return str(node_val)
        elif desc == "t":
            return node_val
        elif desc == "is":
            return [int(v) for v in node_val]
        elif desc == "fs":
            return [float(v) for v in node_val]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not understand the Constant node '{node}' "
                f"specified with descriptor '{desc}'.",
                value,
            )
    elif node.kind() == "prim::ListConstruct":
        if desc == "is":
            for v in node.inputs():
                element_node = v.node()
                if element_node.kind() != "onnx::Constant":
                    raise errors.SymbolicValueError(
                        f"Failed to export a node '{element_node}' "
                        f"(in list node {node}) "
                        f"because it is not constant. "
                        f"Please try to make things (e.g. kernel sizes) static if possible.",
                        value,
                    )
            return [int(_node_get(v.node(), "value")) for v in value.node().inputs()]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not know how to unpack the ListConstruct node that "
                f"is not a list of integers: '{node}'",
                value,
            )

    if arg_name is None or node_name is None:
        raise errors.SymbolicValueError(
            f"Expected node type 'onnx::Constant', got '{node.kind()}'.",
            value,
        )

    raise errors.SymbolicValueError(
        "Expected node type 'onnx::Constant' "
        f"for argument '{arg_name}' of node '{node_name}', got '{node.kind()}'.",
        value,
    )


def _node_get(node: _C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    assert isinstance(node, _C.Node)
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


def _is_onnx_constant(value: _C.Value):
    """Whether a Value is an ONNX constant."""
    return value.node().kind() == "onnx::Constant"


def _maybe_get_const(
    value: _C.Value | torch.Tensor | Number | Sequence | None,
    descriptor: _ValueDescriptor,
):
    # NOTE: prim::Constant at this stage usually means something not compatible in ONNX,
    # otherwise it'd be converted to onnx::Constant
    # TODO(justinchuby): Replace insinstance with _is_value once we figure out mypy
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, "t")
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if not _is_constant(value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected a constant value of the '{arg_name}' argument, "
            f"got '{value}'",
            value,
        )
    return _parse_arg(value, desc)


def _unpack_list(list_value: _C.Value) -> list[_C.Value]:
    list_node = list_value.node()
    if list_node.kind() != "prim::ListConstruct":
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type prim::ListConstruct, got '{list_node}'.",
            list_value,
        )
    return list(list_node.inputs())


def _unpack_tuple(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type 'prim::TupleConstruct', "
            f"got '{tuple_node.kind()}'.",
            tuple_value,
        )
    return tuple(tuple_node.inputs())


def _unpack_quantized_tensor(tuple_value: _C.Value) -> tuple[_C.Value, ...]:
    """Unpacks a quantized tensor into a tuple of tensor and scale/zero_point.
    Args:
        tuple_value: A tuple of tensor, scale, zero_point, and optionally axis.
    Returns:
        A tuple of tensor, scale, zero_point, and optionally axis.
    """
    tuple_node = tuple_value.node()
    # A quantized tensor is represented as tuple of the form (tensor, scale, zero_point, <axis>)
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected the output of `{tuple_node}` to be a quantized "
            f"tensor. Is this likely due to missing support for quantized "
            f"`{tuple_node.kind()}`. Please create an issue on {_constants.PYTORCH_GITHUB_ISSUES_URL}",
            tuple_value,
        )
    unpacked = tuple(tuple_node.inputs())
    assert len(unpacked) == 3 or len(unpacked) == 4
    return unpacked


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be unpacked.
def _is_packed_list(list_value: Any) -> bool:
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


def parse_args(
    *arg_descriptors: _ValueDescriptor,
) -> Callable[[Callable[_Concatenate[_U, _P], _T]], Callable[_Concatenate[_U, _P], _T]]:
    """A decorator which converts args from torch._C.Value to built-in types.

    For example:

    ```
    @parse_args('v', 'i', 'fs')
    foo(g, a, b, c):
        assert isinstance(a, torch._C.Value)
        assert isinstance(b, int)
        assert isinstance(c, list)
        assert isinstance(c[0], float)
    ```

    Args:
        arg_descriptors: list of str, where each element is
            a string that specifies the type to convert to. Valid descriptors:
            "v": no conversion, keep torch._C.Value.
            "i": int
            "is": list of int
            "f": float
            "fs": list of float
            "b": bool
            "s": str
            "t": torch.Tensor
            "none": the variable is unused
    """

    def decorator(
        fn: Callable[_Concatenate[_U, _P], _T],
    ) -> Callable[_Concatenate[_U, _P], _T]:
        fn._arg_descriptors = arg_descriptors  # type: ignore[attr-defined]

        @functools.wraps(fn)
        def wrapper(g: _U, *args: _P.args, **kwargs: _P.kwargs) -> _T:
            # some args may be optional, so the length may be smaller
            FILE_BUG_MSG = (
                "If you believe this is not due to custom symbolic implementation within your code or "
                "an external library, please file an issue at "
                "https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to report this bug."
            )
            assert len(arg_descriptors) >= len(args), (
                f"A mismatch between the number of arguments ({len(args)}) and "
                f"their descriptors ({len(arg_descriptors)}) was found at symbolic function '{fn.__name__}'. "
                f"{FILE_BUG_MSG}"
            )

            try:
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                # FIXME(justinchuby): Avoid catching Exception.
                # Catch a more specific exception instead.
                arg_names = [None] * len(args)  # type: ignore[list-item]
                fn_name = None
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[method-assign]
                for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)
            ]
            # only support _outputs in kwargs
            assert len(kwargs) <= 1, (
                f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single "
                f"key/value entry. "
                f"{FILE_BUG_MSG}"
            )

            if len(kwargs) == 1:
                assert "_outputs" in kwargs, (
                    f"Symbolic function {fn.__name__}'s '**kwargs' can only contain "
                    f"'_outputs' key at '**kwargs'. "
                    f"{FILE_BUG_MSG}"
                )
            return fn(g, *args, **kwargs)

        return wrapper

    return decorator


def quantized_args(
    *arg_q_descriptors: bool,
    scale: float | None = None,
    zero_point: int | None = None,
    quantize_output: bool = True,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """A decorator which extends support for quantized version of the base operator.

    Quantization is detected by examining the arguments that are annotated by
    `arg_q_descriptors`.

    If quantization is detected, the base operator symbolic function will be wrapped with
    argument de-quantization and output quantization.

    Otherwise, only the base symbolic function will be invoked.

    For example:

    ```
    @quantized_args(True, False)
    def foo(g, x, y):
        return x + y
    ```

    is equivalent to

    ```
    def q_foo(g, x, y):
        if is_quantized_tensor(x):
            x = dequantize(x)
            out = foo(g, x, y)
            return quantize(out)
        else:
            return foo(g, x, y)
    ```

    Args:
        arg_q_descriptors: A sequence of bool, where each element represents if the
          argument is QTensor for quantized version of this operator. It defaults
          to False for unspecified (variable length) arguments.
        scale: Quantized output scale. If None, derive from
          the first quantized input scale.
        zero_point: Quantized output zero point. If None,
          derive from the first quantized input zero point.
        quantize_output: If True, quantize the output of the base operator. Default is True
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            nonlocal scale
            nonlocal zero_point
            if scale is not None:
                _scale = g.op("Constant", value_t=torch.tensor(scale))
            else:
                _scale = None
            if zero_point is not None:
                _zero_point = g.op("Constant", value_t=torch.tensor(zero_point))
            else:
                _zero_point = None

            # Support variable length arguments by marking unspecified ones as non-quantized
            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (
                len(args) - len(arg_q_descriptors)
            )
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))

            def _is_arg_quantized(descriptor, arg):
                return descriptor and _is_value(arg) and _is_tuple_construct(arg)

            # Run regular symbolic function if none of the argument is QTensor.
            is_quantized: list[bool] = []
            for descriptor, arg in descriptor_args:
                # ListConstruct
                if _is_packed_list(arg):
                    is_quantized.extend(
                        _is_arg_quantized(descriptor, arg_input)
                        for arg_input in arg.node().inputs()
                    )
                else:
                    is_quantized.append(_is_arg_quantized(descriptor, arg))

            if not any(is_quantized):
                return fn(g, *args, **kwargs)

            # Dequantize arguments that are quantized
            non_quantized_args = []
            for descriptor, arg in descriptor_args:
                if _is_arg_quantized(descriptor, arg):
                    # Quantized arg is a tuple of (value, scale, zero_point)
                    dequantized_arg, arg_scale, arg_zero_point, _ = dequantize_helper(
                        g, arg
                    )
                    non_quantized_args.append(dequantized_arg)
                    # Set scale and zero_point to the first quantized input if not already set
                    if _scale is None:
                        _scale = arg_scale
                    if _zero_point is None:
                        _zero_point = arg_zero_point
                # ListConstruct
                elif _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        if _is_arg_quantized(descriptor, arg_input):
                            # Quantized arg is a tuple of (value, scale, zero_point)
                            (
                                dequantized_arg,
                                arg_scale,
                                arg_zero_point,
                                _,
                            ) = dequantize_helper(g, arg_input)
                            # Set scale and zero_point to the first quantized input if not already set
                            if _scale is None:
                                _scale = arg_scale
                            if _zero_point is None:
                                _zero_point = arg_zero_point
                            arg_input.replaceAllUsesWith(dequantized_arg)
                    non_quantized_args.append(arg)
                else:
                    # Non-quantized arg
                    non_quantized_args.append(arg)
            # TODO(justinchuby): Only single output is supported for now. We may want to
            # support multiple outputs in the future.
            output = fn(g, *non_quantized_args, **kwargs)

            assert _scale is not None, "Bug: Scale must be set for quantized operator"
            assert _zero_point is not None, (
                "Bug: Zero point must be set for quantized operator"
            )

            if quantize_output:
                return quantize_helper(g, output, _scale, _zero_point)
            return output

        return wrapper

    return decorator


def _scalar(x: Any) -> Number | None:
    """Convert a scalar tensor into a Python value."""
    if isinstance(x, torch.Tensor) and x.shape == ():
        return x.item()
    return None


def _if_scalar_type_as(self, tensor):
    """
    Convert self into the same type of tensor, as necessary.
    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, _C.Value):
        return self

    scalar_type = _type_utils.JitScalarType.from_value(
        tensor, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        ty = scalar_type.scalar_name().lower()
        return getattr(self, ty)()
    return self


def _is_none(x: Any) -> bool:
    return x is None or (x.node().mustBeNone() if isinstance(x, _C.Value) else False)


def _is_value(x: Any) -> bool:
    return isinstance(x, _C.Value)


def _is_constant(value: Any) -> bool:
    return not _is_value(value) or value.node().kind() in {
        "onnx::Constant",
        "prim::Constant",
    }


def _is_tensor(x: _C.Value) -> bool:
    return x.type().isSubtypeOf(_C.TensorType.get())


# Note: _C.JitType is not exposed to Python and cannot be checked in runtime.
def _as_list_type(jit_type: _C.JitType) -> _C.ListType | None:
    if isinstance(jit_type, _C.ListType):
        return jit_type
    return None


def _is_list(x: _C.Value) -> bool:
    return _as_list_type(x.type()) is not None


def _is_tensor_list(x: _C.Value) -> bool:
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    return isinstance(x_type.getElementType(), _C.TensorType)


def _is_scalar_list(x: _C.Value) -> bool:
    """Checks if x is a scalar list, for example: List[float], List[int].

    Besides checking the type is ListType, we also check if the data type is
    a valid ONNX data type.
    """
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    scalar_type = _type_utils.JitScalarType.from_value(x)
    return scalar_type.onnx_compatible()


def _is_tuple_construct(x: _C.Value) -> bool:
    return x.node().kind() == "prim::TupleConstruct"


def is_complex_value(x: _C.Value) -> bool:
    assert _is_value(x)
    return _type_utils.JitScalarType.from_value(
        x, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.COMPLEX32,
        _type_utils.JitScalarType.COMPLEX64,
        _type_utils.JitScalarType.COMPLEX128,
    }


def _get_tensor_rank(x: _C.Value) -> int | None:
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()


def _get_tensor_sizes(x: _C.Value, allow_nonstatic: bool = True):
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    if allow_nonstatic:
        # Each individual symbol is returned as None.
        # e.g. [1, "a", "b"] -> [1, None, None]
        return x_type.varyingSizes()
    # returns None, if exists any symbol in sizes.
    # e.g. [1, "a", "b"] -> None
    return x_type.sizes()


def _get_tensor_dim_size(x: _C.Value, dim: int) -> int | None:
    sizes = _get_tensor_sizes(x)
    return sizes[dim] if sizes else None


def _get_dim_for_cross(x: _C.Value, dim: int | None):
    if dim == -1:
        tensor_rank = _get_tensor_rank(x)
        assert tensor_rank is not None
        return dim + tensor_rank
    # If dim is not given, it defaults to the first dimension found with the size 3
    if dim is None:
        sizes = _get_tensor_sizes(x)
        assert sizes is not None
        for index, size in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim


def _unimplemented(op: str, msg: str, value: _C.Value | None = None) -> None:
    # For BC reasons, the behavior for Caffe2 does not raise exception for unimplemented operators
    if GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f"{op}, {msg}", value)


def _onnx_unsupported(op_name: str, value: _C.Value | None = None) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of operator {op_name}. "
        f"Please feel free to request support or submit a pull request "
        f"on PyTorch GitHub: {_constants.PYTORCH_GITHUB_ISSUES_URL}"
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


def _onnx_opset_unsupported(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    value: _C.Value | None = None,
) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of {op_name} in opset {current_opset}. "
        f"Please try opset version {supported_opset}."
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


def _onnx_opset_unsupported_detailed(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    reason: str,
    value: _C.Value | None = None,
) -> NoReturn:
    message = (
        f"Unsupported: ONNX export of {op_name} in "
        f"opset {current_opset}. {reason}. Please try opset version {supported_opset}."
    )
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    raise errors.OnnxExporterError(message)


def _block_list_in_opset(name: str):
    def symbolic_fn(*args, **kwargs):
        raise errors.OnnxExporterError(
            f"ONNX export failed on {name}, which is not implemented for opset "
            f"{GLOBALS.export_onnx_opset_version}. "
            "Try exporting with other opset versions."
        )

    return symbolic_fn


def _try_get_scalar_type(*args) -> _type_utils.JitScalarType | None:
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(
            arg, _type_utils.JitScalarType.UNDEFINED
        )
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    return None


def _type_promote_from_values(*args) -> _type_utils.JitScalarType:
    undef = _type_utils.JitScalarType.UNDEFINED
    jit_types = [_try_get_scalar_type(arg) for arg in args]
    if len(jit_types) == 0:
        return undef
    if len(jit_types) == 1:
        return jit_types[0]  # type: ignore[return-value]
    new_dtype = jit_types[0].dtype()  # type: ignore[union-attr]
    for t in jit_types:
        new_dtype = torch.promote_types(new_dtype, t.dtype())  # type: ignore[union-attr]
    return _type_utils.JitScalarType.from_dtype(new_dtype)


def _maybe_cast_to_type(
    g: jit_utils.GraphContext, value, jit_type: _type_utils.JitScalarType
):
    if (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        != jit_type
    ):
        return g.op(
            "Cast",
            value,
            to_i=jit_type.onnx_type(),
        )
    return value


def _select_helper(g: jit_utils.GraphContext, self, dim, index, apply_reshape=True):
    index_const = _maybe_get_scalar(index)
    index_dim = _get_tensor_rank(index)
    if not _is_value(index_const):
        # Index is a constant scalar. Make it a size 1 constant tensor.
        index = g.op("Constant", value_t=torch.LongTensor([index_const]))
    elif index_dim is not None and apply_reshape:
        if index_dim == 0:
            # Index is a scalar. Reshape it to a size 1 tensor.
            index = _reshape_helper(
                g, index, g.op("Constant", value_t=torch.LongTensor([1]))
            )

    index_scalar_type = _type_utils.JitScalarType.from_value(
        index, _type_utils.JitScalarType.UNDEFINED
    )
    if index_scalar_type not in {
        _type_utils.JitScalarType.INT64,
        _type_utils.JitScalarType.INT,
    }:
        index = g.op("Cast", index, to_i=_C_onnx.TensorProtoDataType.INT64)
    return g.op("Gather", self, index, axis_i=dim)


def _slice_helper(
    g: jit_utils.GraphContext,
    input,
    axes,
    starts,
    ends,
    steps=None,
):
    if g.opset <= 9:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
            _slice as _slice9,
        )

        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (
            _slice as _slice10,
        )

        return _slice10(g, input, axes, starts, ends, steps)


def _is_fp(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    }


def _is_bool(value) -> bool:
    return (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.BOOL
    )


def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    """Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.

    A Tensor is a considered a "wrapped number" if it is
    auto-wrapped from a C++ or Python number type. Integer types are
    wrapped as 0-dim int64 tensors and floating-point types are
    wrapped as 0-dim double tensors.

    The input to this function is constant value. If the data type
    is a floating point type, it is converted to a 0-dim double
    tensor, else it is converted to a 0-dim tensor of its original type
    """
    assert not isinstance(scalar, torch.Tensor)
    if isinstance(scalar, float):
        return g.op("Constant", value_t=torch.tensor(scalar, dtype=torch.double))
    return g.op("Constant", value_t=torch.tensor(scalar))


def _sort_helper(g: jit_utils.GraphContext, input, dim, descending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    dim_size_ = g.op(
        "Gather",
        shape_,
        g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),
    )
    if g.opset <= 10:
        if not descending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, dim_size_, axis_i=dim, largest_i=descending, outputs=2
        )


def _topk_helper(
    g: jit_utils.GraphContext, input, k, dim, largest=True, sorted=False, out=None
):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    if not _is_value(k):
        k = g.op("Constant", value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = _reshape_helper(g, k, g.op("Constant", value_t=torch.tensor([1])))
        if _try_get_scalar_type(k) != _type_utils.JitScalarType.INT64:
            k = g.op("Cast", k, to_i=_C_onnx.TensorProtoDataType.INT64)
    if g.opset <= 10:
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        return g.op(
            "TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2
        )


def _lt_helper(g: jit_utils.GraphContext, input, other):
    if g.opset <= 8:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset8 import lt as _lt8

        return _lt8(g, input, other)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import lt as _lt9

        return _lt9(g, input, other)


def _interpolate_warning(interpolate_mode):
    onnx_op = (
        "onnx:Resize" if GLOBALS.export_onnx_opset_version >= 10 else "onnx:Upsample"
    )
    warnings.warn(
        "You are trying to export the model with "
        + onnx_op
        + " for ONNX opset version "
        "" + str(GLOBALS.export_onnx_opset_version) + ". "
        "This operator might cause results to not match the expected results by PyTorch.\n"
        "ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. "
        "Attributes to determine how to transform the input were added in onnx:Resize in opset 11 "
        "to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\n"
        "We recommend using opset 11 and above for models using this operator.",
        stacklevel=2,
    )


def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if len(axes_i) == 0:
        # unnecessary unsqueeze if axes length==0
        return input
    elif _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Unsqueeze", input, axes)
        return g.op("Unsqueeze", input, axes_i=axes_i)
    # Tensor type
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Unsqueeze with dynamic axes.", input
        )
    return g.op("Unsqueeze", input, axes_i[0])


def _squeeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Squeeze", input, axes)
        return g.op("Squeeze", input, axes_i=axes_i)
    # Tensor type
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Squeeze with dynamic axes.", input
        )
    axes_t = axes_i[0]
    axes_rank = _get_tensor_rank(axes_t)
    assert axes_rank is not None
    if axes_rank > 1:
        raise errors.SymbolicValueError(
            "For Squeeze axses as input, the axes rank must be one in ONNX spec.", input
        )
    elif axes_rank == 0:
        # The axes is a scalar. Unsqueeze it to a rank 1 tensor.
        axes_t = _unsqueeze_helper(g, axes_t, [0])
        return g.op("Squeeze", input, axes_t)
    return g.op("Squeeze", input, axes_t)


def _reducesum_helper(
    g: jit_utils.GraphContext,
    input,
    axes_i=None,
    keepdims_i=1,
    noop_with_empty_axes_i=0,
):
    keepdims_i = _maybe_get_const(keepdims_i, "i")
    if g.opset >= 13:
        if axes_i:
            if not _is_value(axes_i):
                axes_i = g.op(
                    "Constant", value_t=torch.tensor(axes_i, dtype=torch.long)
                )
            return g.op(
                "ReduceSum",
                input,
                axes_i,
                keepdims_i=keepdims_i,
                noop_with_empty_axes_i=noop_with_empty_axes_i,
            )
        return g.op(
            "ReduceSum",
            input,
            keepdims_i=keepdims_i,
            noop_with_empty_axes_i=noop_with_empty_axes_i,
        )
    else:
        return g.op("ReduceSum", input, axes_i=axes_i, keepdims_i=keepdims_i)


def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim):
    output_size = _maybe_get_const(output_size, "is")
    if _is_value(output_size):
        offset = 2
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op("Cast", output_size, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        divisor = _slice_helper(
            g, g.op("Shape", input), axes=[0], ends=[sys.maxsize], starts=[offset]
        )
        divisor = g.op("Cast", divisor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scale_dims = g.op("Div", dividend, divisor)
        scales = g.op("Concat", offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [
            1.0
            if i < 2
            else float(output_size[-(dim - i)])
            / float(input.type().sizes()[-(dim - i)])
            for i in range(dim)
        ]
        scales = g.op(
            "Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32)
        )
    return scales


def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales):
    available_scales = _maybe_get_const(scales[0], "fs") != -1 and not _is_none(
        scales[0]
    )

    if not available_scales:
        return None

    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scales_list = g.op(
        "Constant", value_t=torch.tensor(_maybe_get_const(scales[0], "fs"))
    )
    scales = g.op("Concat", offsets, scales_list, axis_i=0)
    return scales


def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args):
    if mode == "nearest":
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return scales, align_corners


def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim):
    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    scale_factor_rank = _get_tensor_rank(scale_factor)
    if isinstance(scale_factor.type(), _C.ListType) or (
        scale_factor_rank is not None and scale_factor_rank > 0
    ):
        return g.op("Concat", offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op(
            "Cast", scale_factor, to_i=_C_onnx.TensorProtoDataType.FLOAT
        )
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    return scale_factor


def _interpolate_get_scales_and_mode(
    g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners
):
    mode = _maybe_get_const(mode, "s")
    if "linear" in mode:
        mode = "linear"
    if "cubic" in mode:
        mode = "cubic"
    _interpolate_warning(mode)

    align_corners = _maybe_get_const(align_corners, "b")
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented("interpolate", "align_corners == True")

    if not input.type().dim():
        return _unimplemented("interpolate", "missing input shape")
    dim = input.type().dim()

    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = _maybe_get_const(size, "t").dim() == 0
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op("Concat", *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        return _unimplemented(
            "interpolate", "Both size and scales are None in __interpolate"
        )
    return scale_factor, mode


def _argmin_argmax_helper(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
    op_name: str,
):
    def op_wrapper(input, axis_i, keepdims_i):
        if g.opset >= 12:
            return g.op(
                op_name,
                input,
                axis_i=axis_i,
                keepdims_i=keepdims_i,
                select_last_index_i=False,
            )
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)

    if _is_none(dim):
        flattened = _reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        output = op_wrapper(flattened, axis_i=0, keepdims_i=False)
        if keepdim:
            input_shape = g.op("Shape", input)
            input_shape_shape = g.op("Shape", input_shape)
            new_shape = g.op(
                "ConstantOfShape",
                input_shape_shape,
                value_t=torch.tensor([1], dtype=torch.int64),
            )
            output = g.op("Reshape", output, new_shape)
        return output

    dim = _parse_arg(dim, "i")
    return op_wrapper(input, axis_i=dim, keepdims_i=keepdim)


def _interpolate_helper(name, dim, interpolate_mode):
    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = (
            "asymmetric"
            if interpolate_mode == "nearest"
            else "align_corners"
            if align_corners
            else "half_pixel"
        )

        if scales is None:
            input_size = g.op("Shape", input)
            input_size_beg = _slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0]
            )
            output_size = g.op(
                "Cast", output_size, to_i=_C_onnx.TensorProtoDataType.INT64
            )
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )
                empty_scales = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                output_size,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"
        else:
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                mode_s=interpolate_mode,  # nearest, linear, or cubic
                nearest_mode_s="floor",
            )  # only valid when mode="nearest"

    return symbolic_fn


def __interpolate_helper(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
):
    mode = _maybe_get_const(mode, "s")
    if "linear" in mode:
        mode = "linear"
    if "cubic" in mode:
        mode = "cubic"
    align_corners = _maybe_get_const(align_corners, "b")
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = (
        "asymmetric"
        if mode == "nearest"
        else "align_corners"
        if align_corners
        else "half_pixel"
    )

    if not _is_none(size):
        input_size = g.op("Shape", input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        # in some cases size is not a packed list but size is a scalar
        # We need to also verify that (_maybe_get_const(size, "t").dim() == 0)
        # but this information is not always available. Try to get the dim,
        # and if not assume that it is not a scalar.
        try:
            is_scalar = not _is_packed_list(size) and (
                _maybe_get_const(size, "t").dim() == 0
            )
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn(
                    "Cannot verify if the output_size is a scalar "
                    "while exporting interpolate. Assuming that it is not a scalar.",
                    stacklevel=2,
                )

        if is_scalar:
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented(
                    "interpolate (with a scalar output_size)",
                    "missing input shape (try giving an array of output_size values)",
                )
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            size = g.op("Concat", *size, axis_i=0)
        size = g.op("Cast", size, to_i=_C_onnx.TensorProtoDataType.INT64)
        size = g.op("Concat", input_size, size, axis_i=0)

        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op(
                "Constant", value_t=torch.tensor([], dtype=torch.float32)
            )

        return g.op(
            "Resize",
            input,
            empty_roi,
            empty_scales,
            size,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )
    else:  # if not _is_none(scales)
        rank = _get_tensor_rank(input)
        if rank is None:
            return _unimplemented("interpolate (with scales)", "missing input shape")

        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        scales = _interpolate_get_scales(g, scale_factor, rank)
        return g.op(
            "Resize",
            input,
            empty_roi,
            scales,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )  # only valid when mode="nearest"


def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    if g.opset < 11:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            unbind,  # type: ignore[no-redef]
        )
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset13 import (
            unbind,  # type: ignore[no-redef]
        )
    return unbind(g, self, dim, _outputs)


def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            scatter,  # type: ignore[no-redef]
        )
    return scatter(g, self, dim, index, src)


def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    if g.opset <= 12:
        split_out = g.op("Split", self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset13 import split

        repeats = g.op("Constant", value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]


def _repeat_interleave_single_value_repeat_helper(
    g: jit_utils.GraphContext, self, repeats, dim
):
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (
        flatten,
        unsqueeze,
    )

    if not _is_tensor(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))

    const_repeats: bool = _is_constant(repeats)
    reps = _maybe_get_const(repeats, "t")

    # Convert 'repeats' to 1-d if it is 0-d.
    if _get_tensor_rank(repeats) == 0:
        repeats = g.op("Reshape", repeats, g.op("Constant", value_t=torch.tensor([1])))

    # Create a new dim of size 1, then expand it to be 'repeats' long, and finally collapse it.
    unsqueezed = unsqueeze(g, self, dim + 1)

    # repeats_per_dim is 1 for all dims except for the new unsqueezed dim, where it has value 'repeats'.
    if const_repeats:
        # 'Repeats' is a constant, 'repeats_per_dim' can be a constant.
        onehot = torch.ones(_get_tensor_rank(unsqueezed), dtype=torch.int64)  # type: ignore[arg-type]
        onehot[dim + 1] = reps
        repeats_per_dim = g.op("Constant", value_t=onehot)
    else:
        # 'Repeats' is a variable, 'repeats_per_dim' cannot be a constant.
        onehot = g.op(
            "OneHot",
            unsqueeze(g, dim + 1, 0),  # indices, must be >= 1-dimensional
            g.op(
                "Constant", value_t=torch.tensor(_get_tensor_rank(unsqueezed))
            ),  # depth
            g.op(
                "Concat", g.op("Constant", value_t=torch.tensor([1])), repeats, axis_i=0
            ),  # on/off values
        )
        repeats_per_dim = flatten(g, onehot, 0, 1)

    tiled = g.op("Tile", unsqueezed, repeats_per_dim)
    return flatten(g, tiled, dim, dim + 1)


def _arange_cast_helper(
    g: jit_utils.GraphContext, end, start=None, step=None, dtype=None
) -> tuple[
    _type_utils.JitScalarType,
    _C.Value | None,
    _C.Value | None,
    _C.Value | None,
]:
    def _is_all_integral(scalars):
        for scalar in scalars:
            scalar_type = _type_utils.JitScalarType.from_value(
                scalar, _type_utils.JitScalarType.UNDEFINED
            )
            if (
                scalar_type != _type_utils.JitScalarType.INT64
                and scalar_type != _type_utils.JitScalarType.UNDEFINED
            ):
                return False
        return True

    # This logic is based on torch.arange docs. If "dtype" is provided,
    # infer input types from dtype. If not, then check if any of start, stop,
    # or step are floating point, and infer the type from get_default.
    # Otherwise, the dtype is inferred to be torch.int64.
    if dtype is None or (_is_value(dtype) and _is_none(dtype)):
        if _is_all_integral([start, end, step]):
            scalar_type = _type_utils.JitScalarType.INT64
        else:
            scalar_type = _type_utils.JitScalarType.from_dtype(
                torch.get_default_dtype()
            )
    else:
        assert isinstance(dtype, int)
        # TODO(justinchuby): Check if dtype is indeed a int.
        scalar_type = _type_utils.JitScalarType(dtype)

    start = g.op("Cast", start, to_i=scalar_type.onnx_type()) if start else None
    end = g.op("Cast", end, to_i=scalar_type.onnx_type()) if end else None
    step = g.op("Cast", step, to_i=scalar_type.onnx_type()) if step else None
    return scalar_type, end, start, step


def _arange_helper(g: jit_utils.GraphContext, *args):
    if g.opset <= 10:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import arange
    else:
        from torch.onnx._internal.torchscript_exporter.symbolic_opset11 import (
            arange,  # type: ignore[no-redef]
        )
    return arange(g, *args)


def _size_helper(g: jit_utils.GraphContext, self, dim):
    full_shape = g.op("Shape", self)
    from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import select

    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, .
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

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `symbolic_helper.py_docs.md_docs.md`
- **Keyword Index**: `symbolic_helper.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

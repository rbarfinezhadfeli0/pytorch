# Documentation: `torch/_export/serde/serialize.py`

## File Metadata

- **Path**: `torch/_export/serde/serialize.py`
- **Size**: 157,635 bytes (153.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import base64
import copy
import copyreg
import dataclasses
import heapq
import inspect
import io
import json
import keyword
import logging
import math
import operator
import re
import traceback
import typing
from collections import namedtuple, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, cast, final, Optional, Union

import sympy

import torch
import torch.export.exported_program as ep
from torch._export.non_strict_utils import _enable_graph_inputs_of_type_nn_module
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.experimental import symbolic_shapes
from torch.utils import _pytree as pytree
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.symbol import prefix_str, SymT
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._traceback import CapturedTraceback
from torch.utils._triton import has_triton

from ..utils import remove_proxy_from_state_dict
from .schema import (  # type: ignore[attr-defined]
    Argument,
    ArgumentKind,
    BufferMutationSpec,
    ComplexValue,
    ConstantValue,
    CustomObjArgument,
    Device,
    ExportedProgram,
    GradientToParameterSpec,
    GradientToUserInputSpec,
    Graph,
    GraphArgument,
    GraphModule,
    GraphSignature,
    InputSpec,
    InputToBufferSpec,
    InputToConstantInputSpec,
    InputToCustomObjSpec,
    InputTokenSpec,
    InputToParameterSpec,
    InputToTensorConstantSpec,
    Layout,
    LossOutputSpec,
    MemoryFormat,
    ModuleCallEntry,
    ModuleCallSignature,
    NamedArgument,
    NamedTupleDef,
    Node,
    OptionalTensorArgument,
    OutputSpec,
    OutputTokenSpec,
    ParameterMutationSpec,
    RangeConstraint,
    ScalarType,
    SCHEMA_VERSION,
    SchemaVersion,
    SymBool,
    SymBoolArgument,
    SymExpr,
    SymExprHint,
    SymFloat,
    SymFloatArgument,
    SymInt,
    SymIntArgument,
    TensorArgument,
    TensorMeta,
    TokenArgument,
    TREESPEC_VERSION,
    UserInputMutationSpec,
    UserInputSpec,
    UserOutputSpec,
)
from .union import _Union


__all__ = [
    "serialize",
    "GraphModuleSerializer",
    "ExportedProgramSerializer",
    "GraphModuleDeserializer",
    "ExportedProgramDeserializer",
]

log = logging.getLogger(__name__)


class SerializeError(RuntimeError):
    pass


def _reverse_map(d: dict[Any, Enum]):
    return {v.value: k for k, v in d.items()}


MetaType = Union[
    FakeTensor,
    int,
    torch.SymInt,
    float,
    torch.SymFloat,
    bool,
    torch.SymBool,
    ep.CustomObjArgument,
]

DEFAULT_PICKLE_PROTOCOL = 2

ST_DELIMITER = ";"

_TORCH_TO_SERIALIZE_DTYPE = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.uint16: ScalarType.UINT16,
    torch.int16: ScalarType.SHORT,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.float16: ScalarType.HALF,
    torch.float32: ScalarType.FLOAT,
    torch.float64: ScalarType.DOUBLE,
    torch.complex32: ScalarType.COMPLEXHALF,
    torch.complex64: ScalarType.COMPLEXFLOAT,
    torch.complex128: ScalarType.COMPLEXDOUBLE,
    torch.bool: ScalarType.BOOL,
    torch.bfloat16: ScalarType.BFLOAT16,
    torch.float8_e4m3fn: ScalarType.FLOAT8E4M3FN,
    torch.float8_e5m2: ScalarType.FLOAT8E5M2,
    torch.float8_e4m3fnuz: ScalarType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2fnuz: ScalarType.FLOAT8E5M2FNUZ,
}


_SERIALIZE_TO_TORCH_DTYPE = _reverse_map(_TORCH_TO_SERIALIZE_DTYPE)  # type: ignore[arg-type]


_TORCH_TO_SERIALIZE_LAYOUT = {
    torch.sparse_coo: Layout.SparseCoo,
    torch.sparse_csr: Layout.SparseCsr,
    torch.sparse_csc: Layout.SparseCsc,
    torch.sparse_bsr: Layout.SparseBsr,
    torch.sparse_bsc: Layout.SparseBsc,
    torch._mkldnn: Layout._mkldnn,  # type: ignore[attr-defined]
    torch.strided: Layout.Strided,
}


_SERIALIZE_TO_TORCH_LAYOUT = _reverse_map(_TORCH_TO_SERIALIZE_LAYOUT)  # type: ignore[arg-type]


_TORCH_TO_SERIALIZE_MEMORY_FORMAT = {
    torch.contiguous_format: MemoryFormat.ContiguousFormat,
    torch.channels_last: MemoryFormat.ChannelsLast,
    torch.channels_last_3d: MemoryFormat.ChannelsLast3d,
    torch.preserve_format: MemoryFormat.PreserveFormat,
}


_SERIALIZE_TO_TORCH_MEMORY_FORMAT = _reverse_map(_TORCH_TO_SERIALIZE_MEMORY_FORMAT)  # type: ignore[arg-type]

_SYM_OPS = {
    operator.eq,
    operator.ne,
    operator.le,
    operator.ge,
    operator.lt,
    operator.gt,
    operator.neg,
    operator.pos,
    operator.and_,
    operator.or_,
    math.trunc,
    torch.sym_not,
    operator.mul,
    operator.add,
    operator.sub,
    operator.floordiv,
    operator.mod,
    operator.pow,
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_sqrt,
    operator.truediv,
    operator.and_,
}


assert not any(isinstance(op, torch._ops.OpOverload) for op in _SYM_OPS)


@dataclass
class SerializedArtifact:
    exported_program: bytes
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


@dataclass
class _SerializedProgram:
    exported_program: ExportedProgram
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


class LazyMap(dict):
    """
    Dictionary class for deferred instantiation of node metadata values.
    Purpose is to avoid creation of symbolic-shape tensors before relevant shape guards are parsed.
    """

    def __init__(self):
        self.map = {}
        self.evaluated = set()

    def __setitem__(self, k, v):
        self.map[k] = v

    def __getitem__(self, k):
        out = self.map[k]
        if k in self.evaluated:
            return out
        self.evaluated.add(k)
        self.map[k] = out()
        return self.map[k]

    def __repr__(self):
        return self.map.__repr__()


def deserialize_device(d: Device) -> torch.device:
    if d.index is None:
        return torch.device(type=d.type)  # type: ignore[call-overload]
    return torch.device(type=d.type, index=d.index)


def deserialize_size(sizes: Sequence[SymInt]) -> tuple[int, ...]:
    for sym_int_size in sizes:
        assert sym_int_size.type == "as_int", (
            f"Only as_int is supported, got {sym_int_size.type}"
        )
    return tuple(sym_int_size.as_int for sym_int_size in sizes)


def deserialize_stride(strides: Sequence[SymInt]) -> tuple[int, ...]:
    for sym_int_stride in strides:
        assert sym_int_stride.type == "as_int", (
            f"Only as_int is supported, got {sym_int_stride.type}"
        )
    return tuple(sym_int_stride.as_int for sym_int_stride in strides)


def deserialize_scalar_type(st: ScalarType) -> torch.dtype:
    return _SERIALIZE_TO_TORCH_DTYPE[st]


def deserialize_storage_offset(offset: SymInt) -> int:
    assert offset.type == "as_int", f"Only as_int is supported, got {offset.type}"
    return offset.as_int


def _print_sympy(s: Union[torch.SymInt, torch.SymBool, torch.SymFloat, sympy.Expr]):
    if isinstance(s, (torch.SymInt, torch.SymBool, torch.SymFloat)):
        s = s.node.expr
    return sympy.printing.repr.srepr(s)


def serialize_sym_int(s: Union[int, torch.SymInt]) -> SymInt:
    if isinstance(s, (torch.SymInt, sympy.Symbol, int)):
        if symbolic_shapes.is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            assert isinstance(s, (torch.SymInt, sympy.Symbol))
            if s.node.hint is None:
                return SymInt.create(as_expr=SymExpr(_print_sympy(s)))
            else:
                return SymInt.create(
                    as_expr=SymExpr(
                        _print_sympy(s),
                        hint=SymExprHint.create(as_int=s.node.hint),
                    )
                )
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_float(s: Union[float, torch.SymFloat]) -> SymFloat:
    if isinstance(s, (torch.SymFloat, sympy.Symbol, float)):
        if symbolic_shapes.is_concrete_float(s):
            return SymFloat.create(as_float=float(s))
        else:
            assert isinstance(s, (torch.SymFloat, sympy.Symbol))
            if s.node.hint is None:
                return SymFloat.create(as_expr=SymExpr(_print_sympy(s)))
            else:
                return SymFloat.create(
                    as_expr=SymExpr(
                        _print_sympy(s),
                        hint=SymExprHint.create(as_float=s.node.hint),
                    )
                )
    else:
        raise SerializeError(
            f"SymFloat should be either symbol or float, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_bool(s: Union[bool, torch.SymBool]) -> SymBool:
    if isinstance(s, (torch.SymBool, bool)):
        if symbolic_shapes.is_concrete_bool(s):
            return SymBool.create(as_bool=bool(s))
        else:
            return SymBool.create(as_expr=SymExpr(expr_str=_print_sympy(s)))
    else:
        raise SerializeError(
            f"SymBool should be either symbol or bool, got `{s}` of type `{type(s)}`"
        )


def serialize_tensor_meta(t: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `t`.
    """
    return TensorMeta(
        dtype=_TORCH_TO_SERIALIZE_DTYPE[t.dtype],
        sizes=[serialize_sym_int(s) for s in t.shape],
        requires_grad=t.requires_grad,
        device=Device(type=t.device.type, index=t.device.index),
        strides=[serialize_sym_int(s) for s in t.stride()],
        storage_offset=serialize_sym_int(t.storage_offset()),
        layout=_TORCH_TO_SERIALIZE_LAYOUT[t.layout],
    )


_CURRENT_DESERIALIZER: Optional["GraphModuleDeserializer"] = None


def _reduce_fake_tensor(fake_tensor: FakeTensor):
    is_parameter = isinstance(fake_tensor, torch.nn.Parameter)
    tensor_meta = serialize_tensor_meta(fake_tensor)
    tensor_meta_bytes = json.dumps(
        _dataclass_to_dict(tensor_meta), cls=EnumEncoder
    ).encode("utf-8")
    return _reconstruct_fake_tensor, (tensor_meta_bytes, is_parameter)


def _reconstruct_fake_tensor(
    serialized_tensor_meta: bytes, is_parameter: bool
) -> FakeTensor:
    # Deserialize the bytes into a TensorMeta
    json_tensor_meta = json.loads(serialized_tensor_meta.decode("utf-8"))
    tensor_meta = _dict_to_dataclass(TensorMeta, json_tensor_meta)
    # Find the current fake mode
    assert _CURRENT_DESERIALIZER is not None, (
        "Need access to current deserializer state"
    )
    fake_tensor = _CURRENT_DESERIALIZER.deserialize_tensor_meta(tensor_meta)
    if is_parameter:
        fake_tensor = torch.nn.Parameter(fake_tensor)  # type: ignore[assignment]
    # pyrefly: ignore [bad-return]
    return fake_tensor


def serialize_torch_artifact(
    artifact: Optional[Any], pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL
) -> bytes:
    if artifact is None:
        return b""

    assert FakeTensor not in copyreg.dispatch_table, (
        "Refusing to stomp on existing FakeTensor reducer"
    )
    try:
        copyreg.pickle(FakeTensor, _reduce_fake_tensor)
        buffer = io.BytesIO()
        # This is a workaround for backend's tensor deserialization problem:
        # unpickleTensor() always create a tensor on the device where it was originally saved
        # This behavior is bad for multi-gpu training, as we wish to directly load the tensor
        # on the designated device.
        # For now, we simply move the tensor to cpu before saving.
        # TODO: this should be fixed by deserialization instead.
        torch.save(artifact, buffer, pickle_protocol=pickle_protocol)
        return buffer.getvalue()
    finally:
        del copyreg.dispatch_table[FakeTensor]


def deserialize_torch_artifact(
    serialized: Union[dict[str, Any], tuple[Any, ...], bytes],
):
    if isinstance(serialized, (dict, tuple)):
        return serialized
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    # weights_only=False as we want to load custom objects here (e.g. ScriptObject)
    artifact = torch.load(buffer, weights_only=False)
    assert isinstance(artifact, (tuple, dict))
    return artifact


def _sympy_int_to_int(val: sympy.Expr, adjust: str) -> Optional[int]:
    # Convert simple sympy Integers into concrete int
    if val in (sympy.oo, int_oo):
        return None
    if val in (-sympy.oo, -int_oo):
        return None
    if isinstance(val, sympy.Integer):
        return int(val)

    # TODO: Remove this adjustment when Ed gets rid of fractional ranges
    log.warning(
        "Export constraints cannot be non-integer expressions. Found "
        "type %s, and value %s. We will attempt to %s "
        "this value.",
        type(val),
        val,
        adjust,
    )

    if adjust == "floor":
        return math.floor(val)
    elif adjust == "ceil":
        return math.ceil(val)
    else:
        raise RuntimeError(f"Got invalid adjustment {adjust}")


def _int_to_sympy_int(val: Optional[int], default) -> sympy.Expr:
    # Convert concrete int into simple sympy Integers
    if val is None:
        return default
    if val in [-int_oo, int_oo]:
        return val
    if val == math.inf:
        return int_oo
    if val == -math.inf:
        return -int_oo
    return sympy.Integer(val)


def _symbol_index(sym: sympy.Symbol, sym_type: SymT):
    return int(str(sym)[len(prefix_str[sym_type]) :])


def serialize_range_constraints(
    range_constraints: dict[sympy.Symbol, ValueRanges],
) -> dict[str, RangeConstraint]:
    return {
        str(k): RangeConstraint(
            _sympy_int_to_int(v.lower, "ceil"),  # type: ignore[arg-type]
            _sympy_int_to_int(v.upper, "floor"),  # type: ignore[arg-type]
        )
        for k, v in range_constraints.items()
    }


def _get_schema_from_target(target):
    if isinstance(target, torch._ops.OpOverload):
        return target._schema
    elif type(target) in _serialization_registry:
        return _serialization_registry[type(target)].op_schema(target)
    raise RuntimeError(f"Cannot find schema for {type(target)}")


@dataclass
class GraphState:
    inputs: list[Argument] = field(default_factory=list)
    outputs: list[Argument] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    tensor_values: dict[str, TensorMeta] = field(default_factory=dict)
    sym_int_values: dict[str, SymInt] = field(default_factory=dict)
    sym_bool_values: dict[str, SymBool] = field(default_factory=dict)
    sym_float_values: dict[str, SymFloat] = field(default_factory=dict)
    is_single_tensor_return: bool = False
    custom_obj_values: dict[str, CustomObjArgument] = field(default_factory=dict)


class Final(type):
    def __new__(metacls, name, bases, classdict):
        for b in bases:
            if isinstance(b, Final):
                raise TypeError(f"type '{b.__name__}' is not an acceptable base type")
        return type.__new__(metacls, name, bases, dict(classdict))


def is_metadata_matched(config, entry_metadata):
    metadata_attrs = ["num_cpu_threads", "num_warps", "num_stages", "num_ctas"]
    for attr in metadata_attrs:
        if hasattr(config, attr) and hasattr(entry_metadata, attr):
            if getattr(config, attr) != getattr(entry_metadata, attr):
                return False
    return True


def get_triton_kernel_and_cache_entry(node: torch.fx.Node):
    assert (
        node.target
        is torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional
    )

    assert has_triton(), "triton required to serialize triton kernels"
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    assert isinstance(node.kwargs["kernel_idx"], int)
    kernel = torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
        node.kwargs["kernel_idx"]
    )

    # For Autotuner, we need to look at the underlying JITFunction's cache
    # since the Autotuner itself doesn't have a cache
    is_autotuner = isinstance(kernel, Autotuner)
    # pyrefly: ignore [missing-attribute]
    actual_kernel = kernel.fn if is_autotuner else kernel

    if hasattr(actual_kernel, "device_caches"):
        caches = actual_kernel.device_caches
        assert len(caches.keys()) == 1
        cache = next(iter(caches.values()))[0]
    elif hasattr(actual_kernel, "cache"):
        # old path, still used for cpu triton builds
        caches = actual_kernel.cache
        assert len(caches.keys()) == 1
        cache = next(iter(caches.values()))
    else:
        raise AssertionError(
            # pyrefly: ignore [missing-attribute]
            f"kernel caches not found for kernel {actual_kernel.__name__}"
        )

    if len(cache.keys()) == 1:
        return actual_kernel, next(iter(cache.values()))

    has_constexprs = (
        isinstance(actual_kernel, JITFunction)
        and hasattr(actual_kernel, "constexprs")
        and len(actual_kernel.constexprs) > 0
    )

    if has_constexprs:
        constexpr_vals = {}
        # pyrefly: ignore [missing-attribute]
        for constexpr_idx in actual_kernel.constexprs:
            # pyrefly: ignore [missing-attribute]
            if constexpr_idx < len(actual_kernel.arg_names):
                # pyrefly: ignore [missing-attribute]
                param_name = actual_kernel.arg_names[constexpr_idx]
                kwargs_dict = node.kwargs.get("kwargs", {})
                if isinstance(kwargs_dict, dict):
                    if param_name in kwargs_dict:
                        constexpr_vals[param_name] = kwargs_dict[param_name]

        expected_values = [
            # pyrefly: ignore [missing-attribute]
            constexpr_vals[actual_kernel.arg_names[idx]]
            # pyrefly: ignore [missing-attribute]
            for idx in actual_kernel.constexprs
            # pyrefly: ignore [missing-attribute]
            if actual_kernel.arg_names[idx] in constexpr_vals
        ]

        matching_entries = []
        for sig_key, cache_entry in cache.items():
            constexpr_matches = re.findall(r"\('constexpr',\s*([^)]+)\)", sig_key)
            if constexpr_matches:
                constexpr_values = []
                for match in constexpr_matches:
                    if match in ("True", "False"):
                        constexpr_values.append(match == "True")
                    elif "." in match or "e" in match or "E" in match:
                        constexpr_values.append(float(match))
                    else:
                        constexpr_values.append(int(match))

                if constexpr_values == expected_values:
                    matching_entries.append((sig_key, cache_entry))
    else:
        matching_entries = list(cache.items())

    if len(matching_entries) == 0:
        raise AssertionError(
            # pyrefly: ignore [missing-attribute]
            f"couldn't find a kernel cache entry with metadata matching the autotuner configs for kernel {actual_kernel.__name__}. "
            f"Available cache keys: {list(cache.keys())}"
        )

    if len(matching_entries) == 1:
        return actual_kernel, matching_entries[0][1]

    if is_autotuner:
        for _sig_key, cache_entry in matching_entries:
            entry_metadata = cache_entry.metadata
            # pyrefly: ignore [missing-attribute]
            for config in kernel.configs:
                if is_metadata_matched(config, entry_metadata):
                    return actual_kernel, cache_entry

        raise AssertionError(
            # pyrefly: ignore [missing-attribute]
            f"Multiple cache entries found for autotuned kernel {actual_kernel.__name__} "
            f"{'with same constexpr values' if has_constexprs else 'with no constexpr'} "
            f"and couldn't disambiguate using configs. "
        )

    raise AssertionError(
        # pyrefly: ignore [missing-attribute]
        f"Multiple cache entries found for non-autotuned kernel {actual_kernel.__name__} "
        f"{'with same constexpr values' if has_constexprs else 'with no constexpr'}. "
        f"This should not happen. Available cache keys: {[key for key, _ in matching_entries]}"
    )


@final
class GraphModuleSerializer(metaclass=Final):
    def __init__(
        self,
        graph_signature: ep.ExportGraphSignature,
        module_call_graph: list[ep.ModuleCallEntry],
    ):
        self.graph_state = GraphState()
        self.graph_signature = graph_signature
        self.module_call_graph = module_call_graph
        self.custom_objs: dict[str, torch._C.ScriptObject] = {}
        self.duplicate_getitem_nodes: dict[str, str] = {}
        self.treespec_namedtuple_fields: dict[str, NamedTupleDef] = {}

    @contextmanager
    def save_graph_state(self):
        saved = self.graph_state
        self.graph_state = GraphState()
        try:
            yield
        finally:
            self.graph_state = saved

    def handle_placeholder(self, node: torch.fx.Node):
        assert node.op == "placeholder"
        val = node.meta["val"]
        log.debug("[handle_placeholder] %s: %s", node.name, val)
        if isinstance(val, torch.Tensor):
            graph_input = Argument.create(
                as_tensor=self.serialize_tensor_output(node.name, val)
            )
        elif isinstance(val, torch.SymInt):
            graph_input = Argument.create(
                as_sym_int=self.serialize_sym_int_output(node.name, val)
            )
        elif isinstance(val, torch.SymFloat):
            raise AssertionError("SymFloat graph input is not implemented yet.")
        elif isinstance(val, (int, bool, str, float, type(None))):
            graph_input = self.serialize_input(val)
        elif isinstance(val, ep.CustomObjArgument):
            class_fqn = val.class_fqn
            graph_input = Argument.create(
                as_custom_obj=CustomObjArgument(name=node.name, class_fqn=class_fqn)
            )
            self.graph_state.custom_obj_values[node.name] = (
                self.serialize_script_obj_meta(val)
            )
        else:
            raise AssertionError(f"Unimplemented graph input type: {node.meta['val']}")
        self.graph_state.inputs.append(graph_input)

    def handle_output(self, node: torch.fx.Node):
        assert node.op == "output"
        assert len(node.args) == 1, "FX.Node's args should have one arg"
        node_args = node.args[0]
        log.debug("[handle_output] %s: %s", node.name, node_args)
        if isinstance(node_args, torch.fx.Node):
            # For singleton tensor returns
            self.graph_state.is_single_tensor_return = True
            self.graph_state.outputs = [self.serialize_input(node_args)]
        else:
            assert isinstance(node_args, (tuple, list))
            self.graph_state.outputs = [self.serialize_input(arg) for arg in node_args]

    def serialize_operator(self, target) -> str:
        if isinstance(target, str):
            return target
        elif target.__module__.startswith("torch._ops"):
            # TODO(zhxchen17) Maybe provide a function name helper in FX.
            # From torch.fx.node._get_qualified_name
            module = target.__module__.replace("torch._ops", "torch.ops")
            return f"{module}.{target.__name__}"
        else:  # TODO(zhxchen17) Don't catch all here.
            return f"{target.__module__}.{target.__name__}"

    def handle_call_function(self, node: torch.fx.Node):
        assert node.op == "call_function"
        meta_val = node.meta.get("val")
        log.debug(
            "[handle_call_function] %s: %s(%s, {%s}) -> %s",
            node.name,
            node.target,
            node.args,
            node.kwargs,
            meta_val,
        )

        # getitem has been handled in the producer node, skip it here
        if node.target is operator.getitem:
            return

        if node.target in _SYM_OPS or (
            meta_val is not None
            and isinstance(meta_val, (torch.SymInt, torch.SymBool, torch.SymFloat))
        ):
            assert len(node.kwargs) == 0
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_sym_op_inputs(node.target, node.args),
                outputs=[self.serialize_output(node.name, meta_val)],
                metadata=self.serialize_metadata(node),
            )
        elif isinstance(node.target, torch._ops.OpOverload):
            ex_node = Node(
                target=self.serialize_operator(node.target),
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                # TODO: create a new tensor_values here, meta might have faketensor info
                metadata=self.serialize_metadata(node),
            )
        elif isinstance(node.target, torch._ops.HigherOrderOperator):

            def _is_hop_single_tensor_return(node) -> bool:
                assert isinstance(node.target, torch._ops.HigherOrderOperator)
                # HOP schema is not always available, so we look at node.meta["val"]
                meta_val = node.meta.get("val", None)
                return meta_val is not None and isinstance(meta_val, torch.Tensor)

            # Special handle serialization for aoti_call_delegate
            if node.target is torch._higher_order_ops.aoti_call_delegate:
                serializable_args = list(node.args)

                # AOTI lowered module is not serializable, serialize the aoti_path instead
                lowered_module_name: str = node.args[0].name  # type: ignore[assignment, no-untyped-def, union-attr]
                assert hasattr(node.graph.owning_module, lowered_module_name)
                lowered_module = getattr(node.graph.owning_module, lowered_module_name)  # type: ignore[no-untyped-def]
                serializable_args[0] = lowered_module.aoti_path

                # AOTI compiled graph module in node.args[0] is stateful, and will fail the verifier check
                # Skip serializing original_gm as a workaround
                serializable_args[1] = None

                serializable_weight_nodes = []
                if serializable_args[2] is not None and isinstance(
                    serializable_args[2], Iterable
                ):
                    for weight_node in serializable_args[2]:
                        # skip passing custom obj into the weight arg as an hack
                        # The schema of weight input is a list of Tensors.
                        # Downstream runtime is not actively consuming the weighs arg for anything meaningful.
                        if isinstance(weight_node, torch.fx.Node) and isinstance(
                            weight_node.meta.get("val", None), ep.CustomObjArgument
                        ):
                            continue
                        serializable_weight_nodes.append(weight_node)
                    serializable_args[2] = serializable_weight_nodes

                def serialize_tensor_list_output(node):
                    meta_val = node.meta.get("val", None)
                    tensor_args = []
                    for idx, meta in enumerate(meta_val):
                        name = self._output_node_name_at_index(node, idx)
                        tensor_args.append(self.serialize_tensor_output(name, meta))
                    return [Argument.create(as_tensors=tensor_args)]

                ex_node = Node(
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(serializable_args, node.kwargs),
                    outputs=serialize_tensor_list_output(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=False,
                )
            elif (
                node.target
                is torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional
            ):
                kernel, kernel_cache_entry = get_triton_kernel_and_cache_entry(node)
                kernel_cache_metadata = kernel_cache_entry.metadata

                meta_val = node.meta["val"]
                assert isinstance(meta_val, dict)

                output_keys = meta_val.keys()
                output_indices = []

                constexpr_keys = {p.name for p in kernel.params if p.is_constexpr}
                found_constexpr = False
                args_new = ()
                i = 0

                assert isinstance(node.kwargs["kwargs"], dict)
                for k, v in node.kwargs["kwargs"].items():
                    # don't serialize constexpr since they will
                    # be embedded into the binary and don't
                    # need to be passed around as attributes
                    if k in constexpr_keys:
                        found_constexpr = True
                        continue

                    assert not found_constexpr, (
                        "non-constexpr args found after constexpr arg(s)"
                    )

                    if k in output_keys:
                        output_indices.append(i)
                    args_new += (v,)  # type: ignore[assignment]
                    i += 1

                assert isinstance(node.kwargs["grid"], list)

                kernel_name_with_hash = (
                    f"{kernel.fn.__name__}_{kernel_cache_metadata.hash}"
                )
                kwargs_new = {
                    "name": kernel_name_with_hash,
                    "grid": node.kwargs["grid"][0],
                    "output_indices": output_indices,
                    "num_warps": kernel_cache_metadata.num_warps,
                }
                if hasattr(kernel_cache_metadata, "num_cpu_threads"):
                    kwargs_new["num_cpu_threads"] = (
                        kernel_cache_metadata.num_cpu_threads
                    )

                if hasattr(kernel_cache_metadata, "shared"):
                    kwargs_new["shared_memory_bytes"] = kernel_cache_metadata.shared

                ex_node = Node(
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(args_new, kwargs_new),
                    outputs=self.serialize_hoo_outputs(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=_is_hop_single_tensor_return(node),
                )
            else:
                ex_node = Node(
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(node.args, node.kwargs),
                    outputs=self.serialize_hoo_outputs(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=_is_hop_single_tensor_return(node),
                )
        elif type(node.target) in _serialization_registry:
            # Sanity check for unhandled serialization.
            assert type(node.target) in _serialization_registry, (
                f"{type(node.target)} is not supported in export serialization."
            )

            handler = _serialization_registry[type(node.target)]
            namespace = handler.namespace()
            op_name = handler.to_op_name(node.target)
            assert isinstance(namespace, str) and isinstance(op_name, str)
            assert ":" not in namespace and ":" not in op_name
            ex_node = Node(
                target=f"#{namespace}:{op_name}",
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                metadata=self.serialize_metadata(node),
            )
        else:
            raise SerializeError(f"Serializing {node.target} is not supported")

        self.graph_state.nodes.append(ex_node)

    def handle_get_attr(self, node):
        log.debug("[handle_get_attr] %s", node.name)

    def _output_node_at_index(self, node, index) -> Optional[torch.fx.Node]:
        user_node = None
        for user in node.users:
            assert user.target is operator.getitem, f"{user} is not a getitem node"
            if index == user.args[1]:
                if user_node is None:
                    user_node = user
                else:
                    # We want to deduplicate getitem nodes that are trying to
                    # index to the same index
                    self.duplicate_getitem_nodes[user.name] = user_node.name
        return user_node

    def _output_node_name_at_index(self, node, index) -> str:
        user_node = self._output_node_at_index(node, index)
        if user_node is None:
            return f"{node.name}_unused_{index}"
        else:
            return user_node.name

    def serialize_metadata(self, node: torch.fx.Node) -> dict[str, str]:
        ret = {}

        if stack_trace := node.meta.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        if nn_module_stack := node.meta.get("nn_module_stack"):

            def export_nn_module_stack(val):
                assert isinstance(val, tuple) and len(val) == 2
                path, ty = val

                assert isinstance(path, str)
                assert isinstance(ty, str)

                return path + "," + ty

            # Serialize to "key,orig_path,type_str"
            nn_module_list = [
                f"{k},{export_nn_module_stack(v)}" for k, v in nn_module_stack.items()
            ]
            ret["nn_module_stack"] = ST_DELIMITER.join(nn_module_list)

        if source_fn_st := node.meta.get("source_fn_stack"):
            source_fn_list = [
                f"{source_fn[0]},{self.serialize_operator(source_fn[1])}"
                for source_fn in source_fn_st
            ]
            ret["source_fn_stack"] = ST_DELIMITER.join(source_fn_list)

        if torch_fn := node.meta.get("torch_fn"):
            ret["torch_fn"] = ST_DELIMITER.join(list(torch_fn))

        if custom := node.meta.get("custom"):
            try:
                ret["custom"] = json.dumps(custom)
            except Exception as e:
                raise SerializeError(
                    f"Failed to serialize custom metadata for node {node.name} with error {e}"
                ) from e

        return ret

    def serialize_script_obj_meta(
        self, script_obj_meta: ep.CustomObjArgument
    ) -> CustomObjArgument:
        log.debug("[serialize_script_obj_meta] %s", script_obj_meta)
        return CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    def serialize_sym_op_inputs(self, op, args) -> list[NamedArgument]:
        if isinstance(op, torch._ops.OpOverload):
            args_names = [arg.name for arg in op._schema.arguments]
        else:
            assert op in _SYM_OPS
            args_names = list(inspect.signature(op).parameters.keys())
        serialized_args = []
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(
                    name=args_name,
                    arg=self.serialize_input(arg),
                    kind=ArgumentKind.POSITIONAL,
                )
            )
        return serialized_args

    def serialize_inputs(
        self,
        target: Any,  # torch._ops.OpOverload and other custom operator types.
        args,
        kwargs=None,
    ) -> list[NamedArgument]:
        schema = None
        serialized_args = []

        if isinstance(target, torch._higher_order_ops.torchbind.CallTorchBind):
            obj = args[0]
            method = args[1]
            schema = target.schema(obj, method)
        else:
            assert isinstance(
                target, (torch._ops.OpOverload, *_registered_extension_types())
            )
            schema = _get_schema_from_target(target)
        assert schema is not None
        kwargs = kwargs or {}

        for i, schema_arg in enumerate(schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(
                            kwargs[schema_arg.name], schema_arg.type
                        ),
                        kind=ArgumentKind.KEYWORD,
                    )
                )
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(args[i], schema_arg.type),
                        kind=ArgumentKind.POSITIONAL,
                    )
                )
            else:
                # We intentionally don't serialize the missing arguments
                # with default values
                pass

        return serialized_args

    def serialize_hoo_inputs(self, args, kwargs) -> list[NamedArgument]:
        """
        For serializing HOO inputs since HOOs do not have a schema.
        """
        inputs = [
            NamedArgument(
                name="", arg=self.serialize_input(a), kind=ArgumentKind.POSITIONAL
            )
            for a in args
        ]
        inputs.extend(
            [
                NamedArgument(
                    name=name,
                    arg=self.serialize_input(a),
                    kind=ArgumentKind.KEYWORD,
                )
                for name, a in kwargs.items()
            ]
        )
        return inputs

    def is_inductor_sym_int_arg(self, arg) -> bool:
        # This is a special branch for handling SymInt args in inductor's
        # ExternalFallbackNode.
        # For regular FX graph, SymInt arg should be a fx.Node and should be
        # verified with is_sym_int_arg()
        return type(arg) is int or isinstance(arg, torch.SymInt)

    def is_sym_int_arg(self, arg) -> bool:
        return type(arg) is int or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_int_values
        )

    def is_sym_float_arg(self, arg) -> bool:
        return isinstance(arg, float) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_float_values
        )

    def is_sym_bool_arg(self, arg) -> bool:
        return isinstance(arg, bool) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_bool_values
        )

    # should be torch._C.JitType but that annotation is busted
    def serialize_input(self, arg, arg_type: Optional[Any] = None) -> Argument:
        import torch._inductor.ir as inductor_ir

        inductor_tensor_buffers = (
            inductor_ir.Buffer,
            inductor_ir.ReinterpretView,
        )

        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                assert isinstance(arg.target, str)
                attr = getattr(arg.graph.owning_module, arg.target)

                if isinstance(attr, torch.Tensor):
                    raise SerializeError(
                        "getattr nodes containing tensors should not appear in the graph"
                    )
                elif isinstance(attr, torch.fx.GraphModule):
                    with self.save_graph_state():
                        graph = self.serialize_graph(attr)
                    return Argument.create(
                        as_graph=GraphArgument(name=arg.target, graph=graph)
                    )
                elif type(attr).__name__ == "LoweredBackendModule":
                    # Special handling for executorch_call_delegate HOP
                    # It's first argument is a LoweredBackendModule, for which we
                    # serialize name and backend id of the lowered module
                    module_name = getattr(attr, "module_name", None)
                    backend_id = getattr(attr, "backend_id", None)
                    assert module_name is not None, "module_name should not be None"
                    assert backend_id is not None, "backend_id should not be None"
                    return Argument.create(as_string=f"{module_name}-{backend_id}")
                else:
                    raise SerializeError(
                        f"Unsupported getattr attribute {arg.target} with type: {type(attr)}"
                    )
            elif self.is_sym_int_arg(arg):
                return Argument.create(
                    as_sym_int=SymIntArgument.create(as_name=arg.name)
                )
            elif self.is_sym_float_arg(arg):
                return Argument.create(
                    as_sym_float=SymFloatArgument.create(as_name=arg.name)
                )
            elif self.is_sym_bool_arg(arg):
                return Argument.create(
                    as_sym_bool=SymBoolArgument.create(as_name=arg.name)
                )
            elif isinstance(arg.meta["val"], ep.CustomObjArgument):
                return Argument.create(
                    as_custom_obj=CustomObjArgument(
                        name=arg.name, class_fqn=arg.meta["val"].class_fqn
                    )
                )
            elif arg.name in self.duplicate_getitem_nodes:
                dedup_name = self.duplicate_getitem_nodes[arg.name]
                return Argument.create(as_tensor=TensorArgument(name=dedup_name))
            else:
                return Argument.create(as_tensor=TensorArgument(name=arg.name))
        elif isinstance(arg, inductor_tensor_buffers):
            # Other branches are for arguments in fx node.
            # This is a special branch for handling buffers (representing tensor arguments)
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            arg_name = arg.get_name()
            assert arg_name is not None, "Buffer must have valid name"
            return Argument.create(as_tensor=TensorArgument(name=arg_name))
        elif isinstance(arg, inductor_ir.TorchBindObject):
            # This is a special branch for handling TorchBindObject
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            arg_name = arg.get_name()
            assert arg_name is not None, "Buffer must have valid name"
            arg_val = arg.get_real_obj()
            class_fqn = arg_val._type().qualified_name()
            self.custom_objs[arg_name] = arg_val
            return Argument.create(as_custom_obj=CustomObjArgument(arg_name, class_fqn))
        elif isinstance(arg, torch.SymInt):
            # This is a special branch for handling SymInt args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_int_arg(arg) being true
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=str(arg)))
        elif isinstance(arg, torch.SymFloat):
            # This is a special branch for handling SymFloat args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_float_arg(arg) being true
            return Argument.create(
                as_sym_float=SymFloatArgument.create(as_name=str(arg))
            )
        elif type(arg) is bool:
            return Argument.create(as_bool=arg)
        elif type(arg) is str:
            return Argument.create(as_string=arg)
        elif type(arg) is int:
            return Argument.create(as_int=arg)
        elif type(arg) is float:
            return Argument.create(as_float=arg)
        elif type(arg) is complex:
            return Argument.create(
                as_complex=ComplexValue(real=arg.real, imag=arg.imag)
            )
        elif arg is None:
            return Argument.create(as_none=True)
        elif isinstance(arg, (list, tuple)):
            if len(arg) == 0:
                if arg_type is not None:
                    if isinstance(arg_type, torch.OptionalType):
                        arg_type = arg_type.getElementType()  # type: ignore[assignment]
                    assert isinstance(arg_type, torch.ListType)
                    elem_type = arg_type.getElementType()
                    if isinstance(elem_type, torch.OptionalType):
                        elem_type = elem_type.getElementType()

                    if isinstance(elem_type, torch.BoolType):
                        return Argument.create(as_bools=[])
                    elif isinstance(elem_type, torch.IntType):
                        return Argument.create(as_ints=[])
                    elif isinstance(elem_type, torch.FloatType):
                        return Argument.create(as_floats=[])
                    elif isinstance(elem_type, torch.StringType):
                        return Argument.create(as_strings=[])
                    elif isinstance(elem_type, torch.TensorType):
                        return Argument.create(as_tensors=[])
                    else:
                        # I believe empty symint lists default to ints, but
                        # please file an issue if this is not the case
                        raise SerializeError(f"Empty list with type {elem_type} nyi.")
                else:
                    # We could serialize this by default to a tensor list. This
                    # is needed in the HOO case
                    log.warning(
                        "Unsure how to serialize the given empty list, "
                        "as we don't know what is the type of this argument. "
                        "Serializing it as a tensor list by default."
                    )
                    return Argument.create(as_tensors=[])

            if all(type(a) is bool for a in arg):
                return Argument.create(as_bools=list(arg))
            elif all(type(a) is int for a in arg):
                return Argument.create(as_ints=list(arg))
            elif all(type(a) is float for a in arg):
                return Argument.create(as_floats=list(arg))
            elif all(type(a) is str for a in arg):
                return Argument.create(as_strings=list(arg))
            elif all(self.is_inductor_sym_int_arg(a) for a in arg):
                # This is a special branch for handling SymInt args in inductor's
                # ExternalFallbackNode.
                # For regular FX graph, SymInt arg should be a fx.Node
                values = []
                for a in arg:
                    if isinstance(a, torch.SymInt):
                        values.append(SymIntArgument.create(as_name=str(a)))
                    elif type(a) is int:
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(isinstance(a, torch.SymFloat) for a in arg):
                return Argument.create(
                    as_sym_floats=[SymFloatArgument.create(as_name=str(a)) for a in arg]
                )
            elif all(self.is_sym_int_arg(a) for a in arg):
                # list of sym_ints
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymIntArgument.create(as_name=a.name))
                    elif type(a) is int:
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(self.is_sym_float_arg(a) for a in arg):
                # list of sym_float
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymFloatArgument.create(as_name=a.name))
                    elif isinstance(a, float):
                        values.append(SymFloatArgument.create(as_float=a))
                return Argument.create(as_sym_floats=values)
            elif all(self.is_sym_bool_arg(a) for a in arg):
                # list of sym_bools
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymBoolArgument.create(as_name=a.name))
                    elif isinstance(a, bool):
                        values.append(SymBoolArgument.create(as_bool=a))
                return Argument.create(as_sym_bools=values)
            elif all(isinstance(a, torch.fx.Node) for a in arg):
                # list of tensors
                arguments = []
                for a in arg:
                    if a.op == "get_attr":
                        raise SerializeError(
                            "getattr nodes containing tensors should not appear in the graph"
                        )
                    arguments.append(TensorArgument(name=a.name))
                return Argument.create(as_tensors=arguments)
            elif all(isinstance(a, (torch.fx.Node, type(None))) for a in arg):
                # list of optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=True)
                    elif isinstance(a, torch.fx.Node):
                        return OptionalTensorArgument.create(
                            as_tensor=TensorArgument(name=a.name)
                        )
                    else:
                        raise SerializeError(f"Unsupported list/tuple a
```



## High-Level Overview


This Python file contains 20 class(es) and 149 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SerializeError`, `SerializedArtifact`, `_SerializedProgram`, `LazyMap`, `GraphState`, `Final`, `GraphModuleSerializer`, `ExportedProgramSerializer`, `GraphModuleDeserializer`, `Result`, `ExportedProgramDeserializer`, `EnumEncoder`, `type`, `Edges`, `ExtensionHandler`

**Functions defined**: `_reverse_map`, `__init__`, `__setitem__`, `__getitem__`, `__repr__`, `deserialize_device`, `deserialize_size`, `deserialize_stride`, `deserialize_scalar_type`, `deserialize_storage_offset`, `_print_sympy`, `serialize_sym_int`, `serialize_sym_float`, `serialize_sym_bool`, `serialize_tensor_meta`, `_reduce_fake_tensor`, `_reconstruct_fake_tensor`, `serialize_torch_artifact`, `deserialize_torch_artifact`, `_sympy_int_to_int`

**Key imports**: base64, copy, copyreg, dataclasses, heapq, inspect, io, json, keyword, logging


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/serde`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `base64`
- `copy`
- `copyreg`
- `dataclasses`
- `heapq`
- `inspect`
- `io`
- `json`
- `keyword`
- `logging`
- `math`
- `operator`
- `re`
- `traceback`
- `typing`
- `collections`: namedtuple, OrderedDict
- `collections.abc`: Callable, Iterable, Iterator, Sequence
- `contextlib`: contextmanager
- `enum`: Enum
- `sympy`
- `torch`
- `torch.export.exported_program as ep`
- `torch._export.non_strict_utils`: _enable_graph_inputs_of_type_nn_module
- `torch._export.verifier`: load_verifier
- `torch._subclasses.fake_tensor`: FakeTensor, FakeTensorMode
- `torch.fx._symbolic_trace`: _ConstantAttributeType
- `torch.fx.experimental`: symbolic_shapes
- `torch.utils`: _pytree as pytree


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_export/serde`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`schema.yaml_docs.md`](./schema.yaml_docs.md)
- [`export_schema.thrift_docs.md`](./export_schema.thrift_docs.md)
- [`schema.py_docs.md`](./schema.py_docs.md)
- [`dynamic_shapes.py_docs.md`](./dynamic_shapes.py_docs.md)
- [`schema_check.py_docs.md`](./schema_check.py_docs.md)
- [`union.py_docs.md`](./union.py_docs.md)


## Cross-References

- **File Documentation**: `serialize.py_docs.md`
- **Keyword Index**: `serialize.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

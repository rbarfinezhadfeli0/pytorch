# Documentation: `docs/torch/_inductor/ir.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/ir.py_docs.md`
- **Size**: 54,134 bytes (52.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/ir.py`

## File Metadata

- **Path**: `torch/_inductor/ir.py`
- **Size**: 349,299 bytes (341.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import os
import textwrap
import traceback
from collections.abc import Callable, Container, Generator, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from enum import Enum
from functools import partial
from typing import (
    Any,
    cast,
    ClassVar,
    Literal,
    Optional,
    overload,
    SupportsFloat,
    SupportsInt,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import assert_never, Never, override, ParamSpec, Self, TypeIs
from unittest.mock import patch

import sympy
from sympy import Expr, Integer, Symbol

import torch._export.serde.schema as export_schema
import torch._library.utils as library_utils
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._higher_order_ops.auto_functionalize import can_auto_functionalize
from torch._inductor import metrics
from torch._inductor.utils import get_free_symbols
from torch._prims_common import (
    compute_required_storage_length,
    is_boolean_dtype,
    is_float_dtype,
    make_channels_last_strides_for,
    StrideType,
)
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import (
    _remove_effect_token_unbacked_bindings,
    compute_unbacked_bindings,
    free_symbols,
    free_unbacked_symbols,
    IterateExprs,
    rebind_unbacked,
    resolve_unbacked_bindings,
    ShapeEnv,
    SymTypes,
)
from torch.fx.node import Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, Mod, ModularIndexing
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import (
    BackendFeature,
    CodegenSymbol,
    get_scheduling_for_device,
    index_prevent_reordering,
    Kernel,
)
from .dependencies import (
    Dep,
    extract_free_symbols,
    extract_input_node_reduction_ranges,
    extract_read_writes,
    var_builder,
)
from .loop_body import LoopBody
from .ops_handler import OpCounterCSE, OpCountResult, ReductionType, StoreMode
from .runtime.benchmarking import benchmarker
from .runtime.hints import DeviceProperties, ReductionHint
from .utils import (
    argsort,
    argsort_sym,
    cache_on_self,
    cache_on_self_and_args,
    ceildiv,
    convert_shape_to_inductor,
    convert_shape_to_symint,
    developer_warning,
    do_bench_using_profiling,
    dtype_from_size,
    get_dtype_size,
    get_kernel_metadata,
    GPU_ALIGN_BYTES,
    ir_dataclass,
    is_dynamic,
    is_gpu,
    sympy_dot,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
    tensor_is_aligned,
)
from .virtualized import ops, OpsValue, V


if TYPE_CHECKING:
    from torch._library.fake_class_registry import FakeScriptObject
    from torch.fx.experimental.symbolic_shapes import SympyBoolean
    from torch.fx.node import Argument

    from .codegen.cuda.cuda_template import CUDATemplate
    from .codegen.wrapper import PythonWrapperCodegen
    from .graph import GraphLowering
    from .utils import IndentedBuffer

else:
    CUDATemplate: TypeAlias = object


try:
    import triton

    triton_version = triton.__version__
    has_triton = True
except ImportError:
    triton_version = None
    has_triton = False


_P = ParamSpec("_P")
_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")

_IntLike: TypeAlias = Union[int, Expr]
_NumLike: TypeAlias = Union[int, float, Expr]

_OpOverloads: TypeAlias = Union[torch._ops.OpOverload, torch._ops.HigherOrderOperator]

log = logging.getLogger(__name__)
indent = functools.partial(textwrap.indent, prefix="  ")
aten = torch.ops.aten

autotune_warmup = int(os.getenv("TORCH_AUTOTUNE_WARMUP", 25))
autotune_rep = int(os.getenv("TORCH_AUTOTUNE_REP", 100))

""" [Note: Inductor IR]

Inductor's IR is produced by executing 'lowering' code (see lowering.py).  Each
lowering is registered to a particular aten operator, and expects inputs that
correspond to the aten schema.  However, in place of torch Tensor inputs, lowerings
expect Inductor TensorBox inputs.

TensorBox IR represents torch tensors.  Tensors are sometimes single objects owning
storage, and sometimes views of another Tensor's storage.  Mutating tensor operations
(such as add_()) affect the underlying storage and any associated views.  Other operations
(such as .t_()) update metadata about the current view but don't modify the underlying storage.

To model this in Inductor, the IR distinguishes between TensorBox, View, StorageBox and Buffer.

TensorBox is the top level IR construct that any lowering should produce and maps to a torch.Tensor
output from an operation.  But just as torch.Tensors take different forms, TensorBox IR can
reference View IR or directly reference StorageBox IRs.

Some Inductor lowerings produce new sets of 'Box'es, while others (such as .t() or other view ops)
may take an existing TensorBox and point it to a new underlying View IR.

Tensors that directly own storage are represented as a chain of:
TensorBox -> StorageBox -> Buffer
where Buffer is a simple (1D) allocation, and StorageBox introduces the concept of a Layout.

If you mutate the data of such a tensor, we swing the StorageBox pointer to point to a new buffer
(leaving the old buffer unmodified and functionalizing the operation).

Tensors backed by views add one more indirection to the IR.
TensorBox -> View -> StorageBox -> Buffer
In these cases, the underlying StorageBox/Buffer will be shared with the pre-view TensorBox.

Computation is represented by Operation nodes, with each operation producing 1
or more output Buffers. In the case of mutations, these will be new Buffers that have the
mutated buffer listed in its get_mutation_names().

It is also possible to have an InputBuffer for which there is no corresponding Operation,
e.g. it may be a graph input or compile time constant.

"""


_NodeOrNodes: TypeAlias = Union[
    int,
    "TensorBox",
    dict[str, "TensorBox"],
    "Symbol",
    "IRNode",
    Sequence[
        Optional[Union[int, dict[str, "TensorBox"], "TensorBox", "Symbol", "IRNode"]]
    ],
]


def _is_static(x: object) -> TypeIs[Union[int, Integer]]:
    return isinstance(x, (int, Integer))


@dataclasses.dataclass(frozen=True)
class GraphPartitionSignature:
    # symbol inputs that are necessary for codegen
    symbol_inputs: OrderedSet[sympy.Symbol]

    # mapping from partition input name to IRNode or Expr. Need the name str since
    # we cannot get name from Expr.
    input_nodes: dict[str, Union[IRNode, sympy.Expr, TorchBindObject]]
    output_nodes: list[IRNode]

    # mapping from partition input name to a boolean for whether deallocating it
    # in the partition function
    input_deallocation: dict[str, bool]
    skip_cudagraph: bool

    # name of constants read/written by the graph partition
    constant_names: list[str]


def validate_ir(node_or_nodes: Optional[_NodeOrNodes]) -> None:
    def _check_tensorbox(nodes: Optional[_NodeOrNodes]) -> None:
        # Could expand this to check deeper properties
        # (e.g. TensorBox points to View or StorageBox)
        if nodes is None:
            pass
        elif isinstance(nodes, (list, tuple)):
            for node in nodes:
                _check_tensorbox(node)
        elif isinstance(nodes, dict):
            for node in nodes.values():
                _check_tensorbox(node)
        else:
            assert isinstance(
                nodes,
                (
                    ExpandView,
                    DynamicScalar,
                    AssertScalar,
                    TensorBox,
                    sympy.logic.boolalg.Boolean,
                    Expr,
                    int,
                    EffectfulKernel,
                    ShapeAsConstantBuffer,
                ),
            ), (
                f"Found {type(nodes)}, which is not a supported top level IR node. See [Note: Inductor IR]"
            )

    # Be picky about the accepted data structure (don't use pytree here)
    _check_tensorbox(node_or_nodes)


def ops_wrapper(name: str) -> Callable[..., OpsValue]:
    assert isinstance(name, str), type(name)

    def fn(*args: object, **kwargs: object) -> OpsValue:
        return getattr(ops, name)(*args, **kwargs)

    return fn


def inverse_reorder(order: Sequence[int]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def same_reorder(order: Sequence[int]) -> Callable[[Sequence[_T]], Sequence[_T]]:
    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        assert len(index) == len(order)
        return [index[order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(
    reindex1: Callable[[Sequence[_U]], Sequence[_V]],
    reindex2: Callable[[Sequence[_T]], Sequence[_U]],
) -> Callable[[Sequence[_T]], Sequence[_V]]:
    def reindex(index: Sequence[_T]) -> Sequence[_V]:
        return reindex1(reindex2(index))

    return reindex


NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


def get_fill_order(
    seq: Sequence[Union[int, torch.SymInt, Expr]], shape_env: Optional[ShapeEnv] = None
) -> Sequence[int]:
    """
    Convert strides to fill order (argsort)
    """
    if shape_env is None or all(isinstance(s, (int, sympy.Integer)) for s in seq):
        sorted_idx: Sequence[int] = argsort(seq)
    else:
        # argsort_sym handles unbacked symints (with the help of the shape_env)
        sorted_idx = argsort_sym(shape_env, seq)
    return sorted_idx


def stride_order2fill_order(order: Sequence[Union[int, Integer]]) -> Sequence[int]:
    """
    Convert stride order to fill order
    For channel last format,

    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(
    seq: Sequence[Union[int, torch.SymInt, Expr]], shape_env: Optional[ShapeEnv] = None
) -> Sequence[int]:
    """
    Convert strides to stride order
    """
    sorted_idx: Sequence[int] = get_fill_order(seq, shape_env)
    out = [0 for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


@overload
def ir_node_to_tensor(x: None, guard_shape: bool = True) -> None: ...


@overload
def ir_node_to_tensor(x: IRNode, guard_shape: bool = True) -> torch.Tensor: ...


def ir_node_to_tensor(
    x: Optional[IRNode], guard_shape: bool = True
) -> Optional[torch.Tensor]:
    if x is None:
        return None

    shape_fn: Callable[[Union[int, Expr]], Union[int, Expr]]
    if not guard_shape:
        shape_fn = V.graph.sizevars.size_hint
    else:
        shape_fn = identity
    size = [shape_fn(s) for s in x.get_size()]
    stride: StrideType
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]
    else:
        stride = FlexibleLayout.contiguous_strides(size)
    dtype = x.get_dtype()
    device = x.get_device()
    size = convert_shape_to_symint(size)
    # pyrefly: ignore [bad-assignment]
    stride = convert_shape_to_symint(stride)
    with V.graph.sizevars.shape_env.suppress_guards():
        t = torch.empty_strided(
            size=size, stride=stride, dtype=dtype, device=device
        ).zero_()
    return t


def may_convert_to_optional(
    value: Optional[Sequence[_T]],
) -> Optional[Sequence[Optional[_T]]]:
    if isinstance(value, list) and not value:
        # [None] makes sure the cpp wrapper codegen will generate something like
        # {std::nullopt} instead of {}
        return [None]
    return value


def get_device_type(
    x: Union[IRNode, OutputSpec, torch.device, None, str],
) -> Optional[str]:
    if isinstance(x, str) or x is None:
        return x
    elif isinstance(x, torch.device):
        return x.type
    elif isinstance(x, (IRNode, OutputSpec)):
        return get_device_type(x.get_device())
    # pyrefly: ignore [bad-argument-type]
    assert_never(f"get_device_type({x}: {type(x).__name__})")


def is_triton(x: Union[IRNode, torch.device, None, str]) -> bool:
    device = get_device_type(x)
    # Special case cpu and cuda as using the method below
    # to determine if the scheduler is a triton scheduler subclass
    # requires instantiating a scheduler for them
    if device in ["cpu", "cuda"]:
        if getattr(config, f"{device}_backend") == "triton":
            return True
        return False
    if (
        device is None
        or (device_scheduling := get_scheduling_for_device(device)) is None
    ):
        return False
    from .codegen.triton import TritonScheduling

    assert isinstance(device_scheduling, type), type(device_scheduling)
    return issubclass(device_scheduling, TritonScheduling)


def is_cpu(x: Union[IRNode, torch.device, None, str]) -> bool:
    return get_device_type(x) == "cpu"


def is_aligned_realized_tensor(x: Union[Buffer, TensorBox], alignment: int) -> bool:
    if (
        not isinstance(x, IRNode)
        or x.maybe_get_stride() is None
        or free_unbacked_symbols(x.get_stride())
        or free_unbacked_symbols(x.get_size())
    ):
        return False

    aligned_strides = sympy.And(
        *(sympy.Eq(Mod(s, alignment), 0) for s in x.get_stride()[:-1])
    )
    aligned_last_dim = sympy.Or(
        sympy.Eq(x.get_stride()[-1], 1), sympy.Le(x.get_size()[-1], 1)
    )
    is_aligned = sympy.And(aligned_strides, aligned_last_dim)

    # Make sure to guard to recompile when necessary.
    return V.graph.sizevars.guard_or_false(is_aligned)


def significant_strides_equal(
    strides1: Sequence[_IntLike],
    strides2: Sequence[_IntLike],
    shape: Sequence[_IntLike],
) -> bool:
    """
    Returns true if the strides are equal, ignoring dimensions of size 1 .
    """
    assert len(shape) == len(strides1) and len(strides1) == len(strides2)
    for dim, s1, s2 in zip(shape, strides1, strides2):
        if V.graph.sizevars.statically_known_leq(dim, 1):
            continue

        if not V.graph.sizevars.statically_known_equals(
            s1, s2
        ) and V.graph.sizevars.symbolic_hint(s1) != V.graph.sizevars.symbolic_hint(s2):
            return False

    return True


def try_match_insignificant_strides(
    tensor: IRNode,
    strides: Sequence[Union[int, torch.SymInt]],
) -> IRNode:
    """
    Tries to match the strides of the tensor to those in the meta_strides. Strides of insignificant
    dimensions - size 0 or 1 - will be updated.

    If there are real stride differences (NHWC vs NCHW), or the tensor is not realized, then the input will be returned
    """
    if not is_storage_and_layout(tensor):
        return tensor

    if all(
        V.graph.sizevars.statically_known_equals(s1, s2)
        for s1, s2 in zip(strides, tensor.get_stride())
    ):
        return tensor

    if not significant_strides_equal(strides, tensor.get_stride(), tensor.get_size()):
        return tensor

    storage, old_layout = as_storage_and_layout(tensor)
    new_stride = [*old_layout.stride]
    for i, s in enumerate(tensor.get_size()):
        if V.graph.sizevars.statically_known_leq(s, 1):
            new_stride[i] = strides[i]

    new_layout = FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
        old_layout.is_pinned,
    )
    return TensorBox(ReinterpretView(data=storage, layout=new_layout))


def gm_original_output_strides(gm: torch.fx.GraphModule) -> None:
    output_node = gm.graph.find_nodes(op="output")[0]
    output_node.meta["user_visible_output_idxs"] = [
        idx for idx, _ in enumerate(output_node.args)
    ]
    from torch._inductor.compile_fx import record_original_output_strides

    record_original_output_strides(gm)


def get_symbolic_inputs(inputs: Sequence[IRNode]) -> list[Expr]:
    sym_vars: OrderedSet[Expr] = OrderedSet()
    for inp in inputs:
        sym_vars |= get_free_symbols(inp.get_size(), unbacked_only=False)
        sym_vars |= get_free_symbols(inp.get_stride(), unbacked_only=False)

    return list(sym_vars)


class IRNode:
    """Base class for all intermediate representation (IR) nodes in TorchInductor.

    Note:
        This is an abstract base class. Most methods raise NotImplementedError
        and must be overridden by concrete subclasses.
    """

    _current_origins: ClassVar[OrderedSet[Any]] = OrderedSet()

    # NB: These are kinda weird,
    origins: OrderedSet[Any] = dataclasses.field(init=False)
    # traces back to where the IRNode is created in Inductor
    traceback: Optional[list[str]] = dataclasses.field(init=False)
    origin_node: Optional[torch.fx.Node] = dataclasses.field(init=False)

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: OrderedSet[Node]) -> Generator[None, None, None]:
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    @staticmethod
    def is_realized_node(node: IRNode) -> bool:
        return isinstance(
            node,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        )

    def _post_init_setattr(self, attr: str, value: Any) -> None:
        # Intended for use in __post_init__ for enforcing an invariant on a dataclass
        # If you must, can also be used for setting provenance info
        # We would like to try and minimize these usages though
        object.__setattr__(self, attr, value)

    def __post_init__(self) -> None:
        origins = OrderedSet(self._current_origins)
        self._post_init_setattr("origins", origins)
        self._post_init_setattr(
            "traceback", traceback.format_stack() if config.debug_ir_traceback else None
        )
        self._post_init_setattr("origin_node", None)

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_traceback(self) -> Optional[list[str]]:
        return self.traceback

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return self.origin_node

    def get_defining_op(self) -> Optional[Operation]:
        return None

    def get_stack_traces(self) -> OrderedSet[str]:
        # Return stack traces to user model code
        # A single IRNode could correspond to multiple lines of code
        stack_traces: OrderedSet[str] = OrderedSet()
        origins = self.origins
        if isinstance(self, ExternKernel):
            origin_node = self.get_origin_node()
            if self.origin_node:
                origins = OrderedSet([origin_node])
        for node in origins:
            if hasattr(node, "stack_trace") and node.stack_trace:
                # nodes in the backward graph don't have mapping to pre_grad_graph
                stack_traces.add(node.stack_trace)
            else:
                pre_grad_nodes = (
                    torch._inductor.debug._inductor_post_to_pre_grad_nodes.get(
                        "postToPre",
                        {},
                        # pyrefly: ignore [missing-attribute]
                    ).get(node.name, [])
                )
                if not isinstance(pre_grad_nodes, list):
                    continue
                for node_name in pre_grad_nodes:
                    stack_trace = (
                        torch._inductor.debug._inductor_pre_grad_node_stack_trace.get(
                            node_name, None
                        )
                    )
                    if stack_trace:
                        stack_traces.add(stack_trace)
        return stack_traces

    def common_repr(self, shorten: bool = True) -> Sequence[str]:
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # this can get *very* long
            origins = f"{origins[:61]}..."
        if not self.get_stack_traces():
            return [origins]

        stack_trace_str = []
        for stack_trace in self.get_stack_traces():
            stack_trace_str.append("stack_traces = {")
            stack_trace_str += stack_trace.split("\n")
            stack_trace_str.append("}")
        return [origins] + stack_trace_str

    def str_helper(
        self, lines: Sequence[object], shorten: bool = True, multiline: bool = True
    ) -> str:
        lines = list(lines) + list(self.common_repr(shorten))
        lines = list(map(str, lines))
        if multiline:
            # pyrefly: ignore [no-matching-overload]
            new_lines = indent(",\n".join(lines))
            return f"{type(self).__name__}(\n{new_lines}\n)"
        else:
            return f"{type(self).__name__}({lines})"

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def maybe_get_dtype(self) -> Optional[torch.dtype]:
        try:
            return self.get_dtype()
        except NotImplementedError:
            return None

    def get_layout(self) -> Layout:
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    def maybe_get_layout(self) -> Optional[Layout]:
        try:
            return self.get_layout()
        except NotImplementedError:
            return None

    def get_output_spec(self) -> OutputSpec:
        return self.get_layout()

    def maybe_get_output_spec(self) -> Optional[OutputSpec]:
        try:
            return self.get_output_spec()
        except NotImplementedError:
            return None

    def has_tensor_output(self) -> bool:
        """True for single tensor output (excludes MultiOutput)"""
        return isinstance(self.maybe_get_output_spec(), Layout)

    def get_size(self) -> Sequence[Expr]:
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    def maybe_get_size(self) -> Optional[Sequence[_IntLike]]:
        try:
            return self.get_size()
        except NotImplementedError:
            return None

    @property
    def shape(self) -> Union[_IntLike, sympy.Rel, Sequence[_IntLike]]:
        return self.get_size()

    def get_numel(self) -> Expr:
        return sympy_product(self.get_size())

    def is_zero_elements(self) -> bool:
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def realize(self) -> Optional[str]:
        """
        If the IRNode refers to data which has not been materialized (e.g.,
        it is a Pointwise/Reduction that could potentially have more
        compute fused into it), realize the IRNode into physical memory,
        ending the possibility of fusing into it, but allowing, e.g., multiple
        users to access the data without having to recompute.

        Check StorageBox.realize for a particularly notable implementation.

        TODO(ezyang): I think, in principle, every IRNode should have an
        implementation of this, and most of the time no-op is OK, but you
        really do have to audit each IRNode for this, so for now, raise
        an error if it's not implemented.  Note that some code in graph.py
        will catch this thrown error and suppress it with a warning.
        """
        raise NotImplementedError(f"realize NYI on {type(self)}")

    def codegen_reference(self, writer: Optional[IndentedBuffer] = None) -> str:
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    def get_device(self) -> Optional[torch.device]:
        return None

    def get_device_or_error(self) -> torch.device:
        device = self.get_device()
        assert device is not None
        return device

    def has_exceeded_max_reads(self) -> bool:
        return False

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        raise NotImplementedError(type(self).__name__)

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        raise NotImplementedError(type(self).__name__)

    def get_stride(self) -> Sequence[_IntLike]:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_stride(self) -> Optional[Sequence[_IntLike]]:
        try:
            return self.get_stride()
        except NotImplementedError:
            return None

    def get_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_name(self) -> Optional[str]:
        try:
            return self.get_name()
        except NotImplementedError:
            return None

    def is_input_buffer(self) -> bool:
        try:
            return self.get_name() in V.graph.graph_inputs
        except NotImplementedError:
            return False

    def has_large_inner_fn(self, threshold: Optional[int] = None) -> bool:
        return False

    def mark_reuse(self, users: int) -> None:
        pass

    def realize_hint(self) -> None:
        pass

    def unwrap_view(self) -> IRNode:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout(self) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_same_order(self, stride: Sequence[_IntLike]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def get_read_writes(self) -> dependencies.ReadWrites:
        raise NotImplementedError(type(self).__name__)

    def get_reads(self) -> OrderedSet[Dep]:
        return self.get_read_writes().reads

    def num_reads(self) -> int:
        return len(self.get_reads())

    def get_storage_numel(self) -> _IntLike:
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_type(self) -> Optional[str]:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_size(self) -> Sequence[Expr]:
        raise NotImplementedError(type(self).__name__)

    def is_extern(self) -> bool:
        return False

    def is_no_op(self) -> bool:
        return False

    def constant_to_device(self, device: torch.device) -> IRNode:
        raise NotImplementedError(type(self).__name__)

    def get_mutation_names(self) -> Sequence[str]:
        raise NotImplementedError(type(self).__name__)

    def get_operation_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        raise NotImplementedError(type(self).__name__)

    if TYPE_CHECKING:

        @property
        def dtype(self) -> torch.dtype: ...


@ir_dataclass(frozen=False)
class Operation:
    def __post_init__(self) -> None:
        self.operation_name: Optional[str] = None

    def get_device(self) -> Optional[torch.device]:
        raise NotImplementedError

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        assert hasattr(self, "origin_node")
        return self.origin_node

    def get_origins(self) -> OrderedSet[Any]:
        assert hasattr(self, "origins")
        return self.origins

    def get_operation_name(self) -> str:
        assert self.operation_name is not None
        return self.operation_name

    def is_extern(self) -> bool:
        return False

    def is_no_op(self) -> bool:
        return False

    def get_read_writes(self) -> dependencies.ReadWrites:
        raise NotImplementedError

    def is_user_of(self, name: str) -> bool:
        return name in self.get_read_names()

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_reads(self) -> OrderedSet[Dep]:
        return self.get_read_writes().reads

    def get_outputs(self) -> list[Buffer]:
        raise NotImplementedError

    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        """
        When unbacked_only=True:
        Returns the unbacked symbols which are required to be in scope in
        order to successfully perform codegen for this buffer.  For example,
        a buffer that corresponds to an extern kernel call that takes i0 as
        an argument would return {i0} here.  This is used to generate necessary
        dependencies that ensure we actually bind i0 in codegen before you
        try to use it.

        Note that this is NOT transitive; in particular, if this buffer takes
        in as input another buffer with dynamic shape (e.g., (i0,)), we will
        not report it here, because you will already have a dependency
        on that buffer, which will eventually have a dependency on i0 if
        necessary.

        When unbacked_only=False:
        Similar to `unbacked_only=True` but including all free symbols
        instead of only free unbacked symbols.
        """
        return OrderedSet()

    def get_workspace_size(self) -> int:
        """
        Gets extra global memory size needed by this buffer.
        Some algorithms (e.g. group gemm) may require extra global memory in the generated code.
        """
        return 0


@ir_dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: Sequence[_IntLike]

    @cache_on_self_and_args("Loops")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.ranges),
            self.inner_fn_free_symbols(unbacked_only),
        )

    def _to_str(self, names: Sequence[str]) -> str:
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
            + [f"origin_node={self.origin_node!r}"]
        )

    def __post_init__(self) -> None:
        super().__post_init__()

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_device(self) -> Optional[torch.device]:
        return self.device

    def get_origin_node(self) -> Optional[torch.fx.Node]:
        return self.origin_node

    def get_size(self) -> Sequence[Expr]:
        return self.ranges

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    @classmethod
    def create(
        cls, *args: Any, **kwargs: Any
    ) -> Union[TensorBox, ShapeAsConstantBuffer]:
        origin_node = kwargs.pop("origin_node", None)
        tb = kwargs.pop("traceback", None)
        r = cls(*args, **kwargs)
        # Need to explicitly set origin_node here to propagate it down.
        # todo(chilli): I think it would be better for IRNode to directly set
        # origin_node
        r._post_init_setattr("origin_node", origin_node)
        r._post_init_setattr("traceback", tb or r.traceback)
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges: Sequence[_IntLike], prefix: SymT = SymT.INDEX) -> Sequence[Expr]:
        return [
            sympy.S.Zero if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self) -> OpCountResult:
        opcounter = OpCounterCSE(V.MockHandler())
        with (
            V.set_ops_handler(opcounter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            self.inner_fn(*self.inner_fn_args())
            return opcounter.getvalue()

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        return (self._index(self.ranges),)

    @cache_on_self
    def inner_fn_str(self) -> str:
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def has_large_inner_fn(self, threshold: Optional[int] = None) -> bool:
        if threshold is None:
            threshold = 0
        threshold = max(threshold, config.realize_opcount_threshold)
        return self.inner_fn_opcount().num_ops > threshold

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        return extract_free_symbols(self.inner_fn, index, unbacked_only=unbacked_only)

    def get_reads(self) -> OrderedSet[Dep]:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(self.inner_fn_opcount().read_buffers)

    def num_reads(self) -> int:
        return len(self.inner_fn_opcount().read_buffers)

    def get_reduction_size(self) -> Sequence[Expr]:
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    def get_reduction_type(self) -> Optional[str]:
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )


def nop_loader_fn(idx: Union[Expr, Sequence[Expr]], *, dtype: torch.dtype) -> OpsValue:
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        return ops.constant(0, dtype)


@ir_dataclass
class Pointwise(Loops):
    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return []

    def get_reduction_type(self) -> Optional[str]:
        return None

    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        return ops.store(output_name or "unnamed", indexer(vars), loader(vars))

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
        )


@ir_dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[Sequence[Expr]], Expr]
    scatter_mode: StoreMode = None

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            output_indexer=self.output_indexer,
            scatter_mode=self.scatter_mode,
        )

    def store_output(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> Any:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


REDUCTION_COMBINE_FN: dict[str, Callable[..., OpsValue]] = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "dot": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(
    reduction_type: str, dtype: torch.dtype, arg_break_ties_left: bool = True
) -> Callable[..., object]:
    if reduction_type in REDUCTION_COMBINE_FN:
        return REDUCTION_COMBINE_FN[reduction_type]

    elif reduction_type in ("argmax", "argmin"):

        def argmax_combine_fn(
            a: tuple[object, object], b: tuple[object, object]
        ) -> tuple[OpsValue, OpsValue]:
            a_value, a_index = a
            b_value, b_index = b

            if reduction_type == "argmin":
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)

            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))

            tie = (
                ops.lt(a_index, b_index)
                if arg_break_ties_left
                else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))
            return (
                ops.where(mask, a_value, b_value),
                ops.where(mask, a_index, b_index),
            )

        return argmax_combine_fn

    elif reduction_type == "welford_combine":

        def welford_combine_fn(
            a: tuple[OpsValue, OpsValue, OpsValue],
            b: tuple[OpsValue, OpsValue, OpsValue],
        ) -> tuple[OpsValue, OpsValue, OpsValue]:
            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b

            delta = b_mean - a_mean
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (
                a_mean + delta * w2_over_w,
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,
                new_weight,
            )

        return welford_combine_fn

    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")


@ir_dataclass
class Reduction(Loops):
    reduction_ranges: Sequence[_IntLike]
    reduction_type: ReductionType
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self) -> str:
        return self._to_str(("ranges", "reduction_ranges", "reduction_type"))

    __repr__ = __str__

    @cache_on_self_and_args("Reduction")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return super().get_free_symbol_uses(unbacked_only) | OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.reduction_ranges)
        )

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.reduction_ranges

    def get_reduction_type(self) -> Optional[str]:
        return self.reduction_type

    def store_reduction(
        self,
        output_name: Optional[str],
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> None:
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        ops.store_reduction(output_name or "unnamed", indexer(vars), value)

    def index_length(self) -> int:
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return (index, rindex)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return extract_free_symbols(
            self.inner_fn, index, rindex, unbacked_only=unbacked_only
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            reduction_ranges=self.reduction_ranges,
            reduction_type=self.reduction_type,
            src_dtype=self.src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

    @staticmethod
    def num_splits(
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[_P, OpsValue],
        ranges: Sequence[_IntLike],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: Union[ReductionType, Literal["scan"]],
        reduction_numel: Expr,
        input_node: Optional[IRNode] = None,
    ) -> tuple[ReductionHint, _IntLike]:
        reduction_numel_hint = V.graph.sizevars.symbolic_hint(reduction_numel)
        numel_hint = V.graph.sizevars.symbolic_hint(sympy_product(ranges))

        should_split = reduction_type == "scan" or (
            not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
            and reduction_type
            not in (
                "argmax",
                "argmin",
            )
            and config.split_reductions
        )

        if not (_is_static(reduction_numel_hint) and _is_static(numel_hint)):
            # We don't support unbacked symints
            return ReductionHint.DEFAULT, 1

        if reduction_type == "dot":
            # Don't split when doing native matmul
            return ReductionHint.DEFAULT, 1

        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        if should_split:
            inner_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=True
            )
            outer_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=False
            )
        else:

            def inner_reduction_splits(
                reduction_numel_hint: int,
                numel_hint: int,
            ) -> int:
                return 1

            outer_reduction_splits = inner_reduction_splits

        # easy cases
        if numel_hint == 1:
            split = inner_reduction_splits(reduction_numel_hint, numel_hint)
            if split == 1:
                # No need to split.
                return ReductionHint.INNER, split
            if input_node is not None and isinstance(input_node, TensorBox):
                with patch.object(FlexibleLayout, "allow_indexing", True):
                    (
                        new_ranges,
                        new_reduction_ranges,
                    ) = extract_input_node_reduction_ranges(input_node)
                if new_ranges is not None and new_reduction_ranges is not None:
                    extracted_numel_hint = V.graph.sizevars.symbolic_hint(
                        sympy_product(new_ranges + new_reduction_ranges)
                    )
                    if reduction_numel_hint == extracted_numel_hint:
                        log.debug(
                            "Use previous IRNode's range and reduction_ranges instead of split. "
                            "current ranges: %s, current reduction ranges: %s, current split: %d, "
                            "new ranges: %s, new reduction ranges: %s",
                            ranges,
                            reduction_ranges,
                            split,
                            new_ranges,
                            new_reduction_ranges,
                        )
                        # If the input_node or its dependent nodes are also Reduction nodes,
                        # use reduction_sizes of this node or its dependent nodes directly.
                        return ReductionHint.INNER, -1
            return ReductionHint.INNER, split
        if (
            reduction_numel_hint <= min_elements_per_thread
            or numel_hint >= num_sm * 2 * 32
        ):
            return ReductionHint.DEFAULT, 1

        r = Reduction(
            device=device,
            dtype=dst_dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type if reduction_type != "scan" else "sum",
            src_dtype=src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

        def get_read_indices(r: Reduction) -> tuple[Sequence[Expr], bool]:
            device = r.get_device()
            assert device is not None
            cb = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=device,
                    dtype=r.get_dtype(),
                    size=r.get_size(),
                ),
                data=r,
            )
            read_writes = cb.get_read_writes()
            # try finding the full size producer
            # TODO this will fail for something like ((1, N) * (N, 1)).sum()
            # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
            assert read_writes.range_vars is not None
            range_vars = [
                r
                for r in read_writes.range_vars
                if isinstance(r, Expr) and not isinstance(r, sympy.Number)
            ]
            indices = []
            changed = False
            for md in sorted(read_writes.reads, key=lambda x: x.name):
                if all(r in md.index.free_symbols for r in range_vars):
                    indices.append(md.index)
                    if md.name in V.graph.name_to_buffer:
                        buf = V.graph.name_to_buffer[md.name]
                        original_stride = getattr(buf.layout, "stride", None)
                        buf.decide_layout()
                        if getattr(buf.layout, "stride", None) != original_stride:
                            changed = True
            return indices, changed

        indices, changed = get_read_indices(r)
        if changed:
            indices, _ = get_read_indices(r)

        if len(indices) == 0:
            # TODO determine splits when all inputs are broadcast
            return ReductionHint.DEFAULT, 1

        (_, reduction_vars), ranges1 = dependencies.index_vars_squeeze(
            r.get_size(), r.get_reduction_size()
        )
        num_outer = 0
        num_inner = 0
        for i in indices:
            j = V.graph.sizevars.simplify_with_ranges(i, ranges1)
            strides = V.graph.sizevars.stride_hints(
                j, reduction_vars, list(ranges1.keys())
            )
            outer = all(s > 1 for s in strides)
            if outer:
                num_outer += 1
            else:
                num_inner += 1
        if num_inner > num_outer:
            return ReductionHint.INNER, inner_reduction_splits(
                reduction_numel_hint, numel_hint
            )
        else:
            return ReductionHint.OUTER, outer_reduction_splits(
                reduction_numel_hint, numel_hint
            )

    @staticmethod
    def _unroll_reduction_fn(
        inner_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: str,
        src_dtype: torch.dtype,
    ) -> Callable[[Sequence[_IntLike]], OpsValue]:
        """Convert inner_fn from a reduction to an pointwise"""
        reduction_ranges = V.graph.sizevars.guard_int_seq(reduction_ranges)

        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def fn(index: Sequence[_IntLike]) -> Any:
            return funct
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
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `ir.py_docs.md_docs.md`
- **Keyword Index**: `ir.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

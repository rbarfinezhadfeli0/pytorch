# Documentation: `docs/torch/_inductor/graph.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/graph.py_docs.md`
- **Size**: 54,027 bytes (52.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/graph.py`

## File Metadata

- **Path**: `torch/_inductor/graph.py`
- **Size**: 107,123 bytes (104.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import contextlib
import functools
import itertools
import logging
import operator
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, NoReturn, Optional, TYPE_CHECKING, Union

import sympy
from sympy import Expr

import torch
import torch._logging
import torch.fx
from torch import device, Tensor
from torch._decomp import get_decompositions
from torch._dynamo.utils import defake, dynamo_timed
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.utils import get_layout_constraint_tag
from torch._logging import LazyString, trace_structured
from torch._prims_common import (
    compute_required_storage_length,
    make_channels_last_strides_for,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch._utils_internal import full_aoti_runtime_assert
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    _get_placeholder_expr,
    free_unbacked_symbols,
    has_free_symbols,
    resolve_unbacked_bindings,
    RuntimeAssert,
    ShapeEnv,
    SympyBoolean,
    SymTypes,
)
from torch.fx.node import Node
from torch.fx.passes.reinplace import _is_view_op
from torch.utils._mode_utils import no_dispatch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.numbers import int_oo

from . import config, ir, metrics
from .codegen.common import (
    BackendFeature,
    DeviceOpOverrides,
    FileBackedGraphModule,
    get_backend_features,
    get_device_op_overrides,
    get_wrapper_codegen_for_device,
    init_backend_registration,
    WorkspaceArg,
)
from .exc import (
    CppWrapperCodegenError,
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)
from .fx_utils import count_flops_fx
from .ir import (
    assign_origin_node,
    Constant,
    DonatedBuffer,
    FixedLayout,
    get_device_type,
    GraphPartitionSignature,
    InputBuffer,
    Pointwise,
    Reduction,
    ShapeAsConstantBuffer,
    StorageBox,
    TensorBox,
    TorchBindObject,
)
from .lowering import (
    constrain_to_fake_tensors,
    constrain_to_fx_strides,
    FALLBACK_ALLOW_LIST,
    fallback_handler,
    fallback_node_due_to_unsupported_type,
    lowerings,
    make_fallback,
    maybe_layout_constraints,
    needs_realized_inputs,
    require_contiguous,
    tag_to_layout_constraint,
    unsupported_output_tensor,
)
from .runtime import autotune_cache
from .runtime.autotune_cache import AutotuneCacheBundler
from .sizevars import SizeVarAllocator
from .utils import (
    convert_shape_to_inductor,
    gather_origins,
    get_cloned_parameter_buffer_name,
    get_donated_idxs,
    get_sympy_Expr_dtype,
    GraphPartitionMap,
    is_same_tensor,
    maybe_get_suppress_shape_guards_ctx,
    normalize_name,
    should_assume_input_aligned,
    should_fallback_by_default,
    SUPPORTED_MKLDNN_DEVICES,
    ValueWithLineMap,
)
from .virtualized import NullHandler, V


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from types import ModuleType

    from torch._higher_order_ops.effects import _EffectType
    from torch.fx import GraphModule
    from torch.fx.graph import Graph

    from .codegen.wrapper import PythonWrapperCodegen
    from .dependencies import Dep
    from .scheduler import BaseSchedulerNode

    CompiledModule = Union[ModuleType, FileBackedGraphModule]

from torch._inductor.codecache import output_code_log


log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")

aten = torch.ops.aten

_post_grad_graph_counter = itertools.count()

if config.is_fbcode():
    from torch._inductor.fb.utils import log_module_code
else:

    def log_module_code(*args: Any, **kwargs: Any) -> None:
        pass


def may_get_constant_buffer_dtype(constant_buffer: sympy.Expr) -> Optional[torch.dtype]:
    assert isinstance(
        constant_buffer, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
    ), (
        "get_constant_buffer_dtype only supports input of sympy.Symbol, sympy.Expr or sympy.core.numbers.Integer"
    )
    if isinstance(constant_buffer, sympy.core.numbers.Integer):
        return torch.int64

    if isinstance(constant_buffer, sympy.Expr):
        return get_sympy_Expr_dtype(constant_buffer)

    if constant_buffer.is_integer:
        return torch.int64
    elif constant_buffer.is_float:
        return torch.float32
    else:
        return None


def is_magic_method(op: Any) -> bool:
    magic_ops = OrderedSet(method_to_operator(m) for m in magic_methods)
    return op in magic_ops


def getattr_recursive(
    obj: GraphModule, target: str
) -> Union[Tensor, torch._C.ScriptObject, GraphModule]:
    target_atoms = target.split(".")
    attr_itr = obj
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def get_user_visible_output_strides(g: Graph) -> dict[Node, tuple[int, ...]]:
    ret: dict[Node, tuple[int, ...]] = {}
    output_node = g.find_nodes(op="output")[0]

    if "user_visible_output_idxs" not in output_node.meta:
        return ret

    if not isinstance(output_node.args[0], torch.fx.Node):
        output_node_args = output_node.args[0]
    else:
        output_node_args = output_node.args

    for idx, node in enumerate(output_node_args):
        if idx in output_node.meta["user_visible_output_idxs"]:
            ret[node] = output_node.meta["original_output_strides"][idx]
    return ret


def extend_user_visible_output_strides(
    user_visible_outputs: dict[Node, tuple[int, ...]],
) -> dict[Node, object]:
    """
    Extend user_visible_output_strides to include view ops that lead to user-visible outputs.
    """
    result: dict[Node, object] = {**user_visible_outputs}
    queue = [*result.keys()]
    visited = OrderedSet([*queue])
    while queue:
        current = queue.pop()
        if (
            _is_view_op(current.target)
            and current.args
            and isinstance(current.args[0], torch.fx.Node)
        ):
            base = current.args[0]
            if base not in visited:
                result.setdefault(base, None)
                visited.add(base)
                queue.append(base)
    return result


def mark_nodes_dislike_padding(
    g: Graph, user_visible_output_strides: dict[Node, tuple[int, ...]]
) -> None:
    """
    Nodes like convolution/convolution_backward want its input to be dense.
    If we pad their inputs, we result in extra calls to copy kernels!  On the other hand, padding usually helps reduction.

    The pass finds nodes that dislike padding. These are nodes that can be reached
    from a convolution/convolution_backward in the backward direction without
    going thru a reduction.
    """
    if not config.comprehensive_padding:
        return

    extended_user_visible_nodes = extend_user_visible_output_strides(
        user_visible_output_strides
    )
    ops_dislike_padding = OrderedSet(
        [
            aten.convolution,
            aten.convolution_backward,
            aten._scaled_mm,
        ]
    )
    # what's a better way to collect the reduction ops?
    ops_like_padding = OrderedSet(
        [
            aten.var_mean,
            aten.sum,
            aten.mean,
            aten.prod,
            aten.any,
            aten.amin,
            aten.amax,
            aten.min,
            aten.max,
            aten.argmin,
            aten.argmax,
            aten.scatter_reduce,
        ]
    )

    def _get_overload_packet(
        node: torch.fx.Node,
    ) -> Optional[torch._ops.OpOverloadPacket]:
        return (
            node.target._overloadpacket
            if node.op == "call_function"
            # hasattr on OpOverloadPacket is slow, do isinstance first
            and isinstance(node.target, torch._ops.OpOverload)
            and hasattr(node.target, "_overloadpacket")
            else None
        )

    for cur in reversed(g.nodes):
        if isinstance(
            cur.target,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
        ):
            cur.meta["dislike_padding"] = True
            continue

        if (
            isinstance(cur.target, torch._ops.OpOverload)
            and get_layout_constraint_tag(cur.target)
            == torch._C.Tag.needs_exact_strides
        ):
            cur.meta["dislike_padding"] = True
            continue

        op = _get_overload_packet(cur)
        if not op:
            continue
        if op in ops_dislike_padding:
            cur.meta["dislike_padding"] = True

        if cur.meta.get("dislike_padding", False):
            # propagate
            for prior in cur.all_input_nodes:
                prior_op = _get_overload_packet(prior)
                if not prior_op:
                    continue
                if prior_op not in ops_like_padding:
                    prior.meta["dislike_padding"] = True
        # We only want to mark output nodes. So, move it after the above prior nodes process.
        if not config.pad_outputs and cur in extended_user_visible_nodes:
            cur.meta["dislike_padding"] = True


class GraphLowering(torch.fx.Interpreter):
    graph_outputs: list[ir.IRNode]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[Sequence[object]] = None,
        shape_env: Optional[ShapeEnv] = None,
        graph_id: Optional[int] = None,
        cpp_wrapper: bool = False,
        aot_mode: bool = False,
        layout_opt: Optional[bool] = None,
        extern_node_serializer: Optional[
            Callable[[list[ir.ExternKernelNode]], Any]
        ] = None,
        is_inference: bool = False,
        is_backward: bool = False,
        is_const_graph: bool = False,
        const_output_index: Optional[dict[str, int]] = None,
        const_wrapper_code: Optional[str] = None,
        const_kernel_code: Optional[str] = None,
        const_module: Optional[GraphLowering] = None,
        name: Optional[str] = None,
        inputs_to_check: Optional[Sequence[int]] = None,
        fx_wrapper: bool = False,
    ) -> None:
        super().__init__(gm)
        self.example_inputs = example_inputs
        self.layout_opt = (
            layout_opt
            if layout_opt is not None
            else self.decide_layout_opt(gm, is_inference=is_inference)
        )
        self.num_channels_last_conv = 0
        self.is_inference = is_inference
        self.is_backward = is_backward
        self.is_const_graph = is_const_graph
        self.const_wrapper_code = const_wrapper_code
        self.const_kernel_code = const_kernel_code
        self.const_module = const_module
        self.inputs_to_check = inputs_to_check

        self.extra_traceback = False  # we do our own error wrapping
        if shape_env is None:
            shape_env = ShapeEnv()
            self.reuse_shape_env = False
        else:
            self.reuse_shape_env = True
        self._shape_env = shape_env
        # We're going to mutate ras_by_symbol as we finish generating them
        self.ras_by_symbol: dict[Optional[sympy.Symbol], list[RuntimeAssert]] = (
            shape_env.deferred_runtime_asserts.copy()
        )
        self.bound_unbacked_symbols = OrderedSet[sympy.Symbol]()

        self.sizevars = SizeVarAllocator(shape_env)
        self.graph_input_names: list[str] = []
        self.graph_inputs: dict[str, Union[TensorBox, TorchBindObject, sympy.Expr]] = {}
        self.graph_inputs_original: dict[str, InputBuffer] = {}
        self.partition_maps: Optional[list[GraphPartitionMap]] = None
        self.zero_dim_cpu_tensor_list: OrderedSet[str] = OrderedSet()
        self.device_types: OrderedSet[str] = (
            const_module.device_types if const_module else OrderedSet()
        )
        self.device_idxs: OrderedSet[int] = (
            const_module.device_idxs if const_module else OrderedSet()
        )
        self.device_type = "cpu"
        self.additional_buffer_deps: dict[str, OrderedSet[str]] = defaultdict(
            OrderedSet
        )

        # Inplace padding may require Inductor to allocate slightly larger
        # tensor for padding.
        self.buffer_to_padded_size: dict[str, list[int]] = {}

        self.buffers: list[ir.Buffer] = []
        self.operations: list[ir.Operation] = []
        self.const_output_index: dict[str, int] = (
            const_output_index if const_output_index else {}
        )
        self.folded_constants: OrderedSet[str] = (
            OrderedSet(const_output_index.keys())
            if const_output_index
            else OrderedSet()
        )
        self.constants: dict[str, torch.Tensor] = (
            const_module.constants if const_module else {}
        )
        self.named_buffers: dict[str, torch.Tensor] = (
            const_module.named_buffers if const_module else {}
        )
        self.named_parameters: dict[str, torch.Tensor] = (
            const_module.named_parameters if const_module else {}
        )
        self.torchbind_constants: dict[
            str, Union[torch._C.ScriptObject, FakeScriptObject]
        ] = {}
        self.seen_subgraphs: dict[str, ir.Subgraph] = {}
        self.constant_reprs: dict[str, str] = {}
        self.removed_operations: OrderedSet[str] = OrderedSet()
        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.removed_inplace_buffers: OrderedSet[str] = OrderedSet()
        self.mutated_buffers: OrderedSet[str] = OrderedSet()
        self.never_reuse_buffers: OrderedSet[str] = OrderedSet()
        self.inplaced_to_remove: OrderedSet[str] = OrderedSet()
        self.device_ops: DeviceOpOverrides = None  # type: ignore[assignment]
        self.wrapper_code: PythonWrapperCodegen = None  # type: ignore[assignment]

        from torch._inductor.extern_node_serializer import extern_node_json_serializer

        self.extern_node_serializer: Callable[[list[ir.ExternKernelNode]], Any] = (
            extern_node_serializer
            if config.is_fbcode() and extern_node_serializer
            else extern_node_json_serializer
        )

        self.current_node: torch.fx.Node = None  # type: ignore[assignment]
        self.lists: dict[str, list[str]] = {}
        self.mutated_inputs: OrderedSet[str] = OrderedSet()
        self.mutated_input_idxs: list[int] = []
        self.name_to_buffer: dict[str, ir.Buffer] = {}
        self.name_to_users: defaultdict[str, list[ir.IRNode]] = defaultdict(list)
        self.name_to_op: dict[str, ir.Operation] = {}
        self.creation_time = time.time()
        self.name = name  # type: ignore[assignment]
        self.cpp_wrapper = cpp_wrapper
        self.fx_wrapper = fx_wrapper

        # record multi_kernel choice for cpp_wrapper so the second pass knows
        # which sub-kernel is picked. Copy cpp_wrapper to another variable
        # since cpp_wrapper flag is OrderedSet to false for the first pass of codegen.
        self.record_multi_kernel_choice = cpp_wrapper
        self.multi_kernel_to_choice: dict[str, str] = {}

        self.aot_mode = aot_mode
        self.graph_id = graph_id
        self.post_grad_graph_id = next(_post_grad_graph_counter)
        self.scheduler: torch._inductor.scheduler.Scheduler = None  # type: ignore[assignment]

        # record intermediate results for input of UsedDefinedTritonKernels
        # This will be used if autotuning is done in one pass.
        self.autotuning_inputs: Optional[list[torch.Tensor]] = None
        self.autotuning_mapping: Optional[dict[str, dict[str, int]]] = None
        self.autotuning_grids: Optional[dict[str, Any]] = None

        # current_device is set only during codegen of a device-specific kernel
        # a graph can have many devices
        self.current_device: Optional[torch.device] = None

        self.nodes_prefer_channels_last = (
            self.find_nodes_prefer_channels_last() if self.layout_opt else OrderedSet()
        )
        self._warned_fallback = OrderedSet(["aten.convolution_backward"])
        self.user_visible_output_strides = get_user_visible_output_strides(gm.graph)
        mark_nodes_dislike_padding(gm.graph, self.user_visible_output_strides)
        self.cache_key: str = ""  # This is the cache key for the compiled artifact
        self.cache_path: str = ""  # This is the path in the filesystem where the compiled artifact is stored
        self.cache_linemap: list[
            tuple[int, str]
        ] = []  # This is the linemap used by the profiler to mark custom compiled kernels getting run
        # Used if lowering encounters cases where cudagraphs are not supported
        self.disable_cudagraphs_reason: Optional[str] = None

        # only keeping one node per device for stack trace purposes
        self.device_node_mapping: dict[torch.device, torch.fx.Node] = {}
        self.orig_gm: torch.fx.GraphModule = gm.__copy__()
        for k, v in self.orig_gm.named_buffers():
            self.named_buffers[k] = v
        for k, v in self.orig_gm.named_parameters():
            self.named_parameters[k] = v
        self.dynamo_flat_name_to_original_fqn = self.module.meta.get(  # type: ignore[operator, union-attr]
            "dynamo_flat_name_to_original_fqn", {}
        )
        self.allocated_constant_name: dict[str, str] = (
            const_module.allocated_constant_name if const_module is not None else {}
        )
        init_backend_registration()
        self.get_backend_features = functools.lru_cache(None)(get_backend_features)

        self.effectful_ops: dict[_EffectType, ir.Buffer] = {}
        # Track the buffers that we know is unaligned
        # This can either be a graph input or the output of fallback
        # kernels.
        self.unaligned_buffers: OrderedSet[str] = OrderedSet()
        self.no_fuse_buffer_names: OrderedSet[str] = OrderedSet()

        self.low_precision_codegen_ops: OrderedSet[str] = OrderedSet()
        # more aggressive prologue fusion
        self.invoke_quant_ops: OrderedSet[str] = OrderedSet()

        # Below field is related to printing debug intermediate tensor values info for debugging
        self.all_codegen_kernel_names: OrderedSet[str] = OrderedSet()

        # state used by for KernelArgs.workspace
        self.workspace_id = itertools.count()

        # track the current placeholder index that we are processing
        self.placeholder_idx = -1

        self.bw_donated_idxs = get_donated_idxs()

        # Cache for dep size hints to avoid expensive recomputation
        self.dep_size_hint_cache: dict[Dep, int] = {}

    def freeze_runtime_asserts(self) -> None:
        self._shape_env.freeze_runtime_asserts()

    def symbolic_sizes_strides(
        self, ex: torch.Tensor
    ) -> tuple[Sequence[Union[int, Expr]], Sequence[Union[int, Expr]]]:
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
        if self.reuse_shape_env:
            return convert_shape_to_inductor(ex.size()), convert_shape_to_inductor(
                ex.stride()
            )
        else:
            from torch._dynamo.source import ConstantSource

            # TODO: this should not be needed once #93059 lands
            # https://github.com/pytorch/pytorch/pull/94031#discussion_r1096044816
            # TODO: make a dedicated UnknownSource for this?
            # NB: This is using the legacy default behavior from
            # create_symbolic_sizes_strides_storage_offset but we hope we can
            # just delete this entirely
            source = ConstantSource(
                f"__inductor_unknown_tensor_{len(self._shape_env.var_to_val)}"
            )
            (
                size,
                stride,
                _,
            ) = self._shape_env.create_symbolic_sizes_strides_storage_offset(
                ex,
                source,
            )

        r_size = [i.node.expr if isinstance(i, torch.SymInt) else i for i in size]
        r_stride = [i.node.expr if isinstance(i, torch.SymInt) else i for i in stride]
        return r_size, r_stride

    def static_sizes_strides(
        self, ex: torch.Tensor
    ) -> tuple[list[sympy.Expr], list[sympy.Expr]]:
        """
        Primarily used to weights
        """
        size = [sympy.Integer(i) for i in ex.size()]
        stride = [sympy.Integer(i) for i in ex.stride()]
        return size, stride

    def get_allocation_size(
        self,
        node: Union[
            ir.TensorBox, ir.StorageBox, ir.Buffer, WorkspaceArg, ir.TorchBindObject
        ],
    ) -> Sequence[Expr]:
        if isinstance(node, ir.TensorBox):
            node = node.data  # type: ignore[assignment]
        if isinstance(node, ir.StorageBox):
            node = node.data  # type: ignore[assignment]
        if (
            isinstance(node, ir.ComputedBuffer)
            and node.name in self.buffer_to_padded_size
        ):
            # pyrefly: ignore [index-error]
            return self.buffer_to_padded_size[node.name]
        else:
            return node.get_size()

    def get_allocation_storage_size(
        self, node: Union[ir.Buffer, WorkspaceArg, ir.TorchBindObject]
    ) -> Expr:
        layout = node.get_layout()
        size = self.get_allocation_size(node)  # consider inplace padding
        stride = layout.stride
        offset = layout.offset
        return compute_required_storage_length(size, stride, offset)  # type: ignore[arg-type]

    def has_feature(
        self,
        device: Union[torch._inductor.ir.IRNode, device, None],
        feature: BackendFeature,
    ) -> bool:
        assert isinstance(feature, BackendFeature), feature
        return feature in self.get_backend_features(get_device_type(device))

    def get_dep_size_hint(self, dep: Dep) -> int:
        """
        Get the size hint for a dependency with caching to avoid expensive recomputation.
        """
        if dep not in self.dep_size_hint_cache:
            res = 0
            try:
                if not dep.has_unbacked_symbols():
                    res = dep.numbytes_hint()
            except KeyError:
                # In at least one test (test/inductor/test_torchbind.py) we
                # create a StarDep that doesn't exist in the graph and calling
                # `has_unbacked_symbols()` throws an error.
                pass
            self.dep_size_hint_cache[dep] = res
        return self.dep_size_hint_cache[dep]

    def get_current_device_or_throw(self) -> torch.device:
        if device := self.current_device:
            return device
        else:
            raise RuntimeError("No current device")

    @contextlib.contextmanager
    def set_current_device(self, device: torch.device) -> Iterator[None]:
        prior = self.current_device
        self.current_device = device
        try:
            yield
        finally:
            self.current_device = prior

    def get_training_phase(self) -> str:
        if self.is_inference:
            return "inference"
        if self.is_backward:
            return "backward"
        return "forward"

    @staticmethod
    def decide_layout_opt(gm: GraphModule, *, is_inference: bool) -> bool:
        """
        Decide if we should enable layout optimization for this graph based on
        heuristics.
        """
        if not config.layout_optimization:
            return False

        if config.force_layout_optimization:
            return True

        conv_nodes = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.convolution.default
        ]
        nconv = len(conv_nodes)

        if nconv == 0:
            return False

        # For cpu backend and mkldnn enabled, we always use channels_last for better performance.
        if (
            torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
            and all(
                n.args[idx].meta["val"].device.type in SUPPORTED_MKLDNN_DEVICES
                for n in conv_nodes
                for idx in [0, 1]
            )
        ):
            return True

        # Following models are skipped due to this:
        # jx_nest_base
        # volo_d1_224
        if len(list(gm.graph.nodes)) >= 300 * nconv:
            log.debug("Skipped layout opt because only a few conv")
            return False

        if any(
            has_free_symbols(n.args[idx].meta["val"])
            for n in conv_nodes
            for idx in [0, 1]
        ):
            log.debug(
                "See perf regression with dynamic shape. Follow up in https://github.com/pytorch/pytorch/issues/102670"
            )
            return False

        def is_grouped(n: Any) -> bool:
            meta_val = n.args[1].meta["val"]  # type: ignore[union-attr, operator]
            assert isinstance(meta_val, torch.Tensor)
            return n.args[-1] > 1 and meta_val.size(1) > 1  # type: ignore[union-attr, operator]

        def is_in_out_channel(n: torch.fx.Node) -> bool:
            return (
                n.args[1].meta["val"].size(0) * 2 <= n.args[1].meta["val"].size(1)  # type: ignore[union-attr, operator]
                and n.args[1].meta["val"].size(2) > 1  # type: ignore[union-attr, operator]
            )

        def is_small_channel(n: torch.fx.Node) -> bool:
            return (
                n.args[1].meta["val"].size(0) <= 64  # type: ignore[union-attr, operator]
                and n.args[1].meta["val"].size(1) <= 64  # type: ignore[union-attr, operator]
            )

        # only grouped convolutions benchmarked as slower in conv samples for inference only
        if is_inference:
            flop_counts: dict[str, float] = defaultdict(float)
            for node in conv_nodes:
                counted_flops = count_flops_fx(node)
                if counted_flops is None:
                    continue

                if is_grouped(node):
                    node_type = "grouped"
                elif is_small_channel(node):
                    node_type = "small"
                elif is_in_out_channel(node):
                    node_type = "in_out"
                else:
                    node_type = "default"

                flop_counts[node_type] += counted_flops
            else:
                log.debug("Conv inputs meta not found")

            # average benchmarked channels last speedup / slowdown, < 1 is speedup.
            # taken from the set of convolution inputs in benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/
            # To regenerate these numbers follow https://gist.github.com/eellison/55d7a6ed6f39829d68ac56f95f4df5bb
            GROUPED_MULTIPLIER = 1.358
            DEFAULT_MULTIPLIER = 0.823
            IN_OUT_MULTIPLIER = 0.725
            SMALL_MULTIPLIER = 0.783

            total_flops = sum(flop_counts.values())
            # TODO - get different values per hardware
            weighted_flops = (
                flop_counts["grouped"] * GROUPED_MULTIPLIER
                + flop_counts["small"] * SMALL_MULTIPLIER
                + flop_counts["in_out"] * IN_OUT_MULTIPLIER
                + flop_counts["default"] * DEFAULT_MULTIPLIER
            )
            do_layout_opt = weighted_flops <= total_flops
            if not do_layout_opt:
                log.debug(
                    "Skipped layout opt in inference because weighted flops indicate slowdown, default: %d, channels last: %d",
                    total_flops,
                    weighted_flops,
                )
            return do_layout_opt

        # Channels last layout can dramatically hurt grouped conv perf. E.g.
        # Conv with arguments like
        #   {"input_shape": [32, 224, 112, 112], "weight_shape": [224, 112, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 2}
        # slows down 31x using channels last..

        # But a lot of timm models use depthwise separable convolution which will
        # result in grouped convolution with in-channel size == 1.
        # For those grouped convolution, channels last still helps a lot.
        # E.g.
        # Conv with arguments
        #   {"input_shape": [128, 58, 56, 56], "weight_shape": [58, 1, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 58}
        # get 1.86x speedup with channels last layout.
        #
        # The following heuristics skip using channels-last if the model contains
        # grouped convolution with in-channels > 1.
        if any(map(is_grouped, conv_nodes)):
            log.debug(
                "Skip layout opt because found grouped convolution with >1 in_channels!"
            )
            return False

        # For some models that contain convolution with larger in-channel than out-channel, applying
        # channels last hurts performance.
        # Following models are skipped due to this:
        # - pytorch_unet
        # - phlippe_densenet (slightly worse)
        # - Background_Matting (1.22x -> 0.821x)
        # - pytorch_CycleGAN_and_pix2pix (1.597x -> 1.294x)
        if any(map(is_in_out_channel, conv_nodes)):
            log.debug(
                "Skip layout opt because some convolutions have smaller out_channel"
            )
            return False

        # Following models are skipped due to this:
        # - functorch_maml_omniglot
        if all(map(is_small_channel, conv_nodes)):
            log.debug("Skip layout opt because all convolution channels are too small")
            return False

        return True

    def qualify_name(self, name: str) -> str:
        """Prepend the given name with the graph name if any."""
        if self.name is not None:
            return f"{self.name}_{name}"
        return name

    def make_subgraph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
        subgraph_name: str,
    ) -> SubgraphLowering:
        """
        Make a subgraph of the current graph with all inherited parts, except
        the graph module (`gm`) and `example_inputs`.  The subgraphs are lowered
        separately and lifted into a separate function in the parent output
        wrapper code.  The subgraph name is qualified by the parent graph's
        name. Note that the lifting of subgraph is supported for python wrapper
        only. For cpp wrapper, we inline the subgraphs in the parent wrapper.
        """
        return SubgraphLowering(
            parent=self,
            gm=gm,
            example_inputs=example_inputs,
            shape_env=self._shape_env,
            cpp_wrapper=self.cpp_wrapper,
            aot_mode=self.aot_mode,
            extern_node_serializer=self.extern_node_serializer,
            is_inference=self.is_inference,
            is_backward=self.is_backward,
            name=self.qualify_name(subgraph_name),
        )

    def find_nodes_prefer_channels_last(self) -> OrderedSet[Node]:
        """
        The rule to decide if an node prefer channels last is simple.
        1. if it's input/output of a convolution
        2. if one of its user prefers channels last

        We have rule 1 because cudnn runs a faster convolution kernel for channels last inputs;
        Rule 2 is also important. It makes sure that indirect inputs to convolution also prefers
        channels last.

        Consider the scenario: conv -> batch-norm -> relu -> conv
        Without rule 2, batch-norm output may use a contiguous layout. That will cause 2 extra copies:
        1. the output of batch-norm should be channels last initially since its input is a conv's output.
           Forcing the batch-norm's output to be contiguous results in the first copy
        2. The second conv's input is initially contiguous. This layout is propagated from the batch-norm's output.
           We need convert it to channels last layout which results in the second copy.
        With rule 2, we makes sure all the tensors in the chain uses channels last layout. So both copies
        can be saved.
        """
        last_conv = None
        nodes_cannot_propagate = [torch.ops.aten.bmm.default]
        output_set = OrderedSet[Node]()
        for n in reversed(self.module.graph.nodes):  # type: ignore[arg-type, union-attr]
            if n.target is torch.ops.aten.convolution.default:
                output_set.add(n)
                if last_conv is None:
                    last_conv = n
                continue
            if n.target in nodes_cannot_propagate:
                continue
            for user in n.users:
                if user in output_set:
                    output_set.add(n)
                    break

        # need a second pass to add downstream nodes of those channel last nodes to the sets.
        # This pass is especially needed to avoid mix-layout kernel inputs in backward pass.
        #
        # Let's say a conv-batchnorm 's output is passed to relu whose output is in turn returned
        # from the fwd graph. Without this second pass, we will force relu's output to be contiguous.
        # Then in the kernel in backward pass, the contiguous output of relu may be mix with other channels last
        # tensors and passed to a kernel.
        #
        # This pass improve yolov3 training speedup from 1.116x (worse than disabling layout optimization speedup 1.196x) to 1.457x.
        # It also improves dla102 training speedup from 1.240x (worse than disabling layout optimization speedup 1.523x) to 1.835x .
        # This also helps the following models:
        # - res2net101_26w_4s
        # - res2net50_14w_8s
        # - sebotnet33ts_256
        for n in self.module.graph.nodes:  # type: ignore[union-attr]
            # layout propagation ends at last conv node, which will benefit vison transformers.
            if last_conv is not None and n == last_conv:
                break
            if n in output_set:
                for user in n.users:
                    if user.target in nodes_cannot_propagate:
                        continue
                    output_set.add(user)

        return output_set

    def warn_fallback(self, name: str) -> None:
        if name not in self._warned_fallback:
            self._warned_fallback.add(name)
            perf_hint_log.info("Using FallbackKernel: %s", name)

    def add_device_info(self, device: torch.device) -> None:
        self.device_types.add(device.type)
        if device.index is not None:
            self.device_idxs.add(device.index)
        if V.graph.current_node and device not in self.device_node_mapping:
            self.device_node_mapping[device] = V.graph.current_node

    @property
    def fake_mode(self) -> torch._subclasses.fake_tensor.FakeTensorMode:
        return V.fake_mode

    def try_get_buffer(
        self, buffer_name: str
    ) -> Optional[Union[ir.TensorBox, ir.Buffer, ir.TorchBindObject]]:
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name]
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name]
        if buffer_name in self.constants:
            data = V.graph.constants[buffer_name]
            return ir.ConstantBuffer(
                name=buffer_name,
                layout=ir.FixedLayout(
                    data.device, data.dtype, *V.graph.static_sizes_strides(data)
                ),
            )

        return None

    def add_symbol_graph_input(self, symbol: sympy.Expr) -> None:
        raise RuntimeError("Should not be called for the main graph")

    def get_buffer(
        self, buffer_name: str
    ) -> Union[ir.TensorBox, ir.Buffer, ir.TorchBindObject]:
        buf = self.try_get_buffer(buffer_name)
        if buf is not None:
            return buf
        raise RuntimeError(f"Failed to find buffer matching name {buffer_name}")

    def get_dtype(self, buffer_name: str) -> torch.dtype:
        if buffer_name in self.constants:
            return self.constants[buffer_name].dtype
        # For a mutation op we should return the dtype of the buffer being mutated
        if (
            hasattr(self.scheduler, "mutation_real_name")
            and buffer_name in self.scheduler.mutation_real_name
        ):
            mutated_buf = self.scheduler.mutation_real_name[buffer_name]
            if mutated_buf in self.name_to_buffer:
                return self.name_to_buffer[mutated_buf].get_dtype()
            if mutated_buf in self.graph_inputs:
                return self.graph_inputs[mutated_buf].get_dtype()
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_dtype()
        m = re.match(r"(as_strided|reinterpret_tensor)\(([a-zA-Z0-9_]+),", buffer_name)
        if m:
            return self.get_dtype(m.group(1))
        raise KeyError(f"could not find {buffer_name}")

    def get_numel(self, buffer_name: str) -> Union[int, Expr]:
        if buffer_name in self.constants:
            return self.constants[buffer_name].numel()
        if buffer_name in self.name_to_buffer:
            buf = self.name_to_buffer[buffer_name]
            if not buf.has_tensor_output():
                return 1
            return buf.get_numel()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_numel()
        raise KeyError(f"could not find {buffer_name}")

    def run(self, *args: Any) -> Any:  # type: ignore[override]
        with dynamo_timed("GraphLowering.run"):
            return super().run(*args)

    def register_operation(self, op: ir.Operation) -> str:
        assert op.operation_name is None, f"Operation registered twice: {op}"
        assert isinstance(op, ir.Operation)
        name = self.qualify_name(f"op{len(self.operations)}")
        self.operations.append(op)
        self.name_to_op[name] = op
        op.operation_name = name
        return name

    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str:
        name = self.qualify_name(f"buf{len(self.buffers)}")
        self.buffers.append(buffer)
        self.name_to_buffer[name] = buffer
        device = buffer.get_device()
        if (
            # Skip empty CPU tensor so that CUDA graphs can succeed, see https://github.com/pytorch/pytorch/pull/114144
            device is not None
            and not (
                isinstance(buffer, ir.ComputedBuffer)
                and buffer.is_zero_elements()
                and device == torch.device("cpu")
            )
        ):
            self.add_device_info(device)

        if set_name:
            buffer.name = name
        return name

    def register_operation_list(self, operation_names: list[str]) -> str:
        name = self.qualify_name("list_" + "_".join(operation_names))
        self.lists[name] = operation_names
        return name

    def register_users_of(
        self, node_output: Union[Iterable[ir.IRNode], ir.IRNode]
    ) -> None:
        def register(value: Union[Iterable[ir.IRNode], ir.IRNode]) -> None:
            if isinstance(value, (list, tuple)):
                for x in value:
                    register(x)
            if isinstance(value, ir.TensorBox):
                for read_name in value.get_read_names():
                    self.name_to_users[read_name].append(value)

        register(node_output)

    def mark_buffer_mutated(self, name: str) -> None:
        """
        When a buffer is mutated we need to make sure all the reads to
        the old version are realized before the mutation happens.
        """
        assert isinstance(name, str)
        self.mutated_buffers.add(name)

        if name not in self.name_to_users:
            return

        for user in self.name_to_users[name]:
            user.realize()

    def get_original_value_of_constant(self, name: str) -> torch.Tensor:
        """
        In AOTI, module buffers may have been mutated during the tracing and compilation.
        Thus we need to read from previously stored original buffers, to make sure the
        generated model.so uses correct initial values.
        """
        assert name in self.allocated_constant_name and name in self.constants, (
            "Can not find the original value for " + name
        )
        orig_name = get_cloned_parameter_buffer_name(self.allocated_constant_name[name])
        return (
            self.module.meta[orig_name]  # type: ignore[index]
            if orig_name in self.module.meta  # type: ignore[operator]
            else self.constants[name]
        )

    def allocate_non_dup_const_name(
        self, name: Optional[str], data: Union[Tensor]
    ) -> str:
        if not config.aot_inductor.use_runtime_constant_folding:
            for constant_name, value in self.constants.items():
                if is_same_tensor(data, value):
                    return constant_name

        if name is None:
            name = f"constant{len(self.constants)}"
        orig_name = name
        if name[0].isdigit():
            name = f"constant_{name}"
        name = self.qualify_name(name)
        # We may generate a var name for each constant in the codegen.
        # Let's only keep sane characters.
        prefix = normalize_name(name)
        name = prefix
        cnt = 0
        while name in self.constants:
            name = f"{prefix}_{cnt}"
            cnt += 1
        self.constants[name] = data
        self.constant_reprs[name] = (
            f"{data.device!r} {data.dtype!r} "
            f"{tuple(data.size())!r} {tuple(data.stride())!r} "
            f"{hash(data):x}"
        )
        self.allocated_constant_name[name] = orig_name  # type: ignore[assignment]
        return name

    def add_tensor_constant(
        self, data: Tensor, name: Optional[str] = None
    ) -> Union[TensorBox, ir.ShapeAsConstantBuffer]:
        new_name = self.allocate_non_dup_const_name(name, data)
        return TensorBox.create(
            ir.ConstantBuffer(
                name=new_name,
                layout=FixedLayout(
                    data.device, data.dtype, *self.static_sizes_strides(data)
                ),
            )
        )

    def constant_name(self, name: str, device_override: Optional[torch.device]) -> str:
        """
        We AOT copy constants to the devices they are needed on.
        If device_override doesn't match the constant's device, then
        copy it and return a different name.
        """
        if self.constants[name].device == device_override or device_override is None:
            return name
        with torch.utils._python_dispatch._disable_current_modes():
            # caller might have OrderedSet fake tensor mode which will create a fake tensor
            # when calling .to, so unset modes here
            return self.allocate_non_dup_const_name(
                f"{name}_{device_override.type}{device_override.index or 0}",
                self.constants[name].to(device_override),
            )

    # pyrefly: ignore [bad-override]
    def placeholder(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object],  # type: ignore[override]
        kwargs: dict[str, object],
    ) -> Union[Expr, TensorBox, None]:
        self.placeholder_idx += 1
        example = super().placeholder(target, args, kwargs)  # type: ignore[arg-type]
        target = self.qualify_name(target)
        if isinstance(example, SymTypes):
            # TODO fix partitioning issue and re-enable for backward
            # https://github.com/pytorch/pytorch/issues/155468.
            if not V.graph.is_backward:
                expr = _get_placeholder_expr(example.node)
            else:
                expr = example.node.expr
            self.graph_inputs[target] = expr
            self.graph_input_names.append(target)
            return expr
        elif isinstance(example, (int, bool, float)):
            expr = sympy.sympify(example)
            self.graph_inputs[target] = expr
            self.graph_input_names.append(target)
            return expr
        elif isinstance(example, FakeScriptObject):
            obj = TorchBindObject(name=target, value=example)
            self.graph_inputs[target] = obj
            self.graph_input_names.append(target)
            return obj
        elif example is None:
            self.graph_input_names.append(target)
            return None
        if isinstance(example, BackwardState):
            # Ignored arg, must be unused
            # Alternately we could filter this out in AotAutograd
            self.graph_input_names.append(target)
            return None
        # See note: Note: [Generator arguments in AOTDispatcher]
        elif isinstance(example, torch.Generator):
            assert len(V.graph.current_node.users) == 1 and next(
                iter(V.graph.current_node.users)
            ).target in (
                torch._prims.rng_prims.graphsafe_run_with_rng_state,
                torch.ops.higher_order.invoke_subgraph,
            )
            gen = ir.GeneratorState(name=target, device=example.device)
            self.graph_inputs[target] = gen  # type: ignore[assignment]
            self.graph_input_names.append(target)
            return gen

        assert isinstance(example, torch.Tensor), example
        # todo(chilli): We can remove the last check once we turn buffers into
        # static shape tensors. That's a hack to workaround Inductor believing
        # the buffer should be static but us passing in a fake tensor with
        # symbolic shapes.
        if not example._has_symbolic_sizes_strides:
            # the first N inputs are weights
            sizes, strides = self.static_sizes_strides(example)
        else:
            sizes, strides = self.symbolic_sizes_strides(example)  # type: ignore[assignment]

        if (
            self.is_backward
            and self.bw_donated_idxs
            and self.placeholder_idx in self.bw_donated_idxs
        ):
            tensor = TensorBox.create(
                DonatedBuffer(
                    name=target,
                    layout=FixedLayout(example.device, example.dtype, sizes, strides),
                )
            )
        else:
            # TODO(jansel): handle input aliasing
            tensor = TensorBox.create(
                InputBuffer(
                    name=target,
                    layout=FixedLayout(example.device, example.dtype, sizes, strides),
                )
            )

        self.graph_inputs[target] = tensor
        self.graph_input_names.append(target)
        self.graph_inputs_original[target] = tensor.data.data  # type: ignore[union-attr]
        if self.current_node.users:  # cudagraphs should work with an unused CPU input
            self.add_device_info(example.device)

        # Note: [Input Alignment handling in Inductor]
        # Alignment matters for generating efficient code. Some operations,
        # e.g. vectorized loads, can only be performed on aligned inputs.
        #
        # But if we codegen assuming aligned inputs and then get unaligned
        # inputs at runtime, then we are forced to clone - which is bad for
        # both perf and memory usage.
        #
        # One option would be to guard on storage_offset%ALIGNMENT, and then
        # codegen based on this. But storage_offset guards turned out to be
        # expensive and cause recompiles; Instead, we're generating code
        # based on the alignment of the example input without guarding.
        with maybe_get_suppress_shape_guards_ctx():
            if not should_assume_input_aligned(example):
                self.unaligned_buffers.add(target)
        return tensor

    def call_function(self, target: Callable, args: Any, kwargs: dict[str, Any]) -> Any:  # type: ignore[type-arg, override]
        if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
            return super().call_function(target, args, kwargs)

        # hasattr on OpOverloadPacket is slow, check isinstance first
        if not isinstance(target, torch._ops.OpOverloadPacket) and hasattr(
            target, "_inductor_lowering_function"
        ):
            # passthrough lowerings from .pattern_matcher
            return target(*args, **kwargs)

        if target not in lowerings:
            assert isinstance(target, torch._ops.OpOverload), (
                f"{target} is not an OpOverload"
            )
            base_name = target.name().split(".")[0]
            if base_name in FALLBACK_ALLOW_LIST:
                make_fallback(target, warn=False, override_decomp=True)
            elif config.implicit_fallbacks:
                error = (
                    MissingOperatorWithDecomp
                    if get_decompositions([target])
                    else MissingOperatorWithoutDecomp
                )
                log.info(
                    "Creating implicit fallback for:\n%s",
                    error.operator_str(target, args, kwargs),
                )

                tag: Optional[torch._C.Tag] = get_layout_constraint_tag(
                    target, with_default=False
               
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

- **Object-Oriented Design**: Uses classes and constructors
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

- **File Documentation**: `graph.py_docs.md_docs.md`
- **Keyword Index**: `graph.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

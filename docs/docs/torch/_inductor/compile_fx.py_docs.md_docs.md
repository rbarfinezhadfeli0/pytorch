# Documentation: `docs/torch/_inductor/compile_fx.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/compile_fx.py_docs.md`
- **Size**: 54,121 bytes (52.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/compile_fx.py`

## File Metadata

- **Path**: `torch/_inductor/compile_fx.py`
- **Size**: 116,574 bytes (113.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from __future__ import annotations

import contextlib
import copy
import enum
import functools
import io
import itertools
import json
import logging
import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import currentframe
from itertools import count
from operator import attrgetter
from typing import Any, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Never, override, ParamSpec, Protocol, TypedDict, Unpack
from unittest import mock

import torch._inductor.async_compile
import torch.fx
import torch.utils._pytree as pytree
from functorch.compile import min_cut_rematerialization_partition
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import (
    compiled_autograd,
    config as dynamo_config,
    logging as dynamo_logging,
    utils as dynamo_utils,
)
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.repro.after_aot import wrap_compiler_debug
from torch._dynamo.utils import (
    chromium_event_timed,
    CompileEventLogger,
    counters,
    detect_fake_mode,
    dynamo_timed,
    flatten_graph_inputs,
    get_metrics_context,
    lazy_format_graph_code,
    set_feature_use,
)
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.subclass_parametrization import (
    unwrap_tensor_subclass_parameters,
)
from torch._functorch.aot_autograd import (
    aot_export_module,
    GraphOutputName,
    make_boxed_func,
    SerializableAOTDispatchCompiler,
)
from torch._inductor.codecache import code_hash, FxGraphCache, output_code_log
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    format_default_skip_message,
    log_cudagraph_skip_and_bump_counter,
    PlaceholderInfo,
)
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.debug import (
    create_mapping_pre_post_grad_nodes,
    save_args_for_compile_fx_inner,
)
from torch._inductor.output_code import (
    CompiledAOTI,
    CompiledFxGraph,
    CompiledFxGraphConstantsWithGm,
    get_expanded_dims,
    index_expanded_dims,
    OutputCode,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.utils import (
    BoxedBool,
    count_tangents,
    fresh_cache,
    get_all_devices,
    InputType,
    is_gpu,
    should_assume_input_aligned,
    should_use_remote_fx_graph_cache,
    tensor_is_aligned,
)
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._logging import trace_structured
from torch._utils_internal import compile_time_strobelight_meta
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymExprPrinter
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.monitor import _WaitCounter
from torch.utils._ordered_set import OrderedSet

from .._dynamo.backends.common import aot_autograd
from .._dynamo.exc import ShortenTraceback, SkipFrame
from ..fx._lazy_graph_module import _use_lazy_graph_module
from ..fx.graph import _PyTreeCodeGen
from ..utils._triton import has_triton
from . import config, distributed_autotune, metrics
from .codegen.common import get_wrapper_codegen_for_device, init_backend_registration
from .debug import DebugContext
from .decomposition import select_decomp_table
from .exc import InductorError
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import get_device_type, IRNode
from .output_code import complex_memory_overlap  # noqa: F401
from .triton_bundler import TritonBundler
from .utils import (
    align_inputs_from_check_idxs,
    clone_preserve_strides,
    copy_misaligned_inputs,
    get_cloned_parameter_buffer_name,
    get_first_incompatible_cudagraph_node,
    maybe_get_suppress_shape_guards_ctx,
    output_node,
    remove_unaligned_input_idxs,
    shape_env_from_inputs,
)
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._inductor.output_code import _StrideExprStr
    from torch._ops import OpOverload
    from torch.export.pt2_archive._package_weights import Weights

    from .ir import ExternKernelNode


_P = ParamSpec("_P")
_T = TypeVar("_T")

if TYPE_CHECKING or not config.is_fbcode():
    # no-op decorator
    def time_and_log(attr: str) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        return dynamo_utils.identity

    def log_optimus_to_scuba(*args: object, **kwargs: object) -> None:
        pass

else:
    from torch._inductor.fb.utils import log_optimus_to_scuba, time_and_log

if TYPE_CHECKING:
    import types

    from torch._functorch._aot_autograd.schemas import (
        FQN,
        GraphInputName,
        GraphSignature,
    )

    CompileFxOutput = Union[
        Callable[[list[object]], Sequence[torch.Tensor]],
        str,
        list[str],
        Weights,
    ]


class FxCompileMode(enum.Enum):
    NORMAL = 0
    # For testing - use the serde FxCompile scheme to debug serialization and
    # deserialization of GraphMoule and CompiledFxGraph.
    SERIALIZE = 1
    # Compile using a subprocess instead of in-process.
    SUBPROCESS = 2


@dataclass
class FxCompileConfig:
    mode: FxCompileMode
    use_async: bool
    use_progressive: bool


def _fx_compile_mode_default() -> FxCompileConfig:
    name = "TORCHINDUCTOR_FX_COMPILE_MODE"
    value = os.environ.get(name)
    if value is None:
        return FxCompileConfig(FxCompileMode.NORMAL, False, False)

    use_async = False
    use_progressive = False

    if value.lower().startswith("progressive+"):
        use_progressive = True
        value = value[12:]
    if value.lower().startswith("async+"):
        use_async = True
        value = value[6:]

    try:
        value = value.upper()
        return FxCompileConfig(FxCompileMode[value], use_async, use_progressive)
    except KeyError:
        import logging

        log = logging.getLogger(__name__)
        log.error(
            "Invalid value of %s for %s. Expected one of %s. Using default.",
            value,
            name,
            ", ".join(sorted(repr(x) for x in FxCompileMode.__members__)),
        )
        # Remove from the environment so subprocesses don't ALSO complain.
        os.environ.pop(name)
        return FxCompileConfig(FxCompileMode.NORMAL, False, False)


def _get_progression_configs() -> list[dict[str, Any]]:
    # TODO make this configurable
    return [
        {"max_autotune": True},
    ]


_fx_compile_config = _fx_compile_mode_default()
fx_compile_mode = _fx_compile_config.mode
fx_compile_async = _fx_compile_config.use_async
fx_compile_progressive = _fx_compile_config.use_progressive

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
pre_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "pre_grad_graphs")
post_grad_graphs_log = torch._logging.getArtifactLogger(__name__, "post_grad_graphs")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)
inductor_metrics_log = torch._logging.getArtifactLogger(__name__, "inductor_metrics")


def get_static_input_idxs(num_fixed: int) -> list[int]:
    # If we are inlining NNModules, we treat all torch.nn.Parameters as static for the purposes
    # of cudagraphs. Rather than copying these into cudagraph-owned memory
    # like we do for normal inputs on each run, we will re-record a cudagraph if these
    # parameter locations change.
    context = torch._guards.TracingContext.try_get()
    fixed = list(range(num_fixed))
    if not context or not context.fw_metadata:
        return fixed

    return context.fw_metadata.static_input_indices


def record_original_output_strides(gm: GraphModule) -> None:
    output_node = gm.graph.find_nodes(op="output")[0]
    output_strides = []

    if not isinstance(output_node.args[0], torch.fx.Node):
        output_node_args = output_node.args[0]
    else:
        output_node_args = output_node.args

    for output in output_node_args:
        if (
            isinstance(output, torch.fx.Node)
            and (val := output.meta.get("val")) is not None
            and isinstance(val, torch.Tensor)
        ):
            output_strides.append(val.stride())
        else:
            # pyrefly: ignore [bad-argument-type]
            output_strides.append(None)
    output_node.meta["original_output_strides"] = output_strides


def _recursive_record_original_output_strides(gm: GraphModule) -> None:
    # invoke_subgraph HOP requires output strides to be respected
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(gm, node.args[0].target)
        _recursive_record_original_output_strides(subgraph)

    record_original_output_strides(gm)


def _recursive_record_user_visible_output_idxs(gm: GraphModule) -> None:
    # invoke_subgraph HOP requires output strides to be respected
    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        subgraph = getattr(gm, node.args[0].target)

        for node in subgraph.graph.find_nodes(op="output"):
            node.meta["user_visible_output_idxs"] = [
                idx
                for idx in range(len(node.args[0]))
                if isinstance(node.args[0][idx], torch.fx.Node)
            ]
        _recursive_record_user_visible_output_idxs(subgraph)


@functools.lru_cache(None)
def _step_logger() -> Callable[..., None]:
    return dynamo_logging.get_step_logger(log)


@functools.cache
def _warn_tf32_disabled() -> None:
    if (
        torch.cuda.is_available()
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.cuda.get_device_capability() >= (8, 0)
    ):
        warnings.warn(
            "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
            "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
        )


def _resolve_name_collision(mod: GraphModule, gm: GraphModule) -> None:
    """
    In aot_export_module (make_fx), we create get_attr nodes with name prefix
    "_tensor_constant" and "_torchbind_obj". See Tracer.create_arg() in
    torch/fx/_symbolic_trace.py

    However, this might result in name collision if the original mod already
    has a different buffer with the same name.

    We resolve this potential name collision here by changing the target name
    with a new number post fix.
    """

    existing_keys = OrderedSet(
        [name for name, val in mod.named_parameters(remove_duplicate=False)]
    )
    existing_keys.update(
        OrderedSet([name for name, val in mod.named_buffers(remove_duplicate=False)])
    )

    def find_smallest_i(graph: fx.Graph, prefix: str) -> int:
        i = 0
        for node in graph.nodes:
            if node.op == "get_attr" and node.target.startswith(prefix):
                if len(node.target) > len(prefix):
                    post_fix = node.target.split(prefix)[-1]
                    if post_fix.isdigit():
                        i = max(i, int(post_fix))
        for key in existing_keys:
            if key.startswith(prefix):
                if len(key) > len(prefix):
                    post_fix = key.split(prefix)[-1]
                    if post_fix.isdigit():
                        i = max(i, int(post_fix))
        return i + 1

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            target_name = node.target
            if not target_name.startswith(
                "_tensor_constant"
            ) and not target_name.startswith("_torchbind_obj"):
                continue

            if not hasattr(mod, target_name):
                continue
            gm_target = attrgetter(target_name)(gm)
            model_target = attrgetter(target_name)(mod)
            if isinstance(gm_target, FakeScriptObject):
                if (
                    isinstance(model_target, FakeScriptObject)
                    and gm_target.real_obj is model_target.real_obj
                ):
                    continue
            elif (
                gm_target.device == model_target.device
                and gm_target.dtype == model_target.dtype
                and torch.equal(gm_target, model_target)
            ):
                # If tensors with same name from gm and model are indeed the same, we don't need to rename
                # Check device first, to avoid torch.equal(wrapper_CUDA__equal) raise when different device
                continue

            prefix = (
                "_tensor_constant"
                if target_name.startswith("_tensor_constant")
                else "_torchbind_obj"
            )
            new_id = find_smallest_i(gm.graph, prefix)
            new_target_name = f"{prefix}{new_id}"
            node.target = new_target_name
            setattr(gm, new_target_name, gm_target)
            existing_keys.add(new_target_name)


def _unlift_graph(
    mod: GraphModule, gm: GraphModule, graph_signature: GraphSignature
) -> GraphModule:
    from torch.export.unflatten import _assign_attr, _AttrKind

    _resolve_name_collision(mod, gm)

    state_dict: dict[str, Union[torch.nn.parameter.Parameter, torch.Tensor]] = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
        _assign_attr(
            param,
            gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = buffer
        _assign_attr(
            buffer,
            gm,
            name,
            attr_kind=_AttrKind.BUFFER,
        )

    placeholder_nodes = gm.graph.find_nodes(op="placeholder")
    lifted_inputs: list[Optional[FQN]] = []

    # In AOTI, module parameters and buffers are not lifted as graph inputs.
    # As a result, mutation to buffers has side effect which makes their initial
    # values different from Eager. So we clone them here as a copy.
    # We are not cloning for parameters, although it will be needed if we want to
    # support training.
    for node in placeholder_nodes:
        node_name = node.name
        if node_name in graph_signature.inputs_to_parameters:
            parameter_name = graph_signature.inputs_to_parameters[node_name]
            lifted_inputs.append(parameter_name)
        elif node_name in graph_signature.inputs_to_buffers:
            buffer_name = graph_signature.inputs_to_buffers[node_name]
            lifted_inputs.append(buffer_name)
            gm.meta[get_cloned_parameter_buffer_name(buffer_name)] = (
                clone_preserve_strides(state_dict[buffer_name])
            )
        else:
            assert node_name in graph_signature.user_inputs
            lifted_inputs.append(None)

    from torch.export._unlift import _unlift

    outputs: tuple[torch.fx.Node, ...] = tuple(gm.graph.output_node().args[0])  # type: ignore[arg-type]
    mutated_outputs = []
    buffer_mutations = graph_signature.buffers_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    output_tokens = graph_signature.output_tokens
    for idx, out in enumerate(outputs):
        value: Optional[Union[FQN, GraphInputName]] = None

        if idx < len(buffer_mutations) + len(user_input_mutations) + len(output_tokens):
            name = GraphOutputName(out.name)
            if name in buffer_mutations:
                value = buffer_mutations[name]
            elif name in user_input_mutations:
                value = user_input_mutations[name]

        mutated_outputs.append(value)

    unlifted_gm = _unlift(
        gm,
        lifted_inputs,
        mutated_outputs,
        pytree.treespec_leaf(),
        None,
    )
    return unlifted_gm


def _get_subgraph_names(
    gm: GraphModule, skip_invoke_subgraph: bool = False
) -> Generator[str, None, None]:
    all_subgraph_names: OrderedSet[str] = OrderedSet(
        x.target for x in gm.graph.find_nodes(op="get_attr")
    )
    fx_subgraph_names: OrderedSet[str] = OrderedSet()
    for child_name, child_module in gm.named_children():
        # Sometimes an owning_module can have unused children. Skip them
        # by checking them from get_attr node targets.
        if child_name in all_subgraph_names and isinstance(
            child_module, torch.fx.GraphModule
        ):
            fx_subgraph_names.add(child_name)

    if skip_invoke_subgraph:
        for node in gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        ):
            fx_subgraph_names.discard(node.args[0].target)

    yield from fx_subgraph_names


def _recursive_pre_grad_passes(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
) -> GraphModule:
    with dynamo_timed(
        "_recursive_pre_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="pre_grad_pass_time_us",
    ):
        if not config.use_pre_grad_passes:
            return gm

        add_passes = config.add_pre_grad_passes
        remove_passes = config.remove_pre_grad_passes
        for subgraph_name in _get_subgraph_names(gm):
            subgraph = getattr(gm, subgraph_name)
            # as we don't have recursive example inputs, passing empty set here
            new_subgraph = _recursive_pre_grad_passes(subgraph, ())
            setattr(gm, subgraph_name, new_subgraph)
        return pre_grad_passes(gm, example_inputs, add_passes, remove_passes)


def _recursive_joint_graph_passes(
    gm: GraphModule, skip_invoke_subgraph: bool = False
) -> None:
    with dynamo_timed(
        "_recursive_joint_graph_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="joint_graph_pass_time_us",
    ):
        if not config.use_joint_graph_passes:
            return

        # invoke_subgraph already runs the _recursive_joint_graph_passes.  In
        # AOTAutograd, `run_joint_graph_passes_on_hops` partitions the
        # invoke_subgraph HOP before calling the partitioner on the outer graph.
        # AOTAutograd has access to partition_fn, which internally calls the
        # `_recursive_joint_graph_passes` for the subgraph. So, skip recursing
        # skip_invoke_subgraph.
        for subgraph_name in _get_subgraph_names(gm, skip_invoke_subgraph):
            subgraph = getattr(gm, subgraph_name)
            _recursive_joint_graph_passes(subgraph, skip_invoke_subgraph)
        joint_graph_passes(gm)


def _recursive_post_grad_passes(gm: GraphModule, is_inference: bool = False) -> None:
    with dynamo_timed(
        "_recursive_post_grad_passes",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="post_grad_pass_time_us",
    ):
        if not config.use_post_grad_passes:
            return

        for subgraph_name in _get_subgraph_names(gm):
            subgraph = getattr(gm, subgraph_name)
            _recursive_post_grad_passes(subgraph, is_inference)
        post_grad_passes(gm, is_inference)


def split_const_gm(
    gm: GraphModule,
    skip_constructor: bool = True,
    lifted_constant_names: Optional[list[str]] = None,
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> tuple[GraphModule, dict[str, int]]:
    """
    This function takes an GraphModule input "gm".
    The gm will be split into 2 components,
      1) const_gm, which consists the subgraph of gm that can be constant folded.
      2) gm (being inplace modified,) which returns the graph after constant folding.

    If an additional "lifted_constants" argument is passed in, we will assume the gm has
    been lifted and run the transformation accordingly.

    When a "skip_folding_node_fn" callback is passed, we will skip constant folding on
    the nodes for which the callback returns True.

    const_output_index is a mapping of corresponding node name from gm to the
    output index of const_gm.
    Returns (const_gm, const_output_index)
    """
    from torch._inductor.constant_folding import (
        CONST_MODULE_TAG,
        META_TAG,
        MODULE_TAG,
        replace_node_with_constant,
        run_and_get_constant_graph,
    )

    const_gm = run_and_get_constant_graph(
        gm, skip_constructor, lifted_constant_names, skip_folding_node_fn
    )
    const_result = const_gm() if lifted_constant_names is None else None

    const_outputs = {
        x.name: idx for idx, x in enumerate(tuple(const_gm.graph.nodes)[-1].args[0])
    }

    to_erase_node = []
    to_replace_node = []
    const_output_index = {}
    for node in gm.graph.nodes:
        if node.name in const_outputs:
            to_replace_node.append(node)
        elif node.meta[META_TAG] == CONST_MODULE_TAG and node.op != "placeholder":
            to_erase_node.append(node)

    for node in to_replace_node:
        new_const_name = "_FOLDED_CONST_" + node.name
        replace_node_with_constant(
            gm,
            node,
            (
                const_result[const_outputs[node.name]]  # type:ignore[index]
                if lifted_constant_names is None
                else None
            ),
            new_const_name,
        )
        const_output_index[new_const_name] = const_outputs[node.name]
    for node in to_erase_node[::-1]:
        if node.users:
            for n in node.users:
                assert n.meta[META_TAG] == MODULE_TAG, f"node: {node} user not empty."
        else:
            gm.graph.erase_node(node)
    gm.recompile()

    return const_gm, const_output_index


def is_tf32_warning_applicable(gm: GraphModule) -> bool:
    aten = torch.ops.aten
    tf32_ops = OrderedSet(
        [
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
        ]
    )
    for target in tf32_ops:
        for node in gm.graph.find_nodes(op="call_function", target=target):
            if (
                isinstance(node.meta.get("val", None), torch.Tensor)
                and node.meta["val"].dtype == torch.float32
                and node.meta["val"].device.type == "cuda"
            ):
                return True
    return False


def maybe_disable_comprehensive_padding(
    example_inputs: Sequence[InputType],
) -> AbstractContextManager[None, None]:
    """
    For CPU backend, enable comprehensive padding causes some unit tests
    fail due to changing number of generated kernels. Skip for now.
    """
    has_gpu = any(
        is_gpu(t.device.type) for t in example_inputs if isinstance(t, torch.Tensor)
    )

    if config.disable_padding_cpu and config.comprehensive_padding and not has_gpu:
        perf_hint_log.info("Skip comprehensive padding on CPU")
        return config.patch(comprehensive_padding=False)
    elif config.aot_inductor.use_runtime_constant_folding:
        perf_hint_log.info(
            "Skip comprehensive padding for use_runtime_constant_folding"
        )
        return config.patch(comprehensive_padding=False)
    else:
        return contextlib.nullcontext()


def maybe_disable_graph_partition(
    cpp_wrapper: bool, aot_mode: bool
) -> AbstractContextManager[None, None]:
    """
    graph partition does not support cpp_wrapper and aot_mode yet.
    """
    if cpp_wrapper or aot_mode:
        return config.patch(graph_partition=False)
    else:
        return contextlib.nullcontext()


def fake_tensor_prop(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    force_allow_non_fake_inputs: bool = False,
) -> torch._subclasses.FakeTensorMode:
    """
    If we can not detect fake mode from the context of inputs, create one.

    The created fake mode will be returned.
    """
    # Ensure that decomps that support symbolic shapes are used
    with enable_python_dispatcher():
        fake_mode = detect_fake_mode(example_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
            FakeTensorProp(gm, mode=fake_mode).propagate(*example_inputs)
        else:
            ctx = (
                contextlib.nullcontext()
                if not force_allow_non_fake_inputs
                else mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
            )
            with ctx:  # type: ignore[attr-defined]
                FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
                    *example_inputs
                )

    return fake_mode


# pass config dict back to user
def get_patched_config_dict(
    config_patches: Optional[Union[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    with config.patch(config_patches):
        return config.get_config_copy()


@contextlib.contextmanager
def with_fresh_cache_if_config() -> Generator[None, None, None]:
    if config.force_disable_caches:
        # Don't delete the cache dir because it has to survive beyond the
        # compile_fx call. Let's put the temp dirs under the default cache
        # dir so they're easier to locate.
        with fresh_cache(dir=cache_dir(), delete=False):
            yield
    else:
        yield


class _CompileFxKwargs(TypedDict, total=False):
    cudagraphs: Optional[BoxedBool]
    static_input_idxs: Sequence[int]
    is_backward: bool
    graph_id: Optional[int]
    cpp_wrapper: bool
    aot_mode: bool
    is_inference: bool
    layout_opt: Optional[bool]
    extern_node_serializer: Optional[Callable[[list[ExternKernelNode]], Any]]
    boxed_forward_device_index: Optional[BoxedDeviceIndex]
    fx_wrapper: bool


class _CompileFxCallable(Protocol):
    def __call__(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        **kwargs: Unpack[_CompileFxKwargs],
    ) -> OutputCode: ...


def compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    **kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    kwargs.setdefault("cudagraphs", None)
    kwargs.setdefault("static_input_idxs", ())
    kwargs.setdefault("is_backward", False)
    kwargs.setdefault("graph_id", None)
    kwargs.setdefault("cpp_wrapper", False)
    kwargs.setdefault("fx_wrapper", False)
    kwargs.setdefault("is_inference", False)
    kwargs.setdefault("boxed_forward_device_index", None)
    kwargs.setdefault("layout_opt", None)
    kwargs.setdefault("extern_node_serializer", None)

    # Need with_fresh_cache_if_config for compile_fx_inner even if we already have one for
    # compile_fx. The reason is the compilation for backward graph may happen after
    # compile_fx return and we may want to use the _LazyGraphModule for compiling
    # the backward graph as well.
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.utils._python_dispatch._disable_current_modes())
        stack.enter_context(_use_lazy_graph_module(dynamo_config.use_lazy_graph_module))
        stack.enter_context(
            dynamo_utils.dynamo_timed(
                "compile_fx_inner",
                phase_name="inductor_compile",
                log_pt2_compile_event=True,
                log_waitcounter=True,
                waitcounter_name_override="compile_inductor",
                dynamo_compile_column_us="inductor_cumulative_compile_time_us",
            )
        )
        stack.enter_context(with_fresh_cache_if_config())
        stack.enter_context(DebugContext())
        CompileEventLogger.pt2_compile(
            "inductor_compile",
            is_backward=kwargs["is_backward"],
        )
        return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
            gm,
            example_inputs,
            **kwargs,
        )


@time_and_log(attr="compilation time (in seconds)")
def _compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    **graph_kwargs: Unpack[_CompileFxKwargs],
) -> OutputCode:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    aot_mode: bool = V.aot_compilation

    # Clean up Compiled Triton Kernels per inductor compile, as the future objects
    # may not be valid for use after they are run/autotuned
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()

    if dynamo_utils.count_calls(gm.graph) == 0 and not aot_mode:
        # trigger the real recompilation for _LazyGraphModule before returning
        # the forward method.
        from torch._dynamo.utils import CompileEventLogLevel
        from torch.fx._lazy_graph_module import _LazyGraphModule

        _LazyGraphModule.force_recompile(gm)
        compile_id = torch._guards.CompileContext.current_compile_id()
        CompileEventLogger.log_instant_event(
            "backward no-op",
            metadata={"compile_id": compile_id},
            log_level=CompileEventLogLevel.PT2_COMPILE,
        )

        return make_boxed_func(gm.forward)

    static_input_idxs: Sequence[int] = graph_kwargs.setdefault("static_input_idxs", ())
    static_inputs_log.debug("static input idxs compile_fx_inner: %s", static_input_idxs)
    inputs_to_check = get_input_idxs_to_check(example_inputs, static_input_idxs)

    assert isinstance(next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)), (
        f"inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}"
    )

    if graph_kwargs.get("cudagraphs") is None:
        graph_kwargs["cudagraphs"] = BoxedBool(config.triton.cudagraphs)
    if config.save_args:
        save_args_for_compile_fx_inner(
            gm,
            example_inputs,
            **graph_kwargs,
        )

    start = time.time()

    fx_graph_remote_cache = should_use_remote_fx_graph_cache()

    # Check if the registered backend(s) support caching.
    init_backend_registration()
    backends_support_caching = all(
        backend.supports_caching
        for backend in (
            get_wrapper_codegen_for_device(
                device.type, config.cpp_wrapper, config.fx_wrapper
            )
            for device in get_all_devices(gm)
        )
        if backend is not None
    )

    with dynamo_timed(
        "fx_codegen_and_compile", log_pt2_compile_event=True, log_waitcounter=True
    ):
        use_cache = (
            not config.force_disable_caches
            and (config.fx_graph_cache or fx_graph_remote_cache)
            and not aot_mode
            and backends_support_caching
            and not torch._functorch.config.bundled_autograd_cache
        )
        local = config.fx_graph_cache
        remote = fx_graph_remote_cache
        set_feature_use("fx_cache", use_cache)

        log.debug(
            "FX cache status: use_cache=%s, local=%s, remote=%s, aot_mode=%s, force_disable_caches=%s",
            use_cache,
            local,
            remote,
            aot_mode,
            config.force_disable_caches,
        )

        # TODO: This is a hack purely to get some info to extract_tensor_metadata_for_cache_key,
        # figure out how to not have to modify example inputs
        for i, input in enumerate(example_inputs):
            if (
                isinstance(input, torch.Tensor)
                and is_gpu(input.device.type)
                and i in static_input_idxs
            ):
                input._is_inductor_static = True  # type: ignore[attr-defined]

        mb_compiled_graph: Optional[OutputCode] = None
        key_info = None
        cache_info = None
        remote_cache = None
        constants = CompiledFxGraphConstantsWithGm(gm)
        # TODO: this time will be slightly inconsistent with the one computed
        # in prepare_key/load_with_key, dump those settings of "cache_event_time"
        start_time = time.time_ns()

        if use_cache:
            (key_info, cache_info) = FxGraphCache.prepare_key(
                gm, example_inputs, graph_kwargs, inputs_to_check, remote
            )

            # Attempt a cache lookup
            if key_info is not None:
                key, debug_lines = key_info
                log.debug("FX cache key generated: %s", key)
                if remote:
                    remote_cache = FxGraphCache.get_remote_cache()
                    log.debug("Using remote FX cache")
                mb_compiled_graph, cache_info = FxGraphCache.load_with_key(
                    key,
                    debug_lines,
                    example_inputs,
                    local,
                    remote_cache,
                    is_backward=graph_kwargs.get("is_backward", False),
                    constants=constants,
                )
            else:
                log.debug("Failed to generate FX cache key")

        if torch._functorch.config.bundled_autograd_cache:
            assert mb_compiled_graph is None
            assert cache_info is None
            # When using bundled autograd cache, we still want
            # to use the TritonBundler, but we don't want to save
            # the results here. The results will get saved directly
            # to AOTAutogradCache.
            TritonBundler.begin_compile()
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm, example_inputs, inputs_to_check, **graph_kwargs
                )
                assert mb_compiled_graph is not None
                (
                    triton_bundle,
                    triton_bundler_meta,
                ) = TritonBundler.collect()
                mb_compiled_graph.set_triton_bundle(triton_bundle)
            except (ShortenTraceback, SkipFrame):
                raise
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None
            finally:
                TritonBundler.end_compile()

        # CACHE BYPASS: Compile the graph, don't save it to the cache
        # (this can happen either because cache was disabled, or we
        # determined the input is uncacheable)
        elif cache_info is None or cache_info["cache_state"] == "bypass":
            assert mb_compiled_graph is None
            log.debug(
                "FX cache bypass reason: %s",
                (
                    cache_info.get("cache_bypass_reason", "unknown")
                    if cache_info is not None
                    else "FX cache disabled or key generation failed"
                ),
            )
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm, example_inputs, inputs_to_check, **graph_kwargs
                )
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None

        # CACHE MISS: Compile the graph and save to cache
        elif cache_info["cache_state"] == "miss":
            assert mb_compiled_graph is None
            assert key_info is not None
            log.debug("FX cache miss, compiling and saving to cache")
            TritonBundler.begin_compile()
            try:
                mb_compiled_graph = fx_codegen_and_compile(
                    gm, example_inputs, inputs_to_check, **graph_kwargs
                )
                assert mb_compiled_graph is not None
                mb_compiled_graph._time_taken_ns = time.time_ns() - start_time
                cache_key, debug_lines = key_info
                mb_compiled_graph._fx_graph_cache_key = cache_key
                mb_compiled_graph._fx_graph_cache_debug_lines = debug_lines
                (
                    triton_bundle,
                    triton_bundler_meta,
                ) = TritonBundler.collect()
                mb_compiled_graph.set_triton_bundle(triton_bundle)
            except (ShortenTraceback, SkipFrame):
                raise
            except Exception as e:
                raise InductorError(e, currentframe()).with_traceback(
                    e.__traceback__
                ) from None
            finally:
                TritonBundler.end_compile()
            if triton_bundler_meta is not None:
                cache_info["triton_bundler_meta"] = str(triton_bundler_meta)
            cache_info["time_taken_ns"] = mb_compiled_graph._time_taken_ns
            log.debug("Saving compiled graph to FX cache with key: %s", cache_key)
            FxGraphCache._save_graph(
                cache_key,
                mb_compiled_graph,
                example_inputs,
                local,
                remote_cache,
            )

        # CACHE HIT: not much to really do, just make sure the cache key
        # is recorded on the graph
        else:
            assert cache_info["cache_state"] == "hit"
            assert mb_compiled_graph is not None
            assert key_info is not None
            (cache_key, debug_lines) = key_info
            log.debug("FX cache hit with key: %s", cache_key)
            mb_compiled_graph._fx_graph_cache_key = cache_key
            mb_compiled_graph._fx_graph_cache_debug_lines = debug_lines

        assert mb_compiled_graph is not None
        compiled_graph = mb_compiled_graph

        # Logging and observability: we log a single chromium event
        # and a tlparse log for every cache action.
        # In the event of a bypass, we also logged to the remote table earlier
        # with log_cache_bypass.
        cache_state = (
            cache_info["cache_state"] if cache_info is not None else "disabled"
        )
        # Here for grepping:
        # fx_graph_cache_hit
        # fx_graph_cache_miss
        # fx_graph_cache_bypass
        # fx_graph_cache_disabled
        CompileEventLogger.instant(
            f"fx_graph_cache_{cache_state}",
            metadata=cache_info or {},
            time_ns=start_time,
        )
        # Add event data about cache hits/miss
        # TODO: add remote cache get/put timings here too
        CompileEventLogger.pt2_compile(
            "inductor_compile",
            cache_state=cache_state,
            cache_event_time=start_time,
            key=cache_info.get("key") if cache_info else None,
            components=cache_info.get("components") if cache_info else None,
            cache_bypass_reason=(
                cache_info.get("cache_bypass_reason")
                if cache_info
                else "cache not enabled"
            ),
            remote_cache_enabled=remote,
            local_cache_enabled=local,
        )

        # Don't clog up the main tlparse output with disabled cache
        if cache_info is not None:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": f"fx_graph_cache_{cache_state}",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )
        compiled_graph.post_compile(example_inputs, constants, graph_kwargs)

    log.debug("FX codegen and compilation took %.3fs", time.time() - start)

    # This message is for printing overview information of inductor mm counts, shapes,etc after lowering
    if log.isEnabledFor(logging.INFO):
        mm_table_data = []
        for key, value in counters["aten_mm_info"].items():
            parts = key.split("_")
            if len(parts) < 3:
                # Unexpected format, show as-is
                mm_table_data.append([key, "-", "?", "?", "?", value])
                continue

            # Determine if this is a batched operation by checking the operation name
            name = "_".join(parts[:-4]) if len(parts) >= 4 else "_".join(parts[:-3])
            is_batched = name.endswith(("bmm", "baddbmm"))

            if is_batched and len(parts) >= 4:
                # Batched operation: last 4 parts are batch, m, n, k
                batch, m, n, k = parts[-4:]
                name = "_".join(parts[:-4])
                mm_table_data.append([name, batch, m, n, k, value])
            else:
                # Non-batched operation: last 3 parts are m, n, k
                m, n, k = parts[-3:]
                name = "_".join(parts[:-3])
                mm_table_data.append([name, "-", m, n, k, value])

        log.info("Overview info of inductor aten mms: ")
        log.info(
            "{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(  # noqa: G001
                "Name", "B", "M", "N", "K", "Count"
            )
        )
        log.info("-" * 130)
        for row in mm_table_data:
            # pyrefly: ignore [not-iterable]
            log.info("{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(*row))  # noqa: G001
            log.info("-" * 130)

    # Not strictly necessary, but good to clean up straggling futures
    # that are unused to reclaim memory.
    torch._inductor.async_compile.CompiledTritonKernels.cache_clear()

    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if graph_kwargs['is_backward'] else 'FORWARDS'} "
        f"graph {graph_kwargs['graph_id']}",
    )
    return compiled_graph


class _FxCompileStat:
    # Count of successful compiles of this type
    codegen_and_compile: int = 0

    def __repr__(self) -> str:
        return f"codegen_and_compile: {self.codegen_and_compile}"


class FxCompile(ABC):
    """
    An FxCompile represents a mechanism that can turn a GraphModule into an
    OutputCode.
    """

    # Some stats for logging/debugging
    _compile_stats: dict[type[FxCompile], _FxCompileStat] = defaultdict(_FxCompileStat)

    # TODO: We should probably eventually add some kind of async version of this
    # so we can kick off a compile and then go do other things - but we'll need
    # to know what kind of API we want for that first.
    @abstractmethod
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode: ...

    @classmethod
    def _reset_stats(cls) -> None:
        cls._compile_stats.clear()


class _InProcessFxCompile(FxCompile):
    @override
    def codegen_and_compile(
        self,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        inputs_to_check: Sequence[int],
        graph_kwargs: _CompileFxKwargs,
    ) -> OutputCode:
        """
        Generates the OutputCode from the GraphModule and example_inputs.
        """
        # Sorry about the mess, we need graph_kwargs to continue to be able
        # to propagate it further on
        # TODO: _CompileFxKwargs actually has stronger types than in the
        # signature, need to tighten it up

        assert "cudagraphs" in graph_kwargs and graph_kwargs["cudagraphs"] is not None
        cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]
        static_input_idxs: Sequence[int] = graph_kwargs.get("static_input_idxs", ())
        is_backward: bool = graph_kwargs.get("is_backward", False)
        graph_id: Optional[int] = graph_kwargs.get("graph_id", None)
        cpp_wrapper: bool = graph_kwargs.get("cpp_wrapper", False)
        fx_wrapper: bool = graph_kwargs.get("fx_wrapper", False)
        aot_mode: bool = V.aot_compilation
        is_inference: bool = graph_kwargs.get("is_inference", False)
        extern_node_serializer: Optional[Callable[[list[ExternKernelNode]], Any]] = (
            graph_kwargs.get("extern_node_serializer", None)
        )

        with (
            _WaitCounter("pytorch.wait_counter.actual_codegen_and_compile").guard(),
            dynamo_utils.preserve_rng_state(),
        ):
            if (sleep_sec := config.sleep_sec_TESTING_ONLY) is not None:
                import time

                log.warning(
                    "Sleeping for %s since sleep_sec_TESTING_ONLY is set", sleep_sec
                )
                time.sleep(sleep_sec)

            if is_tf32_warning_applicable(gm):
                _warn_tf32_disabled()

            inductor_counters = counters["inductor"].copy()

            # lift the maximum depth of the Python interpreter stack
            # to adapt large/deep models
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))

            _step_logger()(
                logging.INFO,
                "torchinductor compiling "
                f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
                f"graph {graph_id}",
            )

            fd = io.StringIO()
            torch._dynamo.repro.after_aot.save_graph_repro(
                fd, gm, example_inputs, "inductor", save_dir=None
            )
            runnable_graph_str = fd.getvalue()

            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_runnable",
                    "encoding": "string",
                },
                payload_fn=lambda: runnable_graph_str,
            )

            V.debug.fx_graph(gm, example_inputs)
            # TODO: Should we actually dump this?  It should be redundant with the aot
            # structured logs...
            # trace_structured("inductor_input_graph", payload_fn=lambda: gm.print_readable(print_output=False))

            shape_env = gm.shape_env
            if shape_env is None:
                shape_env = shape_env_from_inputs(example_inputs)

            # Convert view to reshape in the graph. This is necessary primarily for
            # layout optimization. Do it unconditionally for uniformity.
            #
            # It's needed because when we do layout optimization, an contiguous tensor
            # in eager mode may becomes a channels last tensor. A view op previously
            # can be applied to the contiguous tensor may not be able to be applied
            # on the channels tensor any more. An error like
            #   RuntimeError: view size is not compatible with input tensor's size and stride
            #   (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            # will be printed.
            #
            # Replace view op to reshape op in this case.
            # As an example, timm_resnest/botnet26t_256/convnext_base etc. will fail if we don't do this.
            #
            # Also this has to be done before FakeTensorProp below to avoid the failed
            # .view() call.
            view_to_reshape(gm)

            with dynamo_timed(
                "additional_fake_tensor_prop", log_pt2_compile_event=True
            ):
                # It is safe to run FakeTensorProp under no_grad because by the time
                # we're in inductor, we assume that AOTAutograd has already "taken care"
                # of autograd, so there should be no more autograd-related API's in the
                # graph.
                with torch.no_grad():
                    fake_mode = fake_tensor_prop(gm, example_inputs)

            _recursive_record_original_output_strides(gm)

            # pattern matcher passes might not preserve striding information
            # on node.meta["val"]. if in the future we rely on these being
            # correct we will need to fix.
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "before_post_grad_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: gm.print_readable(
                    print_output=False, include_stride=True, include_device=True
                ),
            )
            with V.set_fake_mode(fake_mode):
                # has some issues with memory in training
                cuda_context = get_cuda_device_context(gm)
                with cuda_context:
                    _recursive_post_grad_passes(gm, is_inference=is_inference)
                V.debug.fx_graph_transformed(gm, example_inputs)
                post_grad_graphs_log.debug(
                    "%s",
                    lazy_format_graph_code(
                        "AFTER POST GRAD",
                        gm,
                        include_stride=True,
                        include_device=True,
                        colored=True,
                    ),
                )

                # We're printing the graph to be used as a cache key - so a
                # printer which is a little less readable but faster is
                # appropriate.
                inductor_post_grad_graph_str = gm.print_readable(
                    print_output=False,
                    include_stride=True,
                    include_device=True,
                    fast_sympy_print=True,
                )
                # "inductor_post_grad_graph" is used in inductor provenance
                # tracking highlighter front-end.
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "inductor_post_grad_graph",
                        "encoding": "string",
                    },
                    payload_fn=lambda: inductor_post_grad_graph_str,
                )
                if config.
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

- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

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

- **File Documentation**: `compile_fx.py_docs.md_docs.md`
- **Keyword Index**: `compile_fx.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/export/_trace.py`

## File Metadata

- **Path**: `torch/export/_trace.py`
- **Size**: 96,752 bytes (94.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import logging
import re
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager, ExitStack, nullcontext
from itertools import chain
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union
from unittest import mock


if TYPE_CHECKING:
    import weakref

import torch
import torch._dynamo
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.db.logging import (
    exportdb_error_message,
    get_class_if_classified_error,
)
from torch._export.non_strict_utils import (
    _fakify_module_inputs,
    _fakify_script_objects,
    _gather_constant_attrs,
    _NonStrictTorchFunctionHandler,
    _override_builtin_ops,
    make_constraints,
    make_fake_inputs,
    produce_guards_and_solve_constraints,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.passes.lift_constants_pass import (
    _materialize_and_lift_constants,
    ConstantAttrMap,
)
from torch._export.utils import (
    _collect_param_buffer_metadata,
    _compiling_state_context,
    _fakify_params_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _update_gm_meta_if_possible,
    apply_runtime_assertion_pass,
    placeholder_naming_pass,
    placeholder_prefixes,
)
from torch._export.verifier import SpecViolationError
from torch._export.wrappers import _wrap_submodules
from torch._functorch._aot_autograd.graph_capture_wrappers import create_functional_call
from torch._functorch._aot_autograd.input_output_analysis import (
    _graph_input_names,
    _graph_output_names,
)
from torch._functorch._aot_autograd.schemas import GraphSignature
from torch._functorch._aot_autograd.subclass_utils import get_subclass_typing_container
from torch._functorch._aot_autograd.utils import (
    create_tree_flattened_fn,
    register_buffer_assignment_hook,
)
from torch._functorch.aot_autograd import (
    _detect_attribute_assignment,
    aot_export_joint_with_descriptors,
)
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import dtrace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._utils_internal import log_export_usage
from torch.export._leakage_detection_utils import find_legit_leaks_from_referrers
from torch.export._unlift import _check_input_constraints_pre_hook
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _combine_args,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
)
from torch.export.exported_program import OutputKind
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.experimental.proxy_tensor import (
    get_proxy_slot,
    make_fx,
    PreDispatchTorchFunctionMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeInfo
from torch.utils._pytree import TreeSpec
from torch.utils._sympy.value_ranges import ValueRangeError

from .exported_program import (
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import _convert_to_export_graph_signature, ExportGraphSignature


log = logging.getLogger(__name__)

# Type alias for dynamic shapes specification
_DynamicShapesSpec: TypeAlias = Union[dict[str, Any], tuple[Any, ...], list[Any]]


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """

    allow_rnn: bool = True
    reorderable_logging_functions: set[Callable] = dataclasses.field(
        default_factory=set
    )
    # Emit runtime asserts after AOTAutograd instead.
    # This isn't really necessary, and isn't much more efficient since the runtime asserts pass does CSE,
    # but if we want to reason more about what guards/runtime asserts to emit,
    # this makes it a bit cleaner to do from the export side. Also no real point in running this twice.
    do_not_emit_runtime_asserts: bool = True
    specialize_int: bool = True
    specialize_float: bool = True
    assume_static_by_default: bool = False
    automatic_dynamic_shapes: bool = False
    capture_dynamic_output_shape_ops: bool = True
    capture_scalar_outputs: bool = True
    prefer_deferred_runtime_asserts_over_guards: bool = False
    replay_side_effects: bool = False
    side_effect_replay_policy: str = "warn"


@dataclasses.dataclass
class ATenExportArtifact:
    gm: torch.fx.GraphModule
    sig: ExportGraphSignature
    constants: dict[str, _ConstantAttributeType]


@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    aten: ATenExportArtifact
    in_spec: TreeSpec
    out_spec: TreeSpec
    fake_mode: FakeTensorMode
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]]


DEFAULT_EXPORT_DYNAMO_CONFIG = ExportDynamoConfig()
DEFAULT_EXPORT_DYNAMO_CONFIG.reorderable_logging_functions = {
    logging.critical,
    logging.debug,
    logging.error,
    logging.exception,
    logging.info,
    logging.log,
    logging.warning,
    print,
    warnings.warn,
}


@contextmanager
def _ignore_backend_decomps():
    orig_mkldnn_flag = torch.backends.mkldnn.set_flags(False)
    orig_nnpack_flag = torch.backends.nnpack.set_flags(False)
    try:
        yield
    finally:
        torch.backends.mkldnn.set_flags(*orig_mkldnn_flag)
        torch.backends.nnpack.set_flags(*orig_nnpack_flag)


@contextmanager
def _disable_custom_triton_op_functional_decomposition():
    old = torch._functorch.config.decompose_custom_triton_ops
    try:
        # pyrefly: ignore [bad-assignment]
        torch._functorch.config.decompose_custom_triton_ops = False
        yield torch._functorch.config.decompose_custom_triton_ops
    finally:
        torch._functorch.config.decompose_custom_triton_ops = old


def custom_triton_ops_decomposition_disabled():
    return not torch._functorch.config.decompose_custom_triton_ops


def _fixup_key(x):
    return "L__self__" + _strip_root(x)


def _strip_root(x):
    if isinstance(x, str) and x.startswith("_export_root"):
        stripped = x[len("_export_root") :]
        return stripped.removeprefix(".")
    return x


def _is_bogus_const_name(name: str):
    splitted_names = name.split(".")
    if len(splitted_names) < 1:
        return True

    return splitted_names[-1].startswith("lifted_tensor")


def _rewrite_tracepoint_node(gm: torch.fx.GraphModule):
    """
    In-place modify input graph module by replacing the export tracepoint with a new node
    that has the same target and args, but with the _export_root stripped from path.
    """
    for node in gm.graph.nodes:
        if node.target is torch.ops.higher_order._export_tracepoint:
            if "path" in node.kwargs:
                path = _strip_root(node.kwargs["path"])
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.create_node(
                        "call_function",
                        torch.ops.higher_order._export_tracepoint,
                        args=node.args,
                        kwargs={
                            "path": path,
                            "kind": node.kwargs["kind"],
                        },
                    )
                    new_node.meta = node.meta
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(node)


def detect_shape_env(inputs: Any = None):
    shape_envs = []

    for i, flat_input in enumerate(inputs):
        if isinstance(flat_input, torch.SymInt):
            shape_envs.append((flat_input.node.shape_env, "symint input", i))

    if shape_envs:
        shape_env, desc1, i1 = shape_envs[0]
        for m, desc2, i2 in shape_envs[1:]:
            assert shape_env is m, (
                f"shape env ({shape_env}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                f"shape env from {desc1} {i1} allocated at:\n{shape_env.stack}\n"
                f"shape env from {desc2} {i2} allocated at:\n{m.stack}"
            )
        return shape_env
    else:
        return None


def _extract_fake_inputs(gm, args, kwargs):
    """
    Given a graph module, extract fakified input tensors from the metadata of
    its placeholders, and map them to the structure of given args and kwargs.
    Also return the fake mode used to fakify those inputs.
    """
    fake_inps: list[Any] = []
    fake_vals: list[Any] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            fake_inps.append(node.meta.get("val"))
        else:
            fake_vals.append(node.meta.get("example_value"))

    if in_shuffle_graph := getattr(gm, "_in_shuffle_graph", None):
        flat_args = pytree.tree_leaves((args, kwargs))
        node_map = {
            node: i
            for i, node in enumerate(
                next(iter(reversed(in_shuffle_graph.graph.nodes))).args[0]
            )
            if node.op == "placeholder"
        }
        new_fake_inps: list[Any] = []
        for i, node in enumerate(
            in_shuffle_graph.graph.find_nodes(op="placeholder")[1:]
        ):
            if node in node_map:
                new_fake_inps.append(fake_inps[node_map[node]])
            else:
                new_fake_inps.append(flat_args[i])
        fake_inps = new_fake_inps
    # We get both because now we might have a combination of symint and tensor
    # inputs, and we want to check that the shape env is consistent between
    # both. Unfortunately we can't see what fake mode is attached to the shape
    # env, then we can just compare fake modes.
    detected_fake_mode = detect_fake_mode(fake_inps + fake_vals)
    detected_shape_env = detect_shape_env(fake_inps + fake_vals)

    if detected_fake_mode:
        if detected_shape_env:
            assert detected_shape_env is detected_fake_mode.shape_env, (
                "Detected shape env does not match fake mode's shape env"
            )
        fake_mode = detected_fake_mode
    elif detected_shape_env:
        fake_mode = FakeTensorMode(shape_env=detected_shape_env, export=True)
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

    count = 0

    def lookup_fake(x):
        nonlocal count
        val = fake_inps[count] if isinstance(x, (int, torch.Tensor)) else x
        count += 1
        return val

    fake_args = pytree.tree_map(lookup_fake, args)
    fake_kwargs = pytree.tree_map(lookup_fake, kwargs)

    return fake_args, fake_kwargs, fake_mode


def _replace_param_buffer_names(param_buffer_table, sig):
    for spec in sig.input_specs:
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            spec.target = param_buffer_table[spec.target]
    for spec in sig.output_specs:
        if spec.kind in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.GRADIENT_TO_PARAMETER,
        ):
            spec.target = param_buffer_table[spec.target]


def _convert_to_positional_args(orig_arg_names, args, kwargs):
    assert len(orig_arg_names) == len(args) + len(kwargs), (
        f"Total number of arg names is expected to be {len(orig_arg_names)} "
        f"but got {len(args)} positional args, {len(kwargs)} kwargs."
    )
    reordered_kwargs = [kwargs[kw_name] for kw_name in orig_arg_names[len(args) :]]
    return (
        *args,
        *reordered_kwargs,
    )


def _normalize_nn_module_stack(gm_torch_level, root_cls):
    # Append a root module to every nn_module_stack.
    root = "L['self']"
    root_key = re.sub(r"[^a-zA-Z0-9]", "_", root)
    for gm in gm_torch_level.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            add_root = True
            if nn_module_stack := node.meta.get("nn_module_stack", {}):
                path, ty = next(iter(nn_module_stack.values()))
                # After deserializing the class `ty` might not exist anymore so
                # it could be a string
                if inspect.isclass(ty) and issubclass(ty, torch.nn.Module):
                    # TODO Figure out why sometimes we have root sometimes we don't.
                    if path == root and ty is root_cls:
                        add_root = False
                else:
                    assert isinstance(ty, str)
            if add_root:

                def normalize_path(path):
                    if path == "L['self']":
                        return ""
                    if path.startswith("L['self']."):
                        return path[len("L['self'].") :]
                    return path

                nn_module_stack = {
                    root_key: (root, root_cls.__module__ + "." + root_cls.__qualname__),
                    # pyrefly: ignore [unbound-name]
                    **nn_module_stack,
                }
                node.meta["nn_module_stack"] = {
                    key: (normalize_path(path), ty)
                    for key, (path, ty) in nn_module_stack.items()
                }


def _get_param_buffer_mapping(
    original_module: torch.nn.Module,
    traced_module: torch.nn.Module,
) -> dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """

    param_lookup: dict[int, str] = {}
    buffer_lookup: dict[int, str] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        if param_lookup.get(id(param)) is None:
            # we only want to keep the first occurrence of a parameter to guarantee parity of original and traced module.
            param_lookup[id(param)] = name
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup[id(buffer)] = name

    param_buffer_table: dict[str, str] = {}
    for dynamo_name, dynamo_param in traced_module.named_parameters(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_param) in param_lookup:
            param_buffer_table[dynamo_name] = param_lookup[id(dynamo_param)]

    for dynamo_name, dynamo_buffer in traced_module.named_buffers(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_buffer) in buffer_lookup:
            param_buffer_table[dynamo_name] = buffer_lookup[id(dynamo_buffer)]

    return param_buffer_table


def _preserve_requires_grad_pass(
    gm: torch.fx.GraphModule,
    sig: ExportGraphSignature,
    fake_params_buffers: dict[str, torch.Tensor],
    constants: dict[str, _ConstantAttributeType],
    flat_fake_args: list[Any],
):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert len(sig.input_specs) == len(placeholders)
    i = 0
    for node, spec in zip(placeholders, sig.input_specs):
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            assert spec.target is not None
            node.meta["val"].requires_grad = fake_params_buffers[
                spec.target
            ].requires_grad
        elif spec.kind == InputKind.USER_INPUT:
            fake_arg = flat_fake_args[i]
            if isinstance(fake_arg, torch.Tensor):
                node.meta["val"].requires_grad = fake_arg.requires_grad
            i += 1
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            assert spec.target is not None
            constant = constants[spec.target]
            if isinstance(constant, torch.Tensor):
                # If the tensor is not leaf, it should already have a correct requires grad field
                if node.meta["val"].is_leaf:
                    node.meta["val"].requires_grad = constant.requires_grad
                else:
                    assert node.meta["val"].requires_grad == constant.requires_grad
        elif spec.kind in (InputKind.CUSTOM_OBJ, InputKind.TOKEN):
            continue
        else:
            raise AssertionError(spec.kind)


def _remap_constants(
    orig_constant_attrs: ConstantAttrMap,
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
    remap_table: dict[str, list[str]] = {}
    for name, value in constants.items():
        if value in orig_constant_attrs:
            remap_table[name] = orig_constant_attrs[value]

    for spec in graph_signature.input_specs:
        if spec.kind in (
            InputKind.CONSTANT_TENSOR,
            InputKind.CUSTOM_OBJ,
        ):
            orig_target = spec.target
            assert orig_target is not None
            targets = remap_table.get(orig_target, [orig_target])
            spec.target = targets[0]

            constant = constants[orig_target]
            del constants[orig_target]
            for target in targets:
                constants[target] = constant


def _replace_unbacked_bindings(gm: torch.fx.GraphModule) -> None:
    """
    When we run an interpreter-based pass over a GraphModule, execution of data-dependent operators
    will produce example values with new unbacked symbols. To track that the new/old symbols are equivalent,
    we used to rely on the unbacked_renamings mapping. This led to problematic metadata where the unbacked_bindings
    keys mapped new symbols (u2) to paths containing old symbols (u0) in the example values, or worse, backed symbols
    or constants (e.g. if the original unbacked was replaced/specialized). Additionally this created problems with
    de/serialized programs, since we didn't comprehensively serialize ShapeEnv/unbacked renamings/node bindings.

    This pass attempts a simpler way of handling these for export, by throwing away the previously computed bindings, and re-running
    the pattern match used in compute_unbacked_bindings. This ensures we keep the original symbols contained in the example values,
    or delete bindings if they've been replaced/specialized.
    """
    from torch._export.utils import _get_shape_env_from_gm
    from torch.fx.experimental.symbolic_shapes import _free_unbacked_symbols_with_path
    from torch.utils._sympy.symbol import symbol_is_type, SymT

    if (shape_env := _get_shape_env_from_gm(gm)) is None:
        return

    base_unbacked_symbols = {
        symbol
        for symbol in shape_env.var_to_range
        if symbol_is_type(symbol, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
        and symbol not in shape_env.unbacked_renamings
    }
    for node in gm.graph.nodes:
        node.meta.pop("unbacked_bindings", None)
        if (val := node.meta.get("val")) is not None and (
            unbacked_bindings := _free_unbacked_symbols_with_path(
                val,
                (),
                shape_env=shape_env,
                pending=base_unbacked_symbols,
                simplify=True,
            )
        ):
            node.meta["unbacked_bindings"] = unbacked_bindings


def _produce_aten_artifact(
    *,
    gm: torch.fx.GraphModule,
    mod,
    constant_attrs,
    graph_signature,
    pre_dispatch,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    _prettify_placeholder_names=True,
) -> ATenExportArtifact:
    """
    This is a helper function that is shared between export_to_aten_ir and export_to_aten_ir_make_fx
    to produce the aten artifact. (export compatible graph module + signature)

    It does:
    1. Applies runtime assertion pass
    2. Recompute unbacked_bindings pass
    3. Populate meta val when missing
    4. Lift constants as placeholders
    5. Replace raw autograd and autocast ops with HOPs
    6. Prettify names for placeholders
    7. Preserve requires_grad value on node meta val
    """
    # Run runtime asserts pass before creating input/output specs, since size-related CSE/DCE might affect output signature.
    # Overwrite output specs afterwards.
    flat_fake_args = pytree.tree_leaves((fake_args, fake_kwargs))
    gm, graph_signature = apply_runtime_assertion_pass(gm, graph_signature)

    # Simplify unbacked_bindings by recomputing them.
    # Useful for any pass that's interpreter-based and might call rebind_unbacked(),
    # e.g. AOTAutograd in this case.
    _replace_unbacked_bindings(gm)

    total_non_user_inputs = (
        len(graph_signature.parameters)
        + len(graph_signature.buffers)
        + len(graph_signature.input_tokens)
    )
    set_missing_meta_vals(gm, flat_fake_args, total_non_user_inputs)

    export_graph_signature: Optional[ExportGraphSignature]
    export_graph_signature = _convert_to_export_graph_signature(
        graph_signature, gm, _get_non_persistent_buffers(mod)
    )

    # script objects are always stored in constants no matter whether they're initial inputs or
    # they're lifted in aot" before rewrite_script_object_meta
    constants = _materialize_and_lift_constants(
        gm, export_graph_signature, constant_attrs
    )

    if pre_dispatch:
        from torch._export.passes.replace_autocast_with_hop_pass import (
            replace_autocast_with_hop_pass,
        )
        from torch._export.passes.replace_set_grad_with_hop_pass import (
            replace_set_grad_with_hop_pass,
        )

        # Note: replace_set_grad_with_hop_pass need to be after lift_constant_pass because
        # a getattr of a constant tensor doesn't have meta["val"] until after lift_constant_pass.
        # If replace_set_grad_with_hop_pass is before lift_constant_pass,
        # and the constant_tensor is passed as input of the set grad hop, the placeholder's
        # meta["val"] will be None and fails our verifier for placeholder.
        gm, export_graph_signature = replace_set_grad_with_hop_pass(
            gm, export_graph_signature
        )

        gm, export_graph_signature = replace_autocast_with_hop_pass(
            gm, export_graph_signature
        )

    # Remove nn_module_stack, stack_trace metadata from all placeholders/inputs nodes.
    for _mod in gm.modules():
        if not isinstance(_mod, torch.fx.GraphModule):
            continue
        for node in _mod.graph.nodes:
            if node.op in ["placeholder", "output"]:
                node.meta.pop("nn_module_stack", None)
                node.meta.pop("stack_trace", None)

    # Prettify names for placeholder nodes.
    assert export_graph_signature is not None
    if _prettify_placeholder_names:
        placeholder_naming_pass(
            gm,
            export_graph_signature,
            mod,
            fake_args,
            fake_kwargs,
            fake_params_buffers,
            constants,
        )

    _preserve_requires_grad_pass(
        gm, export_graph_signature, fake_params_buffers, constants, flat_fake_args
    )

    return ATenExportArtifact(
        gm,
        export_graph_signature,
        constants,
    )


def _rename_constants_nodes(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> None:
    """
    For strict mode, rename constants nodes that were previously annotated as buffers.
    """
    # handle name collisions with existing constants
    node_names = {node.name for node in gm.graph.nodes}

    def rename_constant(name):
        if name in node_names:
            n = 1
            while (dup_name := f"{name}_{n}") in node_names:
                n += 1
            name = dup_name
        node_names.add(name)
        return name

    # use input specs to map names from buffers to constants
    buffer_prefix = placeholder_prefixes[InputKind.BUFFER]
    const_prefix = placeholder_prefixes[InputKind.CONSTANT_TENSOR]
    buffer_to_constant = {}
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.CONSTANT_TENSOR and not spec.arg.name.startswith(
            const_prefix
        ):
            if spec.arg.name.startswith(buffer_prefix):  # map from buffer to constants
                c_name = rename_constant(
                    const_prefix + spec.arg.name[len(buffer_prefix) :]
                )
            else:  # lifted constant
                c_name = rename_constant(const_prefix + spec.arg.name)
            buffer_to_constant[spec.arg.name] = c_name
            spec.arg.name = c_name
    for spec in graph_signature.output_specs:
        if spec.arg.name in buffer_to_constant:
            spec.arg.name = buffer_to_constant[spec.arg.name]

    # Rename constants nodes for all modules
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.name in buffer_to_constant:
                node.name = node.target = buffer_to_constant[node.name]
        mod.recompile()


def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    Restores the state dict of the traced module to that of the original module.
    """
    param_buffer_table = _get_param_buffer_mapping(original_module, traced_module)
    # Don't want to change the convention of previous call.
    param_buffer_table_reverse = {v: k for k, v in param_buffer_table.items()}

    # Replace state dict attr names with the fqn
    for name, _ in list(
        chain(
            original_module.named_parameters(remove_duplicate=False),
            # pyrefly: ignore [bad-argument-type]
            original_module.named_buffers(remove_duplicate=False),
        )
    ):
        if name in param_buffer_table_reverse:
            dynamo_name = param_buffer_table_reverse[name]
            param = torch.fx.graph_module._get_attr(traced_module, dynamo_name)
            torch.fx.graph_module._assign_attr(param, traced_module, name)
            torch.fx.graph_module._del_attr(traced_module, dynamo_name)

    # Replace graph getattr nodes with the correct name
    for node in traced_module.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            if attr_name in param_buffer_table:
                node.target = param_buffer_table[attr_name]

    traced_module.recompile()


def _get_module_hierarchy(mod: torch.nn.Module) -> dict[str, str]:
    return {
        name: type(m).__name__ for name, m in mod.named_modules(remove_duplicate=False)
    }


def _make_module_call_graph(
    in_spec: TreeSpec,
    out_spec: TreeSpec,
    module_call_signatures: dict[str, ModuleCallSignature],
    forward_arg_names: Optional[list[str]] = None,
) -> list[ModuleCallEntry]:
    original = [
        ModuleCallEntry(fqn=fqn, signature=module_call_signatures.get(fqn))
        for fqn in _EXPORT_MODULE_HIERARCHY  # type: ignore[union-attr]
    ]
    assert original[0].fqn == ""
    original[0].signature = ModuleCallSignature(
        inputs=[],
        outputs=[],
        in_spec=in_spec,
        out_spec=out_spec,
        forward_arg_names=forward_arg_names,
    )
    additional = [
        ModuleCallEntry(fqn=fqn, signature=signature)
        for fqn, signature in module_call_signatures.items()
        if fqn not in _EXPORT_MODULE_HIERARCHY  # type: ignore[operator]
    ]
    return [*original, *additional]


class _ExportModuleSpecTrackerDict(dict):
    pass


def _export_to_torch_ir(
    f: Callable,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    *,
    preserve_module_call_signature: tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    restore_fqn: bool = True,
    _log_export_usage: bool = True,
    same_signature: bool = True,
) -> torch.fx.GraphModule:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a torch.fx.GraphModule in torch IR.
    """

    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_export_to_torch_ir"})

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    kwargs = kwargs or {}

    # Map ints to a wrapper structure to help us mark it as dynamic, if it is
    # dynamic. We will unwrap ints in fakify later.
    args, kwargs = pytree.tree_map_only(int, _IntWrapper, (args, kwargs))

    combined_args = _combine_args(f, args, kwargs)
    _check_dynamic_shapes(combined_args, dynamic_shapes)
    constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)

    # Unwrap static ints -- in the case where we have an empty graph
    # containing just integer computation, dynamo will run its generated
    # bytecode with these args/kwargs, which will error because we cannot
    # directly apply int operations on IntWrapper. So we will just unwrap
    # them here.
    args, kwargs = pytree.tree_map_only(
        _IntWrapper,
        lambda a: a.val
        if a.dynamism is None or a.dynamism.type == _DimHintType.STATIC
        else a,
        (args, kwargs),
    )

    dynamo_cfg = dataclasses.replace(
        DEFAULT_EXPORT_DYNAMO_CONFIG,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
    )

    def use_legacy_dynamo_graph_capture() -> bool:
        return bool(
            constraints  # dynamic shape
            or dynamic_shapes  # dynamic shape
            or isinstance(f, torch.fx.GraphModule)  # retracing
            or preserve_module_call_signature  # unflatten
            or torch._functorch.config.fake_tensor_propagate_real_tensors  # draft
            or torch._export.config.use_legacy_dynamo_graph_capture
        )

    with torch._dynamo.config.patch(dataclasses.asdict(dynamo_cfg)):
        try:
            module_call_specs: dict[str, dict[str, pytree.TreeSpec]] = (
                _ExportModuleSpecTrackerDict()
            )
            ctx = nullcontext()
            if not isinstance(f, torch.fx.GraphModule):
                ctx = _wrap_submodules(  # type: ignore[assignment]
                    f, preserve_module_call_signature, module_call_specs
                )
            with ctx, _ignore_backend_decomps():
                if torch._export.config.use_new_tracer_experimental:
                    from torch._dynamo.functional_export import (
                        _dynamo_graph_capture_for_export,
                        dynamo_graph_capture_for_export,
                    )

                    if use_legacy_dynamo_graph_capture():
                        dynamo_graph_capture = _dynamo_graph_capture_for_export(
                            f, constraints=constraints, dynamic_shapes=dynamic_shapes
                        )
                    else:
                        dynamo_graph_capture = dynamo_graph_capture_for_export(f)
                    # We can't serialize entire fake mode yet, so this is to make sure
                    # things like copy.deepcopy(ep.graph_module) not crash.
                    # see test_export.py::test_custom_tag_metadata_re_export
                    # Once we delete the old strict export, we can use
                    gm_torch_level = dynamo_graph_capture(*args, **kwargs)
                    # We can't serialize entire fake mode yet, so this is to make sure
                    # things like copy.deepcopy(ep.graph_module) not crash.
                    # see test_export.py::test_custom_tag_metadata_re_export
                    # Once we delete the old strict export, we can use this fake mode in the
                    # subsequent logic when lowering to aten IR.
                    del gm_torch_level.meta["fake_mode"]

                else:
                    gm_torch_level, _ = torch._dynamo.export(
                        f,
                        dynamic_shapes=dynamic_shapes,  # type: ignore[arg-type]
                        constraints=constraints,  # type: ignore[arg-type]
                        assume_static_by_default=True,
                        tracing_mode="symbolic",
                        disable_constraint_solver=disable_constraint_solver,
                        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                        _log_export_usage=_log_export_usage,
                        same_signature=same_signature,
                    )(
                        *args,
                        **kwargs,
                    )
                    gm_torch_level.meta["module_call_specs"] = module_call_specs
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904
        except GuardOnDataDependentSymNode as e:
            raise UserError(  # noqa: B904
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._check*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    return gm_torch_level


def _aot_export_joint_with_descriptors(
    stack,
    mod,
    args,
    *,
    kwargs,
    decompositions,
    fake_params_buffers,
    _record_nn_module_stack=True,
):
    from torch._functorch._aot_autograd.graph_compile import aot_stage2_export
    from torch._functorch._aot_autograd.input_output_analysis import (
        create_graph_signature,
    )

    joint_with_descriptors = aot_export_joint_with_descriptors(
        stack,
        mod,
        args,
        kwargs=kwargs,
        decompositions=decompositions,
        _record_nn_module_stack=_record_nn_module_stack,
    )
    # Convert JointWithDescriptors to graph module and ViewAndMutationMeta
    gm, fw_metadata = aot_stage2_export(
        joint_with_descriptors._aot_state,
        joint_with_descriptors._aot_graph_capture,
    )

    assert isinstance(gm, torch.fx.GraphModule)

    # Create GraphSignature from the metadata
    graph_signature = create_graph_signature(
        gm,
        fw_metadata,
        joint_with_descriptors.in_spec,
        joint_with_descriptors.out_spec,
        user_args_flat=pytree.tree_leaves((args, kwargs)),
        params_and_buffers_flat=list(fake_params_buffers.values()),
        param_names=joint_with_descriptors.params_spec,
        buffer_names=joint_with_descriptors.buffers_spec,
        trace_joint=False,
        num_user_fw_outs=None,
        loss_index=None,
    )
    return gm, graph_signature


def _export_to_aten_ir(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    produce_guards_callback=None,
    *,
    transform=lambda x: x,  # TODO(zhxchen17) Revisit if this is needed later.
    pre_dispatch=False,
    decomp_table=None,
    _prettify_placeholder_names: bool = True,
    decompose_custom_triton_ops: bool = False,
) -> ATenExportArtifact:
    custom_triton_ops_decomposition_ctx = (
        nullcontext
        if decompose_custom_triton_ops
        else _disable_custom_triton_op_functional_decomposition
    )
    # This _reparameterize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with ExitStack() as stack:
        stack.enter_context(
            torch.nn.utils.stateless._reparametrize_module(
                mod,
                fake_params_buffers,
                tie_weights=True,
                strict=True,
                stack_weights=True,
            )
        )
        stack.enter_context(_ignore_backend_decomps())
        stack.enter_context(_compiling_state_context())
        stack.enter_context(custom_triton_ops_decomposition_ctx())
        stack.enter_context(torch.no_grad())

        gm, graph_signature = transform(_aot_export_joint_with_descriptors)(
            stack,
            mod,
            fake_args,
            kwargs=fake_kwargs,
            decompositions=decomp_table,
            fake_params_buffers=fake_params_buffers,
            _record_nn_module_stack=True,
        )

    def _maybe_fixup_gm_and_output_node_meta(old_gm, new_gm):
        if isinstance(old_gm, torch.fx.GraphModule):
            if hasattr(old_gm, "meta"):
                new_gm.meta.update(old_gm.meta)
            old_output_node = list(old_gm.graph.nodes)[-1]
            new_output_node = list(new_gm.graph.nodes)[-1]
            assert old_output_node.op == "output" and new_output_node.op == "output"
            # make sure we don't override any meta
            if "desc" in new_output_node.meta:
                del new_output_node.meta["desc"]
            new_output_node.meta.update(old_output_node.meta)

    # TODO unfortunately preserving graph-level metadata and output node's meta
    # is not working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    _maybe_fixup_gm_and_output_node_meta(mod, gm)

    # Run produce guards before we handle runtime asserts.
    # This means we run the export solver before the runtime asserts pass.
    # Right now this doesn't mean much - the export solver is only there for suggested fixes,
    # and we won't even get to constraint solving if that's needed.
    # But if in future we want to control what runtime asserts are emitted for export,
    # or rely on produce_guards + solver for some simplification on runtime asserts, this probably makes sense.
    if produce_guards_callback:
        try:
            produce_guards_callback(gm)
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904

    return _produce_aten_artifact(
        gm=gm,
        mod=mod,
        constant_attrs=constant_attrs,
        graph_signature=graph_signature,
        pre_dispatch=pre_dispatch,
        fake_args=fake_args,
        fake_kwargs=fake_kwargs,
        fake_params_buffers=fake_params_buffers,
        _prettify_placeholder_names=_prettify_placeholder_names,
    )


def _get_forward_arg_names(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
) -> list[str]:
    """
    Gets the argument names to forward that are used, for restoring the
    original signature when unlifting the exported program module.
    - Positional args: retain the original argument names, and enumerate
        *args as args_0, args_1, ...
    - Keyword args: retain the original kwarg names in the order specified
        by the user. This order seems to matter for the current state of
        export lifted modules.
    """
    sig = inspect.signature(mod.forward)
    _args = sig.bind_partial(*args).arguments

    names: list[str] = []
    for name, value in _args.items():
        # handle variable number of positional args
        if sig.parameters[name].kind == inspect._ParameterKind.VAR_POSITIONAL:
            names.extend([f"{name}_{i}" for i, _ in enumerate(value)])
        else:
            names.append(name)
    # order of kwargs matters for input spec
    if kwargs:
        names.extend([kwarg for kwarg, _ in kwargs.items()])

    return names


def _get_non_persistent_buffers(mod: torch.nn.Module) -> set[str]:
    """
    Returns set of non-persistent buffers in a module and its submodules.
    """
    result: set[str] = set()
    for name, m in mod.named_modules(remove_duplicate=False):
        if name:
            result.update(f"{name}.{b}" for b in m._non_persistent_buffers_set)
        else:
            result.update(m._non_persistent_buffers_set)
    return result


def _rewrite_dynamo_tensor_constants(
    orig_mod_buffers: set[torch.Tensor],
    traced_mod_buffers: dict[str, torch.Tensor],
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """
    Dynamo erroneously marks tensor attributes on modules as buffers.
    Rewrite them to be tensor constants.
    """
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER:
            assert spec.target is not None
            value = traced_mod_buffers[spec.target]
            if value not in orig_mod_buffers:
                # This was a tensor constant erroneously marked as a buffer.
                # Convert it into a constant in the graph signature, and add its
                # value to the constants table.
                spec.kind = InputKind.CONSTANT_TENSOR
                constants[spec.target] = value  # type: ignore[arg-type]


def _move_non_persistent_buffers_to_tensor_constants(
    orig_mod: torch.nn.Module,
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """
    Moves non-persistent buffers to tensor constants.
    """
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER and not spec.persistent:
            assert spec.target is not None
            assert spec.target not in constants
            constants[spec.target] = orig_mod.get_buffer(spec.target)  # type: ignore[arg-type]


def _verify_nn_module_stack(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform nn_module_stack checks on the graph.
    Current constraints:
        For the top level graph:
        - populated for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
        For submodule graphs:
        - None for 'placeholder', output'

    TODO(pianpwk): make this a consistent node-level check once nn_module_stack is populated for cond submodules.
    """
    # Check top-level graph for all nodes, all graphs for placeholder & output nodes
    for i, mod in enumerate([graph_module] + list(graph_module.modules())):
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op in ["call_function", "get_attr"]:
                if i == 0:
                    if (
                        nn_module_stack := node.meta.get("nn_module_stack", None)
                    ) is None:
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} is missing nn_module_stack metadata"
                        )
                    if not all(
                        isinstance(k, str)
                        and isinstance(v, tuple)
                        and len(v) == 2
                        and all(isinstance(x, str) for x in v)
                        for k, v in nn_module_stack.items()
                    ):
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} has incorrect nn_module_stack metadata format"
                            f"expected Dict[str, Tuple[str, str]], but got {nn_module_stack}"
                        )
            elif node.op in ["placeholder", "output"]:
                if node.meta.get("nn_module_stack", None):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains nn_module_stack metadata, this should be None"
                    )


def _verify_stack_trace(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform stack trace checks on the graph.
    Constraints:
        - None or non-empty str for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
    """
    for mod in [graph_module, *graph_module.modules()]:
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in graph_module.graph.nodes:
            stack_trace = node.meta.get("stack_trace", None)
            if node.op in ["call_function", "get_attr"]:
                if not (stack_trace is None or isinstance(stack_trace, str)):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} has invalid stack_trace metadata, "
                        f"expected a string or None but instead found: {stack_trace}"
                    )
            elif node.op in ["placeholder", "output"]:
                if stack_trace:
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains stack_trace metadata, "
                        f"expected None but instead found: {stack_trace}"
                    )


def _verify_placeholder_names(
    gm: torch.fx.GraphModule, sig: ExportGraphSignature
) -> None:
    """
    Performs a sanity check on the placeholder node names.
    - User input nodes: no restrictions, should match the original forward() signature
    - Params/buffers/constants/custom_obj/token nodes: should start with prefixes defined in <placeholder_prefixes>
    """
    name_to_kind = {spec.arg.name: spec.kind for spec in sig.input_specs}
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                if node.name not in name_to_kind:
                    continue
                node_kind = name_to_kind[node.name]
                prefix = placeholder_prefixes[node_kind]
                if not node.name.startswith(prefix):
                    raise SpecViolationError(
                        f"Placeholder node name {node.name} does not follow spec for {node_kind}, name should have prefix: {prefix}"
                    )


def get_ep_stats(ep: ExportedProgram) -> dict[str, Any]:
    op_count = 0
    op_set = set()
    for m in ep.graph_module.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        for node in m.graph.nodes:
            if node.op != "call_function":
                continue
            op_count += 1
            assert hasattr(node.target, "__module__")
            assert hasattr(node.target, "__name__")
            op_set.add(f"{node.target.__module__}.{node.target.__name__}")
    return {"op_count": op_count, "op_set": op_set}


_EXPORT_FLAGS: Optional[set[str]] = None
_EXPORT_MODULE_HIERARCHY: Optional[dict[str, str]] = None


def _log_export_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
        try:
            start = time.time()
            ep = fn(*args, **kwargs)
            end = time.time()
            log_export_usage(
                event="export.time",
                metrics=end - start,
                flags=_EXPORT_FLAGS,
                **get_ep_stats(ep),
            )
        except Exception as e:
            t = type(e)
            error_type = t.__module__ + "." + t.__qualname__
            case_name = get_class_if_classified_error(e)
            if case_name is not None:
                log.error(exportdb_error_message(case_name))
                log_export_usage(
                    event="export.error.classified",
                    type=error_type,
                    message=str(e),
                    flags=_EXPORT_FLAGS,
                )
            else:
                log_export_usage(
                    event="export.error.unclassified",
                    type=error_type,
                    message=str(e),
                    flags=_EXPORT_FLAGS,
                )

            if hasattr(e, "partial_fx_graph"):
                print(
                    e.partial_fx_graph,
                    file=sys.stderr,
                )

            raise e
        finally:
            _EXPORT_FLAGS = None
            _EXPORT_MODULE_HIERARCHY = None

        return ep

    return wrapper


def _process_jit_trace_inputs_for_export(example_inputs, example_kwarg_inputs):
    if not isinstance(example_inputs, (tuple, list, dict)):
        example_inputs = (example_inputs,)

    elif isinstance(example_inputs, list):
        example_inputs = tuple(example_inputs)

    elif (
        isinstance(example_inputs, (torch.Tensor, dict))
        and example_kwarg_inputs is None
    ):
        example_inputs = (example_inputs,)

    if example_kwarg_inputs is None:
        example_kwarg_inputs = {}
    return example_inputs, example_kwarg_inputs


def _get_original_state_dict(mod: torch.nn.Module) -> dict[str, Any]:
    # Explicitly not calling mode.state_dict() as we do not want the module state for serialization
    # but the running module state so we can always match by id() the entries here with the graph inputs
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    original_state_dict = named_parameters | named_buffers

    non_persistent_buffers = _get_non_persistent_buffers(mod)
    for k in non_persistent_buffers:
        original_state_dict.pop(k, None)

    return original_state_dict


def _process_export_inputs(
    mod: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: Optional[dict[str, object]],
    dynamic_shapes: Optional[
        Union[
            _DynamicShapesSpec,
            torch.export.AdditionalInputs,
            torch.export.ShapesCollection,
        ]
    ],
) -> tuple[
    tuple[object, ...],
    dict[str, object],
    TreeSpec,
    Option
```



## High-Level Overview


This Python file contains 9 class(es) and 65 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExportDynamoConfig`, `ATenExportArtifact`, `ExportArtifact`, `_ExportModuleSpecTrackerDict`, `Wrapper`

**Functions defined**: `_ignore_backend_decomps`, `_disable_custom_triton_op_functional_decomposition`, `custom_triton_ops_decomposition_disabled`, `_fixup_key`, `_strip_root`, `_is_bogus_const_name`, `_rewrite_tracepoint_node`, `detect_shape_env`, `_extract_fake_inputs`, `lookup_fake`, `_replace_param_buffer_names`, `_convert_to_positional_args`, `_normalize_nn_module_stack`, `normalize_path`, `_get_param_buffer_mapping`, `_preserve_requires_grad_pass`, `_remap_constants`, `_replace_unbacked_bindings`, `_produce_aten_artifact`, `_rename_constants_nodes`

**Key imports**: dataclasses, functools, inspect, logging, re, sys, time, warnings, Callable, contextmanager, ExitStack, nullcontext


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/export`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `functools`
- `inspect`
- `logging`
- `re`
- `sys`
- `time`
- `warnings`
- `collections.abc`: Callable
- `contextlib`: contextmanager, ExitStack, nullcontext
- `itertools`: chain
- `typing`: Any, Optional, TYPE_CHECKING, TypeAlias, Union
- `unittest`: mock
- `weakref`
- `torch`
- `torch._dynamo`
- `torch.fx`
- `torch.utils._pytree as pytree`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._dynamo.exc`: UserError, UserErrorType
- `torch._export.passes.collect_tracepoints_pass`: CollectTracepointsPass
- `torch._export.verifier`: SpecViolationError
- `torch._export.wrappers`: _wrap_submodules
- `torch._functorch._aot_autograd.graph_capture_wrappers`: create_functional_call
- `torch._functorch._aot_autograd.schemas`: GraphSignature
- `torch._functorch._aot_autograd.subclass_utils`: get_subclass_typing_container
- `torch._guards`: detect_fake_mode, tracing, TracingContext
- `torch._library.fake_class_registry`: FakeScriptObject
- `torch._logging`: dtrace_structured
- `torch._subclasses.fake_tensor`: FakeTensorMode


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_remove_auto_functionalized_pass.py_docs.md`](./_remove_auto_functionalized_pass.py_docs.md)
- [`exported_program.py_docs.md`](./exported_program.py_docs.md)
- [`_wrapper_utils.py_docs.md`](./_wrapper_utils.py_docs.md)
- [`_unlift.py_docs.md`](./_unlift.py_docs.md)
- [`_swap.py_docs.md`](./_swap.py_docs.md)
- [`_tree_utils.py_docs.md`](./_tree_utils.py_docs.md)
- [`_safeguard.py_docs.md`](./_safeguard.py_docs.md)
- [`_remove_effect_tokens_pass.py_docs.md`](./_remove_effect_tokens_pass.py_docs.md)


## Cross-References

- **File Documentation**: `_trace.py_docs.md`
- **Keyword Index**: `_trace.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/_functorch/partitioners.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_functorch/partitioners.py_docs.md`
- **Size**: 53,882 bytes (52.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_functorch/partitioners.py`

## File Metadata

- **Path**: `torch/_functorch/partitioners.py`
- **Size**: 117,875 bytes (115.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
import functools
import hashlib
import heapq
import itertools
import logging
import math
import operator
import os
import os.path
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
import torch._inductor.inductor_prims
import torch.distributed
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters, is_node_meta_valid
from torch._functorch._activation_checkpointing.ac_logging_utils import (
    create_structured_trace_for_min_cut_info,
)
from torch._inductor import config as inductor_config
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    find_symbol_binding_fx_nodes,
    free_symbols,
    hint_int,
    is_symbol_binding_fx_node,
    statically_known_false,
    statically_known_true,
)
from torch.fx.passes import graph_drawer
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy

from . import config
from ._activation_checkpointing.graph_info_provider import GraphInfoProvider
from ._activation_checkpointing.knapsack import (
    dp_knapsack,
    greedy_knapsack,
    ilp_knapsack,
)
from ._activation_checkpointing.knapsack_evaluator import KnapsackEvaluator
from ._aot_autograd.descriptors import AOTOutput, SavedForBackwardsAOTOutput
from ._aot_autograd.logging_utils import get_aot_graph_name
from ._aot_autograd.utils import get_cuda_generator_meta_val, is_with_effects
from .compile_utils import fx_graph_cse, get_aten_target, raise_getitems


if TYPE_CHECKING:
    import sympy


AOT_PARTITIONER_DEBUG: bool = config.debug_partitioner
log: logging.Logger = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims


@dataclass
class OpTypes:
    """Class for keeping track of different operator categories"""

    fusible_ops: OrderedSet[Callable]
    compute_intensive_ops: OrderedSet[Callable]
    random_ops: OrderedSet[Callable]
    view_ops: OrderedSet[Callable]
    recomputable_ops: OrderedSet[Callable]

    def is_fusible(self, node: fx.Node):
        return get_aten_target(node) in self.fusible_ops

    def is_compute_intensive(self, node: fx.Node):
        return get_aten_target(node) in self.compute_intensive_ops

    def is_random(self, node: fx.Node):
        return get_aten_target(node) in self.random_ops

    def is_view(self, node: fx.Node):
        return get_aten_target(node) in self.view_ops

    def is_recomputable(self, node: fx.Node):
        return get_aten_target(node) in self.recomputable_ops


@dataclass
class NodeInfo:
    # Be careful about iterating over these explicitly, as their order may not
    # be deterministic
    inputs: list[fx.Node]
    _required_fw_nodes: OrderedSet[fx.Node]
    required_bw_nodes: OrderedSet[fx.Node]
    unclaimed_nodes: OrderedSet[fx.Node]
    fw_order: dict[fx.Node, int]
    # Effectively maps to which of our primals are parameters
    static_lifetime_input_nodes: OrderedSet[fx.Node]

    @functools.cached_property
    def required_fw_nodes(self) -> list[fx.Node]:
        return sorted(
            (n for n in self._required_fw_nodes), key=lambda n: self.fw_order[n]
        )

    def is_required_fw(self, n: fx.Node) -> bool:
        return n in self._required_fw_nodes

    def is_required_bw(self, n: fx.Node) -> bool:
        return n in self.required_bw_nodes

    def is_unclaimed(self, n: fx.Node) -> bool:
        return n in self.unclaimed_nodes

    def get_fw_order(self, n: fx.Node) -> int:
        assert n in self._required_fw_nodes, f"Node {n} not in fw nodes!"
        return self.fw_order[n]


@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool
    ban_if_long_fusible_chains: bool
    ban_if_materialized_backward: bool
    ban_if_not_in_allowlist: bool
    ban_if_reduction: bool


def must_recompute(node: fx.Node) -> bool:
    return node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]


def has_recomputable_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if must_recompute(node):
            return True
    return False


def has_recomputable_rng_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return True
    return False


def sym_node_size(node: fx.Node) -> int:
    if isinstance(node.meta["val"], (torch.SymInt, torch.SymBool)):
        return 1
    assert isinstance(node.meta["val"], torch.SymFloat)
    return 4


class InvalidNodeBase:
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(
    joint_graph: fx.Graph,
    inputs: list[fx.Node],
    outputs: list[fx.Node],
    outputs_descs: list[AOTOutput],
    subgraph: Optional[str] = None,
    ignore_must_be_in_fw_bw: bool = False,
) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        # Can't use node_copy here as we may be turning previous call_function into placeholders
        new_node.meta = node.meta
        # pyrefly: ignore [unsupported-operation]
        env[node] = new_node

    for node in joint_graph.nodes:
        if not ignore_must_be_in_fw_bw:
            if (
                _must_be_in_backward(node)
                and subgraph != "backward"
                and node not in inputs
            ):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue

            if (
                _must_be_in_forward(node)
                and subgraph != "forward"
                and node not in inputs
            ):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue

        if node in env:
            # Node must be one of our inputs. (Any member of env which wasn't an
            # input to start must have been created by this loop and won't be in
            # joint_graph.nodes).
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode  # type: ignore[assignment]
        elif node.op == "call_function":
            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [
                isinstance(env[x], InvalidNodeBase)
                for x in all_args
                if isinstance(x, fx.Node)
            ]
            if any(all_args):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue
            # pyrefly: ignore [unsupported-operation, bad-argument-type]
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "get_attr":
            # pyrefly: ignore [unsupported-operation, bad-argument-type]
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "output":
            pass
    output_values = []
    for x in outputs:
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            assert not isinstance(env[x], InvalidNodeBase), (
                f"Node {x} was invalid, but is output"
            )
            output_values.append(env[x])
        else:
            output_values.append(x)
    out = new_graph.output(tuple(output_values))
    out.meta["desc"] = outputs_descs

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def _is_primal(node: fx.Node) -> bool:
    return (
        node.op == "placeholder"
        and "tangents" not in str(node.target)
        and not _is_bwd_seed_offset(node)
        and not _is_fwd_seed_offset(node)
    )


def _is_tangent(node: fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" in str(node.target)


def _is_bwd_seed_offset(node: fx.Node) -> bool:
    return node.op == "placeholder" and (
        "bwd_seed" in str(node.target) or "bwd_base_offset" in str(node.target)
    )


def _is_fwd_seed_offset(node: fx.Node) -> bool:
    return node.op == "placeholder" and (
        "fwd_seed" in str(node.target) or "fwd_base_offset" in str(node.target)
    )


def _is_backward_state(node: fx.Node) -> bool:
    return node.op == "placeholder" and isinstance(node.meta.get("val"), BackwardState)


def _has_tag_is_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "is_backward"


def _has_tag_must_be_in_forward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_forward"


def _has_tag_must_be_in_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_backward"


def _must_be_in_forward(node: fx.Node) -> bool:
    if _has_tag_must_be_in_forward(node):
        return True
    is_mutable = is_with_effects(node) or (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target._schema.is_mutable
    )
    return (
        not _has_tag_is_backward(node)
        and not _has_tag_must_be_in_backward(node)
        and is_mutable
    )


def _must_be_in_backward(node: fx.Node) -> bool:
    if _has_tag_must_be_in_backward(node):
        return True
    is_mutable = is_with_effects(node) or (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target._schema.is_mutable
    )
    return _has_tag_is_backward(node) and is_mutable


def _extract_fwd_bwd_outputs(
    joint_module: fx.GraphModule, *, num_fwd_outputs
) -> tuple[list[fx.Node], list[fx.Node], list[AOTOutput], list[AOTOutput]]:
    outputs = pytree.arg_tree_leaves(
        *(node.args for node in joint_module.graph.find_nodes(op="output"))
    )
    outputs_descs = pytree.arg_tree_leaves(
        next(iter(joint_module.graph.find_nodes(op="output"))).meta.get(
            "desc", [None] * len(outputs)
        )
    )
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    fwd_outputs_descs = outputs_descs[:num_fwd_outputs]
    bwd_outputs_descs = outputs_descs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs


def _remove_by_name(saved_values: list[fx.Node], name: str):
    for saved_value in saved_values:
        if saved_value.name == name:
            saved_values.remove(saved_value)
            break


def find_first_sym_node(
    fwd_module_outputs: Union[list[fx.Node], tuple[fx.Node]],
) -> int:
    idx = len(fwd_module_outputs)
    for i in range(len(fwd_module_outputs) - 1, -1, -1):
        if not is_sym_node(fwd_module_outputs[i]):
            idx = i + 1
            break
    return idx


def calculate_quantization_scaling(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    max: float = 57344.0,
    min: float = 1e-12,
    position: int = 0,
):
    with graph.inserting_after(node):
        abs_node = graph.call_function(
            torch.ops.aten.abs.default,
            args=(node,),
        )
        abs_node.meta["val"] = torch.ops.aten.abs.default(node.meta["val"])
        abs_node.meta["tensor_meta"] = extract_tensor_metadata(abs_node.meta["val"])
    with graph.inserting_after(abs_node):
        amax_node = graph.call_function(
            torch.ops.aten.amax.default,
            args=(abs_node, [-1], True),
        )
        amax_node.meta["val"] = torch.ops.aten.amax.default(
            abs_node.meta["val"], [-1], True
        )
        amax_node.meta["tensor_meta"] = extract_tensor_metadata(amax_node.meta["val"])
    with graph.inserting_after(amax_node):
        amax_64_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(amax_node, torch.float64),
        )
        amax_64_node.meta["val"] = torch.ops.prims.convert_element_type.default(
            amax_node.meta["val"], torch.float64
        )
        amax_64_node.meta["tensor_meta"] = extract_tensor_metadata(
            amax_64_node.meta["val"]
        )
    with graph.inserting_after(amax_64_node):
        clamp_min_node = graph.call_function(
            torch.ops.aten.clamp_min.default,
            args=(amax_64_node, min),
        )
        clamp_min_node.meta["val"] = torch.ops.aten.clamp_min.default(
            amax_64_node.meta["val"], min
        )
        clamp_min_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_min_node.meta["val"]
        )
    with graph.inserting_after(clamp_min_node):
        reciprocal_node = graph.call_function(
            torch.ops.aten.reciprocal.default,
            args=(clamp_min_node,),
        )
        reciprocal_node.meta["val"] = torch.ops.aten.reciprocal.default(
            clamp_min_node.meta["val"]
        )
        reciprocal_node.meta["tensor_meta"] = extract_tensor_metadata(
            reciprocal_node.meta["val"]
        )
    with graph.inserting_after(reciprocal_node):
        mul_node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(reciprocal_node, max),
        )
        mul_node.meta["val"] = torch.ops.aten.mul.Tensor(
            reciprocal_node.meta["val"], max
        )
        mul_node.meta["tensor_meta"] = extract_tensor_metadata(mul_node.meta["val"])
    with graph.inserting_after(mul_node):
        scale_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(mul_node, torch.float32),
            name=f"fp8_scale_pos_{position}_{node.name}",
        )
        scale_node.meta["val"] = torch.ops.prims.convert_element_type.default(
            mul_node.meta["val"], torch.float32
        )
        scale_node.meta["tensor_meta"] = extract_tensor_metadata(scale_node.meta["val"])
    return scale_node


def perform_quantization(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    scale_node: torch.fx.Node,
    quant_type: torch.dtype,
    clamp_min: float,
    clamp_max: float,
    position: int,
) -> torch.fx.Node:
    with graph.inserting_after(scale_node):
        target_node_32 = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(node, torch.float32),
        )
        target_node_32.meta["val"] = torch.ops.prims.convert_element_type.default(
            node.meta["val"], torch.float32
        )
        target_node_32.meta["tensor_meta"] = extract_tensor_metadata(
            target_node_32.meta["val"]
        )
    with graph.inserting_after(target_node_32):
        scaled_target_node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(target_node_32, scale_node),
        )
        scaled_target_node.meta["val"] = torch.ops.aten.mul.Tensor(
            target_node_32.meta["val"], scale_node.meta["val"]
        )
        scaled_target_node.meta["tensor_meta"] = extract_tensor_metadata(
            scaled_target_node.meta["val"]
        )
    with graph.inserting_after(scaled_target_node):
        clamp_min_scaled_node = graph.call_function(
            torch.ops.aten.clamp_min.default,
            args=(scaled_target_node, clamp_min),
        )
        clamp_min_scaled_node.meta["val"] = torch.ops.aten.clamp_min.default(
            scaled_target_node.meta["val"], clamp_min
        )
        clamp_min_scaled_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_min_scaled_node.meta["val"]
        )
    with graph.inserting_after(clamp_min_scaled_node):
        clamp_max_scaled_node = graph.call_function(
            torch.ops.aten.clamp_max.default,
            args=(clamp_min_scaled_node, clamp_max),
        )
        clamp_max_scaled_node.meta["val"] = torch.ops.aten.clamp_max.default(
            clamp_min_scaled_node.meta["val"], clamp_max
        )
        clamp_max_scaled_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_max_scaled_node.meta["val"]
        )
    with graph.inserting_after(clamp_max_scaled_node):
        quant_activation_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(clamp_max_scaled_node, quant_type),
            name=f"fp8_quant_pos_{position}_{node.name}",
        )
        quant_activation_node.meta["val"] = (
            torch.ops.prims.convert_element_type.default(
                clamp_max_scaled_node.meta["val"], quant_type
            )
        )
        quant_activation_node.meta["tensor_meta"] = extract_tensor_metadata(
            quant_activation_node.meta["val"]
        )
    return quant_activation_node


def calculate_tensor_size(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a PyTorch tensor in megabytes (MB).

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        float: Memory size in MB
    """
    # Get number of elements and size per element
    num_elements = tensor.numel()
    element_size = tensor.element_size()

    return (num_elements * element_size) / (1024 * 1024)


def get_allowed_dtypes() -> list[torch.dtype]:
    allowed_dtypes = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("allowed_dtypes", "torch.bfloat16")
    allowed_dtypes = [
        getattr(torch, dtype.split(".")[-1]) for dtype in allowed_dtypes.split(";")
    ]
    return allowed_dtypes


def should_quantize(node: torch.fx.Node) -> bool:
    allowed_dtypes = get_allowed_dtypes()
    if not is_node_meta_valid(node) or node.meta["val"].dtype not in allowed_dtypes:
        return False
    size_threshold = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("size_in_mb", 100)
    # calculate the size of the node
    size_in_mb = calculate_tensor_size(node.meta["val"])
    if not torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("skip_dynamo_guards", False):
        return size_in_mb >= size_threshold
    else:
        # case 1: we always quantize tensors with dynamic shapes
        if torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ].get("quantize_dynamic_shape", False):
            return statically_known_true(
                size_in_mb >= size_threshold
            ) or not statically_known_false(size_in_mb >= size_threshold)
        else:
            # case 2: we always not quantize tensors with dynamic shapes
            return statically_known_true(size_in_mb >= size_threshold)


def get_quant_type() -> torch.dtype:
    quant_type = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("quant_type", "torch.float8_e5m2")

    return getattr(torch, quant_type.split(".")[-1])


def calculate_range(dtype: torch.dtype) -> tuple:
    """
    Calculate the range of values for a given torch.dtype.
    Args:
        dtype (torch.dtype): The input dtype.
    Returns:
        tuple: A tuple containing the minimum and maximum values.
    """
    info = torch.finfo(dtype)
    return info.min, info.max


def quantize_activation_fw(graph: torch.fx.Graph) -> None:
    output = graph.find_nodes(op="output")[0]
    fwd_outputs = output.args[0]
    quant_type = get_quant_type()
    clamp_min, clamp_max = calculate_range(quant_type)
    position_to_quant = dict()
    tensor_scale_nodes, sym_scale_nodes = [], []
    for position, node in enumerate(fwd_outputs):
        # check if the activation node is the node saved for quantization
        if node.meta.get("saved_for_quantization", False):
            # case: use scaling
            if torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ].get("use_scaling", True):
                # calculating the scale
                scale_node = calculate_quantization_scaling(
                    graph, node, clamp_max, 1e-12, position
                )

                # converting to fp8
                quant_node = perform_quantization(
                    graph, node, scale_node, quant_type, clamp_min, clamp_max, position
                )
                if not is_sym_node(scale_node):
                    tensor_scale_nodes.append(scale_node)
                else:
                    sym_scale_nodes.append(scale_node)
            else:
                # case: do not use scaling
                with graph.inserting_after(node):
                    quant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, quant_type),
                        name=f"fp8_quant_pos_{position}_{node.name}",
                    )
                    quant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], quant_type
                        )
                    )
                    quant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        quant_node.meta["val"]
                    )

            position_to_quant[position] = quant_node

    # Use position-based lookup for building output
    # only update the return node args, and remain all other users unchanged
    output_updated_args = [
        position_to_quant.get(i, node) for i, node in enumerate(fwd_outputs)
    ]
    # add the scale nodes to the output find the first sym_node in the output
    # pyrefly: ignore [bad-argument-type]
    idx = find_first_sym_node(output_updated_args)
    scale_nodes = tensor_scale_nodes + sym_scale_nodes
    if scale_nodes:
        output_updated_args = (
            output_updated_args[:idx] + scale_nodes + output_updated_args[idx:]
        )

    output.update_arg(0, tuple(output_updated_args))
    counters["inductor"]["activation_quantization_fwd_aten_pass"] += 1


def quantize_activation_bw(graph: torch.fx.Graph) -> None:
    bw_inputs = [node for node in graph.nodes if node.op == "placeholder"]
    activation_node = None
    for node in bw_inputs:
        if node.meta.get("saved_for_quantization", False):
            node.meta.pop("saved_for_quantization")
            dequant_type = node.meta.pop("dequant_type")
            # dequantize the node
            if torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ].get("use_scaling", False):
                # case: use scaling
                with graph.inserting_after(node):
                    # find corresponding scale node
                    scale_name = "fp8_scale_" + node.name.replace("fp8_quant_", "")
                    scale_node = next(
                        bwd_input
                        for bwd_input in bw_inputs
                        if bwd_input.name == scale_name
                    )
                with graph.inserting_after(scale_node):
                    activation_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, dequant_type),
                    )
                    activation_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], dequant_type
                        )
                    )
                    activation_node.meta["tensor_meta"] = extract_tensor_metadata(
                        activation_node.meta["val"]
                    )
                with graph.inserting_after(activation_node):
                    divided_target_node_32 = graph.call_function(
                        torch.ops.aten.div.Tensor,
                        args=(activation_node, scale_node),
                    )
                    divided_target_node_32.meta["val"] = torch.ops.aten.div.Tensor(
                        activation_node.meta["val"], scale_node.meta["val"]
                    )
                    divided_target_node_32.meta["tensor_meta"] = (
                        extract_tensor_metadata(divided_target_node_32.meta["val"])
                    )
                with graph.inserting_after(divided_target_node_32):
                    dequant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(divided_target_node_32, dequant_type),
                    )
                    dequant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            divided_target_node_32.meta["val"], dequant_type
                        )
                    )
                    dequant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        dequant_node.meta["val"]
                    )
            else:
                with graph.inserting_after(node):
                    dequant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, dequant_type),
                        name="dequant_" + str(node.name),
                    )
                    dequant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], dequant_type
                        )
                    )
                    dequant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        dequant_node.meta["val"]
                    )
            # find the users of the node and replace them with the new node except the dequant_node
            for user in list(node.users.keys()):
                if user != dequant_node and user != activation_node:
                    user.replace_input_with(node, dequant_node)

    counters["inductor"]["activation_quantization_bwd_aten_pass"] += 1


def perform_fp8_activation_quantization(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    bwd_module_inputs: dict[str, fx.Node],
) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_activation_quantization_fwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: fwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    quantize_activation_fw(fwd_module.graph)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_activation_quantization_fwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: fwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_activation_quantization_bwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: bwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    quant_fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
    # update the corresponding bwd_inputs due to the fwd_outputs quantization
    for fwd_node in quant_fwd_module_outputs:
        if "fp8_quant_" in fwd_node.name:
            bwd_input = bwd_module_inputs[
                re.sub(r"^fp8_quant_pos_\d+_", "", fwd_node.name)
            ]
            with bwd_module.graph.inserting_after(bwd_input):
                quant_bwd_input = bwd_module.graph.placeholder(name=fwd_node.name)
            dequant_type = bwd_input.meta["dequant_type"]
            quant_bwd_input.meta.update(fwd_node.meta)
            quant_bwd_input.meta["saved_for_quantization"] = True
            quant_bwd_input.meta["dequant_type"] = dequant_type
            bwd_input.replace_all_uses_with(quant_bwd_input)
            bwd_module.graph.erase_node(bwd_input)
    # update the bwd_inputs if quantization with scaling is used
    if torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("use_scaling", True):
        quant_bwd_module_inputs = list(bwd_module.graph.find_nodes(op="placeholder"))
        # update the corresponding bwd input nodes find the last non-tangent node
        bwd_input_loc = quant_bwd_module_inputs[-1]
        for bw_input in reversed(quant_bwd_module_inputs):
            if not _is_tangent(bw_input):
                bwd_input_loc = bw_input
                break

        scaled_fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
        for fwd_node in scaled_fwd_module_outputs:
            if "fp8_scale_" in fwd_node.name:
                # fwd node is a scale node
                with bwd_module.graph.inserting_after(bwd_input_loc):
                    scale_bwd_input = bwd_module.graph.placeholder(name=fwd_node.name)
                scale_bwd_input.meta.update(fwd_node.meta)
                bwd_input_loc = scale_bwd_input

    quantize_activation_bw(bwd_module.graph)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_activation_quantization_bwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: bwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )


def enable_activation_quantization(
    saved_values: list[fx.Node],
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> None:
    if (
        inductor_config.post_grad_fusion_options.get(
            "activation_quantization_aten_pass", None
        )
        is None
    ):
        return

    static_input_names = (
        [node.name for node in static_lifetime_input_nodes]
        if static_lifetime_input_nodes
        else []
    )
    saved_values_names = {node.name: node for node in saved_values}
    if torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("exclude_primals", False):
        saved_values_names = {
            node.name: node for node in saved_values if "primals" not in node.name
        }
    fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
    bwd_module_inputs = {
        node.name: node for node in bwd_module.graph.find_nodes(op="placeholder")
    }
    should_perform_fp8_quant = False
    for node in fwd_module_outputs:
        if node.name in saved_values_names and should_quantize(node):
            if node.name in static_input_names:
                log.debug("Skipping quantization of static input %s: ", node.name)
                continue
            node.meta["saved_for_quantization"] = True
            node.meta["dequant_type"] = node.meta["val"].dtype
            # some of the fwd outputs and bwd inputs are not share the same object
            bwd_module_inputs[node.name].meta["saved_for_quantization"] = True
            bwd_module_inputs[node.name].meta["dequant_type"] = node.meta["val"].dtype
            should_perform_fp8_quant = True

    if should_perform_fp8_quant:
        perform_fp8_activation_quantization(fwd_module, bwd_module, bwd_module_inputs)


def _extract_fwd_bwd_modules(
    joint_module: fx.GraphModule,
    saved_values: list[fx.Node],
    saved_sym_nodes: list[fx.Node],
    *,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
        _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    )
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]
    tangent_inputs = [*filter(_is_tangent, placeholders)]
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
    )

    distributed_enabled = torch.distributed.is_available()

    for node in bwd_graph.find_nodes(op="placeholder"):
        # This is to filter out saved values that don't actually end up being used by the backwards pass
        if not node.users:
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        # wait_tensor is a bit special: if we have a "dead activation" that is not used in the bw,
        # but this dead activation is actually a collective,
        # then the collective will generally by followed by a wait_tensor() call.
        # we need to peak one node further to see if this wait_tensor is dead as well.
        elif distributed_enabled and all(
            n.target is torch.ops._c10d_functional.wait_tensor.default
            and len(n.users) == 0
            for n in node.users
        ):
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        elif _is_backward_state(node):
            # BackwardState is saved directly
            _remove_by_name(saved_values, node.name)
            assert backward_state_inputs

    # Now that we have the finalized list of saved values, we need to ensure
    # we propagate all symbols which are referenced by backwards inputs.
    # These are not directly used in the graph but are required for downstream
    # sizevar assignment
    saved_symbols: OrderedSet[sympy.Symbol] = OrderedSet()
    saved_sym_nodes_binding = []
    saved_sym_nodes_derived = []

    # Some symbols may already be bound in the directly saved_sym_nodes,
    # keep track of them so we don't re-bind them
    for node in saved_sym_nodes:
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            saved_symbols.add(symbol)
            saved_sym_nodes_binding.append(node)
        else:
            saved_sym_nodes_derived.append(node)

    # Now go through all of the prospective backward inputs and track any
    # other symbols we need to bind
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values, tangent_inputs):
        if "val" not in node.meta:
            continue
        new_symbols = free_symbols(node.meta["val"]) - saved_symbols
        # NB: Deterministic order please!
        for s in sorted(new_symbols, key=lambda s: s.name):
            # NB: For well formed graphs, the symbol should always be present,
            # but we also have ways to produce ill-formed graphs, e.g., direct
            # make_fx usages, so don't choke in this case
            if s not in symbol_bindings:
                continue
            saved_sym_nodes_binding.append(symbol_bindings[s])
        saved_symbols |= new_symbols

    # Update saved_sym_nodes that are now reordered to have all bindings at
    # front. This can also be used later on to figure out the position of saved
    # sym nodes in the output of fwd graph.
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes,
        fwd_outputs_descs
        + [
            SavedForBackwardsAOTOutput(i)
            for i in range(len(saved_values) + len(saved_sym_nodes))
        ],
        "forward",
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes
        + saved_values
        + tangent_inputs
        + bwd_seed_offset_inputs
        + backward_state_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
    )

    fwd_module = fx._lazy_graph_module._make_graph_module(joint_module, fwd_graph)
    bwd_module = fx._lazy_graph_module._make_graph_module(joint_module, bwd_graph)
    enable_activation_quantization(
        saved_values, fwd_module, bwd_module, static_lifetime_input_nodes
    )
    return fwd_module, bwd_module


def default_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Optional[list[int]] = None,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(
            joint_module,
            _joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
        )
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
        _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    )
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs, fwd_outputs_descs, "forward"
    )
    forward_node_names = OrderedSet(
        node.name for node in forward_only_graph.nodes if node.op != "output"
    )
    order = {node: idx for idx, node in enumerate(joint_module.graph.nodes)}
    saved_values = []
    saved_sym_nodes = []

    def is_mutated_later_in_fw(node):
        if _has_tag_is_backward(node):
            return False
        tensor_arg_aliases = [
            x
            for x in node.args
            if isinstance(x, fx.Node)
            and "val" in x.meta
            and isinstance(x.meta["val"], torch.Tensor)
        ]
        while len(tensor_arg_aliases) > 0:
            a = tensor_arg_aliases.pop()
            for u in a.users:
                if not isinstance(u.target, torch._ops.OpOverload):
                    continue
                # If we witness a mutation on our node later, and that mutation is not "must be in backward",
                # then our node needs to be computed in the forward (otherwise we will compute it on the mutated values)
                if (
                    # one of the args was mutated
                    u.target._schema.is_mutable
                    # and the mutation happens "later"
                    and order[u] > order[node]
                    # and the mutation happened during the forward
                    and not (_has_tag_is_backward(u) or _has_tag_must_be_in_backward(u))
                ):
                    for idx, alias_info in enumerate(u.target._schema.arguments):
                        if alias_info.is_write and u.args[idx] is a:
                            return True
                elif u.target.is_view:
                    tensor_arg_aliases.append(u)
        return False

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            # if a node isn't "required" to be in the forward, but any of its arguments
            # are later mutated in the forward, then it must have been run in the forward
            # (if not, and the node's arg was saved for backward, we would have mutated a saved value)
            # NB: doesn't handle nodes where the input is a list of tensors and one of those tensors is later mutated
            if is_mutated_later_in_fw(node):
                saved_values.append(node)
            continue
        if is_sym_node(node):
            # Symints must be kept separate from tensors so that PythonFunction only calls
            # save_for_backward on tensors and stashes symints in autograd .ctx
            saved_sym_nodes.append(node)
        elif (
            "tensor_meta" not in node.meta
            and node.op == "call_function"
            and not isinstance(node.meta.get("val"), torch._subclasses.FakeTensor)
        ):
            # Since we can't save tuple of tensor values, we need to flatten out what we're saving
            users = node.users
            assert all(user.target is operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [
                n for n in node.users if n.name not in forward_node_names
            ]
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                # If we have a tensor in the forward, where only its sizes/strides are needed in the backward,
                # and not the actual tensor data,
                # then it will be a lot cheaper to save only the sizes/strides, and not the actual tensor.
                #
                # Note that saving the tensor could also cause compilation problems:
                # If the user mutated an input in the forward and uses its sizes/strides in the backward,
                # then we would be obligated to clone the input before saving it to appease autograd.
                # (This is how we originally found this bug).
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)
    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())

    return _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_nodes=static_lifetime_input_nodes,
    )


INT_INF = int(1e6)


def _tensor_nbytes(numel: int, dtype) -> int:
    return numel * dtype.itemsize


def _size_of(node: fx.Node) -> int:
    def object_nbytes(x) -> int:
        if not isinstance(x, torch.Tensor):
            return 0
        return _tensor_nbytes(hint_int(x.numel(), fallback=4096), x.dtype)

    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        # NB: The fallback values here are meaningless, maybe we should respect
        # torch._inductor.config.unbacked_symint_fallback (but this is a
        # layering violation)
        elif isinstance(val, (list, tuple)):
            return sum(object_nbytes(n) for n in val)
        elif isinstance(val, dict):
            return sum(object_nbytes(n) for _, n in val.items())
        elif isinstance(val, torch.Tensor):
            return object_nbytes(val)

        raise RuntimeError(f"Unknown metadata type {type(val)} on node {node}")
    if node.op == "get_attr" or node.target is torch.ops.aten._assert_scalar.default:
        return 0
    raise RuntimeError(
        f"Node {node} didn't have `val` metadata; we should always have `val` metadata on the nodes."
    )


# Used for some investigative purposes
def _count_ops(graph: fx.Graph):
    from collections import defaultdict

    cnt: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        if node.op == "call_function":
            cnt[node.target.__name__] += 1
    log.info("%s", sorted(cnt.items(), key=operator.itemgetter(1), reverse=True))


@functools.cache
def pointwise_ops():
    ops = []
    for attr_name in dir(torch.ops.aten):
        opoverloadpacket = getattr(torch.ops.aten, attr_name)
        if not isinstance(opoverloadpacket, torch._ops.OpOverloadPacket):
            continue

        for overload in opoverloadpacket.overloads():
            op_overload = getattr(opoverloadpacket, overload)
            if torch.Tag.pointwise in op_overload.tags:
                # currently aot autograd uses packet not overload
                ops.append(opoverloadpacket)
                break

    return ops


def sort_depths(args, depth_map: dict[fx.Node, int]) -> list[tuple[fx.Node, int]]:
    arg_depths = {
        arg: depth_map[arg] for arg in args if isinstance(arg, torch.fx.node.Node)
    }
    return sorted(arg_depths.items(), key=operator.itemgetter(1), reverse=True)


def reordering_to_mimic_autograd_engine(gm: fx.GraphModule) -> fx.GraphModule:
    """
    This pass finds the first bwd node in the graph (by looking at users of
    tangents) and then reorders the graph by walking from this node to all the
    way to the end of the graph. At each op in this traversal, we insert this op
    in a new graph and try to bring only the relevant subgraph from the other
    non-bwd edges relevant for this op. This closely mimics the behavior of
    autograd engine.

    Why is this pass required in the first place?

    This is an artifact of how partitioners work today. The starting point of
    partitioner is a joint graph, which is fwd and then bwd graph. In the case
    of checkpointing, we keep portions of fwd graph in their original place in
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """

    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    order = {node: idx for idx, node in enumerate(gm.graph.nodes)}

    def insert_node_in_graph(node):
        cur_nodes = [node]
        insertable_nodes: OrderedSet[fx.Node] = OrderedSet()
        while len(cur_nodes) > 0:
            node = cur_nodes.pop()
            if node in insertable_nodes or node in env:
                continue
            insertable_nodes.add(node)

            # Bias traversal towards the nodes that have higher depth - prioritizes
            # critical path first.
            cur_nodes += node.all_input_nodes

        # pyrefly: ignore [bad-assignment]
        insertable_nodes = sorted(insertable_nodes, key=lambda n: order[n])
        for node in insertable_nodes:
            env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Find first bwd node in the graph
    tangent_inputs = list(filter(_is_tangent, gm.graph.nodes))
    first_node_in_bwd = None
    minimum_order = math.inf
    for tangent in tangent_inputs:
        for user in tangent.users:
            if order[user] < minimum_order:
                minimum_order = order[user]
                first_node_in_bwd = user

    # If gradInp does not depend upon gradOut, we may not find any nodes in the "backwards pass"
    if first_node_in_bwd is None:
        return gm

    # Build the graph op-by-op by starting from the node all the way to the end
    # copy_ can be not using tangents at all, we must copy it.
    for node in list(gm.graph.nodes)[: order[first_node_in_bwd]]:
        if node.op == "call_function" and node.target is torch.ops.aten.copy_.default:
            insert_node_in_graph(node)

    for node in list(gm.graph.nodes)[order[first_node_in_bwd] :]:
        insert_node_in_graph(node)

    # The output node is already built by the traversal.
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm


def app
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_functorch`):

- [`python_key.py_docs.md_docs.md`](./python_key.py_docs.md_docs.md)
- [`deprecated.py_docs.md_docs.md`](./deprecated.py_docs.md_docs.md)
- [`autograd_function.py_docs.md_docs.md`](./autograd_function.py_docs.md_docs.md)
- [`partitioners.py_kw.md_docs.md`](./partitioners.py_kw.md_docs.md)
- [`aot_autograd.py_kw.md_docs.md`](./aot_autograd.py_kw.md_docs.md)
- [`predispatch.py_kw.md_docs.md`](./predispatch.py_kw.md_docs.md)
- [`apis.py_docs.md_docs.md`](./apis.py_docs.md_docs.md)
- [`benchmark_utils.py_docs.md_docs.md`](./benchmark_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `partitioners.py_docs.md_docs.md`
- **Keyword Index**: `partitioners.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/_inductor/fx_passes/group_batch_fusion.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/group_batch_fusion.py`
- **Size**: 59,332 bytes (57.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import collections
import logging
import operator
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch._dynamo.utils import counters, is_node_meta_valid
from torch._logging import trace_structured
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..pattern_matcher import (
    CallFunctionVarArgs,
    get_arg_value,
    stable_topological_sort,
)
from ..utils import OPTIMUS_EXCLUDE_POST_GRAD


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False

aten = torch.ops.aten

log = logging.getLogger(__name__)

DEFAULT_BETA = 1
DEFAULT_ALPHA = 1

MIN_FUSE_SET_SIZE = 5
MAX_FUSE_SET_SIZE = 300
MAX_FUSE_SEARCH_DEPTH = 5
# The maximum tensor size that can go into the fusion group
MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR = 4096
# Whether we only fuse nodes with same parent node
FUSE_NODES_WITH_SAME_PARENT = False
# Whether we enable the add broadcast in batch linear
SHAPE_BROADCAST_BATCH_LINEAR = False
# Whether we enable the fuse nodes with same users
Fuse_NODES_WITH_SAME_USERS = False

# exclude these nodes from BFS
# excluding get item improves optimizer compilation time by 60s
SEARCH_EXCLUSIONS = OrderedSet([operator.getitem])


default_graph_search_options = {
    "min_fuse_set_size": MIN_FUSE_SET_SIZE,
    "max_fuse_set_size": MAX_FUSE_SET_SIZE,
    "max_fuse_search_depth": MAX_FUSE_SEARCH_DEPTH,
    "max_fuse_tensor_size_group_linear": MAX_FUSE_TENSOR_SIZE_GROUP_LINEAR,
    "fuse_nodes_with_same_parent": FUSE_NODES_WITH_SAME_PARENT,
    "shape_broadcast_batch_linear": SHAPE_BROADCAST_BATCH_LINEAR,
    "fuse_nodes_with_same_users": Fuse_NODES_WITH_SAME_USERS,
}

graph_search_options = default_graph_search_options


def update_stack_example_value(node, metadata, dim=0, op=torch.stack):
    """
    Update the example value of the node in the graph to enable followup split cat opt.
    """
    if node is not None and hasattr(node, "meta"):
        if op is torch.stack:
            example_value = torch.stack(metadata, dim=dim)
        elif op is torch.unbind:
            example_value = torch.unbind(metadata, dim=dim)  # type: ignore[assignment]
        else:
            return
        node.meta["example_value"] = example_value


def update_pointwise_example_value(pointwise_node, input, other, op):
    """
    Update the example value of the add node in the graph to enable followup split cat opt.
    """
    if pointwise_node is not None and hasattr(pointwise_node, "meta"):
        if op is torch.add:
            example_value = torch.add(input, other)
        elif op is torch.mul:
            example_value = torch.mul(input, other)
        else:
            return
        pointwise_node.meta["example_value"] = example_value


class GroupBatchFusionBase:
    def __init__(self, **kwargs) -> None:
        self.graph_search_options = kwargs.pop(
            "graph_search_options", default_graph_search_options
        )

    def match(self, node):
        raise NotImplementedError("match called on base")

    def fuse(self, graph, subset):
        raise NotImplementedError("fuse called on base")


PRE_GRAD_FUSIONS: dict[str, GroupBatchFusionBase] = {}
POST_GRAD_FUSIONS: dict[str, GroupBatchFusionBase] = {}


def register_fusion(name: str, pre_grad=True):
    def decorator(fusion_cls: GroupBatchFusionBase):
        if pre_grad:
            PRE_GRAD_FUSIONS[name] = fusion_cls
        else:
            POST_GRAD_FUSIONS[name] = fusion_cls
        return fusion_cls

    return decorator


def list_group_batch_fusions(pre_grad=True) -> list[str]:
    if pre_grad:
        return list(PRE_GRAD_FUSIONS.keys())
    else:
        return list(POST_GRAD_FUSIONS.keys())


def decompose_stack(graph: torch.fx.GraphModule, input_tensors: list[Any]) -> Any:
    unsqueezed_inputs = []
    unsqueezed_inputs_meta = []
    for input_tensor in input_tensors:
        unsqueezed_input = graph.call_function(  # type: ignore[operator]
            aten.unsqueeze, args=(input_tensor,), kwargs={"dim": 0}
        )
        unsqueezed_inputs.append(unsqueezed_input)
        unsqueezed_input.meta["val"] = aten.unsqueeze(input_tensor.meta["val"], dim=0)  # type: ignore[assignment]
        unsqueezed_inputs_meta.append(unsqueezed_input.meta["val"])
    stacked_inputs = graph.call_function(  # type: ignore[operator]
        aten.cat, args=(unsqueezed_inputs,), kwargs={"dim": 0}
    )
    stacked_inputs.meta["val"] = aten.cat(unsqueezed_inputs_meta, dim=0)  # type: ignore[assignment]
    return stacked_inputs


class GroupFusion(GroupBatchFusionBase):
    """
    Fuse ops in a group way, e.g, fuse mm/addmm of arbitrary input shapes with fbgemm.gmm.
    """


class BatchFusion(GroupBatchFusionBase):
    """
    Fuse ops in a batch way, e.g, fuse mm/addmm of same input shapes with bmm.
    """


class BatchPointwiseOpsFusionFactory(BatchFusion):
    def __init__(self, op, **kwargs) -> None:
        super().__init__(**kwargs)
        self.op = op


@register_fusion("batch_linear_post_grad", pre_grad=False)
class PostGradBatchLinearFusion(BatchFusion):
    """
    Fuse ops in a batch way in post grad (aten level).
    """

    def _addmm_node_can_be_fused(self, node: torch.fx.Node) -> bool:
        # pyre-fixme[7]: Incompatible return type
        return (
            node.kwargs.get("beta", DEFAULT_BETA) == DEFAULT_BETA
            and node.kwargs.get("alpha", DEFAULT_ALPHA) == DEFAULT_ALPHA  # type: ignore[return-value]
        )

    def _is_input_2d(self, input: torch.fx.Node) -> bool:
        input_shapes = input.meta["val"].shape
        return (
            len(input_shapes) == 2
            and isinstance(input_shapes[0], int)
            and isinstance(input_shapes[1], int)
        )

    def match(self, node: torch.fx.Node) -> tuple[str, int, int, int, bool, str] | None:
        if CallFunctionVarArgs(aten.mm).match(node):
            input_m, weight_m = node.args
            bias_m = None

        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias_m, input_m, weight_m = node.args
        else:
            return None
        # get the user of the node
        if self.graph_search_options.get("fuse_nodes_with_same_users", False):
            users = [user.target for user in node.users]
        else:
            users = ""  # type: ignore[assignment]
        # only handle the cases where inputs are 2D tensors
        if not self._is_input_2d(input_m) or not self._is_input_2d(weight_m):  # type: ignore[arg-type]
            return None
        m, k = input_m.meta["val"].shape  # type: ignore[union-attr]
        n = weight_m.meta["val"].shape[1]  # type: ignore[union-attr]
        batch_key = ("batch_linear_post_grad", m, k, n, bias_m is not None, str(users))
        return batch_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_nodes = []
        batch_inputs_meta = []
        batch_weights_meta = []
        batch_biases_meta = []

        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            elif CallFunctionVarArgs(aten.mm.default).match(node):
                input, weight = node.args
                bias = None
            batch_nodes.append(node)
            batch_inputs.append(input)  # type: ignore[possibly-undefined]
            batch_weights.append(weight)  # type: ignore[possibly-undefined]
            batch_biases.append(bias)  # type: ignore[possibly-undefined]
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_weights_meta.append(weight.meta)  # type: ignore[possibly-undefined, union-attr]
            if bias is not None:  # type: ignore[possibly-undefined]
                batch_biases_meta.append(bias.meta)  # type: ignore[possibly-undefined, union-attr]
            else:
                batch_biases_meta.append(None)

        with graph.inserting_before(subset[-1]):  # type: ignore[operator]
            fused_inputs = decompose_stack(graph, batch_inputs)
            fused_weights = decompose_stack(graph, batch_weights)
            fused_inputs_meta_val = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            fused_weights_meta_val = torch.stack(
                [weight["val"] for weight in batch_weights_meta]
            )
            fused_bmm = graph.call_function(  # type: ignore[operator]
                aten.bmm,
                args=(fused_inputs, fused_weights),
            )
            fused_bmm.meta["val"] = aten.bmm(
                fused_inputs_meta_val, fused_weights_meta_val
            )
        for i, original_mm in enumerate(batch_nodes):
            has_bias = False
            with graph.inserting_after(fused_bmm):  # type: ignore[operator]
                new_mm = graph.call_function(aten.select, args=((fused_bmm, 0, i)))  # type: ignore[operator]
                new_mm.meta["val"] = aten.select(fused_bmm.meta["val"], 0, i)
                if batch_biases[i]:
                    has_bias = True
                    # broadcast the bias to the same shape as the mm output
                    if self.graph_search_options.get(
                        "shape_broadcast_batch_linear", False
                    ):
                        broadcast_shape = torch.broadcast_shapes(
                            batch_biases_meta[i]["val"].shape, new_mm.meta["val"].shape
                        )
                        broadcast_bias = graph.call_function(  # type: ignore[operator]
                            aten.broadcast_to.default,
                            args=(batch_biases[i],),
                            kwargs={"size": broadcast_shape},
                        )
                        broadcast_bias.meta["val"] = aten.broadcast_to(
                            batch_biases_meta[i]["val"], broadcast_shape
                        )  # type: ignore[assignment]
                        new_bias_add = graph.call_function(  # type: ignore[operator]
                            aten.add.Tensor, args=((broadcast_bias, new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            broadcast_bias.meta["val"], new_mm.meta["val"]
                        )
                    else:
                        new_bias_add = graph.call_function(  # type: ignore[operator]
                            aten.add, args=((batch_biases[i], new_mm))
                        )
                        new_bias_add.meta["val"] = aten.add.Tensor(
                            batch_biases_meta[i]["val"], new_mm.meta["val"]
                        )
            new_mm_cont = new_bias_add if has_bias else new_mm  # type: ignore[possibly-undefined]
            original_mm.replace_all_uses_with(new_mm_cont)
            new_mm_cont.meta.update(original_mm.meta)
            graph.erase_node(original_mm)  # type: ignore[operator]
        counters["inductor"]["batch_linear_post_grad"] += 1


@register_fusion("group_linear", pre_grad=False)
class GroupLinearFusion(GroupFusion):
    def _addmm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[2].meta["val"].shape  # type: ignore[union-attr]
        return (
            node.kwargs.get("beta", DEFAULT_BETA) == DEFAULT_BETA
            and node.kwargs.get("alpha", DEFAULT_ALPHA) == DEFAULT_ALPHA
            and len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def _mm_node_can_be_fused(self, node: torch.fx.Node):
        input_shape = node.args[0].meta["val"].shape  # type: ignore[union-attr]
        weight_shape = node.args[1].meta["val"].shape  # type: ignore[union-attr]
        return (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and all(x % 2 == 0 for x in input_shape + weight_shape)
            and all(
                shape <= self.graph_search_options["max_fuse_tensor_size_group_linear"]
                for shape in input_shape + weight_shape
            )
        )

    def match(self, node: torch.fx.Node) -> tuple[str, bool] | None:
        if CallFunctionVarArgs(aten.mm.default).match(
            node
        ) and self._mm_node_can_be_fused(node):
            group_key = ("group_linear", True)
        elif CallFunctionVarArgs(aten.addmm.default).match(
            node
        ) and self._addmm_node_can_be_fused(node):
            bias = node.args[0]
            group_key = ("group_linear", bias is None)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        group_inputs = []
        group_weights = []
        group_biases = []
        group_nodes = []
        for node in subset:
            if CallFunctionVarArgs(aten.addmm.default).match(node):
                bias, input, weight = node.args
            else:
                assert CallFunctionVarArgs(aten.mm.default).match(node)
                input, weight = node.args
                bias = None

            group_nodes.append(node)
            group_inputs.append(input)
            group_weights.append(weight)
            group_biases.append(bias)

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            fused_mm = graph.call_function(  # type: ignore[operator]
                torch.ops.fbgemm.gmm.default,
                args=(group_inputs, group_weights, group_biases),
                kwargs={"smart_fused": True},
            )

        for i, original_mm in enumerate(group_nodes):
            with graph.inserting_after(fused_mm):  # type: ignore[operator]
                new_mm = graph.call_function(operator.getitem, args=(fused_mm, i))  # type: ignore[operator]
            original_mm.replace_all_uses_with(new_mm)
            new_mm.meta.update(original_mm.meta)
            graph.erase_node(original_mm)  # type: ignore[operator]
        counters["inductor"]["group_linear"] += 1


class BatchPointwiseMathOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise math operator (e.g., add, mul) in post grad pass.
    """

    def __init__(self, op, **kwargs) -> None:
        super().__init__(op, **kwargs)
        self.op = op

    def _pointwise_node_can_be_fused(self, node: torch.fx.Node):
        # note: we only consider the case where the inputs are tensors
        # for mixed precision training, we need to make sure the inputs
        # of the aten.cat when do the stack should be the same dtype
        # otherwise, the output of the aten.cat may be not the same as
        # its inputs, and cause dtype not same error in mm or addmm
        input, other = node.args
        return (
            input.meta["val"].shape == other.meta["val"].shape  # type: ignore[union-attr]
            # input and other can be scalars, where they have no attribute 'meta'
            if hasattr(input, "meta")
            and hasattr(other, "meta")
            and is_node_meta_valid(input)  # type: ignore[arg-type, union-attr]
            and is_node_meta_valid(other)  # type: ignore[arg-type, union-attr]
            # torch.SymInt or torch.SymFloat object has no attribute 'shape'
            and isinstance(input.meta["val"], torch.Tensor)  # type: ignore[union-attr]
            and isinstance(other.meta["val"], torch.Tensor)  # type: ignore[union-attr]
            else False
        )

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(self.op).match(
            node
        ) and self._pointwise_node_can_be_fused(node):
            alpha = node.kwargs.get("alpha", DEFAULT_ALPHA)
            rounding_mode = node.kwargs.get("rounding_mode", None)
            input, other = node.args
            shape = list(input.meta["val"].shape)  # type: ignore[union-attr]
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # only consider the linear case so far
                # pyre-fixme[16]
                if input.target is aten.select or other.target is aten.select:  # type: ignore[union-attr]
                    parent = (
                        # pyre-fixme[16]
                        input.args[0]  # type: ignore[union-attr]
                        # pyre-fixme[16]
                        if input.target is aten.select  # type: ignore[union-attr]
                        else other.args[0]  # type: ignore[union-attr]
                    )
                else:
                    parent = ""
            else:
                parent = ""
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(shape),
                str(input.meta["val"].dtype),  # type: ignore[union-attr]
                str(other.meta["val"].dtype),  # type: ignore[union-attr]
                str(alpha),
                str(rounding_mode),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_inputs, batch_others = [], []
        alpha = subset[0].kwargs.get("alpha", DEFAULT_ALPHA)
        batch_inputs_meta, batch_others_meta = [], []

        for node in subset:
            input, other = node.args
            batch_inputs.append(input)
            batch_others.append(other)
            batch_inputs_meta.append(input.meta)  # type: ignore[possibly-undefined, union-attr]
            batch_others_meta.append(other.meta)  # type: ignore[possibly-undefined, union-attr]

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = decompose_stack(graph, batch_inputs)
            stack_others = decompose_stack(graph, batch_others)
            stack_inputs_meta = torch.stack(
                [input["val"] for input in batch_inputs_meta]
            )
            stack_others_meta = torch.stack(
                [other["val"] for other in batch_others_meta]
            )

            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs, stack_others),
                kwargs={"alpha": alpha} if self.op == aten.add.Tensor else {},
            )
            batch_op.meta["val"] = self.op(stack_inputs_meta, stack_others_meta)
            for i, original_add in enumerate(subset):
                with graph.inserting_after(batch_op):  # type: ignore[operator]
                    new_add = graph.call_function(  # type: ignore[operator]
                        torch.ops.aten.select, args=((batch_op, 0, i))
                    )
                original_add.replace_all_uses_with(new_add)
                new_add.meta.update(original_add.meta)
                graph.erase_node(original_add)  # type: ignore[operator]
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


@register_fusion("batch_linear_lhs")
class BatchLinearLHSFusion(BatchFusion):
    """
    Batch linear left-hand side fusion. This pass tries to fuse the following patterns:

        torch.nn.functional.linear(x, w1), linear(x, w2),... * linear(x, wn)
        -> torch.mm(x, torch.cat([w1, w2,... * wn]).transpose(0, 1))

    We have a separate pass to eliminate contiguous transpose in a generic way.
    """

    def match(self, node: torch.fx.Node) -> tuple[str, bool, Any] | None:
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            bias = get_arg_value(node, 2, "bias")
            group_key = ("batch_linear_lhs", bias is None, input)
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_input = None
        batch_weights, batch_weights_meta = [], []
        batch_biases, batch_biases_meta = [], []
        split_sections = []
        for node in subset:
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            batch_nodes.append(node)
            if batch_input is None:
                batch_input = input
            else:
                assert batch_input is input
            batch_weights.append(weight)
            batch_weights_meta.append(weight.meta["example_value"])
            if bias:
                batch_biases.append(bias)
                batch_biases_meta.append(bias.meta["example_value"])
            split_sections.append(weight.meta["example_value"].shape[0])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            cat_weights = graph.call_function(  # type: ignore[operator]
                torch.cat, args=(batch_weights,), kwargs={"dim": 0}
            )
            cat_weights.meta["example_value"] = torch.cat(batch_weights_meta, dim=0)
            transposed_weights = graph.call_function(  # type: ignore[operator]
                torch.transpose, args=(cat_weights, 0, 1)
            )
            transposed_weights.meta["example_value"] = torch.transpose(
                cat_weights.meta["example_value"], 0, 1
            )
            if len(batch_biases) > 0:
                cat_biases = graph.call_function(  # type: ignore[operator]
                    torch.cat, args=(batch_biases,), kwargs={"dim": 0}
                )
                cat_biases.meta["example_value"] = torch.cat(batch_biases_meta, dim=0)
                fused_lhs = graph.call_function(  # type: ignore[operator]
                    torch.addmm,
                    args=(cat_biases, batch_input, transposed_weights),
                )
                fused_lhs.meta["example_value"] = torch.addmm(
                    cat_biases.meta["example_value"],
                    batch_input.meta["example_value"],  # type: ignore[union-attr]
                    transposed_weights.meta["example_value"],
                )
            else:
                fused_lhs = graph.call_function(  # type: ignore[operator]
                    torch.mm,
                    args=(batch_input, transposed_weights),
                )
                fused_lhs.meta["example_value"] = torch.mm(
                    batch_input.meta["example_value"],  # type: ignore[union-attr]
                    transposed_weights.meta["example_value"],
                )
            fused_lhs_list = graph.call_function(  # type: ignore[operator]
                torch.split, args=(fused_lhs, split_sections), kwargs={"dim": 1}
            )

        for i, node in enumerate(batch_nodes):
            with graph.inserting_after(fused_lhs_list):  # type: ignore[operator]
                new_node = graph.call_function(  # type: ignore[operator]
                    operator.getitem, args=(fused_lhs_list, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_linear_lhs"] += 1


# Poor person's check for if a node in the graph mutates its input.
# (the graph is torch IR, so we will see torch fns and python operators)
def _is_mutable_node(tgt):
    if str(tgt).endswith("_"):
        # e.g. torch.mul_, torch.Tensor.mul_
        return True
    if (
        hasattr(tgt, "__module__")
        and tgt.__module__ == "_operator"
        and tgt.__name__.startswith("i")
    ):
        # e.g. operator.iand, operator.imul
        return True
    return False


def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, "input")
    weight = get_arg_value(node, 1, "weight")
    return (
        is_node_meta_valid(node)
        and is_node_meta_valid(input)
        and is_node_meta_valid(weight)
        and len(input.meta["example_value"].shape) == 2
        and len(weight.meta["example_value"].shape) == 2
        # the mm -> bmm transform adds an unbind() op,
        # which is not safe for autograd when the output of the mm is mutated.
        # don't pattern match if any users of the mm mutate the input.
        and not any(_is_mutable_node(user.target) for user in node.users)
    )


@register_fusion("batch_linear")
class PreGradBatchLinearFusion(BatchFusion):
    """
    Batch linear fusion in pre grad pass.
    Fuse linear with same size with torch.baddmm
    """

    def _getitem_args(self, getitem_node: torch.fx.Node):
        if getitem_node.target != operator.__getitem__ or (
            getitem_node.op != "call_function"
        ):
            return None
        return getitem_node.args[0]

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.linear).match(
            node
        ) and is_linear_node_can_be_fused(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 1, "weight")
            bias = get_arg_value(node, 2, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                "batch_linear",
                self._getitem_args(input),
                str(input.meta["example_value"].shape),
                str(weight.meta["example_value"].shape),
                bias is None,
                str(users),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_weights = []
        batch_biases = []
        batch_inputs_metadata = []
        batch_weights_metadata = []
        batch_biases_metadata = []
        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])
            weight = get_arg_value(node, 1, "weight")
            batch_weights.append(weight)
            batch_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 2, "bias")
            batch_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                batch_biases_metadata.append(bias.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            stack_weights = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_weights,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_weights, batch_weights_metadata)
            transpose_weight = graph.call_function(  # type: ignore[operator]
                torch.transpose, args=(stack_weights, 1, 2)
            )
            transpose_weight.meta["example_value"] = torch.transpose(
                stack_weights.meta["example_value"], 1, 2
            )
            if all(bias is None for bias in batch_biases):
                bmm = graph.call_function(  # type: ignore[operator]
                    torch.bmm,
                    args=(stack_inputs, transpose_weight),
                )
                bmm.meta["example_value"] = torch.bmm(
                    stack_inputs.meta["example_value"],
                    transpose_weight.meta["example_value"],
                )
                bmm_meta = bmm.meta["example_value"]
            else:
                stack_biases = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(batch_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_biases, batch_biases_metadata)
                unsqueeze_biases = graph.call_function(  # type: ignore[operator]
                    torch.unsqueeze, args=(stack_biases, 1)
                )
                unsqueeze_biases.meta["example_value"] = torch.unsqueeze(
                    stack_biases.meta["example_value"], 1
                )
                bmm = graph.call_function(  # type: ignore[operator]
                    torch.baddbmm,
                    args=(unsqueeze_biases, stack_inputs, transpose_weight),
                )
                try:
                    # it will have runtime error to broadcast when it has dynamic shape included
                    # in the meta data, so we need to skip the update meta data
                    bmm.meta["example_value"] = torch.baddbmm(
                        unsqueeze_biases.meta["example_value"],
                        stack_inputs.meta["example_value"],
                        transpose_weight.meta["example_value"],
                    )
                    bmm_meta = bmm.meta["example_value"]
                except Exception as e:
                    log.debug(
                        f" exception when update bmm meta data with stack error tracekey {e}"  # noqa: G004
                    )
                    bmm_meta = None

            bmm = graph.call_function(torch.unbind, args=(bmm,), kwargs={"dim": 0})  # type: ignore[operator]
            if bmm_meta is not None:
                bmm.meta["example_value"] = torch.unbind(bmm_meta, dim=0)
            for i, linear in enumerate(batch_nodes):
                with graph.inserting_after(bmm):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(bmm, i))  # type: ignore[operator]
                linear.replace_all_uses_with(getitem)
                getitem.meta.update(linear.meta)
                graph.erase_node(linear)  # type: ignore[operator]
        counters["inductor"]["batch_linear"] += 1


@register_fusion("batch_layernorm")
class BatchLayernormFusion(BatchFusion):
    """
    Batch layer norm fusion in pre grad pass
    """

    def match(self, node: torch.fx.Node):
        if CallFunctionVarArgs(torch.nn.functional.layer_norm).match(node):
            input = get_arg_value(node, 0, "input")
            weight = get_arg_value(node, 2, "weight")
            bias = get_arg_value(node, 3, "bias")
            if self.graph_search_options.get("fuse_nodes_with_same_users", False):
                users = [user.target for user in node.users]
            else:
                users = ""  # type: ignore[assignment]
            group_key = (
                (
                    "batch_layernorm",
                    str(input.meta["example_value"].shape),
                    str(weight.meta["example_value"].shape)
                    if weight is not None
                    else "",
                    str(bias.meta["example_value"].shape) if bias is not None else "",
                    str(get_arg_value(node, 1, "normalized_shape")),
                    str(get_arg_value(node, 4, "eps")),
                    str(users),
                )
                if "example_value" in input.meta
                and is_node_meta_valid(weight)
                and is_node_meta_valid(bias)
                else None
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        group_inputs = []
        group_shapes = []
        group_weights = []
        group_biases = []
        group_epss = []
        group_nodes = []
        group_inputs_metadata = []
        group_biases_metadata = []
        group_weights_metadata = []
        for node in subset:
            group_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            group_inputs.append(input)
            group_inputs_metadata.append(input.meta["example_value"])
            group_shapes.append(get_arg_value(node, 1, "normalized_shape"))
            weight = get_arg_value(node, 2, "weight")
            group_weights.append(weight)
            if weight is not None and hasattr(weight, "meta"):
                group_weights_metadata.append(weight.meta["example_value"])
            bias = get_arg_value(node, 3, "bias")
            group_biases.append(bias)
            if bias is not None and hasattr(bias, "meta"):
                group_biases_metadata.append(bias.meta["example_value"])
            eps = get_arg_value(node, 4, "eps")
            if eps is None:
                eps = 1e-5
            group_epss.append(eps)
        stack_dim = -1 - len(group_shapes[-1])

        if all(bias is None for bias in group_biases):
            group_biases = None  # type: ignore[assignment]
        if all(weight is None for weight in group_weights):
            group_weights = None  # type: ignore[assignment]
        assert all(eps == group_epss[0] for eps in group_epss), (
            "all epsilon values must be equal"
        )

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_input = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(group_inputs,), kwargs={"dim": stack_dim}
            )
            update_stack_example_value(stack_input, group_inputs_metadata, stack_dim)
            if group_weights is not None:
                stack_weight = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(group_weights,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_weight, group_weights_metadata)
            else:
                stack_weight = None
            if group_biases is not None:
                stack_bias = graph.call_function(  # type: ignore[operator]
                    torch.stack, args=(group_biases,), kwargs={"dim": 0}
                )
                update_stack_example_value(stack_bias, group_biases_metadata)
            else:
                stack_bias = None

            batch_layer_norm = graph.call_function(  # type: ignore[operator]
                torch.nn.functional.layer_norm,
                args=(stack_input, group_shapes[-1]),
                kwargs={"eps": group_epss[-1]},
            )
            batch_layer_norm.meta["example_value"] = stack_input.meta["example_value"]

            if group_weights is not None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(  # type: ignore[operator]
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                batch_layer_norm = graph.call_function(  # type: ignore[operator]
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )
            elif group_weights is not None and group_biases is None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                # pyrefly: ignore [not-callable]
                batch_layer_norm = graph.call_function(
                    torch.mul, args=(stack_weight, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_weight.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.mul,
                )
            elif group_weights is None and group_biases is not None:
                previous_batch_layer_norm_meta = batch_layer_norm.meta["example_value"]
                # pyrefly: ignore [not-callable]
                batch_layer_norm = graph.call_function(
                    torch.add, args=(stack_bias, batch_layer_norm)
                )
                update_pointwise_example_value(
                    batch_layer_norm,
                    # pyrefly: ignore [missing-attribute]
                    stack_bias.meta["example_value"],
                    previous_batch_layer_norm_meta,
                    torch.add,
                )

            batch_layer_norm_unbind = graph.call_function(  # type: ignore[operator]
                torch.unbind,
                args=(batch_layer_norm,),
                kwargs={"dim": stack_dim},
            )
            update_stack_example_value(
                batch_layer_norm_unbind,
                batch_layer_norm.meta["example_value"],
                op=torch.unbind,
                dim=stack_dim,
            )

        for i, node in enumerate(group_nodes):
            with graph.inserting_after(batch_layer_norm_unbind):  # type: ignore[operator]
                new_node = graph.call_function(  # type: ignore[operator]
                    operator.getitem, args=(batch_layer_norm_unbind, i)
                )
            node.replace_all_uses_with(new_node)
            new_node.meta.update(node.meta)
            graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_layernorm"] += 1


class BatchPointwiseOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in pre grad pass.
    We fuse it in random place, and the introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs) -> None:
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            if self.graph_search_options.get("fuse_nodes_with_same_parent", False):
                # pyre-fixme[16]
                parent = node.args[0]
                parent = parent.target if parent is not None else ""  # type: ignore[union-attr]
            else:
                parent = ""
            # for relu op, we also use the inplace to construct the key
            group_key = (
                "batch_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["example_value"].shape),
                str(node.kwargs.get("inplace", False)),
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            if self.op is torch.nn.functional.relu:
                batch_op = graph.call_function(  # type: ignore[operator]
                    self.op,
                    args=(stack_inputs,),
                    kwargs={"inplace": subset[0].kwargs.get("inplace", False)},
                )
                batch_op.meta["example_value"] = self.op(
                    stack_inputs.meta["example_value"],
                    # pyrefly: ignore [bad-argument-type]
                    inplace=subset[0].kwargs.get("inplace", False),
                )
            else:
                batch_op = graph.call_function(  # type: ignore[operator]
                    self.op,
                    args=(stack_inputs,),
                )
                batch_op.meta["example_value"] = self.op(
                    stack_inputs.meta["example_value"]
                )
            unbind_op = graph.call_function(  # type: ignore[operator]
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            unbind_op.meta["example_value"] = torch.unbind(
                batch_op.meta["example_value"], dim=0
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


class BatchPointwiseOpsPostGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch pointwise ops (e.g., sigmoid, relu, tanh) fusion in post grad pass.
    The introduced stack node may be merged in split cat.
    """

    def __init__(self, op, **kwargs) -> None:
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # for relu op, we also use the inplace to construct the key
            # we batch the ops with same parent to enable followup split cat
            parent = node.args[0]
            parent = (
                parent.target  # type: ignore[union-attr]
                if self.graph_search_options.get("fuse_nodes_with_same_parent", False)
                else ""
            )
            group_key = (
                "batch_aten_" + self.op.__name__.lower().split(".")[0],
                str(input.meta["val"].shape),
                str(node.kwargs.get("inplace", False)),
                # pyre-fixme[16]
                str(parent),
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["val"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = decompose_stack(graph, batch_inputs)
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs,),
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(batch_op):  # type: ignore[operator]
                    getitem = graph.call_function(aten.select, args=(batch_op, 0, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"][
            "batch_aten_" + self.op.__name__.lower().split(".")[0]
        ] += 1


class BatchMathOpsPreGradFusion(BatchPointwiseOpsFusionFactory):
    """
    Batch simple match related ops such as nan_to_num in pre grad pass.
    """

    def __init__(self, op, **kwargs):
        super().__init__(op, **kwargs)
        self.op = op

    def match(self, node: torch.fx.Node):
        input = get_arg_value(node, 0, "input")
        if CallFunctionVarArgs(self.op).match(node) and is_node_meta_valid(node):
            # check the input has the same shape and its users have the same target
            # check all clamp operators have the same min and max values, and
            # nan_to_num operators use the same default value.
            child = next(iter(node.users.keys()))
            group_key = (
                str(input.meta["example_value"].shape)
                + str(node.kwargs)
                + str(child.target)
            )
        else:
            group_key = None
        return group_key

    def fuse(self, graph: torch.fx.GraphModule, subset: list[torch.fx.Node]):
        batch_nodes = []
        batch_inputs = []
        batch_inputs_metadata = []
        kwargs = subset[0].kwargs

        for node in subset:
            batch_nodes.append(node)
            input = get_arg_value(node, 0, "input")
            batch_inputs.append(input)
            batch_inputs_metadata.append(input.meta["example_value"])

        with graph.inserting_before(subset[0]):  # type: ignore[operator]
            stack_inputs = graph.call_function(  # type: ignore[operator]
                torch.stack, args=(batch_inputs,), kwargs={"dim": 0}
            )
            update_stack_example_value(stack_inputs, batch_inputs_metadata)
            batch_op = graph.call_function(  # type: ignore[operator]
                self.op,
                args=(stack_inputs,),
                kwargs=kwargs,
            )
            batch_op.meta["example_value"] = self.op(
                stack_inputs.meta["example_value"], **kwargs
            )
            unbind_op = graph.call_function(  # type: ignore[operator]
                torch.unbind, args=(batch_op,), kwargs={"dim": 0}
            )
            unbind_op.meta["example_value"] = torch.unbind(
                batch_op.meta["example_value"], dim=0
            )
            for i, node in enumerate(batch_nodes):
                with graph.inserting_after(unbind_op):  # type: ignore[operator]
                    getitem = graph.call_function(operator.getitem, args=(unbind_op, i))  # type: ignore[operator]
                node.replace_all_uses_with(getitem)
                getitem.meta.update(node.meta)
                graph.erase_node(node)  # type: ignore[operator]
        counters["inductor"]["batch_" + self.op.__name__.lower().split(".")[0]] += 1


@register_fusion("batch_tanh")
class BatchTanhPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(torch.tanh, **kwargs)


@register_fusion("batch_sigmoid")
class BatchSigmoidPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(torch.sigmoid, **kwargs)


@register_fusion("batch_relu")
class BatchReLuPreGradFusion(BatchPointwiseOpsPreGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(torch.nn.functional.relu, **kwargs)


@register_fusion("batch_detach")
class BatchDetachPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.detach, **kwargs)


@register_fusion("batch_nan_to_num")
class BatchNanToNumPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.nan_to_num, **kwargs)


@register_fusion("batch_clamp")
class BatchClampPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.clamp, **kwargs)


@register_fusion("batch_dropout")
class BatchDropoutPreGradFusion(BatchMathOpsPreGradFusion):
    def __init__(self, **kwargs):
        super().__init__(torch.nn.functional.dropout, **kwargs)


@register_fusion("batch_aten_tanh", pre_grad=False)
class BatchTanhPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.tanh.default, **kwargs)


@register_fusion("batch_aten_sigmoid", pre_grad=False)
class BatchSigmoidPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.sigmoid.default, **kwargs)


@register_fusion("batch_aten_relu", pre_grad=False)
class BatchReLuPostGradFusion(BatchPointwiseOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.relu.default, **kwargs)


@register_fusion("batch_aten_add", pre_grad=False)
class BatchAddPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
        super().__init__(aten.add.Tensor, **kwargs)


@register_fusion("batch_aten_sub", pre_grad=False)
class BatchSubPostGradFusion(BatchPointwiseMathOpsPostGradFusion):
    def __init__(self, **kwargs) -> None:
```



## High-Level Overview


This Python file contains 28 class(es) and 65 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GroupBatchFusionBase`, `GroupFusion`, `BatchFusion`, `BatchPointwiseOpsFusionFactory`, `PostGradBatchLinearFusion`, `GroupLinearFusion`, `BatchPointwiseMathOpsPostGradFusion`, `BatchLinearLHSFusion`, `PreGradBatchLinearFusion`, `BatchLayernormFusion`, `BatchPointwiseOpsPreGradFusion`, `BatchPointwiseOpsPostGradFusion`, `BatchMathOpsPreGradFusion`, `BatchTanhPreGradFusion`, `BatchSigmoidPreGradFusion`, `BatchReLuPreGradFusion`, `BatchDetachPreGradFusion`, `BatchNanToNumPreGradFusion`, `BatchClampPreGradFusion`, `BatchDropoutPreGradFusion`

**Functions defined**: `update_stack_example_value`, `update_pointwise_example_value`, `__init__`, `match`, `fuse`, `register_fusion`, `decorator`, `list_group_batch_fusions`, `decompose_stack`, `__init__`, `_addmm_node_can_be_fused`, `_is_input_2d`, `match`, `fuse`, `_addmm_node_can_be_fused`, `_mm_node_can_be_fused`, `match`, `fuse`, `__init__`, `_pointwise_node_can_be_fused`

**Key imports**: collections, logging, operator, OrderedDict, Iterable, Iterator, Any, torch, counters, is_node_meta_valid, trace_structured, GraphTransformObserver


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `logging`
- `operator`
- `collections.abc`: Iterable, Iterator
- `typing`: Any
- `torch`
- `torch._dynamo.utils`: counters, is_node_meta_valid
- `torch._logging`: trace_structured
- `torch.fx.passes.graph_transform_observer`: GraphTransformObserver
- `torch.utils._ordered_set`: OrderedSet
- `..`: config
- `..utils`: OPTIMUS_EXCLUDE_POST_GRAD
- `deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  `
- `torch.fx._lazy_graph_module`: _LazyGraphModule


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/fx_passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fuse_attention.py_docs.md`](./fuse_attention.py_docs.md)
- [`efficient_conv_bn_eval.py_docs.md`](./efficient_conv_bn_eval.py_docs.md)
- [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- [`numeric_utils.py_docs.md`](./numeric_utils.py_docs.md)
- [`dedupe_symint_uses.py_docs.md`](./dedupe_symint_uses.py_docs.md)
- [`post_grad.py_docs.md`](./post_grad.py_docs.md)
- [`joint_graph.py_docs.md`](./joint_graph.py_docs.md)
- [`fsdp.py_docs.md`](./fsdp.py_docs.md)


## Cross-References

- **File Documentation**: `group_batch_fusion.py_docs.md`
- **Keyword Index**: `group_batch_fusion.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

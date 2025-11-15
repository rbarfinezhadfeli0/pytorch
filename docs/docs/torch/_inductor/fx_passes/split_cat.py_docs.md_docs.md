# Documentation: `docs/torch/_inductor/fx_passes/split_cat.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/split_cat.py_docs.md`
- **Size**: 53,419 bytes (52.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/fx_passes/split_cat.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/split_cat.py`
- **Size**: 121,453 bytes (118.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import itertools
import logging
import operator
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import free_symbols, guard_or_false
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    CallMethodVarArgs,
    FailedMatch,
    get_arg_value,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MatchContext,
    MULTIPLE,
    PatternExpr,
    PatternMatcherPass,
    register_graph_pattern,
    RepeatedExpr,
)
from .group_batch_fusion import is_node_meta_valid, POST_GRAD_FUSIONS, PRE_GRAD_FUSIONS


log = logging.getLogger(__name__)

_Arguments: TypeAlias = tuple[torch.fx.node.Argument, ...]
_TransformParam: TypeAlias = tuple[
    _Arguments | None,
    _Arguments | None,
    _Arguments | None,
    _Arguments | None,
]
_Range: TypeAlias = tuple[int, int]


PRE_GRAD_PATTERNS: dict[str, PatternMatcherPass] = {}
POST_GRAD_PATTERNS: dict[str, PatternMatcherPass] = {}

pre_grad_pass_names = [
    "normalization_pass",
    "remove_split_with_size_one_pass",
    "merge_getitem_cat_pass",
    "merge_stack_tahn_unbind_pass",
    "merge_splits_pass",
    "mutate_cat_pass",
    "split_cat_pass",
    "unbind_stack_pass",
    "split_cat_to_slices_pass",
    "unbind_cat_to_view_pass",
    "split_stack_to_cats_pass",
    "unbind_stack_to_slices_pass",
    "move_reshape_out_of_split_stack_pass",
    "einsum_to_pointwise_pass",
]

post_grad_pass_names = [
    "normalization_aten_pass",
    "decompose_mm_pass",
    "unbind_stack_aten_pass",
    "shape_padding_multiplier",
    "pad_aten_mm_pass",
    "split_cat_aten_pass",
    "select_cat_aten_pass",
    "move_view_after_cat_aten_pass",
]

backend = os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_BACKEND", "inductor")

for pass_name in pre_grad_pass_names:
    # exclude all passes from the group batch fusion
    # they do not use pattern matcher
    if pass_name in PRE_GRAD_FUSIONS:
        continue
    PRE_GRAD_PATTERNS[pass_name] = PatternMatcherPass(
        pass_name=pass_name,
    )

for pass_name in post_grad_pass_names:
    # exclude all passes from the group batch fusion
    # they do not use pattern matcher
    if pass_name in POST_GRAD_FUSIONS:
        continue
    POST_GRAD_PATTERNS[pass_name] = PatternMatcherPass(
        pass_name=pass_name,
    )


def construct_pattern_matcher_pass(pass_name: str):
    """
    Return the specific pattern_matcher_pass given the pass name.
    """
    if pass_name in PRE_GRAD_PATTERNS:
        return PRE_GRAD_PATTERNS[pass_name]
    else:
        return POST_GRAD_PATTERNS[pass_name]


def _get_split_args_default(split_node):
    input_kwarg = "tensor"
    split_size_kwarg = "split_size_or_sections"
    dim_kwarg = "dim"
    default_dim_value = 0
    if split_node.op == "call_method":
        split_size_kwarg = "split_size"
    return (
        get_arg_value(split_node, 0, input_kwarg),
        get_arg_value(split_node, 1, split_size_kwarg),
        get_arg_value(split_node, 2, dim_kwarg) or default_dim_value,
    )


def _get_dim(node: Any):
    assert isinstance(node, torch.fx.Node)
    if "dim" in node.kwargs:
        assert isinstance(node.kwargs["dim"], int)
        return node.kwargs["dim"]
    if node.target is torch.unbind:
        if len(node.args) == 2:
            assert isinstance(node.args[-1], int)
            return node.args[-1]
        return 0  # defaults to dim=0
    if node.target is torch.split:
        if len(node.args) == 3:
            assert isinstance(node.args[-1], int)
            return node.args[-1]
        return 0  # defaults to dim=0
    raise AssertionError(
        f"Can't extract `dim` from {node.target} {node.args} {node.kwargs}"
    )


# noqa: W605
# ############The pattern to be optimized is#########
#         unbind (dim=0)
#       /   ...    \
# getitem      getitem   -> user=1
#    |            |
#  split         split  -> dim=1, user=1, split_section_size=1
#    |            |
#  getitem       getitem  -> user=1
#    \           /
#        cat (dim=1)  -> user=1
#          |

# ################After transformation#############
#          unbind (dim=0)
#        /    ...   \
#    getitem       getitem  -> user=1
#       \          /
#        cat (dim=1)  -> user=1
#         |


def normalize_split_base(
    match: Match,
    _get_split_args: Callable[
        [torch.fx.Node], tuple[torch.fx.Node | None, Any | None, int | None]
    ],
):
    """
    Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
    subsequent optimizations
    """
    split_node = match.nodes[0]
    graph = match.graph
    split_input, split_size, split_dim = _get_split_args(split_node)
    if split_input is None or split_dim is None or split_size is None:
        log.debug("couldn't find split args")
        return
    if not is_node_meta_valid(split_node):
        log.debug("example value absent for node: %s", split_node)
        return
    assert isinstance(split_node.meta["example_value"], (list, tuple))
    split_sections = [t.size()[split_dim] for t in split_node.meta["example_value"]]

    if any(isinstance(section, torch.SymInt) for section in split_sections):
        # TODO dynamic_shapes with assume_static_by_default=False fails while AOT Autograd tracing.
        return
    if split_dim < 0:  # Normalize split dim
        split_dim += split_input.meta["example_value"].dim()

    new_args = (split_input, split_sections)
    new_kwargs = {"dim": split_dim}
    if (
        split_node.args == new_args
        and split_node.kwargs == new_kwargs
        and split_node.op == "call_function"
    ):
        return

    with graph.inserting_after(split_node):
        new_split_node = graph.call_function(
            torch.split,
            args=new_args,
            kwargs=new_kwargs,  # type: ignore[arg-type]
        )
    split_node.replace_all_uses_with(new_split_node)
    new_split_node.meta.update(split_node.meta)
    graph.erase_node(split_node)
    counters[backend]["normalization_pass"] += 1


@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("split", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_split_default(match: Match, *args, **kwargs):
    return normalize_split_base(match, _get_split_args_default)


@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("remove_split_with_size_one_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("split", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("remove_split_with_size_one_pass"),
)
def remove_split_with_size_one(match: Match, *args, **kwargs):
    graph = match.graph
    split_node = match.nodes[0]
    split_input, split_size, split_dim = _get_split_args_default(split_node)
    if split_input is None or split_dim is None or split_size is None:
        log.debug("couldn't find split args")
        return
    if not is_node_meta_valid(split_node):
        log.debug("example value absent for node: %s", split_node)
        return
    assert isinstance(split_node.meta["example_value"], (list, tuple))
    split_sections = [t.size()[split_dim] for t in split_node.meta["example_value"]]

    if any(isinstance(section, torch.SymInt) for section in split_sections):
        # TODO dynamic_shapes with assume_static_by_default=False fails while AOT Autograd tracing.
        return
    # remove the dummy split whose split sections size is one
    # theoretically nodes with no users should be removed, but we have seen the corner case
    # thus we add its users check to walk around the StopIteration error.
    if len(split_sections) == 1 and len(split_node.users.keys()) > 0:
        # find the grand children of the split_node
        next_users = find_next_users(split_node)
        user = next(iter(split_node.users.keys()))
        # replace the users of grand child node with the input node
        for next_user in next_users:
            next_user.replace_input_with(user, split_input)
        # erase the split node and its child
        graph.erase_node(user)
        graph.erase_node(split_node)
        counters[backend]["remove_split_with_size_one_pass"] += 1


@register_graph_pattern(
    CallFunctionVarArgs(torch.unbind, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("unbind", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_unbind_default(match: Match, *args, **kwargs):
    node = match.nodes[0]
    graph = match.graph
    input = get_arg_value(node, 0, "input")
    dim = get_arg_value(node, 1, "dim")
    if dim is None:
        axis = node.kwargs.get("axis")
        if axis is not None:
            dim = axis
        else:
            dim = 0
    if input is None:
        log.debug("couldn't find unbind args")
        return
    if not is_node_meta_valid(input):
        log.debug("example value absent for node: %s", input)
        return
    ndim = input.meta["example_value"].ndim
    # pyrefly: ignore [unsupported-operation]
    if dim < 0:  # Normalize unbind dim
        dim += ndim
    with graph.inserting_after(node):
        new_node = graph.call_function(
            torch.unbind,
            args=(input,),
            kwargs={"dim": dim},
        )
    node.replace_all_uses_with(new_node)
    new_node.meta.update(node.meta)
    graph.erase_node(node)
    counters[backend]["normalization_pass"] += 1


@register_graph_pattern(
    CallFunctionVarArgs([torch.cat, torch.concat], users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_cat_default(match: Match, *args, **kwargs):
    cat_node = match.nodes[0]
    graph = match.graph
    tensors = get_arg_value(cat_node, 0, "tensors")
    cat_dim = get_arg_value(cat_node, 1, "dim")
    if cat_dim is None:
        cat_axis = cat_node.kwargs.get("axis")
        if cat_axis is not None:
            cat_dim = cat_axis
        else:
            cat_dim = 0
    if tensors is None or cat_dim is None:
        log.debug("couldn't find cat args")
        return
    assert isinstance(tensors, (list, tuple))
    for tensor in itertools.chain([cat_node], tensors):
        if not is_node_meta_valid(tensor):
            log.debug("example value absent for node: %s", tensor)
            return

    ndim = cat_node.meta["example_value"].dim()

    def is_empty_tensor(x):
        # special case where torch.cat supports cat'ing with an empty tensor
        x_shape = x.meta["example_value"].shape
        return len(x_shape) == 1 and guard_or_false(x_shape[0] == 0)

    assert all(
        ndim == x.meta["example_value"].dim() or is_empty_tensor(x) for x in tensors
    )

    # pyrefly: ignore [unsupported-operation]
    if cat_dim < 0:  # Normalize cat dim
        cat_dim += ndim

    new_args = (tensors,)
    new_kwargs = {"dim": cat_dim}
    if (
        cat_node.args == new_args
        and cat_node.kwargs == new_kwargs
        and cat_node.op == "call_function"
        and cat_node.target is torch.cat
    ):
        return

    with graph.inserting_after(cat_node):
        new_cat_node = graph.call_function(
            torch.cat,
            args=new_args,
            kwargs=new_kwargs,
        )
    cat_node.replace_all_uses_with(new_cat_node)
    new_cat_node.meta.update(cat_node.meta)
    graph.erase_node(cat_node)
    counters[backend]["normalization_pass"] += 1


@register_graph_pattern(
    CallFunctionVarArgs(torch.stack, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_stack_default(match: Match, *args, **kwargs):
    node = match.nodes[0]
    graph = match.graph
    tensors = get_arg_value(node, 0, "tensors")
    dim = get_arg_value(node, 1, "dim") or 0
    if tensors is None or dim is None:
        log.debug("couldn't find stack args")
        return
    assert isinstance(tensors, (list, tuple))

    # A bug in pytorch, some nodes miss the example_value metadata
    for tensor in itertools.chain([node], tensors):
        if not is_node_meta_valid(tensor):
            log.debug("example value absent for node: %s", tensor)
            return

    ndim = node.meta["example_value"].dim()
    if dim < 0:  # Normalize dim
        dim += ndim

    with graph.inserting_after(node):
        new_node = graph.call_function(
            node.target,  # type: ignore[arg-type]
            args=(tensors,),
            kwargs={"dim": dim},
        )
    node.replace_all_uses_with(new_node)
    new_node.meta.update(node.meta)
    graph.erase_node(node)
    counters[backend]["normalization_pass"] += 1


def find_next_users(split_node: torch.fx.Node) -> list[torch.fx.Node]:
    next_users = []
    for getitem_node in split_node.users:
        for getitem_user in getitem_node.users:
            if getitem_user not in next_users:
                next_users.append(getitem_user)
    return next_users


@register_graph_pattern(
    CallMethodVarArgs("squeeze", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_squeeze_default(match: Match, *args, **kwargs):
    squeeze_node = match.nodes[0]
    squeeze_input = get_arg_value(squeeze_node, 0)

    if "dim" in squeeze_node.kwargs:
        assert len(squeeze_node.args) == 1
        dim = squeeze_node.kwargs["dim"]
    elif len(squeeze_node.args) == 1:
        # squeeze(Tensor)
        dim = None
    elif len(squeeze_node.args) == 2:
        # squeeze(Tensor self, int dim)
        # squeeze(Tensor self, int[] dim)
        dim = squeeze_node.args[1]
    else:
        # squeeze(Tensor self, int[] dim) (called with varargs)
        dim = squeeze_node.args[1:]

    if isinstance(dim, Sequence) and len(dim) == 1:
        dim = dim[0]

    with match.graph.inserting_after(squeeze_node):
        if dim is None:
            new_squeeze_node = match.graph.call_function(
                torch.squeeze, args=(squeeze_input,)
            )
        else:
            new_squeeze_node = match.graph.call_function(
                torch.squeeze, args=(squeeze_input,), kwargs={"dim": dim}
            )
    squeeze_node.replace_all_uses_with(new_squeeze_node)
    new_squeeze_node.meta.update(squeeze_node.meta)
    match.graph.erase_node(squeeze_node)


@register_graph_pattern(
    CallMethodVarArgs("reshape", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_reshape_default(match: Match, *args, **kwargs):
    reshape_node = match.nodes[0]
    if not is_node_meta_valid(reshape_node):
        log.debug("example value absent for node: %s", reshape_node)
        return
    reshape_input = get_arg_value(reshape_node, 0)

    if free_symbols(reshape_node.meta["example_value"].shape):
        log.debug("dynamic shape not supported: %s", reshape_node)
        return

    with match.graph.inserting_after(reshape_node):
        new_reshape_node = match.graph.call_function(
            torch.reshape,
            args=(reshape_input, tuple(reshape_node.meta["example_value"].shape)),
        )
    reshape_node.replace_all_uses_with(new_reshape_node)
    new_reshape_node.meta.update(reshape_node.meta)
    match.graph.erase_node(reshape_node)


@register_graph_pattern(
    CallMethodVarArgs("clamp", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallFunctionVarArgs(torch.clamp, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_clamp_default(match: Match, *args, **kwargs):
    clamp_node = match.nodes[0]
    if not is_node_meta_valid(clamp_node):
        log.debug("example value absent for node: %s", clamp_node)
        return

    if free_symbols(clamp_node.meta["example_value"].shape):
        log.debug("dynamic shape not supported: %s", clamp_node)
        return
    if len(clamp_node.args) > 1:
        args = (get_arg_value(clamp_node, 0),)
        kwargs = {
            "min": get_arg_value(clamp_node, 1, kwarg_name="min"),
            "max": get_arg_value(clamp_node, 2, kwarg_name="max"),
        }
    else:
        args = clamp_node.args
        kwargs = clamp_node.kwargs
    with match.graph.inserting_after(clamp_node):
        new_clamp_node = match.graph.call_function(
            torch.clamp,
            args=args,
            kwargs=kwargs,
        )
    clamp_node.replace_all_uses_with(new_clamp_node)
    new_clamp_node.meta.update(clamp_node.meta)
    match.graph.erase_node(clamp_node)


@register_graph_pattern(
    CallMethodVarArgs("detach", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_detach_default(match: Match, *args, **kwargs):
    detach_node = match.nodes[0]
    if not is_node_meta_valid(detach_node):
        log.debug("example value absent for node: %s", detach_node)
        return

    if free_symbols(detach_node.meta["example_value"].shape):
        log.debug("dynamic shape not supported: %s", detach_node)
        return

    with match.graph.inserting_after(detach_node):
        new_detach_node = match.graph.call_function(
            torch.detach,
            args=detach_node.args,
        )
    detach_node.replace_all_uses_with(new_detach_node)
    new_detach_node.meta.update(detach_node.meta)
    match.graph.erase_node(detach_node)


class TorchSplit(CallFunction):
    """
    Matches a call to torch.split if it is in a normalized form. Ensures that all users of
    splits are unique getitems.
    """

    def __init__(self, arg, sizes, func=torch.split) -> None:
        # using KeywordArg("dim") for `dim` checks they all match
        super().__init__(func, arg, sizes, _users=MULTIPLE, dim=KeywordArg("dim"))

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = super()._match(node, ctx)
        if not m:
            return m
        split_sections = node.args[1]
        if not isinstance(split_sections, (list, tuple)):
            return FailedMatch("split not normalized")
        # check users are all unique getitems
        seen_idxs = OrderedSet[int]()
        for user in node.users:
            if not CallFunction(operator.getitem, Arg(), Arg()).match(user):
                # This should ideally never happen. Split user should always be a getitem
                return FailedMatch(f"user of split not a getitem: {user}")
            if not isinstance(user.args[1], int):
                return FailedMatch("only integer getitems are handled")
            if user.args[1] in seen_idxs:
                return FailedMatch(f"duplicate getitem {user.args[1]}")
            if user.args[-1] < 0:  # type: ignore[operator]
                # This shouldn't ideally happen as dynamo normalizes indexes to positive
                return FailedMatch("negative index")
            seen_idxs.add(user.args[1])
        return m


@register_graph_pattern(
    TorchSplit(
        CallFunction(
            operator.getitem,
            TorchSplit(
                KeywordArg("first_split_input"),
                KeywordArg("first_split_sections"),
            ),
            Ignored(),
        ),
        KeywordArg("next_split_sections"),
    ),
    pass_dict=construct_pattern_matcher_pass("merge_splits_pass"),
)
def merge_splits(
    match: Match,
    first_split_input: torch.fx.Node,
    first_split_sections: list[int],
    next_split_sections: list[int],
    # Note: dim is implicitly passed by TorchSplit, as it internally uses a pattern with dim
    dim: int,
):
    node = match.output_node()
    # it is possible that the split has no users,
    # we check the corner case and skip the pattern
    if len(node.users.keys()) == 0:
        return
    graph = match.graph
    first_split = node.args[0].args[0]  # type: ignore[union-attr]
    next_split_index = node.args[0].args[1]  # type: ignore[union-attr]

    new_split_sections = list(first_split_sections)
    new_split_sections[next_split_index : next_split_index + 1] = next_split_sections  # type: ignore[operator, misc]

    first_split_dim = _get_dim(first_split)

    to_remove = []

    with graph.inserting_before(first_split):  # type: ignore[arg-type]
        # Add the new split node
        new_split = graph.call_function(
            torch.split,
            args=(first_split_input, new_split_sections),
            kwargs={"dim": first_split_dim},
        )
        if is_node_meta_valid(first_split_input):
            new_split.meta["example_value"] = torch.split(
                first_split_input.meta["example_value"],
                new_split_sections,
                dim=first_split_dim,
            )
        first_split_num_to_user = {
            user.args[1]: user
            for user in first_split.users  # type: ignore[union-attr]
        }

        new_split_num = 0
        for split_num in range(len(first_split_sections)):
            if split_num not in first_split_num_to_user:
                new_split_num += 1
                continue
            old_getitem = first_split_num_to_user[split_num]
            if split_num != next_split_index:
                old_getitem.update_arg(0, new_split)
                old_getitem.update_arg(1, new_split_num)
                new_split_num += 1
            else:
                next_split_num_to_user = {user.args[1]: user for user in node.users}
                # It is not necessary all getitems from the split node are used.
                for next_split_num in range(len(next_split_sections)):
                    with graph.inserting_after(new_split):
                        new_getitem = graph.call_function(
                            operator.getitem, args=(new_split, new_split_num)
                        )
                    new_split_num += 1
                    if next_split_num not in next_split_num_to_user:
                        continue
                    next_getitem = next_split_num_to_user[next_split_num]
                    new_getitem.meta.update(next_getitem.meta)
                    next_getitem.replace_all_uses_with(new_getitem)
                    to_remove.append(next_getitem)
                to_remove.append(node)
                to_remove.append(old_getitem)

        to_remove.append(first_split)  # type: ignore[arg-type]
    for node in to_remove:
        graph.erase_node(node)

    counters[backend]["merge_splits_pass"] += 1


class SplitCatSimplifier:
    """
    Helper class to simplify split-cat pattern. In simple cases, both split and cat node can be removed in a "split->cat"
    pattern. However, there are various cases where they can't and we need to simplify split/ add transforms before cat.
    Some such cases are:
        1. Final node has additional args (not coming from the initial split)
        2. Shuffling of args between split/cat
        3. Some final nodes are non-(cat/stack)
        4. Split-dim != cat-dim (but equal split)

    Note that any combination of the above cases can happen.

    To deal with 1, 2, & 3 - we iterate over all users of split. And figure out common "ranges" that can be merged.
    Then, we simplify the split accordingly. In the best case, split can be entirely removed.

    To deal with 4, we add some transformations (unflatten + movedim) (See `get_transform_params`).

    Finally, depending on final node being cat or stack, unsqueeze/flatten needs to be added.

    """

    def simplify(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: list[int],
    ):
        # Find the next users (i.e. users after the getitem)
        next_users = find_next_users(split_node)
        # Gather inputs of the next users. When inputs come from `split_node`, they are instead represented by
        # a tuple indicating the split ranges. See `get_user_input_list` for more details
        user_inputs_list = self.get_user_input_list(split_node, next_users)
        # Simplify the split_sections based on user_inputs_list. In simpler cases, len(simplified_split_ranges) == 1 and
        # we can simply replace the split node. Otherwise, we simplify it.
        simplified_split_ranges = self.get_simplified_split_ranges(
            split_sections, next_users, user_inputs_list
        )
        if not simplified_split_ranges:  # Simplification not possible
            return
        transform_params_list = self.get_transform_params(
            split_node, next_users, user_inputs_list
        )
        if not transform_params_list:
            return

        # Start actual replacement
        user_inputs_list_new = self.replace_split(
            graph, split_node, split_sections, user_inputs_list, simplified_split_ranges
        )
        self.replace_cat(
            graph,
            split_node,
            next_users,
            user_inputs_list_new,
            transform_params_list,  # type: ignore[arg-type]
        )
        self.erase_old_nodes(graph, split_node, next_users)  # type: ignore[arg-type]
        counters[backend]["unbind_stack_pass"] += 1

    def get_user_input_list(
        self, split_node: torch.fx.Node, next_users: list[torch.fx.Node]
    ) -> list[list[torch.fx.Node | _Range]]:
        """
        Returns list of inputs to the following user nodes, in order. The outer list represents the user node. The inner
        list represents the inputs to that particular node. This list can either contain
          - a tuple representing the ranges of get_items that should go into the cat (closed interval)
          - torch.fx.Node representing "other" inputs (which are not coming from our split)
        """
        user_inputs_list: list[list[torch.fx.Node | _Range]] = []
        for user in next_users:
            if user.target in (torch.cat, torch.stack):
                user_inputs_list.append(self.get_merged_user_inputs(split_node, user))
            else:
                user_inputs_list.append(self.get_non_cat_node_input(split_node, user))  # type: ignore[arg-type]
        return user_inputs_list

    def get_merged_user_inputs(
        self, split_node: torch.fx.Node, cat_node: torch.fx.Node
    ) -> list[torch.fx.Node | _Range]:
        user_inputs = get_arg_value(cat_node, 0, "tensors")
        simplified_user_inputs = []
        split_users = OrderedSet(split_node.users.keys())
        for user_input in user_inputs:
            if user_input not in split_users:
                simplified_user_inputs.append(user_input)
            else:
                # Add which "getitem" cat depends on
                simplified_user_inputs.append(user_input.args[1])
        return self.merge_consecutive_inputs(simplified_user_inputs)

    def get_non_cat_node_input(
        self, split_node: torch.fx.Node, node: torch.fx.Node
    ) -> list[_Range]:
        """
        Get input for a non cat node in the same format as `get_merged_user_inputs`
        """
        node_input = []
        split_users = OrderedSet(split_node.users.keys())
        for node_arg in node.all_input_nodes:
            if node_arg in split_users:
                getitem_num = get_arg_value(node_arg, 1)
                node_input.append((getitem_num, getitem_num))
        return node_input

    def merge_consecutive_inputs(
        self, inputs: list[torch.fx.Node | int]
    ) -> list[torch.fx.Node | _Range]:
        """
        Merge consecutive inputs going into a user node.

        For e.g.
        [arg0, 0, 1, 2, arg1] -> [arg0, (0, 2), arg1]
        """
        merged_ranges = []
        cur_range = None
        for input_ in inputs:
            if isinstance(input_, int):
                if not cur_range:
                    cur_range = [input_, input_]
                elif input_ == cur_range[1] + 1:
                    cur_range[1] += 1
                else:
                    merged_ranges.append(tuple(cur_range))
                    cur_range = [input_, input_]
            else:
                if cur_range:
                    merged_ranges.append(tuple(cur_range))
                    cur_range = None
                merged_ranges.append(input_)  # type: ignore[arg-type]
        if cur_range:
            merged_ranges.append(tuple(cur_range))
        return merged_ranges  # type: ignore[return-value]

    def get_simplified_split_ranges(
        self,
        split_sections,
        next_users,
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[_Range] | None:
        ranges = OrderedSet[Any]()
        for user_inputs in user_inputs_list:
            ranges.update(u for u in user_inputs if isinstance(u, tuple))

        cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()
        split_ranges = sorted(
            [(cumulative_sizes[r[0]], cumulative_sizes[r[1] + 1]) for r in ranges]
        )

        if not self.has_non_overlapping_ranges(
            split_ranges,
        ):  # This need not be a strict condition
            # However, we keep it now for simplicity.
            return None
        split_ranges = self.fill_gaps(split_ranges, 0, cumulative_sizes[-1])
        if len(split_sections) == len(split_ranges):  # Simplification not possible
            return None
        counters[backend]["scmerge_split_sections_removed"] = len(split_sections) - len(
            split_ranges
        )
        return split_ranges

    def has_non_overlapping_ranges(self, ranges: list[_Range]) -> bool:
        for range_, next_range in itertools.pairwise(ranges):
            if range_[1] > next_range[0]:
                return False
        return True

    def fill_gaps(self, ranges: list[_Range], min_: int, max_: int) -> list[_Range]:
        cur = min_
        filled_ranges = []
        for a, b in ranges:
            if cur < a:
                filled_ranges.append((cur, a))
            filled_ranges.append((a, b))
            cur = b
        if filled_ranges[-1][1] < max_:
            filled_ranges.append((filled_ranges[-1][1], max_))
        return filled_ranges

    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[list[_TransformParam]] | None:
        """
        Figure out what transforms are needed for each input to each cat node.

        We replace a split node with an unflatten followed by a movedim
        """
        split_dim = _get_dim(split_node)
        split_sections = split_node.args[1]
        transform_params_list: list[list[_TransformParam]] = []

        for user_node, user_inputs in zip(next_users, user_inputs_list):
            if user_node.target not in (torch.cat, torch.stack):
                transform_params_list.append([])
                continue

            cat_dim = get_arg_value(user_node, 1, "dim")
            transform_params: list[_TransformParam] = []
            for user_input in user_inputs:
                if split_dim == cat_dim and user_node.target is torch.cat:
                    # No transform needed
                    transform_params.append((None, None, None, None))
                elif isinstance(user_input, tuple):  # Split being simplified
                    # Verify equal split
                    subset_split_sections = split_sections[  # type: ignore[index]
                        user_input[0] : user_input[1]
                        + 1  # type: ignore[index]
                    ]
                    # All sections should be equal
                    if len(OrderedSet(subset_split_sections)) != 1:  # type: ignore[arg-type]
                        return None

                    num_splits = len(subset_split_sections)  # type: ignore[arg-type]
                    unflatten_params = (split_dim, (num_splits, -1))
                    movedim_params = (
                        (split_dim, cat_dim) if split_dim != cat_dim else None
                    )
                    transform_params.append(
                        (unflatten_params, movedim_params, None, None)
                    )
                elif (
                    user_node.target is torch.stack or split_dim != cat_dim
                ):  # We need to unsqueeze inputs not coming through split
                    transform_params.append((None, None, (cat_dim,), None))
                else:  # Non-split inputs
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list

    def replace_split(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: list[int],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
        split_ranges: list[_Range],
    ) -> list[list[torch.fx.Node]]:
        """
        Replace the split node. It can either remove the split node if len(split_ranges) == 1, or simplify it
        into a split with lesser sections if len(split_ranges) > 1.

        Returns the new `user_inputs_list`, with tuples replaced with new getitems from the newer split node.
        """
        split_input = split_node.args[0]
        split_dim = _get_dim(split_node)
        if len(split_ranges) == 1:  # We can completely eliminate the split node
            split_items = [split_input]
        else:
            with graph.inserting_after(split_node):
                new_split = graph.call_function(
                    torch.split,
                    args=(
                        split_input,
                        [r[1] - r[0] for r in split_ranges],
                    ),
                    kwargs={"dim": split_dim},
                )
                if is_node_meta_valid(split_input):  # type: ignore[arg-type, union-attr]
                    new_split.meta["example_value"] = torch.split(
                        split_input.meta["example_value"],  # type: ignore[union-attr]
                        [r[1] - r[0] for r in split_ranges],
                        dim=split_dim,
                    )
                counters[backend]["scmerge_split_added"] += 1
            split_items = []
            with graph.inserting_after(new_split):
                for i in range(len(split_ranges)):
                    getitem = graph.call_function(operator.getitem, args=(new_split, i))
                    if is_node_meta_valid(new_split):
                        getitem.meta["example_value"] = new_split.meta["example_value"][
                            i
                        ]
                        split_items.append(getitem)
        # Now assign the right getitem to the right input
        cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()
        new_user_inputs_list = []
        for user_inputs in user_inputs_list:
            new_user_inputs = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    # Find the correct new getitem (present in split_items)
                    new_user_inputs.append(
                        # pyrefly: ignore [bad-argument-type]
                        split_items[
                            split_ranges.index(
                                (
                                    cumulative_sizes[user_input[0]],
                                    cumulative_sizes[user_input[1] + 1],
                                )
                            )
                        ]
                    )
                else:
                    new_user_inputs.append(user_input)
            new_user_inputs_list.append(new_user_inputs)
        return new_user_inputs_list  # type: ignore[return-value]

    def replace_cat(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
        user_inputs_list_new,
        transform_params_list: list[list[_TransformParam]],
    ):
        split_dim = _get_dim(split_node)
        split_users = split_node.users.keys()
        new_cats = []
        for user_node, user_inputs_new, transform_params in zip(
            next_users, user_inputs_list_new, transform_params_list
        ):
            if user_node.target not in (torch.cat, torch.stack):
                # Change the args and kwargs of non-cat/stack nodes. Replace old getitems (belonging to
                # the original split node) with the newer getitems
                next_cat_input = 0
                for input_node in user_node.all_input_nodes:
                    if input_node in split_users:
                        user_node.replace_input_with(
                            input_node, user_inputs_new[next_cat_input]
                        )
                        next_cat_input += 1
                continue

            # Handle cat/stack user nodes
            cat_dim = get_arg_value(user_node, 1, "dim")
            user_inputs_new_transformed, user_inputs_new_transformed_meta = [], []
            # For `unsqueeze` transform, we will combine consecutive inputs with the same unsqueeze params, and stack them
            to_stack, to_stack_meta = [], []
            stack_dim = None
            with graph.inserting_before(user_node):
                for user_input_new, transform_param in zip(
                    user_inputs_new, transform_params
                ):
                    # pyrefly: ignore [bad-argument-type]
                    if not is_node_meta_valid(user_input_new):
                        log.debug("example value absent for node: %s", user_input_new)
                        return
                    # Apply transforms
                    (
                        unflatten_params,
                        movedim_params,
                        unsqueeze_params,
                        flatten_params,
                    ) = transform_param
                    if unsqueeze_params and (
                        stack_dim is None or stack_dim == unsqueeze_params[0]
                    ):
                        to_stack.append(user_input_new)
                        # pyrefly: ignore [missing-attribute]
                        to_stack_meta.append(user_input_new.meta["example_value"])
                        stack_dim = unsqueeze_params[0]
                        continue
                    elif to_stack:
                        stacked_input = graph.call_function(
                            torch.stack, args=(to_stack,), kwargs={"dim": stack_dim}
                        )
                        stacked_input.meta["example_value"] = torch.stack(  # type: ignore[arg-type]
                            to_stack_meta,
                            dim=stack_dim,  # type: ignore[arg-type]
                        )
                        to_stack, to_stack_meta = [], []
                        stack_dim = None
                        user_inputs_new_transformed.append(stacked_input)
                        user_inputs_new_transformed_meta.append(
                            stacked_input.meta["example_value"]
                        )
                        if unsqueeze_params:
                            to_stack.append(user_input_new)
                            stack_dim = unsqueeze_params[0]
                            # pyrefly: ignore [missing-attribute]
                            to_stack_meta.append(user_input_new.meta["example_value"])
                            continue

                    if unflatten_params:
                        # pyrefly: ignore [missing-attribute]
                        user_input_new_meta = user_input_new.meta["example_value"]
                        user_input_new = graph.call_function(
                            torch.unflatten, args=(user_input_new, *unflatten_params)
                        )
                        user_input_new.meta["example_value"] = torch.unflatten(  # type: ignore[arg-type]
                            user_input_new_meta,  # type: ignore[arg-type]
                            *unflatten_params,  # type: ignore[arg-type]
                        )
                    if movedim_params:
                        # pyrefly: ignore [missing-attribute]
                        user_input_new_meta = user_input_new.meta["example_value"]
                        user_input_new = graph.call_function(
                            torch.movedim, args=(user_input_new, *movedim_params)
                        )
                        user_input_new.meta["example_value"] = torch.movedim(  # type: ignore[arg-type]
                            user_input_new_meta,  # type: ignore[arg-type]
                            *movedim_params,  # type: ignore[arg-type]
                        )
                    if flatten_params:
                        # pyrefly: ignore [missing-attribute]
                        user_input_new_meta = user_input_new.meta["example_value"]
                        user_input_new = graph.call_function(
                            torch.flatten, args=(user_input_new, *flatten_params)
                        )
                        user_input_new.meta["example_value"] = torch.flatten(  # type: ignore[arg-type]
                            user_input_new_meta,
                            *flatten_params,  # type: ignore[arg-type]
                        )
                    user_inputs_new_transformed.append(user_input_new)
                    user_inputs_new_transformed_meta.append(
                        # pyrefly: ignore [missing-attribute]
                        user_input_new.meta["example_value"]
                    )
                if to_stack:
                    stacked_input = graph.call_function(
                        torch.stack, args=(to_stack,), kwargs={"dim": stack_dim}
                    )
                    stacked_input.meta["example_value"] = torch.stack(  # type: ignore[arg-type]
                        to_stack_meta,
                        dim=stack_dim,  # type: ignore[arg-type]
                    )
                    user_inputs_new_transformed.append(stacked_input)
                    user_inputs_new_transformed_meta.append(
                        stacked_input.meta["example_value"]
                    )

            with graph.inserting_after(user_node):
                if len(user_inputs_new_transformed) > 1:
                    new_cat_node = graph.call_function(
                        torch.cat,
                        args=(user_inputs_new_transformed,),
                        kwargs={"dim": cat_dim},
                    )
                    new_cat_node.meta["example_value"] = torch.cat(
                        user_inputs_new_transformed_meta,
                        dim=cat_dim,
                    )
                    counters[backend]["scmerge_cat_added"] += 1
                else:
                    new_cat_node = user_inputs_new_transformed[-1]
                    new_cat_node.meta["example_value"] = (
                        user_inputs_new_transformed_meta[-1]
                    )

            if (
                user_node.target is torch.cat
                and split_dim != cat_dim
                and split_node.target is torch.split
            ):
                with graph.inserting_after(new_cat_node):
                    new_cat_node_meta = new_cat_node.meta["example_value"]
                    new_cat_node = graph.call_function(
                        torch.flatten, args=(new_cat_node, cat_dim, cat_dim + 1)
                    )
                    new_cat_node.meta["example_value"] = torch.flatten(
                        new_cat_node_meta,
                        cat_dim,
                        cat_dim + 1,
                    )
            user_node.replace_all_uses_with(new_cat_node)
            new_cats.append(new_cat_node)

    def erase_old_nodes(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
    ):
        to_remove = [split_node]
        counters[backend]["scmerge_split_removed"] += 1
        to_remove.extend(split_node.users.keys())
        for next_user in next_users:
            if next_user.target not in (torch.cat, torch.stack):
                continue
            counters[backend]["scmerge_cat_removed"] += 1
            to_remove.append(next_user)
        for node in reversed(to_remove):
            if len(node.users.keys()) == 0:
                graph.erase_node(node)


class UnbindCatRemover(SplitCatSimplifier):
    """
    Helper class to merge Unbind->Cat/Stack. Many of the cases are similar to SplitCatSimplifier.

    Unbind can't be simplified like splits. So, we can only remove the unbind node. Other than this,
    other cases like multiple users, additional args, dim mismatch are similar to `SplitCatSimplifier`,
    hence we extend that class.
    """

    def remove_unbind(
        self,
        graph: torch.fx.Graph,
        unbind_node: torch.fx.Node,
    ):
        if not is_node_meta_valid(unbind_node):
            return
        # we need to check if the getitem indices from unbind are consecutive and all go to the same cat node
        # before we do the unbind remove, otherwise it will hit the error when we unbind part of them
        getitem_indices = [getitem_node.args[1] for getitem_node in unbind_node.users]
        if not is_sorted_and_consecutive(getitem_indices) or len(  # type: ignore[arg-type]
            getitem_indices
        ) != len(unbind_node.meta["example_value"]):
            return
        num_unbind = len(getitem_indices)
        split_sections = [1 for _ in range(num_unbind)]  # type: ignore[operator, arg-type]

        super().simplify(graph, unbind_node, split_sections)

    def get_simplified_split_ranges(
        self,
        split_sections: list[int],
        next_users: list[torch.fx.Node],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[_Range] | None:
        simplified_split_ranges = super().get_simplified_split_ranges(
            split_sections, next_users, user_inputs_list
        )
        if not simplified_split_ranges or len(simplified_split_ranges) != 1:
            return None
        return simplified_split_ranges

    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: list[torch.fx.Node],
        user_inputs_list: list[list[torch.fx.Node | _Range]],
    ) -> list[list[_TransformParam]] | None:
        """
        Figure out what transforms are needed for each input to each cat node.

        Here is the rough transforms we apply:

        x -> unbind -> stack => x -> movedim

        x -> unbind -> cat => x -> movedim -> flatten

        When cat/stack nodes have additional args:

             addn ---|              addn -> unsqueeze ---|
        x -> unbind -> stack  =>           x -> movedim  -> cat

             addn ---|                            addn ---|
        x -> unbind -> cat  =>   x -> movedim -> flatten  -> cat

        (Note application of these depends on the dims as well)


        """
        split_dim = _get_dim(split_node)
        transform_params_list: list[list[_TransformParam]] = []
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            cat_dim = get_arg_value(user_node, 1, "dim") or 0
            transform_params: list[_TransformParam] = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    # User input is coming from unbind
                    movedim_params = (
                        (split_dim, cat_dim) if split_dim != cat_dim else None
                    )
                    flatten_params = None
                    if user_node.target is torch.cat:
                        flatten_params = (cat_dim, cat_dim + 1)
                    transform_params.append(
                        (None, movedim_params, None, flatten_params)
                    )
                elif (
                    user_node.target is torch.stack
                ):  # We need to unsqueeze inputs not coming through unbind into cat
                    transform_params.append((None, None, (cat_dim,), None))
                else:  # Non-unbind inputs
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list


class GetItem(CallFunction):
    def __init__(self, arg, index, _users=1) -> None:
        super().__init__(operator.getitem, arg, index, _users=_users)

    def find_anchor_nodes(self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]):
        # We generally match GetItem with arg being an Arg(). So, we never return the anchor
        # nodes as the stored node in ctx.pattern_to_node is returned. Here we override find_anchor_nodes
        # to not use ctx.pattern_to_node
        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/fx_passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/_inductor/fx_passes`):

- [`dedupe_symint_uses.py_kw.md_docs.md`](./dedupe_symint_uses.py_kw.md_docs.md)
- [`overlap_preserving_bucketer.py_kw.md_docs.md`](./overlap_preserving_bucketer.py_kw.md_docs.md)
- [`pre_grad.py_docs.md_docs.md`](./pre_grad.py_docs.md_docs.md)
- [`b2b_gemm.py_docs.md_docs.md`](./b2b_gemm.py_docs.md_docs.md)
- [`freezing_patterns.py_kw.md_docs.md`](./freezing_patterns.py_kw.md_docs.md)
- [`fsdp.py_docs.md_docs.md`](./fsdp.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`replace_random.py_kw.md_docs.md`](./replace_random.py_kw.md_docs.md)
- [`joint_graph.py_kw.md_docs.md`](./joint_graph.py_kw.md_docs.md)
- [`numeric_utils.py_docs.md_docs.md`](./numeric_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `split_cat.py_docs.md_docs.md`
- **Keyword Index**: `split_cat.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

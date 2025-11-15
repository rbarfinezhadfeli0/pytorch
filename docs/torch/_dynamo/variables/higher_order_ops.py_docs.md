# Documentation: `torch/_dynamo/variables/higher_order_ops.py`

## File Metadata

- **Path**: `torch/_dynamo/variables/higher_order_ops.py`
- **Size**: 175,315 bytes (171.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

"""
This module contains classes and utilities for handling higher-order operators in Dynamo.
It provides functionality for tracing and transforming control flow constructs like
conditions (torch.cond), loops (torch.while_loop), maps (torch.ops.higher_order.map),
and other higher-order operations.

The module includes specialized VariableTracker classes for different types of
higher-order operations, along with utilities for:
- Speculating and capturing subgraphs
- Managing control flow
- Handling autograd function applications
- Supporting function transformations
- Processing activation checkpoints

These classes work together to enable Dynamo to correctly trace and compile code
containing complex control flow patterns and higher-order functions while preserving
their semantic behavior.
"""

import contextlib
import functools
import inspect
import itertools
import logging
import types
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, TYPE_CHECKING

import torch._C
import torch.fx
import torch.nn
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import get_fake_value
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.ctx_manager import RepararametrizeModuleContextVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch._ops import HigherOrderOperator
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree

from .. import graph_break_hints, variables
from ..exc import (
    ObservedException,
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
)
from ..source import AttrSource, DictGetItemSource
from ..utils import proxy_args_kwargs, set_example_value
from .base import VariableTracker
from .dicts import ConstDictVariable
from .lazy import LazyVariableTracker
from .lists import ListVariable, TupleVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")


@dataclass
class OutputSpec:
    """
    Contains the treespec of the output of the speculated subgraph, and the
    information to mask out the constant values from the output during
    flattening and inserting them back during unflattening. Cleaning up
    constants from the graph makes the graph simpler for AOTDispatcher and
    Inductor.
    """

    treespec: pytree.TreeSpec
    # list of True/False to identify the locations of const values in the
    # subgraph output. True means that value at that index is a constant.
    masks_to_filter_const_values: Optional[list[bool]] = None
    # The actual constant values that were present in the subgraph output. Note
    # that this is the same length as the mask, we just look at the indices
    # where mask is True.
    const_values: Optional[list[Any]] = None
    # Number of intermediate nodes that are also made subgraph outputs.
    num_intermediate_nodes_as_outputs: int = 0

    def __post_init__(self):
        if (
            self.masks_to_filter_const_values is not None
            or self.const_values is not None
        ):
            assert len(self.masks_to_filter_const_values) == len(self.const_values)


def raise_hard_error_if_graph_break(reason):
    def deco(fn):
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (Unsupported, ObservedException) as e:
                import sys

                if isinstance(e, Unsupported):
                    exc = UncapturedHigherOrderOpError(
                        f"{reason} Got {e.msg}", e.real_stack
                    )
                else:
                    msg = e.msg if hasattr(e, "msg") else type(e)
                    real_stack = e.real_stack if hasattr(e, "real_stack") else None
                    exc = UncapturedHigherOrderOpError(
                        f"{reason} Got {msg}", real_stack
                    )
                raise exc.with_traceback(sys.exc_info()[2]) from None

        return graph_break_as_hard_error

    return deco


# This function is a syntax sugar for creating a dummy new subtracer so that
# newly added nodes are added to a separate subgraph in this subtracer instead of affecting
# the main graph. This is useful for creating sample inputs for tracing the subgraph.
# For example, in FlexAttentionHigherOrderVariable, we want to create several scalars
# to trace the score_mod function but we don't want the operators that creates the scalar to
# show up in the graph, we could this function to discard the graph changes.
# Example usage:
# with discard_graph_changes():
#   sample_input= create_sample_inputs()
# speculate_subgraph(tx, f, sample_inputs, {})
@contextlib.contextmanager
def discard_graph_changes(tx):
    ctx = tx.output.subtracer("subgraph_wrapper", None)
    try:
        ctx.__enter__()
        yield
    finally:
        ctx.__exit__(None, None, None)


def check_meta_consistency_vt(
    vars1: list[VariableTracker],
    vars2: list[VariableTracker],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    from torch._higher_order_ops.utils import check_meta_consistency

    from . import TensorVariable

    def _unwrap_var(var):
        if isinstance(var, TensorVariable):
            return var.proxy.node.meta["example_value"]
        elif isinstance(var, SymNodeVariable):
            return var.sym_num
        elif isinstance(var, ConstantVariable):
            return var.as_python_constant()
        else:
            unimplemented(
                gb_type="cannot unwrap variable for check_meta_consistency",
                context=str(var),
                explanation=f"Expected {var} to be TensorVariable, SymNodeVariable, or ConstantVariable",
                hints=[],
            )

    unwrapped1 = [_unwrap_var(var) for var in vars1]
    unwrapped2 = [_unwrap_var(var) for var in vars2]

    return check_meta_consistency(
        unwrapped1,
        unwrapped2,
        lhs_name,
        rhs_name,
        include_contiguity=include_contiguity,
    )


@contextlib.contextmanager
def dynamo_enable_grad(tx: "InstructionTranslator", enable=True):
    from . import GradModeVariable

    org_value = torch.is_grad_enabled()
    try:
        GradModeVariable.create(tx, enable, initialized=True)
        yield
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)


@contextlib.contextmanager
def dynamo_under_activation_checkpoint(tx: "InstructionTranslator"):
    orig_val = tx.output.current_tracer.under_activation_checkpoint
    try:
        tx.output.current_tracer.under_activation_checkpoint = True
        yield
    finally:
        tx.output.current_tracer.under_activation_checkpoint = orig_val


def find_mismatched_vars(var, types, allow_none=False):
    """
    Recursively finds variables whose type is not an instance of the specified types.
    Args:
        var: The variable to check.
        types: A tuple of allowed types.
        allow_none (bool): Whether to allow None values. Defaults to False.
    Returns:
        A set of variables whose type is not an instance of the specified types.
    """
    mismatched_vars = set()
    if isinstance(var, (list, tuple)):
        for item in var:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, (TupleVariable, ListVariable)):
        for item in var.items:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, ConstDictVariable):
        for value in var.items.values():
            mismatched_vars.update(find_mismatched_vars(value, types, allow_none))
    else:

        def _is_none(var):
            return var.is_python_constant() and var.as_python_constant() is None

        if not isinstance(var, types) and not (allow_none and _is_none(var)):
            mismatched_vars.add(var)
    return mismatched_vars


def only_consist_of(var, types, allow_none=False):
    mismatch_vars = find_mismatched_vars(var, types, allow_none=allow_none)
    return len(mismatch_vars) == 0


# A more read-able syntax sugar for creating a UserFunctionVariable for f
# and run call_function on it. Make it return a function to preserve the calling
# convention of the original f.
def _make_inlined(tx: "InstructionTranslator", f):
    assert callable(f), "Expect f to be a python callable."

    def inline_call(*args, **kwargs):
        return UserFunctionVariable(f).call_function(tx, args, kwargs)

    return inline_call


def _call_function_with_auto_output_flattening(
    tx: "InstructionTranslator",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    body_r: Optional[VariableTracker],
    graph_output_vts: VariableTracker | tuple[VariableTracker, ...],
) -> Optional[VariableTracker]:
    """
    Create HOP call node and reproxify output VTs for HOPs with auto output semantics.

    This function is used by HOPs with auto output semantics (see speculate_subgraph_with_auto_output_flattening)
    to create the actual HOP call in the FX graph and properly handle the output variable trackers.

    The key operation is "reproxifying" - updating the proxies of the original tensor VTs
    (from body_r) to point to the HOP call outputs, ensuring the outer graph correctly
    references the HOP outputs while allowing body_r to contain arbitrary Python objects.

    Args:
        tx: The instruction translator
        fn: The HOP function to call
        args: Arguments for the HOP call (typically includes the subgraph node)
        kwargs: Keyword arguments for the HOP call
        flat_example_value: Example value for the HOP output
        body_r: The output VT structure that Dynamo continues tracing with (may be None)
        graph_output_vts: Tensor/symint VTs that were actual graph outputs

    Returns:
        The body_r VT (unchanged), which Dynamo will continue tracing with
    """
    from .builder import wrap_fx_proxy

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=tx.output.create_proxy(
            "call_function",
            fn,
            args=args,
            kwargs=kwargs,
        ),
        example_value=flat_example_value,
    )

    # wrap_fx_proxy creates fresh variable trackers. However, the main program
    # after the speculate subgraph can still use the original tensor vts that
    # are still pointing to the nodes present in the subgraph. So, we reproxify
    # the original tensor vts with the subgraph outputs. This way, whenever the
    # outer graph uses an original vt, it uses the subgraph output.
    #
    # This is critical for maintaining the separation between:
    # - `body_r`: The output VT structure that Dynamo continues tracing (may
    #   contain non-proxyable objects, nested structures, etc.)
    # - `graph_output_vts`: Only the tensor/symint VTs that were actual graph
    #   outputs from speculate_subgraph
    #
    # By overwriting the proxies of VTs in `body_r` with the proxies from the
    # HOP call, we ensure the outer graph correctly references the HOP outputs
    # while still allowing `body_r` to contain arbitrary Python objects.
    if body_r is not None:
        for orig_vt, subgraph_vt in zip(graph_output_vts, flat_variable.items):
            if isinstance(
                orig_vt, (variables.SymNodeVariable, variables.TensorVariable)
            ):
                assert isinstance(
                    subgraph_vt, (variables.SymNodeVariable, variables.TensorVariable)
                )
                orig_vt.proxy = subgraph_vt.proxy
    return body_r


def _call_function_and_unflatten_output(
    tx, fn, args, kwargs, flat_example_value, ret_spec, body_r
):
    from .builder import wrap_fx_proxy

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=tx.output.create_proxy(
            "call_function",
            fn,
            args=args,
            kwargs=kwargs,
        ),
        example_value=flat_example_value,
    )

    # wrap_fx_proxy creates fresh variable trackers. However, the main program
    # after the speculate subgraph can still use the original tensor vts that
    # are still pointing to the nodes present in the subgraph. So, we reproxify
    # the original tensor vts with the subgraph outputs. This way, whenever the
    # outer graph uses an original vt, it uses the subgraph output.
    if body_r is not None:
        for orig_vt, subgraph_vt in zip(body_r.items, flat_variable.items):
            if isinstance(
                orig_vt, (variables.SymNodeVariable, variables.TensorVariable)
            ):
                assert isinstance(
                    subgraph_vt, (variables.SymNodeVariable, variables.TensorVariable)
                )
                orig_vt.proxy = subgraph_vt.proxy

    if ret_spec.num_intermediate_nodes_as_outputs:
        # The treespec was computed w/o any extra intermediate outputs. At this
        # point, it is safe to just get rid of the extra outputs
        flat_variable = TupleVariable(
            flat_variable.items[: -ret_spec.num_intermediate_nodes_as_outputs]
        )

    if ret_spec.masks_to_filter_const_values:
        from torch._dynamo.external_utils import insert_const_values_with_mask

        # During flattening, we removed the constant values. To ensure Dynamo
        # can trace correctly, insert back the constant values in the output.
        flat_variable = _make_inlined(tx, insert_const_values_with_mask)(
            flat_variable, ret_spec.masks_to_filter_const_values, ret_spec.const_values
        )

    # Transform variable back into a list (previously made into a tuple by
    # speculate_subgraph function) so as to respect the pytree API typing.
    flat_list_variable = BuiltinVariable(list).call_function(tx, [flat_variable], {})
    return (
        _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, ret_spec.treespec)
        if ret_spec.treespec
        else flat_variable
    )


def _assert_tensors_nonaliasing(inputs, outputs):
    input_tensor_ids = {
        id(t) for t in pytree.tree_leaves(inputs) if isinstance(t, torch.Tensor)
    }
    output_tensor_ids = {
        id(t) for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
    }
    assert input_tensor_ids.isdisjoint(output_tensor_ids), (
        "inputs to function body cannot alias outputs"
    )


def _check_all_tensorvariable(args):
    from . import TensorVariable

    if not all(type(a.realize()) is TensorVariable for a in args):
        unimplemented(
            gb_type="HOP: non torch.Tensor leaf",
            context=f"args types: {[type(a.realize()) for a in args]}",
            explanation="Expected all leaves to be of torch.Tensor type.",
            hints=[],
        )


def _check_supported_callable_arg(
    tx: "InstructionTranslator", func_var: VariableTracker, arg_name
):
    is_callable = (
        BuiltinVariable(callable).call_function(tx, [func_var], {}).as_python_constant()
    )
    if not is_callable:
        unimplemented(
            gb_type="HOP: non-callable variable",
            context=f"arg name: {arg_name}, func_var type: {str(func_var)}",
            explanation=f"{arg_name} should be a callable but is of type {str(func_var)}.",
            hints=[],
        )


def _call_while_loop(
    self: VariableTracker,
    tx: "InstructionTranslator",
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
    stack_output: bool,
) -> VariableTracker:
    from torch._higher_order_ops.while_loop import _create_unbacked_symint

    from . import TensorVariable

    args, kwargs = LazyVariableTracker.realize_all((args, kwargs))
    cond_fn, body_fn, operands, additional_inputs = args

    # Input checks
    for i, k in enumerate(["cond_fn", "body_fn", "operands"]):
        if v := kwargs.pop(k, None):
            assert i == len(args), (
                "did not provide the right number of non-keyword args"
            )
            args.append(v)

    if kwargs or len(args) != 4:
        unimplemented(
            gb_type="torch.while_loop: improper args/kwargs",
            context=f"args: {args}, kwargs: {kwargs}",
            explanation=f"torch.while_loop expects 4 positional arguments (got {len(args)}) "
            f"and no keyword arguments (got {len(kwargs)}) "
            "Usage: while_loop(cond_fn, body_fn, operands)",
            hints=[
                *graph_break_hints.USER_ERROR,
            ],
        )

    # cond_fn and body_fn input check
    _check_supported_callable_arg(tx, cond_fn, "cond_fn")
    _check_supported_callable_arg(tx, body_fn, "body_fn")

    # operands input check
    operands_seq = operands.unpack_var_sequence(tx)

    # additional_inputs input check
    if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
        unimplemented(
            gb_type="torch.while_loop: improper additional_inputs",
            context=str(additional_inputs),
            explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )
    additional_inputs_seq = additional_inputs.unpack_var_sequence(tx)

    with discard_graph_changes(tx):
        # Note: this must be run under discard graph changes.
        def unspecialize_carried_inputs(tx, carry) -> VariableTracker:
            # See NOTE [unspecialize int carry with unbacked symints]
            if (
                isinstance(carry, ConstantVariable) and carry.python_type() is int
            ) or isinstance(carry, SymNodeVariable):
                example_value = _create_unbacked_symint(
                    tx.output.fake_mode, ignore_fresh_unbacked_symbols=True
                )
                proxy = tx.output.current_tracer.create_graph_input(
                    "unbacked_symint", type(example_value), example_value
                )
                return SymNodeVariable.create(tx, proxy, example_value)
            else:
                # See NOTE [unspecialize constant tensor carry]
                assert isinstance(carry, TensorVariable)
                cloned_carry = carry.clone()
                cloned_carry.proxy.node.meta["example_value"].constant = None
                return cloned_carry

        # clone inputs across subgraphs, to avoid unbacked memoization in fake prop
        cond_operands_seq = [
            unspecialize_carried_inputs(
                tx,
                (
                    carry.call_method(tx, "clone", args=(), kwargs={})
                    if isinstance(carry, TensorVariable)
                    else carry
                ),
            )
            for carry in operands_seq
        ]
        body_operands_seq = [
            unspecialize_carried_inputs(
                tx,
                (
                    carry.call_method(tx, "clone", args=(), kwargs={})
                    if isinstance(carry, TensorVariable)
                    else carry
                ),
            )
            for carry in operands_seq
        ]

    # create cond subgrpahs
    (
        (cond_r, _cond_treespec),
        cond_graph,
        cond_lifted_freevars,
    ) = speculate_subgraph(
        tx,
        cond_fn,
        cond_operands_seq + additional_inputs_seq,
        {},
        "while_loop",
        source_target=self.value,
        # NOTE [why we cannot use "automatic" for while_loop]:
        # The reason is that we want to enforce
        # the ordering of inputs and outputs to be consistent and the ordering
        # of cond_fn and body_fn to the consistent.
        # e.g. suppose we use "automatic" and we have:
        #
        # def body_fn(ph1, ph2):
        #   new_a, new_b = ph2.cos(), ph1.sin()
        #   return new_a, new_b
        #
        # a, b = torch.randn(3), torch.randn(3)
        # new_a, new_b = body_fn(a, b)
        #
        # Using automatic, the ordering of arguments will be the order that they're
        # used. In this example, the capture graph looks like:
        #
        # def captured_body(ph1, ph2):
        #   new_a, new_b = ph1.cos(), ph2.add_(1)
        #   return new_a, new_b
        #
        # This is fine when we change the calling convention of captured_body to be
        # new_a, new_b = captured_body(b, a).
        # But for while_loop, the next iteration's input is previous iteration output
        # we'll end up feeding captured_body(new_a, new_b) instead.
        # So it's best we always enforce the ordering of carried_inputs the same as outputs
        # with "flatten_manual".
        set_subgraph_inputs="flatten_manual",
        supports_input_mutation=self.supports_input_mutation,
        supports_aliasing=self.supports_aliasing,
        remove_consts_from_outputs=False,
    )
    cond_nn_modules = dict(tx.output.nn_modules)
    validate_subgraph_output_types(cond_r)
    if isinstance(cond_r, TensorVariable):
        cond_r_meta = _extract_tensor_metadata(
            cond_r.proxy.node.meta["example_value"], include_contiguity=False
        )
        if cond_r_meta.dtype != torch.bool or cond_r_meta.shape != torch.Size([]):
            unimplemented(
                gb_type="torch.while_loop: unsupported cond_fn return type",
                context=str(cond_r),
                explanation=f"Expected cond_fn to return a scalar tensor or a bool but got {cond_r_meta.shape}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
    elif isinstance(cond_r, ConstantVariable):
        # short-circuiting while_loop when cond_fn returns a constant such as 0, 1 True or False
        pred = cond_r.as_python_constant()
        if pred:
            unimplemented(
                gb_type="torch.while_loop: infinite loop detected",
                context=str(cond_r),
                explanation=f"Infinite loop detected because while_loop's cond_fn always returns the same value {pred}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
        else:
            return operands

    # create body subgraph
    (
        (body_r, body_treespec),
        body_graph,
        body_lifted_freevars,
    ) = speculate_subgraph(
        tx,
        body_fn,
        body_operands_seq + additional_inputs_seq,
        {},
        "while_loop",
        source_target=self.value,
        set_subgraph_inputs="flatten_manual",
        should_flatten_outputs=True,
        supports_input_mutation=False,
        supports_aliasing=False,
        remove_consts_from_outputs=False,
    )
    validate_subgraph_output_types(body_r)

    # We set include contiguity=False because we have vmap x HOP tests, where if
    # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
    # "querying is_contiguous inside of vmap for memory_format other than
    # torch.contiguous_format is not yet implemented". This is okay because stride
    # is still checked.
    check_meta_consistency_vt(
        body_r.unpack_var_sequence(tx),
        operands_seq,
        "body_fn_output",
        "carried_inputs",
        include_contiguity=False,
    )

    (
        cond_graph,
        body_graph,
        cond_shared,
        _body_shared,
        cond_unique,
        body_unique,
    ) = _merge_graph_inputs(
        cond_graph,
        cond_lifted_freevars,
        "cond_fn",
        body_graph,
        body_lifted_freevars,
        "body_fn",
    )

    # Note: cond_shared and body_shared refer to the same proxy in parent graph
    # so using either of them is OK. Use cond_shared as it doesn't matter.
    additional_lifted_inputs = cond_shared + cond_unique + body_unique

    body_nn_modules = dict(tx.output.nn_modules)

    cond_gm = torch.fx.GraphModule(cond_nn_modules, cond_graph)
    body_gm = torch.fx.GraphModule(body_nn_modules, body_graph)
    cond_name = tx.output.install_subgraph("cond_fn", cond_gm)
    body_name = tx.output.install_subgraph("body_fn", body_gm)

    cond_node = make_attr(tx, cond_name)
    body_node = make_attr(tx, body_name)

    operands_proxy = tuple(operand.as_proxy() for operand in operands_seq)
    additional_inputs_proxy = tuple(
        [inp.as_proxy() for inp in additional_inputs_seq] + additional_lifted_inputs
    )
    p_args = (
        cond_node,
        body_node,
        operands_proxy,
        additional_inputs_proxy,
    )
    return _call_function_and_unflatten_output(
        tx,
        self.value,
        p_args,
        {},
        None,
        body_treespec,
        body_r,
    )


def are_same_graph_modules(fn_name, a_mod, b_mod, fake_mode):
    from torch._subclasses._fake_tensor_utils import _CacheKeyState
    from torch._subclasses.fake_tensor import extract_tensor_metadata

    # Maps the equivalent nodes from a to b
    node_map = {}

    def check_all_args(a_nodes, b_nodes):
        for arg_a, arg_b in zip(a_nodes, b_nodes):
            if isinstance(arg_a, torch.fx.Node):
                if node_map[arg_a] != arg_b:
                    return False
            elif isinstance(arg_a, slice):
                if not isinstance(arg_b, slice):
                    return False
                if not check_all_args(
                    (arg_a.start, arg_a.stop, arg_a.step),
                    (arg_b.start, arg_b.stop, arg_b.step),
                ):
                    return False
            elif arg_a != arg_b:
                # This is a catch-all for everything else. `slice` was a
                # surprise but can there be other data structures that can
                # contain fx.Nodes in them?
                return False
        return True

    for a_node, b_node in zip(a_mod.graph.nodes, b_mod.graph.nodes):
        if a_node.op != b_node.op:
            return False

        if a_node.op == "placeholder":
            a_value = a_node.meta["example_value"]
            b_value = b_node.meta["example_value"]

            if isinstance(a_value, torch.Tensor):
                if not isinstance(b_value, torch.Tensor):
                    return False
                # Extract fake tensor metadata for a and b and then compare
                a_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                a_metadata = extract_tensor_metadata(a_value)
                a_metadata._flatten_into(a_result, fake_mode, state)

                b_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                b_metadata = extract_tensor_metadata(b_value)
                b_metadata._flatten_into(b_result, fake_mode, state)
                if a_result != b_result:
                    return False
            elif isinstance(a_value, torch.SymInt):
                if not isinstance(b_value, torch.SymInt):
                    return False
                if a_value is not b_value:
                    return False
        elif a_node.op == "call_function":
            if a_node.target is not b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug(
                    "%s: Graph comparison failed at node (call_function): %s",
                    fn_name,
                    a_node,
                )
                return False
        elif a_node.op == "call_method":
            if a_node.target != b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug(
                    "%s: Graph comparison failed at node (call_method) : %s",
                    fn_name,
                    a_node,
                )
                return False
        elif a_node.op == "output":
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug("%s: Graph comparison failed at the output node", fn_name)
                return False
        elif a_node.op == "get_attr":
            a_attr = getattr(a_mod, a_node.target)
            b_attr = getattr(b_mod, b_node.target)
            if isinstance(a_attr, torch.fx.GraphModule):
                if not isinstance(b_attr, torch.fx.GraphModule):
                    return False
                # This is an example of a HOP inside a HOP
                if not are_same_graph_modules(fn_name, a_attr, b_attr, fake_mode):
                    return False
            else:
                # TODO - write an example with tensor as a graph attribute in
                # the Fx graph
                raise NotImplementedError(f"get_attr with {type(a_attr)}")
        else:
            # TODO - call_module is not supported because Dynamo Fx graph does
            # not install a call_module
            raise NotImplementedError(f"Graph equivalence check saw a {a_node.op}")

        # Two nodes are equal - add them to them map
        node_map[a_node] = b_node

    return True


def validate_args_and_maybe_create_graph_inputs(
    sub_args,
    tracer,
    tx,
    set_subgraph_inputs,
    description,
    sub_args_names=None,
):
    from . import AutogradFunctionContextVariable
    from .builder import wrap_fx_proxy_cls

    assert tracer.parent is not None

    if set_subgraph_inputs == "flatten_manual":
        flat_args, tree_spec = _make_inlined(tx, pytree.tree_flatten)(
            ListVariable(sub_args)
        ).unpack_var_sequence(tx)

        flat_inputs = validate_args_and_maybe_create_graph_inputs(
            flat_args.unpack_var_sequence(tx),
            tracer,
            tx,
            set_subgraph_inputs="manual",
            description=description,
        )

        return _make_inlined(tx, pytree.tree_unflatten)(
            ListVariable(flat_inputs), tree_spec
        ).unpack_var_sequence(tx)
    else:
        if sub_args_names is not None:
            # Can be greater if user passes some args as kwargs
            assert len(sub_args_names) >= len(sub_args)
        args = []
        for idx, a in enumerate(sub_args):
            assert isinstance(a, VariableTracker)
            if set_subgraph_inputs == "automatic":
                args.append(a)
                continue
            elif set_subgraph_inputs == "semi_automatic":
                if isinstance(a, AutogradFunctionContextVariable):
                    example_value = a.as_proxy().node.meta["example_value"]
                    arg_name = (
                        a.as_proxy().node.name
                        if sub_args_names is None
                        else sub_args_names[idx]
                    )
                    tracer.create_graph_input(arg_name, a.python_type(), example_value)
                elif a.maybe_fx_node() is not None:
                    node = a.maybe_fx_node()
                    example_value = node.meta["example_value"]
                    arg_name = (
                        a.as_proxy().node.name
                        if sub_args_names is None
                        else sub_args_names[idx]
                    )
                    new_proxy = tracer.create_graph_input(
                        arg_name, a.python_type(), example_value
                    )
                    example_value = node.meta.get("example_value", None)
                    a = wrap_fx_proxy_cls(
                        target_cls=type(a),
                        tx=tx,
                        proxy=new_proxy,
                        example_value=example_value,
                    )
                args.append(a)
                continue

            if a.is_python_constant():
                # This arg is not used in the body of the higher order op.
                # Currently, this new input is added to make the calls
                # happy, which expect a fixed number of arguments. In
                # future, we can clean this up.
                arg_name = (
                    "const_unused"
                    if sub_args_names is None
                    else f"const_unused_{sub_args_names[idx]}"
                )
                tracer.create_graph_input(
                    arg_name, a.python_type(), a.as_python_constant()
                )
                new_arg = a
            # Weird special case, we probably want to delete it or fold it
            # into the next case (of `a` being placeable into a graph)
            elif isinstance(a, AutogradFunctionContextVariable):
                example_value = a.as_proxy().node.meta["example_value"]
                arg_name = (
                    a.as_proxy().node.name
                    if sub_args_names is None
                    else sub_args_names[idx]
                )
                tracer.create_graph_input(arg_name, a.python_type(), example_value)
                new_arg = a
            # If `a` can be put into a graph
            elif a.maybe_fx_node() is not None:
                node = a.maybe_fx_node()
                example_value = node.meta.get("example_value", None)
                arg_name = node.name if sub_args_names is None else sub_args_names[idx]
                new_proxy = tracer.create_graph_input(
                    arg_name, a.python_type(), example_value
                )
                new_arg = wrap_fx_proxy_cls(
                    target_cls=type(a),
                    tx=tx,
                    proxy=new_proxy,
                    example_value=example_value,
                )
            # If `a` cannot be put into a graph
            else:
                # HOPs work much better if they use speculate_subgraph(set_subgraph_inputs="automatic").
                unimplemented(
                    gb_type="HOP body taking non-Tensor as input",
                    context=str(sub_args),
                    explanation=f"{description} with body that accepts non-Tensors as input. "
                    f"Got type {a.python_type()} at index {idx}.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            args.append(new_arg)
        return args


# This helper function is used to make sure two graphs share the same input signature. For example,
# in torch.cond, two branches might lift different set of tensors as inputs. This function helps to
# dedup the inputs and modify the graphs to take the same set of inputs.
def _merge_graph_inputs(
    l_graph, l_lifted_freevars, l_name, r_graph, r_lifted_freevars, r_name
):
    def dedup_and_sort_lifted_freevars(l_lifted_freevars, r_lifted_freevars):
        # The nn module attributes are guaranteed to be registered into the top-level graph module during
        # higher order op speculation. Therefore, get_attr nodes in two branches with the same
        # target refer to the same attribute and we can safely deduplicate them with their target.
        #
        # Note: ideally, dynamo should just create a single proxy for the same attribute of a nn module. But
        # true_branch and false_branch belong to two separate tracing contexts, they may register the same
        # attribute to top level separately. This creates two get_attr proxies for the same attribute
        # that have different meta data such as stack_trace (one stack trace for the true_branch,
        # and the other for false_branch). It seems better to discard the proxy explicitly in cond
        # than make dynamo create a single proxy for the same get_attr target.
        def shared_getattrs(l_lifted_proxies, r_lifted_proxies):
            true_targets = {
                proxy.node.target: proxy
                for proxy in l_lifted_proxies
                if proxy.node.op == "get_attr"
            }
            l_shared_getattrs = {}
            r_shared_getattrs = {}

            for false_proxy in r_lifted_proxies:
                if (
                    false_proxy.node.op == "get_attr"
                    and false_proxy.node.target in true_targets
                ):
                    true_proxy = true_targets[false_proxy.node.target]
                    l_shared_getattrs[true_proxy] = true_proxy
                    r_shared_getattrs[false_proxy] = true_proxy
            return l_shared_getattrs, r_shared_getattrs

        l_shared_getattrs, r_shared_getattrs = shared_getattrs(
            l_lifted_freevars.keys(), r_lifted_freevars.keys()
        )

        l_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            l_shared_getattrs.keys()
        )
        r_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            r_shared_getattrs.keys()
        )
        unique_l_freevars = l_lifted_freevars.keys() - l_shared_freevars
        unique_r_freevars = r_lifted_freevars.keys() - r_shared_freevars

        def _sort_by_name(vars):
            return sorted(vars, key=lambda var: var.node.name)

        return (
            list(_sort_by_name(list(l_shared_freevars))),
            list(_sort_by_name(list(r_shared_freevars))),
            list(_sort_by_name(list(unique_l_freevars))),
            list(_sort_by_name(list(unique_r_freevars))),
        )

    (l_shared, r_shared, unique_l, unique_r) = dedup_and_sort_lifted_freevars(
        l_lifted_freevars, r_lifted_freevars
    )

    # Let's say we capture cond(pred, true_fn, false_fn, (x,))
    # With set_graph_input set to automatic,
    # true_fn has lifted variables x, a, b, c
    # false_fn has lifted variables x, a, b, d
    # Then fixup_branch_inps make sure both branches have the same signature, i.e.:
    # - true_fn(x, a, b, c_true_branch, d_false_branch)
    # - false_fn(x, a, b, c_true_branch, d_false_branch)
    #
    # More formally, the signature has three parts in the following order:
    # 1. used in both branches: x, a, b
    # 2. only used in true branches: c, suffixed with _true_branch
    # 3. only used in false branches: d, suffixed with _false_branch
    # Within each part, we re-order the nodes by name to have a derterministic ordering for testing.
    def fixup_branch_inps(graph, lifted_freevars, shared, unique_l, unique_r):
        def _insert_or_replace_phs(new_args, name_suffix):
            for arg in new_args:
                new_ph = graph.placeholder(arg.node.name + name_suffix)
                new_ph.meta = arg.node.meta
                # Override with new_ph if there exists a old placeholder.
                if arg in lifted_freevars:
                    old_ph = lifted_freevars[arg].node
                    old_ph.replace_all_uses_with(new_ph)
                    # replace_all_uses_with doesn't clean users. Clean it manually so that we could erase it.
                    old_ph.users = {}
                    graph.erase_node(old_ph)

        first_not_ph_node = next(
            node for node in graph.nodes if node.op != "placeholder"
        )
        with graph.inserting_before(first_not_ph_node):
            _insert_or_replace_phs(shared, "")
            _insert_or_replace_phs(unique_l, "_" + l_name)
            _insert_or_replace_phs(unique_r, "_" + r_name)

    fixup_branch_inps(l_graph, l_lifted_freevars, l_shared, unique_l, unique_r)
    fixup_branch_inps(r_graph, r_lifted_freevars, r_shared, unique_l, unique_r)
    return l_graph, r_graph, l_shared, r_shared, unique_l, unique_r


# NOTE: [HigherOrderOperator subgraph input ordering]
# The input ordering of the higher order ops is determined by the order of
# the creation of the placeholder.
# Manually created inputs are created in validate_args_and_maybe_create_graph_inputs before
# speculating subgraph.
# During subgraph speculation, we may lift closured tensors and free symbols as inputs,
# their ordering is determined by the time they are lifted: earlier lifted ones precede later
# lifted ones.
#
# Suppose the placeholders are
# O1, O2, X1, O3, O4, X2, X3, O5 where Xs are lifted phs
# The following code re-order the placeholders to
# O1, O2, O3, O4, O5, X1, X2, X3
def move_lifted_freevars_phs_to_end(
    graph: torch.fx.Graph, lifted_freevars: tuple[torch.fx.Node]
):
    lifted_ph_set = {child_p.node for child_p in lifted_freevars.values()}

    prev_phs = [n for n in graph.nodes if n.op == "placeholder"]

    # No need to reorder when graph doesn't have args or doesn't
    # have lifted freevars or all inputs are lifted freevars.
    if (
        len(prev_phs) == 0
        or len(lifted_ph_set) == 0
        or len(prev_phs) == len(lifted_ph_set)
    ):
        return

    # Step 1: find first X1
    for x1 in prev_phs:
        if x1 in lifted_ph_set:
            break

    assert x1 is not None and x1.op == "placeholder"
    # Step 2: starting from the X1, skip Xs and prepend Os before X1.
    cand_x = x1.next
    while cand_x is not None and cand_x.op == "placeholder":
        if cand_x in lifted_ph_set:
            cand_x = cand_x.next
        else:
            nxt = cand_x.next
            cand_x._remove_from_list()
            x1.prepend(cand_x)
            cand_x = nxt

    # Step 3: assert that all placeholders are in the correct order as .
    # in lifted_freevars
    after_phs = [node for node in graph.nodes if node.op == "placeholder"][
        -len(lifted_freevars) :
    ]
    assert len(after_phs) == len(lifted_freevars)
    for child_proxy, ph in zip(lifted_freevars.values(), after_phs):
        assert child_proxy.node is ph, (
            "The order of placeholders is different from the order of lifted_freevars"
        )

    graph.lint()


def check_aliasing_and_input_mutation(
    subtracer, graph, supports_input_mutation, supports_aliasing, source_target
):
    if not supports_input_mutation:
        mutation_info = subtracer.has_input_mutation()
        if mutation_info.has_mutation:
            context = f"{mutation_info.msg} in\n {graph}"
            unimplemented(
                gb_type="Encountered input mutation during higher order op tracing",
                context=context,
                explanation=f"Higher order ops do not support input mutation. Found in {source_target.name()}",
                hints=[
                    "Consider using the debug context to change user code to avoid mutation.",
                    "Please open an issue.",
                ],
            )

    if not supports_aliasing:
        aliasing_info = subtracer.has_aliasing()
        if aliasing_info.has_aliasing:
            context = f"{aliasing_info.msg} in\n {graph}"
            unimplemented(
                gb_type="Encountered aliasing during higher order op tracing",
                context=context,
                explanation=f"Higher order ops do not support aliasing. Found in {source_target.name()}",
                hints=[
                    "Replace `return input` with `return input.clone()` to avoid aliasing.",
                    "Consider using the debug context to change user code to avoid aliasing.",
                    "Please open an issue.",
                ],
            )


def trace_hop_function(
    f,
    tx,
    subtracer,
    enable_grad,
    under_activation_checkpoint,
    restore_side_effects,
    args,
    sub_kwargs,
):
    autograd_ctx = (
        dynamo_enable_grad(tx, enable_grad)
        if enable_grad is not None
        else contextlib.nullcontext()
    )
    checkpoint_ctx = (
        dynamo_under_activation_checkpoint(tx)
        if under_activation_checkpoint
        else contextlib.nullcontext()
    )

    # For handling side effects, we can make an argument that we don't
    # have to do anything here. The side effects infra does a good job
    # of graph breaking if we mutate any nonlocal or global variable
    # while subtracing. As a result if tracing succeeds, side effects
    # data structure will only contain read-only data structures that
    # are put there for tracking purposes.
    # But on the other hand, there is an argument that if we ever write
    # a new side effect in Dynamo which does not go through the side
    # effect infra, we can end up in bad state.
    # Therefore we restore the side effects after tracing. The catch is
    # that we have to special handle tensor variables. If we have seen a
    # nonlocal variable tensor during subtracing, we want to keep a
    # track of that tensor, so that later subtracing or the root tracer
    # itself does not create a new proxy for the already observed tensor
    # variable.
    if restore_side_effects:
        prev_side_effects = tx.output.side_effects.clone()

    with autograd_ctx, checkpoint_ctx:
        output = f.call_function(tx, args, sub_kwargs)

    if restore_side_effects:
        new_side_effects = tx.output.side_effects.clone()
        prev_side_effects.track_runahead_tensor_and_symvar_side_effects(
            new_side_effects
        )
        tx.output.side_effects = prev_side_effects
    return output


def get_hop_args(
    tx, f, subtracer, sub_args, sub_kwargs, set_subgraph_inputs, description
):
    sub_args_names = maybe_positional_arg_names(f)
    # User mismatch in the number of args. Will eventually lead to an error.
    if sub_args_names is not None and len(sub_args_names) < len(sub_args):
        sub_args_names = None
    args = validate_args_and_maybe_create_graph_inputs(
        sub_args,
        subtracer,
        tx,
        set_subgraph_inputs,
        description,
        sub_args_names,
    )

    validate_args_and_maybe_create_graph_inputs(
        sub_kwargs.values(),
        subtracer,
        tx,
        set_subgraph_inputs="automatic",
        description=description,
    )
    return args


# TODO - The eventual goal is to replace
# speculate_subgraph_with_auto_output_flattening with speculate_subgraph or
# merge them two into one. We are following a staged approach because of
# existing implementation complexity for control flow ops.
def speculate_subgraph_with_auto_output_flattening(
    tx: "InstructionTranslator",
    f: VariableTracker,
    sub_args: Sequence[VariableTracker],
    sub_kwargs: Optional[dict[str, VariableTracker]],
    description: str,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target: Optional[HigherOrderOperator] = None,
    enable_grad: Optional[bool] = None,
    # TODO - We can probably just make everyone use automatic for wrap_semantics
    set_subgraph_inputs: Literal[
        "automatic", "semi_automatic", "flatten_manual", "manual"
    ] = "automatic",
    # Make default False
    restore_side_effects: bool = True,
    under_activation_checkpoint: bool = False,
    # TODO - supports input_mutation and aliasing should be False by default for strictness
    supports_input_mutation: bool = True,
    supports_aliasing: bool = True,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer: Optional["torch._dynamo.output_graph.SubgraphTracer"] = None,
) -> tuple[
    VariableTracker,  # output: The VT that Dynamo continues tracing with
    torch.fx.Graph,  # graph: The FX graph representing the subgraph computation
    dict[
        torch.fx.Proxy, torch.fx.Proxy
    ],  # lifted_freevars: Free variables lifted as inputs
    VariableTracker
    | tuple[
        VariableTracker, ...
    ],  # graph_output_vts: Tensor/symint VTs that are actual FX graph outputs
]:
    """
    Speculate subgraph for Higher-Order Operators (HOPs) with automatic output flattening.

    ## Automatic output flattening

    For many HOPs, the representation exists only as a container for the
    subgraph. In later compiler stages or at runtime, the HOP is desugared and
    simply executes the subgraph directly, as if it were inlined. For such hops,
    we follow automatic output flattening.
    For example:
    - invoke_subgraph
    - activation checkpointing (torch.utils.checkpoint.checkpoint)
    - autograd.Function
    - nested_compile_region

    This is in contrast to control flow HOPs which do not follow this desugaring:
    - torch.cond (conditional execution based on predicate)
    - torch.while_loop (iterative execution)
    - torch.map (parallel execution over batch dimension)

    For control flow HOPs, the HOP behavior is fundamentally different from just
    running the body function once.

    ## Key Advantage: Disentangling VTs from Graph Outputs

    Desugaring simplify HOP processing by allowing us to disentangle the output
    variable trackers (VTs) from the HOP subgraph outputs. This mirrors typical
    Dynamo processing where:
    - VTs "run ahead" representing the program state for continued tracing
    - The graph is a side data structure tracking computation seen so far

    This separation is crucial for HOPs with non-proxyable outputs (e.g., custom
    user-defined objects containing tensors). The function may return complex Python
    objects for Dynamo to continue tracing, but only the tensor/symint VTs need to
    be registered as actual FX graph outputs.

    Example:
       
```



## High-Level Overview

"""This module contains classes and utilities for handling higher-order operators in Dynamo.It provides functionality for tracing and transforming control flow constructs likeconditions (torch.cond), loops (torch.while_loop), maps (torch.ops.higher_order.map),and other higher-order operations.The module includes specialized VariableTracker classes for different types ofhigher-order operations, along with utilities for:- Speculating and capturing subgraphs- Managing control flow- Handling autograd function applications- Supporting function transformations- Processing activation checkpointsThese classes work together to enable Dynamo to correctly trace and compile codecontaining complex control flow patterns and higher-order functions while preservingtheir semantic behavior.

This Python file contains 39 class(es) and 117 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OutputSpec`, `Foo`, `Foo`, `TorchHigherOrderOperatorVariable`, `CustomFunctionHigherOrderOperatorVariable`, `CondHigherOrderVariable`, `CallTorchbindHigherOrderVariable`, `WhileLoopHigherOrderVariable`, `WhileLoopStackOutputHigherOrderVariable`, `AssociativeScanHigherOrderVariable`, `ScanHigherOrderVariable`, `MapHigherOrderVariable`, `ExecutorchCallDelegateHigherOrderVariable`, `FunctorchHigherOrderVariable`, `FunctionalCallVariable`, `ReparametrizeModuleCallVariable`, `WrapHigherOrderVariable`, `WrapWithSetGradEnabledHigherOrderVariable`, `WrapWithAutocastHigherOrderVariable`, `HintsWrapperHigherOrderVariable`

**Functions defined**: `__post_init__`, `raise_hard_error_if_graph_break`, `deco`, `graph_break_as_hard_error`, `discard_graph_changes`, `check_meta_consistency_vt`, `_unwrap_var`, `dynamo_enable_grad`, `dynamo_under_activation_checkpoint`, `find_mismatched_vars`, `_is_none`, `only_consist_of`, `_make_inlined`, `inline_call`, `_call_function_with_auto_output_flattening`, `_call_function_and_unflatten_output`, `_assert_tensors_nonaliasing`, `_check_all_tensorvariable`, `_check_supported_callable_arg`, `_call_while_loop`

**Key imports**: contextlib, functools, inspect, itertools, logging, types, warnings, Sequence, dataclass, Any, Literal, Optional, TYPE_CHECKING


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo/variables`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `functools`
- `inspect`
- `itertools`
- `logging`
- `types`
- `warnings`
- `collections.abc`: Sequence
- `dataclasses`: dataclass
- `typing`: Any, Literal, Optional, TYPE_CHECKING
- `torch._C`
- `torch.fx`
- `torch.nn`
- `torch._dispatch.python`: enable_python_dispatcher
- `torch._dynamo.utils`: get_fake_value
- `torch._dynamo.variables.builtin`: BuiltinVariable
- `torch._dynamo.variables.constant`: ConstantVariable
- `torch._dynamo.variables.ctx_manager`: RepararametrizeModuleContextVariable
- `torch._dynamo.variables.functions`: UserFunctionVariable
- `torch._dynamo.variables.nn_module`: UnspecializedNNModuleVariable
- `torch._dynamo.variables.tensor`: SymNodeVariable
- `torch._guards`: Source
- `torch._ops`: HigherOrderOperator
- `torch.fx.passes.shape_prop`: _extract_tensor_metadata
- `torch.utils`: _pytree as pytree
- `..`: graph_break_hints, variables
- `..source`: AttrSource, DictGetItemSource
- `..utils`: proxy_args_kwargs, set_example_value
- `.base`: VariableTracker
- `.dicts`: ConstDictVariable


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

Files in the same folder (`torch/_dynamo/variables`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`streams.py_docs.md`](./streams.py_docs.md)
- [`nn_module.py_docs.md`](./nn_module.py_docs.md)
- [`tensor.py_docs.md`](./tensor.py_docs.md)
- [`torch.py_docs.md`](./torch.py_docs.md)
- [`constant.py_docs.md`](./constant.py_docs.md)
- [`dicts.py_docs.md`](./dicts.py_docs.md)
- [`distributed.py_docs.md`](./distributed.py_docs.md)
- [`lists.py_docs.md`](./lists.py_docs.md)


## Cross-References

- **File Documentation**: `higher_order_ops.py_docs.md`
- **Keyword Index**: `higher_order_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

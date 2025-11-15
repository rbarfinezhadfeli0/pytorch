# Documentation: `docs/torch/_export/passes/replace_autocast_with_hop_pass.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/passes/replace_autocast_with_hop_pass.py_docs.md`
- **Size**: 10,437 bytes (10.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/passes/replace_autocast_with_hop_pass.py`

## File Metadata

- **Path**: `torch/_export/passes/replace_autocast_with_hop_pass.py`
- **Size**: 7,163 bytes (7.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._higher_order_ops.wrap import wrap_with_autocast

from ..utils import node_inline_, nodes_filter, nodes_first, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target
        in [
            torch.amp.autocast_mode._enter_autocast,
            torch.amp.autocast_mode._exit_autocast,
        ]
    )


def _is_enter_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch.amp.autocast_mode._enter_autocast
    )


def _is_exit_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch.amp.autocast_mode._exit_autocast
    )


def _is_autocast_sub_mod(node: torch.fx.Node) -> bool:
    """
    Check if the first non-placeholder node is `torch.amp.autocast_mode._enter_autocast`.
    """
    if node.op == "call_module":
        assert isinstance(node.target, str)
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target is torch.amp.autocast_mode._enter_autocast
        ):
            # TODO: check if current auto-cast type is the same as the args of
            # _enter_autocast. If so, return False, i.e. do not create a submodule.
            return True
    return False


def _check_valid_autocast_block(
    enter_autocast_node: torch.fx.Node, exit_autocast_node: torch.fx.Node
) -> None:
    assert _is_enter_autocast_node(enter_autocast_node)
    assert _is_exit_autocast_node(exit_autocast_node)
    assert exit_autocast_node.args[0] == enter_autocast_node


def _replace_with_hop(node: torch.fx.Node) -> None:
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    assert graph.owning_module is not None
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    autocast_nodes = nodes_filter(sub_graph.nodes, _is_autocast_node)
    if len(autocast_nodes) > 0:
        assert len(autocast_nodes) > 1  # need at least an enter node and an exist node
        enter_autocast_node = autocast_nodes[0]
        exit_autocast_node = autocast_nodes[-1]
        _check_valid_autocast_block(enter_autocast_node, exit_autocast_node)

        _replace_with_hop_helper(node, enter_autocast_node, wrap_with_autocast)
        sub_graph.erase_node(exit_autocast_node)
        sub_graph.erase_node(enter_autocast_node)


def _split_autocast(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    split_autocast creates a new graph module that splits the input graph module into multiple submodules
    based on the `_enter_autocast` and `_exit_autocast` nodes. It doesn't mutate the input graph module.

    Nodes between the **outer-most** `_enter_autocast` and `_exit_autocast(_enter_autocast)` are split
    into a submodule. Nested autocast regions are not split.
    `_enter_autocast` and `_exit_autocast(_enter_autocast)` nodes are in the submodule as well.

    Below is an example of splitting. A, B, C, D, E are blocks of non-autocast nodes in the original graph
    module. Nodes marked with the same number are grouped into the same submodule.
    A               # 0
    enter_autocast  # 1
    B               # 1
    exit_autocast   # 1
    C               # 2
    enter_autocast  # 3
    D               # 3
    exit_autocast   # 3
    E               # 4
    """
    enter_autocast_node_stack: list[torch.fx.Node] = []
    first_node_after_outer_most_exit: bool = False

    def node_call_back(node: torch.fx.Node) -> bool:
        nonlocal enter_autocast_node_stack, first_node_after_outer_most_exit
        increment_id = False
        if first_node_after_outer_most_exit or (
            len(enter_autocast_node_stack) == 0 and _is_enter_autocast_node(node)
        ):
            assert len(enter_autocast_node_stack) == 0
            first_node_after_outer_most_exit = False
            increment_id = True
        if _is_enter_autocast_node(node):
            enter_autocast_node_stack.append(node)
        elif _is_exit_autocast_node(node):
            assert len(enter_autocast_node_stack) > 0
            last_enter_autocast_node = enter_autocast_node_stack.pop()
            assert node.args[0] == last_enter_autocast_node
            if len(enter_autocast_node_stack) == 0:
                # next node should be in the next submodule since
                # autocast block ends
                first_node_after_outer_most_exit = True
        return increment_id

    return sequential_split(gm, node_call_back)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_autocast_with_hop_pass().
    Split the graph module into multiple subgraphs based on the autocast nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    Nodes between `_enter_autocast` and `_exit_autocast(_enter_autocast)` are considered
    as a subgraph.
    """
    need_replacing = any(_is_autocast_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # split_autocast returns a new graph module that could have different output
    # args names. We need to fix the graph signature in `_sequential_split_and_maybe_inline_subgraphs_helper`.
    new_gm = _split_autocast(gm)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node) -> None:
        if _is_autocast_sub_mod(node):
            _replace_with_hop(node)
        else:
            assert node.op == "call_module"
            assert isinstance(node.target, str)
            node_inline_(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_autocast_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )

```



## High-Level Overview

"""

This Python file contains 0 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_is_autocast_node`, `_is_enter_autocast_node`, `_is_exit_autocast_node`, `_is_autocast_sub_mod`, `_check_valid_autocast_block`, `_replace_with_hop`, `_split_autocast`, `node_call_back`, `_sequential_split_and_maybe_inline_subgraphs`, `_maybe_inline_or_replace_with_hop`, `replace_autocast_with_hop_pass`

**Key imports**: annotations, TYPE_CHECKING, torch, wrap_with_autocast, node_inline_, nodes_filter, nodes_first, sequential_split, ExportGraphSignature


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `typing`: TYPE_CHECKING
- `torch`
- `torch._higher_order_ops.wrap`: wrap_with_autocast
- `..utils`: node_inline_, nodes_filter, nodes_first, sequential_split
- `torch.export.graph_signature`: ExportGraphSignature


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/_export/passes`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_node_metadata_hook.py_docs.md`](./_node_metadata_hook.py_docs.md)
- [`replace_set_grad_with_hop_pass.py_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md)
- [`functionalize_side_effectful_ops_pass.py_docs.md`](./functionalize_side_effectful_ops_pass.py_docs.md)
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `replace_autocast_with_hop_pass.py_docs.md`
- **Keyword Index**: `replace_autocast_with_hop_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_export/passes`):

- [`replace_set_grad_with_hop_pass.py_docs.md_docs.md`](./replace_set_grad_with_hop_pass.py_docs.md_docs.md)
- [`_node_metadata_hook.py_docs.md_docs.md`](./_node_metadata_hook.py_docs.md_docs.md)
- [`replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md`](./replace_view_ops_with_view_copy_ops_pass.py_kw.md_docs.md)
- [`lift_constants_pass.py_kw.md_docs.md`](./lift_constants_pass.py_kw.md_docs.md)
- [`remove_runtime_assertions.py_kw.md_docs.md`](./remove_runtime_assertions.py_kw.md_docs.md)
- [`lift_constants_pass.py_docs.md_docs.md`](./lift_constants_pass.py_docs.md_docs.md)
- [`constant_folding.py_docs.md_docs.md`](./constant_folding.py_docs.md_docs.md)
- [`remove_runtime_assertions.py_docs.md_docs.md`](./remove_runtime_assertions.py_docs.md_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_kw.md_docs.md)
- [`constant_folding.py_kw.md_docs.md`](./constant_folding.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `replace_autocast_with_hop_pass.py_docs.md_docs.md`
- **Keyword Index**: `replace_autocast_with_hop_pass.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

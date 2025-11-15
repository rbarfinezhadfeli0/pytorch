# Documentation: `torch/_export/passes/replace_set_grad_with_hop_pass.py`

## File Metadata

- **Path**: `torch/_export/passes/replace_set_grad_with_hop_pass.py`
- **Size**: 4,284 bytes (4.18 KB)
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
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

from ..utils import node_inline_, nodes_filter, nodes_first, nodes_map, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_set_grad_enabled_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch._C._set_grad_enabled
    )


def _is_set_grad_enabled_sub_mod(
    node: torch.fx.Node, omit_if_same_with_ambient: bool = False
) -> bool | torch.Tensor:
    if node.op == "call_module":
        assert isinstance(node.target, str)
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target is torch._C._set_grad_enabled
        ):
            return (
                first_non_ph.args[0] != torch.is_grad_enabled()
                if omit_if_same_with_ambient
                else True
            )
    return False


def _replace_with_hop(node: torch.fx.Node) -> None:
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    assert graph.owning_module is not None
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    set_grad_nodes = nodes_filter(sub_graph.nodes, _is_set_grad_enabled_node)
    if len(set_grad_nodes) > 0:
        assert len(set_grad_nodes) == 1
        set_grad_node = set_grad_nodes[0]
        _replace_with_hop_helper(node, set_grad_node, wrap_with_set_grad_enabled)
        sub_graph.erase_node(set_grad_node)


def _remove_set_grad_and_inline(node: torch.fx.Node) -> None:
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    assert graph.owning_module is not None
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    nodes_map(
        sub_graph.nodes,
        lambda n: sub_graph.erase_node(n) if _is_set_grad_enabled_node(n) else n,
    )
    node_inline_(node)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
    need_replacing = any(_is_set_grad_enabled_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # sequential_split returns a new graph module that could have different output
    # args names. We need to fix the graph signature.
    new_gm = sequential_split(gm, _is_set_grad_enabled_node)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node):
        if _is_set_grad_enabled_sub_mod(node, omit_if_same_with_ambient=True):
            _replace_with_hop(node)
        else:
            _remove_set_grad_and_inline(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_set_grad_with_hop_pass(
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


This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_is_set_grad_enabled_node`, `_is_set_grad_enabled_sub_mod`, `_replace_with_hop`, `_remove_set_grad_and_inline`, `_sequential_split_and_maybe_inline_subgraphs`, `_maybe_inline_or_replace_with_hop`, `replace_set_grad_with_hop_pass`

**Key imports**: annotations, TYPE_CHECKING, torch, wrap_with_set_grad_enabled, node_inline_, nodes_filter, nodes_first, nodes_map, sequential_split, ExportGraphSignature


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
- `torch._higher_order_ops.wrap`: wrap_with_set_grad_enabled
- `..utils`: node_inline_, nodes_filter, nodes_first, nodes_map, sequential_split
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
- [`functionalize_side_effectful_ops_pass.py_docs.md`](./functionalize_side_effectful_ops_pass.py_docs.md)
- [`insert_custom_op_guards.py_docs.md`](./insert_custom_op_guards.py_docs.md)
- [`constant_folding.py_docs.md`](./constant_folding.py_docs.md)
- [`replace_autocast_with_hop_pass.py_docs.md`](./replace_autocast_with_hop_pass.py_docs.md)
- [`add_runtime_assertions_for_constraints_pass.py_docs.md`](./add_runtime_assertions_for_constraints_pass.py_docs.md)
- [`replace_quantized_ops_with_standard_ops_pass.py_docs.md`](./replace_quantized_ops_with_standard_ops_pass.py_docs.md)
- [`replace_with_hop_pass_util.py_docs.md`](./replace_with_hop_pass_util.py_docs.md)


## Cross-References

- **File Documentation**: `replace_set_grad_with_hop_pass.py_docs.md`
- **Keyword Index**: `replace_set_grad_with_hop_pass.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

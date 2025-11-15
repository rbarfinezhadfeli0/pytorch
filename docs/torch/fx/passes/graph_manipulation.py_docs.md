# Documentation: `torch/fx/passes/graph_manipulation.py`

## File Metadata

- **Path**: `torch/fx/passes/graph_manipulation.py`
- **Size**: 3,965 bytes (3.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Any, NamedTuple, Optional

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_arg, Node, Target
from torch.fx.passes.shape_prop import ShapeProp


__all__ = [
    "replace_target_nodes_with",
    "size_bytes",
    "get_size_of_all_nodes",
    "get_tensor_meta",
    "get_size_of_node",
]


@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
):
    """Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,
    and updates them to match the new op code and target"""
    new_graph = Graph()
    val_map: dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            val_map[node] = new_graph.create_node(
                new_op, new_target, args, kwargs, node.name
            )
        else:
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    fx_module.graph = new_graph


@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int


@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(
    fx_module: GraphModule, args: Optional[list[torch.Tensor]] = None
) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    if args is not None:
        # Mark shape and dtype for each node (node.shape and node.dtype)
        ShapeProp(fx_module).propagate(*args)
    # Calculate the total size of the whole fx graph
    for node in fx_module.graph.nodes:
        if node.op == "output":
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return


@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any:
    tensor_meta = node.meta.get("tensor_meta")

    if not tensor_meta:
        raise RuntimeError(
            f"Node {node} has no tensor metadata associated with it! "
            f"Check that shape propagation has run."
        )

    return tensor_meta


@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
    # Total num of elements
    total_num_of_elems = 0
    # For a module, consider all parameters
    if node.op == "call_module":
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        # Parameters are named tuples
        for _name, p in parameters:
            total_num_of_elems += p.numel()
    # Don't forget the output size
    # node.shape is the shape of this node's output
    tensor_meta = get_tensor_meta(node)
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    # Assume for now if it's quantized then it's qint8 or quint8
    if tensor_meta.is_quantized:
        size_per_elem_bytes = torch._empty_affine_quantized(
            [], dtype=tensor_meta.dtype
        ).element_size()
    else:
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)

```



## High-Level Overview

"""Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `size_bytes`

**Functions defined**: `replace_target_nodes_with`, `get_size_of_all_nodes`, `get_tensor_meta`, `get_size_of_node`

**Key imports**: Any, NamedTuple, Optional, torch, compatibility, Graph, GraphModule, map_arg, Node, Target, ShapeProp


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any, NamedTuple, Optional
- `torch`
- `torch.fx._compatibility`: compatibility
- `torch.fx.graph`: Graph
- `torch.fx.graph_module`: GraphModule
- `torch.fx.node`: map_arg, Node, Target
- `torch.fx.passes.shape_prop`: ShapeProp


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

Files in the same folder (`torch/fx/passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`operator_support.py_docs.md`](./operator_support.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_drawer.py_docs.md`](./graph_drawer.py_docs.md)
- [`shape_prop.py_docs.md`](./shape_prop.py_docs.md)
- [`split_utils.py_docs.md`](./split_utils.py_docs.md)
- [`runtime_assert.py_docs.md`](./runtime_assert.py_docs.md)
- [`splitter_base.py_docs.md`](./splitter_base.py_docs.md)
- [`graph_transform_observer.py_docs.md`](./graph_transform_observer.py_docs.md)
- [`fake_tensor_prop.py_docs.md`](./fake_tensor_prop.py_docs.md)


## Cross-References

- **File Documentation**: `graph_manipulation.py_docs.md`
- **Keyword Index**: `graph_manipulation.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

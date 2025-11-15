# Documentation: `torch/_dynamo/graph_utils.py`

## File Metadata

- **Path**: `torch/_dynamo/graph_utils.py`
- **Size**: 3,558 bytes (3.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections import deque
from typing import Any, Optional

import torch
from torch.fx import Graph, map_arg, Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten


# flattens with support for slices
# Note: a better way to do this would
# be register/unregister slices as pytree nodes
# but there is no unregister API in the pytorch
# pytree impl
def _get_flat_args(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> list[Node]:
    args = list[Any]()
    map_arg((node.args, node.kwargs), args.append)
    if node in node_to_additional_deps:
        args.extend(node_to_additional_deps[node])
    return args


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _detect_cycles(
    graph: Graph, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> str:
    current_path: deque[Node] = deque()
    current_path_set: set[Node] = set()
    pending: deque[tuple[Node, Node]] = deque()

    def add_to_current_path(node: Node) -> None:
        current_path.append(node)
        current_path_set.add(node)

    def pop_current_path() -> None:
        node = current_path.pop()
        current_path_set.remove(node)

    def current_path_head() -> Node:
        return current_path[-1]

    for origin in graph.find_nodes(op="output"):
        current_path.clear()
        current_path_set.clear()
        add_to_current_path(origin)
        for child in _get_flat_args_unique(origin, node_to_additional_deps):
            pending.append((child, origin))

        while pending:
            cur_node, parent = pending.pop()

            # handle backtracking
            while current_path and current_path_head() != parent:
                pop_current_path()

            if not isinstance(cur_node, Node):
                continue

            if cur_node in current_path_set:
                current_path.append(cur_node)
                return f"cycle detected in path: {current_path}"

            add_to_current_path(cur_node)

            for child in _get_flat_args_unique(cur_node, node_to_additional_deps):
                pending.append((child, cur_node))

    return "no cycle detected"


def _graph_device_type(graph: Optional[Graph]) -> str:
    if graph is None:
        return "cpu"

    def _device_type(x: Any) -> str:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return "cpu"

    def _flatten_meta(node: Node, key: str) -> list[Any]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                return _device_type(obj)

        # Check for device conversions
        if node.op == "call_method":
            for gpu in ["cuda", "xpu"]:
                if node.target == gpu:
                    return gpu
                if node.target == "to" and gpu in node.args:
                    return gpu

        # Check args/kwargs for non-CPU device specs
        flat_args, _ = tree_flatten((node.args, node.kwargs))
        for obj in flat_args:
            return _device_type(obj)
    return "cpu"

```



## High-Level Overview


This Python file contains 0 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_get_flat_args`, `_get_flat_args_unique`, `_detect_cycles`, `add_to_current_path`, `pop_current_path`, `current_path_head`, `_graph_device_type`, `_device_type`, `_flatten_meta`

**Key imports**: deque, Any, Optional, torch, Graph, map_arg, Node, OrderedSet, tree_flatten


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: deque
- `typing`: Any, Optional
- `torch`
- `torch.fx`: Graph, map_arg, Node
- `torch.utils._ordered_set`: OrderedSet
- `torch.utils._pytree`: tree_flatten


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `graph_utils.py_docs.md`
- **Keyword Index**: `graph_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

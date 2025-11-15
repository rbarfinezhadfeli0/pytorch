# Documentation: `torch/_inductor/augmented_graph_helper.py`

## File Metadata

- **Path**: `torch/_inductor/augmented_graph_helper.py`
- **Size**: 7,057 bytes (6.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections import defaultdict
from typing import Optional

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


class AugmentedGraphHelper:
    """
    Graph helper that augments the original graph with additional
    dependencies and uses, plus tracks node equivalences for coalescing.

    TODO: if this becomes too large of compile time, consider binding
    graphcycles.cc
    """

    def __init__(
        self,
        graph: fx.Graph,
        node_ancestors: Optional[dict[fx.Node, OrderedSet[fx.Node]]] = None,
    ):
        # Each node starts in its own singleton set
        self.graph = graph
        self.merge_sets = {node: OrderedSet([node]) for node in graph.nodes}

        # Extra dependencies: node depends on dep (dep must come before node)
        self.extra_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Extra uses: reverse of extra_deps (node is used by user)
        self.extra_uses: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        # Note: only reflect original ancestors, not maintained through additional deps
        # or merge sets
        self.node_ancestors = node_ancestors

    def add_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        """Add extra dependency: node depends on dep."""
        self.extra_deps[n].add(dep)
        self.extra_uses[dep].add(n)

    def remove_extra_dep(self, *, n: fx.Node, dep: fx.Node) -> None:
        if dep in self.extra_deps[n]:
            self.extra_deps[n].discard(dep)
            self.extra_uses[dep].discard(n)

    def merge_to_set(self, existing_node: fx.Node, new_node: fx.Node) -> None:
        """
        Merge new_node into existing_node's set. The new node must be a singleton set.
        """
        existing_set = self.merge_sets[existing_node]
        new_set = self.merge_sets[new_node]
        assert len(new_set) == 1

        # Add all nodes from new_set to existing_set
        existing_set.update(new_set)

        # Update all nodes from new_set to point to existing_set
        for node in new_set:
            self.merge_sets[node] = existing_set

    def unmerge_node(self, node: fx.Node) -> None:
        """Remove a node from its merge set, making it singleton."""
        old_set = self.merge_sets[node]

        # If already singleton, nothing to do
        if len(old_set) == 1:
            return

        # Remove from old set
        old_set.remove(node)

        # Make node singleton
        self.merge_sets[node] = OrderedSet([node])

    def get_merged_deps(self, node: fx.Node) -> OrderedSet[fx.Node]:
        """
        Get all dependencies of a node considering merges and extra deps.
        Combines:
        1. Direct deps (all_input_nodes) of node and its merge equivalents
        2. Extra deps of node and its merge equivalents
        """
        deps: OrderedSet[fx.Node] = OrderedSet()

        # For each node in the merge set
        for merged_node in self.merge_sets[node]:
            # Add direct dependencies from all_input_nodes
            deps.update(merged_node.all_input_nodes)
            # Add extra dependencies
            deps.update(self.extra_deps[merged_node])

        return deps

    def has_cycle(self) -> bool:
        merged_deps = {n: self.get_merged_deps(n) for n in self.graph.nodes}
        return torch._dynamo.graph_deduplication._has_cycle(self.graph, merged_deps)

    def has_path(self, source: fx.Node, target: fx.Node) -> bool:
        """Check if there's a path from source to target."""
        # we should not be checking path from node to itself
        assert self.merge_sets[source] is not self.merge_sets[target]

        # search backwards from target to source
        visited: OrderedSet[fx.Node] = OrderedSet()
        queue = [target]
        visited.add(target)

        while queue:
            current = queue.pop()

            for dep in self.get_merged_deps(current):
                # Check if we reached source or its equivalent
                if dep in self.merge_sets[source]:
                    return True

                if dep in visited:
                    continue

                # We are searching from target, so this node is necessarily an ancestor
                # of target.
                # If dep is an ancestor of source, any path through dep to source would imply a cycle
                if self.node_ancestors:
                    source_set = self.merge_sets[source]
                    is_ancestor_of_source = any(
                        dep in self.node_ancestors[s] for s in source_set
                    )
                    # Add to visited to avoid recomputing this check if we see dep again
                    if is_ancestor_of_source:
                        visited.add(dep)
                        continue

                visited.add(dep)
                queue.append(dep)

        return False

    def transfer_erased_node_deps(self, erased_to_new: dict[fx.Node, fx.Node]) -> None:
        """
        Transfer all extra dependencies from erased nodes to their replacements, handling
        cross-dependencies between erased nodes correctly.
        """
        erased_merge_sets: dict[fx.Node, fx.Node] = {}

        for replaced, new in erased_to_new.items():
            for equiv in self.merge_sets[replaced]:
                erased_merge_sets[equiv] = new

        # Transfer dependencies
        for old_node, new_node in erased_merge_sets.items():
            # Transfer dependencies FROM old_node (what old_node depended on)
            for extra_dep in self.extra_deps[old_node]:
                # Redirect if dep is also being erased
                updated_dep = erased_merge_sets.get(extra_dep, extra_dep)
                self.extra_deps[new_node].add(updated_dep)
                self.extra_uses[updated_dep].discard(old_node)
                self.extra_uses[updated_dep].add(new_node)

            # Transfer dependencies TO old_node (what depended on old_node)
            for extra_use in self.extra_uses[old_node]:
                # Redirect if this user is also being erased
                updated_use = erased_merge_sets.get(extra_use, extra_use)

                # Update the user's deps to point to new_node
                self.extra_deps[updated_use].discard(old_node)
                self.extra_deps[updated_use].add(new_node)
                self.extra_uses[new_node].add(updated_use)

        # Clean up erased nodes
        for old_node in erased_merge_sets:
            self.extra_deps[old_node].clear()
            self.extra_uses[old_node].clear()
            del self.merge_sets[old_node]

    def get_all_extra_deps(self) -> dict[fx.Node, OrderedSet[fx.Node]]:
        """
        Get all extra dependencies in a format suitable for topological sort.
        Returns a copy to avoid external modifications.
        """
        return {
            node: OrderedSet(deps)
            for node, deps in self.extra_deps.items()
            if deps  # Only include nodes with non-empty deps
        }

```



## High-Level Overview

"""    Graph helper that augments the original graph with additional    dependencies and uses, plus tracks node equivalences for coalescing.    TODO: if this becomes too large of compile time, consider binding    graphcycles.cc

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AugmentedGraphHelper`

**Functions defined**: `__init__`, `add_extra_dep`, `remove_extra_dep`, `merge_to_set`, `unmerge_node`, `get_merged_deps`, `has_cycle`, `has_path`, `transfer_erased_node_deps`, `get_all_extra_deps`

**Key imports**: defaultdict, Optional, torch, torch.fx as fx, OrderedSet


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: defaultdict
- `typing`: Optional
- `torch`
- `torch.fx as fx`
- `torch.utils._ordered_set`: OrderedSet


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `augmented_graph_helper.py_docs.md`
- **Keyword Index**: `augmented_graph_helper.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

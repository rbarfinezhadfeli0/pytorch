# Documentation: `torch/package/_digraph.py`

## File Metadata

- **Path**: `torch/package/_digraph.py`
- **Size**: 5,630 bytes (5.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from collections import deque


class DiGraph:
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    def __init__(self):
        # Dict of node -> dict of arbitrary attributes
        self._node = {}
        # Nested dict of node -> successor node -> nothing.
        # (didn't implement edge data)
        self._succ = {}
        # Nested dict of node -> predecessor node -> nothing.
        self._pred = {}

        # Keep track of the order in which nodes are added to
        # the graph.
        self._node_order = {}
        self._insertion_idx = 0

    def add_node(self, n, **kwargs):
        """Add a node to the graph.

        Args:
            n: the node. Can we any object that is a valid dict key.
            **kwargs: any attributes you want to attach to the node.
        """
        if n not in self._node:
            self._node[n] = kwargs
            self._succ[n] = {}
            self._pred[n] = {}
            self._node_order[n] = self._insertion_idx
            self._insertion_idx += 1
        else:
            self._node[n].update(kwargs)

    def add_edge(self, u, v):
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
        # add nodes
        self.add_node(u)
        self.add_node(v)

        # add the edge
        self._succ[u][v] = True
        self._pred[v][u] = True

    def successors(self, n):
        """Returns an iterator over successor nodes of n."""
        try:
            return iter(self._succ[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    def predecessors(self, n):
        """Returns an iterator over predecessors nodes of n."""
        try:
            return iter(self._pred[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    @property
    def edges(self):
        """Returns an iterator over all edges (u, v) in the graph"""
        for n, successors in self._succ.items():
            for succ in successors:
                yield n, succ

    @property
    def nodes(self):
        """Returns a dictionary of all nodes to their attributes."""
        return self._node

    def __iter__(self):
        """Iterate over the nodes."""
        return iter(self._node)

    def __contains__(self, n):
        """Returns True if ``n`` is a node in the graph, False otherwise."""
        try:
            return n in self._node
        except TypeError:
            return False

    def forward_transitive_closure(self, src: str) -> set[str]:
        """Returns a set of nodes that are reachable from src"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.successors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def backward_transitive_closure(self, src: str) -> set[str]:
        """Returns a set of nodes that are reachable from src in reverse direction"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def all_paths(self, src: str, dst: str):
        """Returns a subgraph rooted at src that shows all the paths to dst."""

        result_graph = DiGraph()
        # First compute forward transitive closure of src (all things reachable from src).
        forward_reachable_from_src = self.forward_transitive_closure(src)

        if dst not in forward_reachable_from_src:
            return result_graph

        # Second walk the reverse dependencies of dst, adding each node to
        # the output graph iff it is also present in forward_reachable_from_src.
        # we don't use backward_transitive_closures for optimization purposes
        working_set = deque(dst)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n in forward_reachable_from_src:
                    result_graph.add_edge(n, cur)
                    # only explore further if its reachable from src
                    working_set.append(n)

        return result_graph.to_dot()

    def first_path(self, dst: str) -> list[str]:
        """Returns a list of nodes that show the first path that resulted in dst being added to the graph."""
        path = []

        while dst:
            path.append(dst)
            candidates = self._pred[dst].keys()
            dst, min_idx = "", None
            for candidate in candidates:
                idx = self._node_order.get(candidate, None)
                if idx is None:
                    break
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                    dst = candidate

        return list(reversed(path))

    def to_dot(self) -> str:
        """Returns the dot representation of the graph.

        Returns:
            A dot representation of the graph.
        """
        edges = "\n".join(f'"{f}" -> "{t}";' for f, t in self.edges)
        return f"""\
digraph G {{
rankdir = LR;
node [shape=box];
{edges}
}}
"""

```



## High-Level Overview

"""Really simple unweighted directed graph data structure to track dependencies.    The API is pretty much the same as networkx so if you add something just    copy their API.

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DiGraph`

**Functions defined**: `__init__`, `add_node`, `add_edge`, `successors`, `predecessors`, `edges`, `nodes`, `__iter__`, `__contains__`, `forward_transitive_closure`, `backward_transitive_closure`, `all_paths`, `first_path`, `to_dot`

**Key imports**: deque


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/package`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: deque


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`package_exporter.py_docs.md`](./package_exporter.py_docs.md)
- [`_package_pickler.py_docs.md`](./_package_pickler.py_docs.md)
- [`glob_group.py_docs.md`](./glob_group.py_docs.md)
- [`file_structure_representation.py_docs.md`](./file_structure_representation.py_docs.md)
- [`_directory_reader.py_docs.md`](./_directory_reader.py_docs.md)
- [`_mangling.py_docs.md`](./_mangling.py_docs.md)
- [`mangling.md_docs.md`](./mangling.md_docs.md)
- [`package_importer.py_docs.md`](./package_importer.py_docs.md)
- [`find_file_dependencies.py_docs.md`](./find_file_dependencies.py_docs.md)


## Cross-References

- **File Documentation**: `_digraph.py_docs.md`
- **Keyword Index**: `_digraph.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/fx/passes/utils/matcher_with_name_node_map_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/utils/matcher_with_name_node_map_utils.py_docs.md`
- **Size**: 7,406 bytes (7.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/passes/utils/matcher_with_name_node_map_utils.py`

## File Metadata

- **Path**: `torch/fx/passes/utils/matcher_with_name_node_map_utils.py`
- **Size**: 4,241 bytes (4.14 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from torch.fx import Graph, GraphModule, Node
from torch.fx._compatibility import compatibility

from .matcher_utils import InternalMatch, SubgraphMatcher


__all__ = ["SubgraphMatcherWithNameNodeMap"]


def _split_to_graph_and_name_node_map(
    gm: GraphModule,
) -> tuple[GraphModule, dict[str, Node]]:
    from torch.fx.graph import _PyTreeInfo
    from torch.utils._pytree import tree_flatten, tree_unflatten

    name_node_map = {}
    for n in gm.graph.nodes:
        if n.op == "output":
            assert gm._out_spec is not None
            output = tree_unflatten(n.args[0], gm._out_spec)
            assert isinstance(output, tuple), (
                "Expecting the pattern graph to return a tuple"
            )
            assert len(output) >= 2, (
                "Expecting the pattern graph to have at least two outputs"
            )
            *out, name_node_map = output
            flattened, out_spec = tree_flatten(out)
            assert isinstance(name_node_map, dict), (
                "Expecting the input graph to have a dict output as the last element"
            )
            n.args = (flattened,)
            orig_pytree_info = gm._graph._codegen.pytree_info  # type: ignore[attr-defined]
            gm._graph._codegen.pytree_info = _PyTreeInfo(  # type: ignore[attr-defined]
                orig_pytree_info.orig_args, orig_pytree_info.in_spec, out_spec
            )
    gm.recompile()
    return gm, name_node_map


@compatibility(is_backward_compatible=False)
class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    """Extends SubgraphMatcher to support querying the matched subgraph nodes through node name,
    this requires pattern to have specific format (returning and additional dictionary at the output,
    that has node name as key, and the node in the pattern graph as value, see Example for more details)

    Difference with SubgraphMatcher is that it takes a `pattern_gm` GraphModule as input during
    initialization since we need to modify the graph (which requires `recompile` the GraphModule)

    Example::
        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            return relu, {"conv": conv, "relu": relu}


        def target_graph(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu *= 2
            return relu


        pattern_gm = export_for_training(pattern, example_inputs).module()
        target_gm = export_for_training(target_graph, example_inputs).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        matches = matcher.match(target_gm)
        for match in matches:
            match.name_node_map["conv"].meta["annotation"] = ...

    """

    def __init__(
        self,
        pattern_gm: GraphModule,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
    ) -> None:
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        self.name_node_map = name_node_map
        super().__init__(
            pattern_gm.graph,
            match_output,
            match_placeholder,
            remove_overlapping_matches,
            ignore_literals,
        )

    def match(self, graph: Graph, node_name_match: str = "") -> list[InternalMatch]:
        """The returned InternalMatch will have name_node_map populated with a map
        from node name (str) to the target node, e.g.
        {"conv": target_conv_ndoe, "relu": target_relu_node}

        this requires the pattern graph returns an additional
        output of node name to node, e.g. instead of:
        ```
        def pattern(...):
            ...
            return relu
        ```
        we should do:
        ```
        def pattern(...):
            ...
            return relu, {"conv": conv, "relu": relu}
        ``` instead
        """
        internal_matches = super().match(graph, node_name_match)
        for internal_match in internal_matches:
            for k, n in self.name_node_map.items():
                internal_match.name_node_map[k] = internal_match.nodes_map[n]
        return internal_matches

```



## High-Level Overview

"""Extends SubgraphMatcher to support querying the matched subgraph nodes through node name,    this requires pattern to have specific format (returning and additional dictionary at the output,    that has node name as key, and the node in the pattern graph as value, see Example for more details)    Difference with SubgraphMatcher is that it takes a `pattern_gm` GraphModule as input during    initialization since we need to modify the graph (which requires `recompile` the GraphModule)    Example::

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SubgraphMatcherWithNameNodeMap`

**Functions defined**: `_split_to_graph_and_name_node_map`, `pattern`, `target_graph`, `__init__`, `match`, `pattern`, `pattern`

**Key imports**: Graph, GraphModule, Node, compatibility, InternalMatch, SubgraphMatcher, _PyTreeInfo, tree_flatten, tree_unflatten


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.fx`: Graph, GraphModule, Node
- `torch.fx._compatibility`: compatibility
- `.matcher_utils`: InternalMatch, SubgraphMatcher
- `torch.fx.graph`: _PyTreeInfo
- `torch.utils._pytree`: tree_flatten, tree_unflatten


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

Files in the same folder (`torch/fx/passes/utils`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`matcher_utils.py_docs.md`](./matcher_utils.py_docs.md)
- [`fuser_utils.py_docs.md`](./fuser_utils.py_docs.md)
- [`source_matcher_utils.py_docs.md`](./source_matcher_utils.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)


## Cross-References

- **File Documentation**: `matcher_with_name_node_map_utils.py_docs.md`
- **Keyword Index**: `matcher_with_name_node_map_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/passes/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/fx/passes/utils`):

- [`common.py_docs.md_docs.md`](./common.py_docs.md_docs.md)
- [`common.py_kw.md_docs.md`](./common.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`fuser_utils.py_kw.md_docs.md`](./fuser_utils.py_kw.md_docs.md)
- [`source_matcher_utils.py_kw.md_docs.md`](./source_matcher_utils.py_kw.md_docs.md)
- [`matcher_utils.py_docs.md_docs.md`](./matcher_utils.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`matcher_utils.py_kw.md_docs.md`](./matcher_utils.py_kw.md_docs.md)
- [`fuser_utils.py_docs.md_docs.md`](./fuser_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `matcher_with_name_node_map_utils.py_docs.md_docs.md`
- **Keyword Index**: `matcher_with_name_node_map_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

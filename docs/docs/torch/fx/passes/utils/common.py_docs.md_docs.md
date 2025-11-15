# Documentation: `docs/torch/fx/passes/utils/common.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/utils/common.py_docs.md`
- **Size**: 5,740 bytes (5.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/passes/utils/common.py`

## File Metadata

- **Path**: `torch/fx/passes/utils/common.py`
- **Size**: 3,161 bytes (3.09 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.nn import Module


__all__ = ["HolderModule", "lift_subgraph_as_module", "compare_graphs"]


@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """

    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self.add_module(k, v)


@compatibility(is_backward_compatible=False)
def lift_subgraph_as_module(
    gm: GraphModule,
    subgraph: Graph,
    comp_name: str = "",
    class_name: str = "GraphModule",
) -> tuple[GraphModule, dict[str, str]]:
    """
    Create a GraphModule for subgraph, which copies the necessary attributes from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (Graph): a valid subgraph that contains copied nodes from the parent graph

        comp_name (str): name for the new component

        class_name (str): name for the submodule

    """

    # Loop through all module calls (call_module) and param fetches (get_attr)
    # in this component, creating HolderModules as necessary to match the path.
    # e.g. if in the original module there's a get_attr node fetches "conv.weight".
    # We create a HolderModule as root -> add a HolderModule named "conv" ->
    # make "weight" a attribute of "conv" HolderModule and point to conv.weight in
    # the original module.
    submodule = HolderModule({})
    orig_to_split_fqn_mapping: dict[str, str] = {}
    for n in subgraph.nodes:
        if n.op not in ("call_module", "get_attr"):
            continue

        target = n.target
        assert isinstance(target, str)
        target_name_parts = target.split(".")
        curr = submodule
        orig_gm = gm

        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                # pyrefly: ignore [missing-attribute]
                curr.add_module(name, HolderModule({}))

            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)

        leaf_node_name = target_name_parts[-1]
        leaf_node = getattr(orig_gm, leaf_node_name)

        orig_to_split_fqn_mapping[target] = f"{comp_name}.{target}"
        # Relies on custom __setattr__ magic.
        setattr(curr, leaf_node_name, leaf_node)

    return GraphModule(submodule, subgraph, class_name), orig_to_split_fqn_mapping


@compatibility(is_backward_compatible=False)
def compare_graphs(left: Graph, right: Graph) -> bool:
    """
    Return True if two graphs are identical, i.e they
        - have the same number of outputs in the same order
        - have the same number of inputs in the same order
        - have the same set of nodes, and identical connectivity
    """

    matcher = SubgraphMatcher(left, match_output=True, match_placeholder=True)
    matches = matcher.match(right)

    return len(matches) > 0

```



## High-Level Overview

"""    HolderModule is used to copy all the attributes from original module to submodules    that uses the attributes

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HolderModule`

**Functions defined**: `__init__`, `lift_subgraph_as_module`, `compare_graphs`

**Key imports**: compatibility, Graph, GraphModule, SubgraphMatcher, Module


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.fx._compatibility`: compatibility
- `torch.fx.graph`: Graph
- `torch.fx.graph_module`: GraphModule
- `torch.fx.passes.utils.matcher_utils`: SubgraphMatcher
- `torch.nn`: Module


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`torch/fx/passes/utils`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`matcher_with_name_node_map_utils.py_docs.md`](./matcher_with_name_node_map_utils.py_docs.md)
- [`matcher_utils.py_docs.md`](./matcher_utils.py_docs.md)
- [`fuser_utils.py_docs.md`](./fuser_utils.py_docs.md)
- [`source_matcher_utils.py_docs.md`](./source_matcher_utils.py_docs.md)


## Cross-References

- **File Documentation**: `common.py_docs.md`
- **Keyword Index**: `common.py_kw.md`
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
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/fx/passes/utils`):

- [`common.py_kw.md_docs.md`](./common.py_kw.md_docs.md)
- [`matcher_with_name_node_map_utils.py_docs.md_docs.md`](./matcher_with_name_node_map_utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`fuser_utils.py_kw.md_docs.md`](./fuser_utils.py_kw.md_docs.md)
- [`source_matcher_utils.py_kw.md_docs.md`](./source_matcher_utils.py_kw.md_docs.md)
- [`matcher_utils.py_docs.md_docs.md`](./matcher_utils.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`matcher_utils.py_kw.md_docs.md`](./matcher_utils.py_kw.md_docs.md)
- [`fuser_utils.py_docs.md_docs.md`](./fuser_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common.py_docs.md_docs.md`
- **Keyword Index**: `common.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

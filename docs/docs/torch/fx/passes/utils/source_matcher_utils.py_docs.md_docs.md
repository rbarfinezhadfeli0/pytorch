# Documentation: `docs/torch/fx/passes/utils/source_matcher_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/utils/source_matcher_utils.py_docs.md`
- **Size**: 8,284 bytes (8.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/passes/utils/source_matcher_utils.py`

## File Metadata

- **Path**: `torch/fx/passes/utils/source_matcher_utils.py`
- **Size**: 5,781 bytes (5.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Node


__all__ = ["get_source_partitions", "check_subgraphs_connected", "SourcePartition"]


# Set`PYTORCH_MATCHER_LOGLEVEL=INFO` to see debug logs
def _init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)

    level = os.environ.get("PYTORCH_MATCHER_LOGLEVEL", "WARNING").upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = _init_logger()


@compatibility(is_backward_compatible=False)
@dataclass
class SourcePartition:
    # Nodes in a particular partition
    nodes: list[Node]

    # The source these nodes decomposed from
    source: Any

    # Nodes in the graph that are needed as inputs to the partition
    # These do not include the params of the partition
    input_nodes: list[Node] = field(default_factory=list)

    # Nodes in the partition that are being used by nodes outside of the
    # partition
    output_nodes: list[Node] = field(default_factory=list)

    # Parameters that are being used
    params: list[Node] = field(default_factory=list)


@compatibility(is_backward_compatible=False)  # type: ignore[misc]
def get_source_partitions(
    graph: Graph,
    wanted_sources: list[Any],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> dict[Any, list[SourcePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_sources: List of sources of nodes that were decomposed from this
            source. This can be a function (ex. torch.nn.functional.linear) or a
            leaf module type (ex. torch.nn.Linear).

    Returns:
        Dictionary mapping sources that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        source.
    """
    modules: dict[type, dict[str, list[Node]]] = {}

    for node in graph.nodes:
        # The metadata source_fn should contain a tuple of a unique name for the
        # source, and the source function if the node is decomposed from a
        # function, or the type of module if the node is decomposed from a leaf
        # module

        # TODO: Bypass "torch_fn" when "source_fn_stack" because now "torch_fn" can
        # be different from "source_fn_stack", for example for the add_ node
        # decomposed from batch norm. We should remove the check on "source_fn_stack"
        # after we fix "torch_fn". T199561090
        if (source_fn_st := node.meta.get("source_fn_stack", None)) is None and (
            torch_fn := node.meta.get("torch_fn", None)
        ) is not None:
            node_fqn, source_fn = torch_fn
            source_fn_name = source_fn.split(".")[1]
            if source_fn_name in wanted_sources:
                diff_modules = modules.setdefault(source_fn_name, {})
                partition = diff_modules.setdefault(node_fqn, [])
                partition.append(node)

        if (source_fn_st := node.meta.get("source_fn_stack", None)) is not None:
            source_fn = source_fn_st[-1]
            if source_fn[1] in wanted_sources:
                diff_modules = modules.setdefault(source_fn[1], {})
                partition = diff_modules.setdefault(source_fn[0], [])
                partition.append(node)

    def make_partition(nodes: list[Node], module_type: type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes and arg.op != "get_attr":
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)
                # get_attr nodes won't be output nodes
                continue

            for user in node.users:
                if user not in nodes:
                    output_nodes.add(node)

        return SourcePartition(
            nodes,
            module_type,
            list(input_nodes),
            list(output_nodes),
            list(params),  # type: ignore[arg-type]
        )

    ret: dict[type[Any], list[SourcePartition]] = {}

    if filter_fn:
        # for each partition, we apply filter_fn to filter out all partitions that doesn't satisfy the
        # filter condition
        filtered_modules = {}
        for tp, name_to_partition in modules.items():
            filtered_name_to_partition = {
                name: partition
                for name, partition in name_to_partition.items()
                if all(map(filter_fn, partition))
            }
            filtered_modules[tp] = filtered_name_to_partition
        modules = filtered_modules

    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]

    return ret


@compatibility(is_backward_compatible=False)  # type: ignore[misc]
def check_subgraphs_connected(
    subgraph1: SourcePartition, subgraph2: SourcePartition
) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).
    """

    for node in reversed(subgraph1.nodes):
        for user in node.users:
            if user in subgraph2.nodes:
                return True
    return False

```



## High-Level Overview


This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SourcePartition`

**Functions defined**: `_init_logger`, `get_source_partitions`, `make_partition`, `check_subgraphs_connected`

**Key imports**: logging, os, Callable, dataclass, field, Any, Optional, compatibility, Graph, Node


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `os`
- `collections.abc`: Callable
- `dataclasses`: dataclass, field
- `typing`: Any, Optional
- `torch.fx._compatibility`: compatibility
- `torch.fx.graph`: Graph
- `torch.fx.node`: Node


## Code Patterns & Idioms

### Common Patterns

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
- [`common.py_docs.md`](./common.py_docs.md)


## Cross-References

- **File Documentation**: `source_matcher_utils.py_docs.md`
- **Keyword Index**: `source_matcher_utils.py_kw.md`
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

- [`common.py_docs.md_docs.md`](./common.py_docs.md_docs.md)
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

- **File Documentation**: `source_matcher_utils.py_docs.md_docs.md`
- **Keyword Index**: `source_matcher_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

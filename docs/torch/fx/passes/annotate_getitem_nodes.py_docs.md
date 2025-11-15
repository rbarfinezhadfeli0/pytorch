# Documentation: `torch/fx/passes/annotate_getitem_nodes.py`

## File Metadata

- **Path**: `torch/fx/passes/annotate_getitem_nodes.py`
- **Size**: 2,761 bytes (2.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import operator

import torch


def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (Graph): The graph to be annotated
    """
    for node in graph.nodes:
        if node.target is operator.getitem:
            sequence_node, index_node = node.args
            if not sequence_node.type:
                continue
            # container types
            if hasattr(sequence_node.type, "_name"):
                parameterized_types = sequence_node.type.__args__
                if sequence_node.type._name == "Tuple":
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        assert len(parameterized_types) > index_node
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type._name == "List":
                    assert len(parameterized_types) == 1
                    node.type = parameterized_types[0]
            # Generic Alias Type
            elif hasattr(sequence_node.type, "__origin__"):
                parameterized_types = sequence_node.type.__args__
                if sequence_node.type.__origin__ is tuple:
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        assert len(parameterized_types) > index_node
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type.__origin__ is list:
                    assert len(parameterized_types) == 1
                    node.type = parameterized_types[0]
            # NamedTuple type
            elif hasattr(sequence_node.type, "__annotations__"):
                if sequence_node.type == torch.Tensor:
                    continue
                sequence_node_field_types = sequence_node.type.__annotations__
                field_name = sequence_node.type._fields[index_node]
                node.type = sequence_node_field_types[field_name]

```



## High-Level Overview

"""    Annotate the type of getitem nodes, inferred from the type of sequence node.    If sequence node is not annotated with a type, do nothing.    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.    This is helpful since annotations on local names within function are lost during FX transforms.    Adding back known type annotation for getitem nodes to improve jit scriptability.    Args:        graph (Graph): The graph to be annotated

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `annotate_getitem_nodes`

**Key imports**: operator, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

- **File Documentation**: `annotate_getitem_nodes.py_docs.md`
- **Keyword Index**: `annotate_getitem_nodes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

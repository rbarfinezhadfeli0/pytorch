# Keyword Index: `torch/ao/ns/fx/n_shadows_utils.py`

## File Information

- **Original File**: [torch/ao/ns/fx/n_shadows_utils.py](../../../../../torch/ao/ns/fx/n_shadows_utils.py)
- **Documentation**: [`n_shadows_utils.py_docs.md`](./n_shadows_utils.py_docs.md)
- **Folder**: `torch/ao/ns/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`M`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`OutputProp`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)

### Functions

- **`__init__`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_add_placeholder`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_get_attr_name`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_get_attr_wrapper_name`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_get_dedup_subgraphs`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_get_logger_for_subgraph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_get_subgraph_containing_node`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_get_weight_info_from_shadow_wrapper`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_order_nodes`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`create_add_loggers_graph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`create_n_transformed_and_logged_copies_of_subgraph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`create_one_transformed_and_logged_copy_of_subgraph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`create_results_comparison`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`create_submodule_from_subgraph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`extract_weight_comparison`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`fetch_attr`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`forward`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`group_results_by_subgraph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`load_arg`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`maybe_remap_node_to_shadow`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`print_n_shadows_summary`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`propagate`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)

### Imports

- **`Any`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`Callable`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`Graph`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`NSResultsType`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`OutputComparisonLogger`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`QConfigAny`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`QConfigMapping`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_MatchResult`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`_maybe_get_fqn`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`collections`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`collections.abc`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`copy`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`getattr_from_fqn`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`operator`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`tabulate`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.ns._numeric_suite_fx`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.ns.fx.graph_passes`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.ns.fx.ns_types`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.ns.fx.utils`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.quantization`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.quantization.fx.match_utils`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.ao.quantization.utils`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.fx`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`torch.utils._pytree`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`tree_map`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)
- **`typing`**: [n_shadows_utils.py_docs.md](./n_shadows_utils.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*

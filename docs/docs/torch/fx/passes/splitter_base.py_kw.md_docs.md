# Documentation: `docs/torch/fx/passes/splitter_base.py_kw.md`

## File Metadata

- **Path**: `docs/torch/fx/passes/splitter_base.py_kw.md`
- **Size**: 7,575 bytes (7.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/fx/passes/splitter_base.py`

## File Information

- **Original File**: [torch/fx/passes/splitter_base.py](../../../../torch/fx/passes/splitter_base.py)
- **Documentation**: [`splitter_base.py_docs.md`](./splitter_base.py_docs.md)
- **Folder**: `torch/fx/passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CustomDrawer`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`FxNetAccNodesFinder`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`FxNetSplitterInternalError`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`NodeEvent`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`NodeEventTracker`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`SimpleModule`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`SplitResult`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`_SplitterBase`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`_SplitterSettingBase`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`class`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`from`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)

### Functions

- **`__call__`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`__init__`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`_draw_graph_based_on_node_support`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`_find_culprit`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`_get_node_style`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`_lower_model_to_backend`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`add`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`clean_up_handles`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`dump`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`dump_selected_nodes`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`extend_acc_subgraph`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`find_deps`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`find_parent_nodes_of_subgraph`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`find_reverse_deps`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`fn`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`forward`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`generate_inputs_for_submodules`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`generate_split_results`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_bytes`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_dtype`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_inputs`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_node_submodule_map`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_submod_inputs`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`node_support_preview`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`pre_forward`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`print_all`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`print_node`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`put_nodes_into_subgraphs`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`reduce_acc_nodes_non_tensor_input`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`reduce_acc_nodes_non_tensor_input_helper`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`reduce_acc_nodes_non_tensor_output`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`remove_small_acc_subgraphs`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`split`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`split_preview`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`starter_nodes`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`tag`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`to_dict`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`to_str`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`update_deps_for_fusions`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`update_reverse_deps_for_fusions`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`writeln`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)

### Imports

- **`.graph_drawer`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`.operator_support`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`.shape_prop`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`.split_utils`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`.tools_common`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`Any`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`FxGraphDrawer`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`Iterable`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`ShapeProp`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`argparse`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`collections`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`collections.abc`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`compatibility`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`copy`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`dataclass`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`dataclasses`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`defaultdict`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_node_target`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`get_size_of_node`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`json`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`logging`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`map_arg`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`os`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`split_by_tags`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`torch`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`torch._logging`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`torch.fx._compatibility`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`torch.fx.node`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`torch.fx.passes.graph_manipulation`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`trace_structured`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)
- **`typing`**: [splitter_base.py_docs.md](./splitter_base.py_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/torch/fx/passes`):

- [`split_utils.py_kw.md_docs.md`](./split_utils.py_kw.md_docs.md)
- [`fake_tensor_prop.py_kw.md_docs.md`](./fake_tensor_prop.py_kw.md_docs.md)
- [`tools_common.py_kw.md_docs.md`](./tools_common.py_kw.md_docs.md)
- [`param_fetch.py_kw.md_docs.md`](./param_fetch.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_manipulation.py_docs.md_docs.md`](./graph_manipulation.py_docs.md_docs.md)
- [`annotate_getitem_nodes.py_docs.md_docs.md`](./annotate_getitem_nodes.py_docs.md_docs.md)
- [`split_module.py_docs.md_docs.md`](./split_module.py_docs.md_docs.md)
- [`pass_manager.py_kw.md_docs.md`](./pass_manager.py_kw.md_docs.md)
- [`tools_common.py_docs.md_docs.md`](./tools_common.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `splitter_base.py_kw.md_docs.md`
- **Keyword Index**: `splitter_base.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

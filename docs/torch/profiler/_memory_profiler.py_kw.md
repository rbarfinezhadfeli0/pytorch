# Keyword Index: `torch/profiler/_memory_profiler.py`

## File Information

- **Original File**: [torch/profiler/_memory_profiler.py](../../../torch/profiler/_memory_profiler.py)
- **Documentation**: [`_memory_profiler.py_docs.md`](./_memory_profiler.py_docs.md)
- **Folder**: `torch/profiler`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Action`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`Category`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`DataFlowEdge`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`DataFlowGraph`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`DataFlowNode`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`Key`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`MemoryProfile`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`MemoryProfileTimeline`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`OpTree`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`SchemaMatcher`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`SizeMap`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`TensorKey`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`class`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`is`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)

### Functions

- **`__eq__`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`__getitem__`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`__hash__`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`__init__`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`__lt__`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`__repr__`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_any_version_depends_on_gradient`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_as_sortable`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_category_snapshot`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_coalesce_timeline`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_determine_edges`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_extract_leaf_events`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_extract_parameters_and_gradients`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_flat_tensor_inputs`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_is_gradient`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_make`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_activations`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_autograd_detail`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_gradients_and_temporaries`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_inputs`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_optimizer_state`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_parameters_using_data_flow`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_set_parameters_using_python_tracer`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_types_match`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_update_values`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`bump`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`children_fn`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`delete`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`dfs`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`export_memory_timeline`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`export_memory_timeline_html`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`export_memory_timeline_raw`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`extract_gradients`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`extract_parameters`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`flow_nodes`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`from_allocation`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`from_tensor`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`get`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`get_category_index`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`get_scopes`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`inputs`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`inputs_are_mutable`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`intermediates`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`is_allocation`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`is_deletion`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`leaf_events`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`leaf_op`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`lookup`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`lookup_schemas`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`match_schemas`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`matches`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`outputs`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`set_by_id`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`set_by_key`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`set_by_version`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`setdefault_by_version`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`sorted_nodes`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`start_time`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`timeline`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`update`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`validate`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)

### Imports

- **`Any`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`FunctionSchema`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`Iterator`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`NamedTemporaryFile`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_ProfilerResult`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_element_size`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`_utils`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`b64encode`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`base64`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`collections`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`collections.abc`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`dataclasses`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`enum`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`importlib.util`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`itertools`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`json`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`logging`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`matplotlib.pyplot`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`numpy`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`os`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`remove`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`tempfile`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`torch`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`torch._C`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`torch._C._autograd`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`torch._C._profiler`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`torch._utils`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`torch.profiler`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)
- **`typing`**: [_memory_profiler.py_docs.md](./_memory_profiler.py_docs.md)


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

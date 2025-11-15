# Keyword Index: `torch/_functorch/_aot_autograd/utils.py`

## File Information

- **Original File**: [torch/_functorch/_aot_autograd/utils.py](../../../../torch/_functorch/_aot_autograd/utils.py)
- **Documentation**: [`utils.py_docs.md`](./utils.py_docs.md)
- **Folder**: `torch/_functorch/_aot_autograd`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`PytreeThunk`**: [utils.py_docs.md](./utils.py_docs.md)

### Functions

- **`_collect_fwd_nodes_from_subgraph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_copy_metadata_to_bw_nodes_in_subgraph`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_autocast_states`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_get_symint_hints`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_is_backward_node_with_seq_nr`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_is_forward_node_with_seq_nr`**: [utils.py_docs.md](./utils.py_docs.md)
- **`_map_assigned_buffer_to_proxy`**: [utils.py_docs.md](./utils.py_docs.md)
- **`call_and_expect_output_descs`**: [utils.py_docs.md](./utils.py_docs.md)
- **`call_func_at_runtime_with_args`**: [utils.py_docs.md](./utils.py_docs.md)
- **`contain_metadata_mutation_ops`**: [utils.py_docs.md](./utils.py_docs.md)
- **`copy_fwd_metadata_to_bw_nodes`**: [utils.py_docs.md](./utils.py_docs.md)
- **`create_tree_flattened_fn`**: [utils.py_docs.md](./utils.py_docs.md)
- **`do`**: [utils.py_docs.md](./utils.py_docs.md)
- **`f`**: [utils.py_docs.md](./utils.py_docs.md)
- **`flat_fn`**: [utils.py_docs.md](./utils.py_docs.md)
- **`fn_wrappers`**: [utils.py_docs.md](./utils.py_docs.md)
- **`g`**: [utils.py_docs.md](./utils.py_docs.md)
- **`get_cuda_generator_meta_val`**: [utils.py_docs.md](./utils.py_docs.md)
- **`inner`**: [utils.py_docs.md](./utils.py_docs.md)
- **`is_with_effects`**: [utils.py_docs.md](./utils.py_docs.md)
- **`is_with_effects_op`**: [utils.py_docs.md](./utils.py_docs.md)
- **`make_boxed_compiler`**: [utils.py_docs.md](./utils.py_docs.md)
- **`make_boxed_func`**: [utils.py_docs.md](./utils.py_docs.md)
- **`maybe_to_fresh_input`**: [utils.py_docs.md](./utils.py_docs.md)
- **`normalize_as_list`**: [utils.py_docs.md](./utils.py_docs.md)
- **`partial_flatten_asdict`**: [utils.py_docs.md](./utils.py_docs.md)
- **`register_buffer_assignment_hook`**: [utils.py_docs.md](./utils.py_docs.md)
- **`rewrite_output`**: [utils.py_docs.md](./utils.py_docs.md)
- **`rewrite_with_effects_input_token`**: [utils.py_docs.md](./utils.py_docs.md)
- **`root_module_when_exporting_non_strict`**: [utils.py_docs.md](./utils.py_docs.md)
- **`saved_tensors_hooks_are_inlineable`**: [utils.py_docs.md](./utils.py_docs.md)
- **`set`**: [utils.py_docs.md](./utils.py_docs.md)
- **`simple_wraps`**: [utils.py_docs.md](./utils.py_docs.md)
- **`strict_zip`**: [utils.py_docs.md](./utils.py_docs.md)
- **`top_saved_tensors_hooks`**: [utils.py_docs.md](./utils.py_docs.md)
- **`unflatten`**: [utils.py_docs.md](./utils.py_docs.md)
- **`unlift_tokens`**: [utils.py_docs.md](./utils.py_docs.md)
- **`without_output_descs`**: [utils.py_docs.md](./utils.py_docs.md)

### Imports

- **`.descriptors`**: [utils.py_docs.md](./utils.py_docs.md)
- **`AOTOutput`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Any`**: [utils.py_docs.md](./utils.py_docs.md)
- **`BackwardState`**: [utils.py_docs.md](./utils.py_docs.md)
- **`Callable`**: [utils.py_docs.md](./utils.py_docs.md)
- **`FakeScriptObject`**: [utils.py_docs.md](./utils.py_docs.md)
- **`FakeTensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`FunctionalTensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`ParamSpec`**: [utils.py_docs.md](./utils.py_docs.md)
- **`collections.abc`**: [utils.py_docs.md](./utils.py_docs.md)
- **`contextlib`**: [utils.py_docs.md](./utils.py_docs.md)
- **`copy`**: [utils.py_docs.md](./utils.py_docs.md)
- **`dataclasses`**: [utils.py_docs.md](./utils.py_docs.md)
- **`functools`**: [utils.py_docs.md](./utils.py_docs.md)
- **`getArtifactLogger`**: [utils.py_docs.md](./utils.py_docs.md)
- **`lazy_format_graph_code`**: [utils.py_docs.md](./utils.py_docs.md)
- **`logging`**: [utils.py_docs.md](./utils.py_docs.md)
- **`nullcontext`**: [utils.py_docs.md](./utils.py_docs.md)
- **`operator`**: [utils.py_docs.md](./utils.py_docs.md)
- **`py_sym_types`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch._dynamo.utils`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch._library.fake_class_registry`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch._logging`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.fx.experimental._backward_state`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [utils.py_docs.md](./utils.py_docs.md)
- **`torch.utils._pytree`**: [utils.py_docs.md](./utils.py_docs.md)
- **`typing`**: [utils.py_docs.md](./utils.py_docs.md)
- **`typing_extensions`**: [utils.py_docs.md](./utils.py_docs.md)
- **`warnings`**: [utils.py_docs.md](./utils.py_docs.md)
- **`wraps`**: [utils.py_docs.md](./utils.py_docs.md)


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

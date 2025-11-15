# Keyword Index: `torch/_export/converter.py`

## File Information

- **Original File**: [torch/_export/converter.py](../../../torch/_export/converter.py)
- **Documentation**: [`converter.py_docs.md`](./converter.py_docs.md)
- **Folder**: `torch/_export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ExplainTS2FXGraphConverter`**: [converter.py_docs.md](./converter.py_docs.md)
- **`TS2EPConverter`**: [converter.py_docs.md](./converter.py_docs.md)
- **`TS2FXGraphConverter`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_DictMock`**: [converter.py_docs.md](./converter.py_docs.md)

### Functions

- **`__contains__`**: [converter.py_docs.md](./converter.py_docs.md)
- **`__getitem__`**: [converter.py_docs.md](./converter.py_docs.md)
- **`__init__`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_check_prim_loop_support`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_check_set_attr_in_if_block`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_convert_as_noop`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_convert_block_to_subgraph`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_convert_prim_iterator`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_convert_prim_unpack_iterator`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_convert_standard_operators`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_create_jit_graph`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_dfs_get_attr`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_dfs_get_attr_dependency`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_get_param_count_list`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_identify_inputs_as_arguments`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_is_get_attr_node`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_map_blocks_to_lifted_attrs`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_trace_and_get_graph_from_model`**: [converter.py_docs.md](./converter.py_docs.md)
- **`add_subgraph`**: [converter.py_docs.md](./converter.py_docs.md)
- **`construct_fqn`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_Bool`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_Float`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_Int`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten___getitem__`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten__convolution`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_add`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_append`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_div`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_tensor`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_aten_to`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_call_function_op`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_graph_inputs`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_graph_outputs`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_node`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_CallMethod`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_Constant`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_CreateObject`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_DictConstruct`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_Enter`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_Exit`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_GetAttr`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_If`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_ListConstruct`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_ListUnpack`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_Loop`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_NumToTensor`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_SetAttr`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_TupleConstruct`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_TupleUnpack`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_Uninitialized`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_device`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_prim_tolist`**: [converter.py_docs.md](./converter.py_docs.md)
- **`convert_profiler__record_function_exit`**: [converter.py_docs.md](./converter.py_docs.md)
- **`disable_logging`**: [converter.py_docs.md](./converter.py_docs.md)
- **`execute_subgraph_from_prim_loop`**: [converter.py_docs.md](./converter.py_docs.md)
- **`explain`**: [converter.py_docs.md](./converter.py_docs.md)
- **`forward`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_args_kwargs`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_attr`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_attribute_fqn_from_ts_node`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_block_to_lifted_attrs`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_dtype_as_int`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_fqn`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_fx_value_by_fqn`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_fx_value_by_ir_value`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_ir_value_parent_name_and_attr_name`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_node_as_placeholder_or_get_attr`**: [converter.py_docs.md](./converter.py_docs.md)
- **`get_op_overload`**: [converter.py_docs.md](./converter.py_docs.md)
- **`inplace_optimize_sym_size_div`**: [converter.py_docs.md](./converter.py_docs.md)
- **`ir_name_to_func_name`**: [converter.py_docs.md](./converter.py_docs.md)
- **`is_top_level_graph`**: [converter.py_docs.md](./converter.py_docs.md)
- **`is_valid_for_codegen`**: [converter.py_docs.md](./converter.py_docs.md)
- **`lift_get_attr`**: [converter.py_docs.md](./converter.py_docs.md)
- **`list_add`**: [converter.py_docs.md](./converter.py_docs.md)
- **`list_append`**: [converter.py_docs.md](./converter.py_docs.md)
- **`normalize_name`**: [converter.py_docs.md](./converter.py_docs.md)
- **`pattern`**: [converter.py_docs.md](./converter.py_docs.md)
- **`replacement`**: [converter.py_docs.md](./converter.py_docs.md)
- **`retrace_as_exported_program`**: [converter.py_docs.md](./converter.py_docs.md)
- **`target`**: [converter.py_docs.md](./converter.py_docs.md)
- **`to_dynamic_tensor`**: [converter.py_docs.md](./converter.py_docs.md)
- **`to_float_tensor`**: [converter.py_docs.md](./converter.py_docs.md)

### Imports

- **`Any`**: [converter.py_docs.md](./converter.py_docs.md)
- **`Callable`**: [converter.py_docs.md](./converter.py_docs.md)
- **`ExportedProgram`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_C`**: [converter.py_docs.md](./converter.py_docs.md)
- **`_tree_map_with_path`**: [converter.py_docs.md](./converter.py_docs.md)
- **`builtins`**: [converter.py_docs.md](./converter.py_docs.md)
- **`collections.abc`**: [converter.py_docs.md](./converter.py_docs.md)
- **`contextlib`**: [converter.py_docs.md](./converter.py_docs.md)
- **`contextmanager`**: [converter.py_docs.md](./converter.py_docs.md)
- **`logging`**: [converter.py_docs.md](./converter.py_docs.md)
- **`operator`**: [converter.py_docs.md](./converter.py_docs.md)
- **`subgraph_rewriter`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch._export.passes.replace_quantized_ops_with_standard_ops_pass`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch.export._trace`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch.export.dynamic_shapes`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch.export.exported_program`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch.export.graph_signature`**: [converter.py_docs.md](./converter.py_docs.md)
- **`torch.fx`**: [converter.py_docs.md](./converter.py_docs.md)
- **`typing`**: [converter.py_docs.md](./converter.py_docs.md)
- **`warnings`**: [converter.py_docs.md](./converter.py_docs.md)


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

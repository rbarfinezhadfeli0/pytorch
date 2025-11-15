# Keyword Index: `torch/_dynamo/codegen.py`

## File Information

- **Original File**: [torch/_dynamo/codegen.py](../../../torch/_dynamo/codegen.py)
- **Documentation**: [`codegen.py_docs.md`](./codegen.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`PyCodegen`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`class`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`uses`**: [codegen.py_docs.md](./codegen.py_docs.md)

### Functions

- **`__call__`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`__init__`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`add_cache`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`add_graph_output`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`add_push_null`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`append_output`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`call_function`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`call_method`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`call_reconstruct`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`clear_tos`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`collect_temp_sources`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_binary_subscr`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_call_function_kw`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_delete`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_import_name`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_attr`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_attrs`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_closure`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_const`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_const_unchecked`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_deref`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_global`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_load_python_module`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_store`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_store_attr`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`create_store_deref`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`dup_top`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`extend_output`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`extract_nested_sources`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`foreach`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`gen_fn`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`get_instructions`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`graph_output_vars`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`load_attr`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`load_deref`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`load_function_name`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`load_graph_output`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`load_import_from`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`load_method`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`make_call_generated_code`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`make_function_with_closure`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`mark_source_temp`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`pop_top`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`restore_stack`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`rot_n`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`setup_globally_cached`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`store`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`store_attr`**: [codegen.py_docs.md](./codegen.py_docs.md)

### Imports

- **`.`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.bytecode_transformation`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.exc`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.source`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.symbolic_convert`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.utils`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.variables.base`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.variables.functions`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.variables.nn_module`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.variables.tensor`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`.variables.torch_function`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`Any`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`AttrSource`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`Callable`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`Counter`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`GraphArg`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`IncorrectUsage`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`InstructionTranslatorBase`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`NNModuleVariable`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`OrderedSet`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`TensorWithTFOverrideVariable`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`ValueMutationExisting`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`collections`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`collections.abc`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`config`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`dataclasses`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`is_safe_constant`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`re`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`sys`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`torch._dynamo.variables.builder`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`torch.nn`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`torch.utils._ordered_set`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`types`**: [codegen.py_docs.md](./codegen.py_docs.md)
- **`typing`**: [codegen.py_docs.md](./codegen.py_docs.md)


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

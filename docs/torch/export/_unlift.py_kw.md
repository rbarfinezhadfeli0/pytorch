# Keyword Index: `torch/export/_unlift.py`

## File Information

- **Original File**: [torch/export/_unlift.py](../../../torch/export/_unlift.py)
- **Documentation**: [`_unlift.py_docs.md`](./_unlift.py_docs.md)
- **Folder**: `torch/export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GuardsFn`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_StatefulGraphModule`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_StatefulGraphModuleFactory`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`for`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`that`**: [_unlift.py_docs.md](./_unlift.py_docs.md)

### Functions

- **`_`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`__call__`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`__init__`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_check_input_constraints_for_module`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_check_input_constraints_pre_hook`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_check_inputs_match`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_convert_guards_code_to_fn`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_create`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_create_stateful_graph_module`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_force_ep_signature_match`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_force_gm_signature_match`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_get_codegen`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_get_input_guards_for_graph`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_get_input_paths`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_insert_copy_for_mutations`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_match_normalized_structure`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_normalize_type`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_ok_to_generate_guards_fn`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_register_attrs_to_new_gm`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_replace_sources`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_unlift`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_unlift_exported_program_lifted_states`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_unlift_inputs_as_getattr`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`eq_spec`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`forward`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`handle_symint`**: [_unlift.py_docs.md](./_unlift.py_docs.md)

### Imports

- **`._remove_effect_tokens_pass`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`._tree_utils`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`.exported_program`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`Any`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`NodeSource`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`SYMPY_INTERP`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`Sequence`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`ValueRanges`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_PyTreeCodeGen`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_assign_attr`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_check_input_constraints_for_graph`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_pytree_subclasses_that_lose_info`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`_remove_effect_tokens`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`ast`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`chain`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`collections.abc`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`copy`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`inspect`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`itertools`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`math`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`re`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`reorder_kwargs`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`sympy`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch._export.non_strict_utils`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch._export.passes.add_runtime_assertions_for_constraints_pass`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch._export.utils`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.export.unflatten`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.fx.graph`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.fx.traceback`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.utils._pytree`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.utils._sympy.solve`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`try_solve`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`typing`**: [_unlift.py_docs.md](./_unlift.py_docs.md)
- **`warnings`**: [_unlift.py_docs.md](./_unlift.py_docs.md)


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

# Keyword Index: `torch/_dynamo/side_effects.py`

## File Information

- **Original File**: [torch/_dynamo/side_effects.py](../../../torch/_dynamo/side_effects.py)
- **Documentation**: [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`SideEffects`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`instance`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`mro`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`objects`**: [side_effects.py_docs.md](./side_effects.py_docs.md)

### Functions

- **`__contains__`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`__eq__`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`__getitem__`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`__init__`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`_get_modified_vars`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`_manual_dict_setitem`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`_manual_list_update`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`_track_obj`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`allow_externally_visible_side_effects_in_subtracer`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`allow_side_effects_under_checkpoint`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`check_allowed_side_effect`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`clear`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`clone`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`cls_supports_mutation_side_effects`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`codegen_hooks`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`codegen_save_tempvars`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`codegen_update_mutated`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`diff`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`disallow_side_effects_in_generator`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`gen_fn`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`get_ca_final_callbacks_var`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`get_example_value`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`get_variable_cls`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`has_existing_dict_mutation`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`has_pending_mutation`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`has_pending_mutation_of_attr`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`ignore_mutations_on`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_attribute_mutation`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_empty`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_live`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_modified`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_reconstructing_generator`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`load_attr`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`load_cell`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`load_global`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`load_new_method`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`mutation`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`prune_dead_object_new`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`register_hook`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`remove_hook`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`should_allow_externally_visible_side_effects_in_subtracer`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`should_allow_side_effects_under_checkpoint`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`stop_ignoring_mutations_on`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`store_attr`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`store_cell`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`store_global`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_cell_existing`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_cell_new`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_global_existing`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_new_user_defined_object`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_object_existing`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_object_new`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_runahead_tensor_and_symvar_side_effects`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`track_save_for_backward`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`visit`**: [side_effects.py_docs.md](./side_effects.py_docs.md)

### Imports

- **`.`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.bytecode_transformation`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.codegen`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.exc`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.source`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.utils`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.variables.base`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.variables.ctx_manager`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.variables.torch_function`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`.variables.user_defined`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`Any`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`AutogradFunctionContextVariable`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`CellType`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`FrozenDataClassVariable`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`Generator`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`GenericContextWrappingVariable`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`GlobalSource`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`InstructionTranslatorBase`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`ListVariable`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`OutputGraph`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`PyCodegen`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`SideEffectsError`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`TorchFunctionMode`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`TorchFunctionModeVariable`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`ValueMutationNew`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`collections`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`collections.abc`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`contextlib`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`graph_break_hints`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`inspect`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_forbidden_context_manager`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`is_frozen_dataclass`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`torch._dynamo.output_graph`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`torch._dynamo.symbolic_convert`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`torch._dynamo.variables.lists`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`torch._dynamo.variables.misc`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`torch.nn`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`torch.overrides`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`types`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`typing`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`warnings`**: [side_effects.py_docs.md](./side_effects.py_docs.md)
- **`weakref`**: [side_effects.py_docs.md](./side_effects.py_docs.md)


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

# Keyword Index: `torch/_C/_functorch.pyi`

## File Information

- **Original File**: [torch/_C/_functorch.pyi](../../../torch/_C/_functorch.pyi)
- **Documentation**: [`_functorch.pyi_docs.md`](./_functorch.pyi_docs.md)
- **Folder**: `torch/_C`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CFunctionalizeInterpreterPtr`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`CGradInterpreterPtr`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`CInterpreter`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`CJvpInterpreterPtr`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`CVmapInterpreterPtr`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`DynamicLayer`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`RandomnessType`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`TransformType`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)

### Functions

- **`__init__`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_add_batch_dim`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_grad_decrement_nesting`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_grad_increment_nesting`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_jvp_decrement_nesting`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_jvp_increment_nesting`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_maybe_unsafe_set_level`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_set_dynamic_layer_keys_included`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_unwrap_batched`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_unwrap_for_grad`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_unwrap_functional_tensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_vmap_decrement_nesting`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_vmap_increment_nesting`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_wrap_for_grad`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`_wrap_functional_tensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`batchSize`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`count_jvp_interpreters`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`current_level`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`deserialize`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`functionalizeAddBackViews`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`get_dynamic_layer_stack_depth`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`get_inplace_requires_grad_allowed`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`get_interpreter_stack`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`get_single_level_autograd_function_allowed`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`get_unwrapped`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`is_batchedtensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`is_functionaltensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`is_functorch_wrapped_tensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`is_gradtrackingtensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`is_legacy_batchedtensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`key`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`level`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`lift`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`maybe_current_level`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`maybe_get_bdim`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`maybe_get_level`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`peek_interpreter_stack`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`pop_dynamic_layer_stack`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`pop_dynamic_layer_stack_and_undo_to_depth`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`prevFwdGradMode`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`prevGradMode`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`push_dynamic_layer_stack`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`randomness`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`serialize`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`set_inplace_requires_grad_allowed`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`set_single_level_autograd_function_allowed`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`unwrap_if_dead`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)

### Imports

- **`Enum`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`Tensor`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`enum`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)
- **`torch`**: [_functorch.pyi_docs.md](./_functorch.pyi_docs.md)


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

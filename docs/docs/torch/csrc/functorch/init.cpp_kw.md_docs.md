# Documentation: `docs/torch/csrc/functorch/init.cpp_kw.md`

## File Metadata

- **Path**: `docs/torch/csrc/functorch/init.cpp_kw.md`
- **Size**: 4,876 bytes (4.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/csrc/functorch/init.cpp`

## File Information

- **Original File**: [torch/csrc/functorch/init.cpp](../../../../torch/csrc/functorch/init.cpp)
- **Documentation**: [`init.cpp_docs.md`](./init.cpp_docs.md)
- **Folder**: `torch/csrc/functorch`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`APIs`**: [init.cpp_docs.md](./init.cpp_docs.md)

### Functions

- **`_add_batch_dim`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_assert_wrapped_functional`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_func_decrement_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_func_increment_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_grad_decrement_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_grad_increment_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_jvp_decrement_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_jvp_increment_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_movedim`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_propagate_functional_input_mutation`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_remove_batch_dim`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_set_dynamic_layer_keys_included`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_unwrap_for_grad`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_unwrap_functional_tensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_vmap_decrement_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_vmap_increment_nesting`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_wrap_for_grad`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`_wrap_functional_tensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`currentLevel`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`dlevel`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`dump_dls`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`dump_local_tls`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`dump_tensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`get_randomness_enum`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`get_unwrapped`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`has_level`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`if`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`initFuncTorchBindings`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`is_batchedtensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`is_functionaltensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`is_gradtrackingtensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`is_legacy_batchedtensor`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`maybe_get_bdim`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`maybe_get_level`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`maybe_unsafe_set_level`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`popDynamicLayerStackToDepth`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`tls_set_vmap_excluded`**: [init.cpp_docs.md](./init.cpp_docs.md)

### Includes

- **`ATen/FunctionalTensorWrapper.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/WrapDimUtils.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/BatchRulesHelper.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/BatchedFallback.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/BatchedTensorImpl.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/DynamicLayer.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/Interpreter.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/LegacyVmapTransforms.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/PlumbingHelper.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`ATen/functorch/TensorWrapper.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`c10/core/AutogradState.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`iostream`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`torch/csrc/functorch/init.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`torch/csrc/utils/python_raii.h`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`torch/python.h`**: [init.cpp_docs.md](./init.cpp_docs.md)

### Namespaces

- **`at`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`static`**: [init.cpp_docs.md](./init.cpp_docs.md)
- **`torch`**: [init.cpp_docs.md](./init.cpp_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/csrc/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/functorch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/functorch`):

- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)
- [`init.h_docs.md_docs.md`](./init.h_docs.md_docs.md)
- [`init.h_kw.md_docs.md`](./init.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `init.cpp_kw.md_docs.md`
- **Keyword Index**: `init.cpp_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

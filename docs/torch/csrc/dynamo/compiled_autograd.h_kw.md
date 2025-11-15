# Keyword Index: `torch/csrc/dynamo/compiled_autograd.h`

## File Information

- **Original File**: [torch/csrc/dynamo/compiled_autograd.h](../../../../torch/csrc/dynamo/compiled_autograd.h)
- **Documentation**: [`compiled_autograd.h_docs.md`](./compiled_autograd.h_docs.md)
- **Folder**: `torch/csrc/dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AutogradCompilerCall`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`CacheKey`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`CacheKeyBuffer`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`CompiledNodeArgs`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`IValuePacker`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`LiftedIValueArg`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`LiftedIValueArgs`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`NodeCall`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`NodeCalls`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`PackedArgs`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`SizeInput`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`Stashed`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`StashedVars`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`SwapSavedVariables`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`TORCH_API`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`TensorArg`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`TensorArgs`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`TraceState`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`for`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`std`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)

### Functions

- **`add`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`add_post_acc_grad_hook`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`add_post_hook`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`add_pre_hook`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`add_size_input`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`add_tensor_pre_hook`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`after`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`before`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`call_accumulate_grad`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`call_copy_slices_epilogue`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`call_copy_slices_prologue`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`call_function`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`call_unpack`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`clear`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`collect`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`collect_hooks_from`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`collect_pynode_objs`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`cond`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`constexpr`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`debug_assert`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`debug_asserts`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`defined`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`emplace_hook`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`emplace_packed_input`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`hash`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`if`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`index`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`key`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`mark_output`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`pack`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`pack_TensorOptions`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`pack_saved_data`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`packed_type`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`restore`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`save`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`set_active_node_call_idx`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`set_default_dyn_type`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`specialize_on_bytes`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`unpack`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`unpack_TensorOptions`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)

### Includes

- **`ATen/TensorGeometry.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`ATen/core/ivalue.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`c10/core/impl/TorchDispatchModeTLS.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`c10/util/flat_hash_map.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`torch/csrc/autograd/function.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`torch/csrc/autograd/input_metadata.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`torch/csrc/autograd/saved_variable.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`torch/csrc/autograd/variable_info.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`torch/csrc/utils/python_stub.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`torch/csrc/utils/torch_dispatch_mode.h`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`typeindex`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)
- **`vector`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)

### Namespaces

- **`torch`**: [compiled_autograd.h_docs.md](./compiled_autograd.h_docs.md)


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

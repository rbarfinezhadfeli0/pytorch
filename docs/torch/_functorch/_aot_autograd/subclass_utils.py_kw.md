# Keyword Index: `torch/_functorch/_aot_autograd/subclass_utils.py`

## File Information

- **Original File**: [torch/_functorch/_aot_autograd/subclass_utils.py](../../../../torch/_functorch/_aot_autograd/subclass_utils.py)
- **Documentation**: [`subclass_utils.py_docs.md`](./subclass_utils.py_docs.md)
- **Folder**: `torch/_functorch/_aot_autograd`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`desugaring`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`from`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`fw`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`info`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`into`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`is`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`metadata`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`mutation`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`outputs`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`requires`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`sizes`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`tensor`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`tensors`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`types`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)

### Functions

- **`_get_types_for_subclass`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`compute_inner_mutated_inp_indices_from_subclass_meta`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`compute_symint_placeholders`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`create_subclass_meta`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`create_subclass_metadata`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`enumerate_filter_symints`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`flatten_subclass`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`get_subclass_typing_container`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`maybe_suggest_memory_format`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`remap_unwrapped_subclass_arg_indices`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`requires_subclass_dispatch`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`runtime_unwrap_tensor_subclasses`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`symint_check`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`unwrap_tensor_subclasses`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`unwrap_tensor_subclasses_with_indices_to_original`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`wrap_tensor_subclasses`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`wrap_tensor_subclasses_maybe_joint`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)

### Imports

- **`.descriptors`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`.schemas`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`.utils`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`Any`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`Callable`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`IntLikeType`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`MemoryFormatMeta`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`SubclassCreationMeta`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`SymInt`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`collections`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`collections.abc`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`get_plain_tensors`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`is_traceable_wrapper_subclass`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`strict_zip`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`torch`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`torch._functorch._aot_autograd.schemas`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`torch.types`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`torch.utils._python_dispatch`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`torch.utils._pytree`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)
- **`typing`**: [subclass_utils.py_docs.md](./subclass_utils.py_docs.md)


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

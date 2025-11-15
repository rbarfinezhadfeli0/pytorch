# Documentation: `docs/torch/distributed/tensor/_api.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_api.py_kw.md`
- **Size**: 5,199 bytes (5.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/tensor/_api.py`

## File Information

- **Original File**: [torch/distributed/tensor/_api.py](../../../../torch/distributed/tensor/_api.py)
- **Documentation**: [`_api.py_docs.md`](./_api.py_docs.md)
- **Folder**: `torch/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DTensor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`_FromTorchTensor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`_ToTorchTensor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`attribute`**: [_api.py_docs.md](./_api.py_docs.md)
- **`constructor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`could`**: [_api.py_docs.md](./_api.py_docs.md)
- **`of`**: [_api.py_docs.md](./_api.py_docs.md)
- **`that`**: [_api.py_docs.md](./_api.py_docs.md)

### Functions

- **`__coerce_same_metadata_as_tangent__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__coerce_tangent_metadata__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__create_chunk_list__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__create_write_items__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__get_tensor_shard__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__init__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__metadata_guard__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__new__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__repr__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__tensor_flatten__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__tensor_unflatten__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`__torch_dispatch__`**: [_api.py_docs.md](./_api.py_docs.md)
- **`_dtensor_init_helper`**: [_api.py_docs.md](./_api.py_docs.md)
- **`_raise_if_contains_partial_placements`**: [_api.py_docs.md](./_api.py_docs.md)
- **`_shard_tensor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`backward`**: [_api.py_docs.md](./_api.py_docs.md)
- **`device_mesh`**: [_api.py_docs.md](./_api.py_docs.md)
- **`distribute_module`**: [_api.py_docs.md](./_api.py_docs.md)
- **`distribute_tensor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`empty`**: [_api.py_docs.md](./_api.py_docs.md)
- **`forward`**: [_api.py_docs.md](./_api.py_docs.md)
- **`from_local`**: [_api.py_docs.md](./_api.py_docs.md)
- **`full`**: [_api.py_docs.md](./_api.py_docs.md)
- **`full_tensor`**: [_api.py_docs.md](./_api.py_docs.md)
- **`ones`**: [_api.py_docs.md](./_api.py_docs.md)
- **`placements`**: [_api.py_docs.md](./_api.py_docs.md)
- **`rand`**: [_api.py_docs.md](./_api.py_docs.md)
- **`randn`**: [_api.py_docs.md](./_api.py_docs.md)
- **`redistribute`**: [_api.py_docs.md](./_api.py_docs.md)
- **`replicate_module_params_buffers`**: [_api.py_docs.md](./_api.py_docs.md)
- **`to_local`**: [_api.py_docs.md](./_api.py_docs.md)
- **`zeros`**: [_api.py_docs.md](./_api.py_docs.md)

### Imports

- **`Any`**: [_api.py_docs.md](./_api.py_docs.md)
- **`Callable`**: [_api.py_docs.md](./_api.py_docs.md)
- **`DTensorSpec`**: [_api.py_docs.md](./_api.py_docs.md)
- **`_mesh_resources`**: [_api.py_docs.md](./_api.py_docs.md)
- **`check_tensor_meta`**: [_api.py_docs.md](./_api.py_docs.md)
- **`collections.abc`**: [_api.py_docs.md](./_api.py_docs.md)
- **`copy`**: [_api.py_docs.md](./_api.py_docs.md)
- **`deprecated`**: [_api.py_docs.md](./_api.py_docs.md)
- **`inspect`**: [_api.py_docs.md](./_api.py_docs.md)
- **`mark_subclass_constructor_exportable_experimental`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch._export.wrappers`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.checkpoint.planner_helpers`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.device_mesh`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor._collective_utils`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor._dispatch`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor._random`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor._redistribute`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor._utils`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch.nn`**: [_api.py_docs.md](./_api.py_docs.md)
- **`torch_xla.distributed.spmd`**: [_api.py_docs.md](./_api.py_docs.md)
- **`typing`**: [_api.py_docs.md](./_api.py_docs.md)
- **`typing_extensions`**: [_api.py_docs.md](./_api.py_docs.md)
- **`warnings`**: [_api.py_docs.md](./_api.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/distributed/tensor`):

- [`device_mesh.py_docs.md_docs.md`](./device_mesh.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`placement_types.py_kw.md_docs.md`](./placement_types.py_kw.md_docs.md)
- [`device_mesh.py_kw.md_docs.md`](./device_mesh.py_kw.md_docs.md)
- [`_sharding_prop.py_kw.md_docs.md`](./_sharding_prop.py_kw.md_docs.md)
- [`_random.py_kw.md_docs.md`](./_random.py_kw.md_docs.md)
- [`_collective_utils.py_kw.md_docs.md`](./_collective_utils.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_collective_utils.py_docs.md_docs.md`](./_collective_utils.py_docs.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_api.py_kw.md_docs.md`
- **Keyword Index**: `_api.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

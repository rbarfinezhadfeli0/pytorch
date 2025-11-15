# Documentation: `docs/torch/distributed/fsdp/_fully_shard/_fully_shard.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/fsdp/_fully_shard/_fully_shard.py_kw.md`
- **Size**: 6,492 bytes (6.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/fsdp/_fully_shard/_fully_shard.py`

## File Information

- **Original File**: [torch/distributed/fsdp/_fully_shard/_fully_shard.py](../../../../../torch/distributed/fsdp/_fully_shard/_fully_shard.py)
- **Documentation**: [`_fully_shard.py_docs.md`](./_fully_shard.py_docs.md)
- **Folder**: `torch/distributed/fsdp/_fully_shard`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FSDPModule`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`UnshardHandle`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_UnshardHandleImpl`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`and`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`for`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`itself`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)

### Functions

- **`__init__`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`__new__`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_apply`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_assert_all_fsdp_modules`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_get_fsdp_state`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_set_unshard_async_op`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_unimplemented_deepcopy`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`disable_fsdp_module_new_init`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`fully_shard`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`get_cls_to_fsdp_cls`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`register_fsdp_forward_method`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`reshard`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_all_reduce_hook`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_allocate_memory_from_process_group_for_comm`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_custom_all_gather`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_custom_reduce_scatter`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_force_sum_reduction_for_comms`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_gradient_divide_factor`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_is_last_backward`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_modules_to_backward_prefetch`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_modules_to_forward_prefetch`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_post_optim_event`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_reduce_scatter_divide_factor`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_requires_all_reduce`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_requires_gradient_sync`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_reshard_after_backward`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_reshard_after_forward`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`set_unshard_in_backward`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`share_comm_ctx`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`unshard`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`wait`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`wrapped_method`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)

### Imports

- **`._fsdp_api`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`._fsdp_common`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`._fsdp_init`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`._fsdp_param_group`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`._fsdp_state`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`AllGather`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`Any`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`Callable`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`DeviceMesh`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`FSDPMeshInfo`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`FSDPParamGroup`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`__future__`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_get_module_fsdp_state`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`_get_root_modules`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`annotations`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`collections.abc`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`contextlib`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`contextmanager`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`contract`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`deprecated`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`functools`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`share_comm_ctx`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`torch`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`torch.distributed._composable`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`torch.distributed.fsdp`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`torch.distributed.tensor`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`torch.distributed.utils`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`torch.nn`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`typing`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)
- **`typing_extensions`**: [_fully_shard.py_docs.md](./_fully_shard.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/distributed/fsdp/_fully_shard`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/fsdp/_fully_shard`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/fsdp/_fully_shard`):

- [`_fsdp_common.py_docs.md_docs.md`](./_fsdp_common.py_docs.md_docs.md)
- [`_fsdp_collectives.py_kw.md_docs.md`](./_fsdp_collectives.py_kw.md_docs.md)
- [`_fsdp_init.py_docs.md_docs.md`](./_fsdp_init.py_docs.md_docs.md)
- [`_fsdp_param.py_kw.md_docs.md`](./_fsdp_param.py_kw.md_docs.md)
- [`_fsdp_state.py_kw.md_docs.md`](./_fsdp_state.py_kw.md_docs.md)
- [`_fsdp_collectives.py_docs.md_docs.md`](./_fsdp_collectives.py_docs.md_docs.md)
- [`_fully_shard.py_docs.md_docs.md`](./_fully_shard.py_docs.md_docs.md)
- [`_fsdp_init.py_kw.md_docs.md`](./_fsdp_init.py_kw.md_docs.md)
- [`_fsdp_api.py_docs.md_docs.md`](./_fsdp_api.py_docs.md_docs.md)
- [`_fsdp_state.py_docs.md_docs.md`](./_fsdp_state.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_fully_shard.py_kw.md_docs.md`
- **Keyword Index**: `_fully_shard.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

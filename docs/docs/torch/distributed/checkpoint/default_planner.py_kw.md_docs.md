# Documentation: `docs/torch/distributed/checkpoint/default_planner.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/checkpoint/default_planner.py_kw.md`
- **Size**: 6,278 bytes (6.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/checkpoint/default_planner.py`

## File Information

- **Original File**: [torch/distributed/checkpoint/default_planner.py](../../../../torch/distributed/checkpoint/default_planner.py)
- **Documentation**: [`default_planner.py_docs.md`](./default_planner.py_docs.md)
- **Folder**: `torch/distributed/checkpoint`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DefaultLoadPlanner`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`DefaultSavePlanner`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_EmptyStateDictLoadPlanner`**: [default_planner.py_docs.md](./default_planner.py_docs.md)

### Functions

- **`__init__`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_check_box_bounds`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_check_box_overlap`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_create_default_local_metadata`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_create_global_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_create_global_plan_with_caching`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_dedup_save_plans`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_finish_plan_with_caching`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_should_include_key`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_validate_global_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`commit_tensor`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`create_default_global_load_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`create_default_global_save_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`create_default_local_load_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`create_default_local_save_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`create_global_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`create_local_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`finish_plan`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`load_bytes`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`lookup_object`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`lookup_tensor`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`resolve_data`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`resolve_tensor`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`set_up_planner`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`transform_object`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`transform_tensor`**: [default_planner.py_docs.md](./default_planner.py_docs.md)

### Imports

- **`.`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`Any`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`ChainMap`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`DTensor`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_flatten_sharded_tensors`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`_version`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`bisect`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`bisect_right`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`collections`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`dataclasses`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`dedup_save_plans`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`find_state_dict_object`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`io`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`logging`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`math`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`narrow_tensor_by_index`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`set_element`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`sys`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed._shard._utils`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint._dedup_save_plans`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint._nested_dict`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint._sharded_tensor_utils`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint._traverse`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint.metadata`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint.planner`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint.planner_helpers`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.checkpoint.utils`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`torch.distributed.tensor`**: [default_planner.py_docs.md](./default_planner.py_docs.md)
- **`typing`**: [default_planner.py_docs.md](./default_planner.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/distributed/checkpoint`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/checkpoint`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributed/checkpoint`):

- [`storage.py_docs.md_docs.md`](./storage.py_docs.md_docs.md)
- [`api.py_kw.md_docs.md`](./api.py_kw.md_docs.md)
- [`_async_process_executor.py_kw.md_docs.md`](./_async_process_executor.py_kw.md_docs.md)
- [`stateful.py_kw.md_docs.md`](./stateful.py_kw.md_docs.md)
- [`state_dict_loader.py_kw.md_docs.md`](./state_dict_loader.py_kw.md_docs.md)
- [`_async_executor.py_kw.md_docs.md`](./_async_executor.py_kw.md_docs.md)
- [`_state_dict_stager.py_kw.md_docs.md`](./_state_dict_stager.py_kw.md_docs.md)
- [`_extension.py_kw.md_docs.md`](./_extension.py_kw.md_docs.md)
- [`resharding.py_docs.md_docs.md`](./resharding.py_docs.md_docs.md)
- [`format_utils.py_docs.md_docs.md`](./format_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `default_planner.py_kw.md_docs.md`
- **Keyword Index**: `default_planner.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

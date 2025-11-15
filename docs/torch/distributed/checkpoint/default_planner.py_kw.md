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

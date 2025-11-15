# Documentation: `docs/torch/distributed/tensor/_redistribute.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_redistribute.py_kw.md`
- **Size**: 5,357 bytes (5.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/tensor/_redistribute.py`

## File Information

- **Original File**: [torch/distributed/tensor/_redistribute.py](../../../../torch/distributed/tensor/_redistribute.py)
- **Documentation**: [`_redistribute.py_docs.md`](./_redistribute.py_docs.md)
- **Folder**: `torch/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DTensorRedistributePlanner`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`DistState`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`Redistribute`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_TransformInfo`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`is`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)

### Functions

- **`_ShardOrder_to_dict`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`__eq__`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`__hash__`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`__init__`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`__post_init__`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`__repr__`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`__str__`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_compute_hash`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_dict_to_ShardOrder`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_gen_transform_infos`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_gen_transform_infos_non_cached`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_to_tuple`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`backward`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`clear_redistribute_planner_cache`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`find_min_cost_path`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`forward`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`generate_graph_based_transform_infos`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`generate_greedy_transform_infos`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`get_logical_shape`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`get_next_state`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`get_redistribute_planner`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`redistribute_local_tensor`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`setup_collective_cost`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`stringify_transform_infos`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)

### Imports

- **`DeviceMesh`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`Sequence`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`_are_we_tracing`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`cache`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`cast`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`collections`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`collections.abc`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`contextlib`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`dataclasses`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`defaultdict`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`functools`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`get_active_debug_mode`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`heapq`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`itertools`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`logging`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch.distributed._functional_collectives`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch.distributed.tensor._api`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch.distributed.tensor.device_mesh`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`torch.utils._debug_mode`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`typing`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)
- **`weakref`**: [_redistribute.py_docs.md](./_redistribute.py_docs.md)


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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `_redistribute.py_kw.md_docs.md`
- **Keyword Index**: `_redistribute.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

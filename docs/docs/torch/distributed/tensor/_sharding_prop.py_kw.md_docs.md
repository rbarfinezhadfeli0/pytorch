# Documentation: `docs/torch/distributed/tensor/_sharding_prop.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/_sharding_prop.py_kw.md`
- **Size**: 4,770 bytes (4.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/tensor/_sharding_prop.py`

## File Information

- **Original File**: [torch/distributed/tensor/_sharding_prop.py](../../../../torch/distributed/tensor/_sharding_prop.py)
- **Documentation**: [`_sharding_prop.py_docs.md`](./_sharding_prop.py_docs.md)
- **Folder**: `torch/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`LocalLRUCache`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`ShardingPropagator`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)

### Functions

- **`__call__`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`__init__`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_adjust_shape_and_stride_args`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_create_output_spec_with_new_tensor_meta`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_length`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_propagate_tensor_meta`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_propagate_tensor_meta_non_cached`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_select_strategy`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_wrap_with_op_strategy`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`cache_clear`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`cache_info`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`propagate`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`propagate_op_sharding_non_cached`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`propagate_tensor_meta`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`register_op_strategy`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`register_sharding_prop_rule`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`spec_to_strategy`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)

### Imports

- **`Callable`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`DTensorSpec`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`FakeTensorMode`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`OpOverload`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`_are_we_tracing`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`cast`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`chain`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`collections.abc`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`contextlib`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`detect_fake_mode`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`disable_proxy_modes_tracing`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`functools`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`itertools`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`lru_cache`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`threading`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch._guards`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch._ops`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch._subclasses`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch.distributed._functional_collectives`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch.distributed.tensor._op_schema`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch.distributed.tensor._utils`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)
- **`typing`**: [_sharding_prop.py_docs.md](./_sharding_prop.py_docs.md)


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

- This file appears to involve **GPU/parallel computing** capabilities.
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
- [`_random.py_kw.md_docs.md`](./_random.py_kw.md_docs.md)
- [`_collective_utils.py_kw.md_docs.md`](./_collective_utils.py_kw.md_docs.md)
- [`_utils.py_kw.md_docs.md`](./_utils.py_kw.md_docs.md)
- [`_collective_utils.py_docs.md_docs.md`](./_collective_utils.py_docs.md_docs.md)
- [`_utils.py_docs.md_docs.md`](./_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_sharding_prop.py_kw.md_docs.md`
- **Keyword Index**: `_sharding_prop.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

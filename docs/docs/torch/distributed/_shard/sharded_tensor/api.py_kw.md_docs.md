# Documentation: `docs/torch/distributed/_shard/sharded_tensor/api.py_kw.md`

## File Metadata

- **Path**: `docs/torch/distributed/_shard/sharded_tensor/api.py_kw.md`
- **Size**: 5,829 bytes (5.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/distributed/_shard/sharded_tensor/api.py`

## File Information

- **Original File**: [torch/distributed/_shard/sharded_tensor/api.py](../../../../../torch/distributed/_shard/sharded_tensor/api.py)
- **Documentation**: [`api.py_docs.md`](./api.py_docs.md)
- **Folder**: `torch/distributed/_shard/sharded_tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ShardedTensor`**: [api.py_docs.md](./api.py_docs.md)
- **`ShardedTensorBase`**: [api.py_docs.md](./api.py_docs.md)
- **`class`**: [api.py_docs.md](./api.py_docs.md)
- **`from`**: [api.py_docs.md](./api.py_docs.md)
- **`to`**: [api.py_docs.md](./api.py_docs.md)

### Functions

- **`__del__`**: [api.py_docs.md](./api.py_docs.md)
- **`__getstate__`**: [api.py_docs.md](./api.py_docs.md)
- **`__hash__`**: [api.py_docs.md](./api.py_docs.md)
- **`__init__`**: [api.py_docs.md](./api.py_docs.md)
- **`__new__`**: [api.py_docs.md](./api.py_docs.md)
- **`__repr__`**: [api.py_docs.md](./api.py_docs.md)
- **`__setstate__`**: [api.py_docs.md](./api.py_docs.md)
- **`__torch_dispatch__`**: [api.py_docs.md](./api.py_docs.md)
- **`__torch_function__`**: [api.py_docs.md](./api.py_docs.md)
- **`_create_tensor_from_params`**: [api.py_docs.md](./api.py_docs.md)
- **`_get_preferred_device`**: [api.py_docs.md](./api.py_docs.md)
- **`_init_from_local_shards`**: [api.py_docs.md](./api.py_docs.md)
- **`_init_from_local_shards_and_global_metadata`**: [api.py_docs.md](./api.py_docs.md)
- **`_init_from_local_tensor`**: [api.py_docs.md](./api.py_docs.md)
- **`_init_rpc`**: [api.py_docs.md](./api.py_docs.md)
- **`_normalize_pg`**: [api.py_docs.md](./api.py_docs.md)
- **`_post_init`**: [api.py_docs.md](./api.py_docs.md)
- **`_prepare_init`**: [api.py_docs.md](./api.py_docs.md)
- **`_raise_if_mismatch`**: [api.py_docs.md](./api.py_docs.md)
- **`_register_remote_shards`**: [api.py_docs.md](./api.py_docs.md)
- **`cpu`**: [api.py_docs.md](./api.py_docs.md)
- **`cuda`**: [api.py_docs.md](./api.py_docs.md)
- **`dispatch`**: [api.py_docs.md](./api.py_docs.md)
- **`find_sharded_tensor`**: [api.py_docs.md](./api.py_docs.md)
- **`gather`**: [api.py_docs.md](./api.py_docs.md)
- **`is_pinned`**: [api.py_docs.md](./api.py_docs.md)
- **`local_shards`**: [api.py_docs.md](./api.py_docs.md)
- **`local_tensor`**: [api.py_docs.md](./api.py_docs.md)
- **`metadata`**: [api.py_docs.md](./api.py_docs.md)
- **`remote_shards`**: [api.py_docs.md](./api.py_docs.md)
- **`reshard`**: [api.py_docs.md](./api.py_docs.md)
- **`shard_size`**: [api.py_docs.md](./api.py_docs.md)
- **`sharding_spec`**: [api.py_docs.md](./api.py_docs.md)
- **`to`**: [api.py_docs.md](./api.py_docs.md)

### Imports

- **`.metadata`**: [api.py_docs.md](./api.py_docs.md)
- **`.reshard`**: [api.py_docs.md](./api.py_docs.md)
- **`.shard`**: [api.py_docs.md](./api.py_docs.md)
- **`.utils`**: [api.py_docs.md](./api.py_docs.md)
- **`Callable`**: [api.py_docs.md](./api.py_docs.md)
- **`DEPRECATE_MSG`**: [api.py_docs.md](./api.py_docs.md)
- **`Shard`**: [api.py_docs.md](./api.py_docs.md)
- **`ShardMetadata`**: [api.py_docs.md](./api.py_docs.md)
- **`ShardedTensorMetadata`**: [api.py_docs.md](./api.py_docs.md)
- **`__future__`**: [api.py_docs.md](./api.py_docs.md)
- **`_get_current_process_group`**: [api.py_docs.md](./api.py_docs.md)
- **`_get_device_module`**: [api.py_docs.md](./api.py_docs.md)
- **`_pytree`**: [api.py_docs.md](./api.py_docs.md)
- **`_remote_device`**: [api.py_docs.md](./api.py_docs.md)
- **`annotations`**: [api.py_docs.md](./api.py_docs.md)
- **`cast`**: [api.py_docs.md](./api.py_docs.md)
- **`collections.abc`**: [api.py_docs.md](./api.py_docs.md)
- **`copy`**: [api.py_docs.md](./api.py_docs.md)
- **`dataclass`**: [api.py_docs.md](./api.py_docs.md)
- **`dataclasses`**: [api.py_docs.md](./api.py_docs.md)
- **`deprecated`**: [api.py_docs.md](./api.py_docs.md)
- **`distributed_c10d`**: [api.py_docs.md](./api.py_docs.md)
- **`functools`**: [api.py_docs.md](./api.py_docs.md)
- **`operator`**: [api.py_docs.md](./api.py_docs.md)
- **`reduce`**: [api.py_docs.md](./api.py_docs.md)
- **`reshard_local_shard`**: [api.py_docs.md](./api.py_docs.md)
- **`threading`**: [api.py_docs.md](./api.py_docs.md)
- **`torch`**: [api.py_docs.md](./api.py_docs.md)
- **`torch._utils`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed._shard._utils`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed._shard.api`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed._shard.metadata`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed._shard.sharding_spec`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed._shard.sharding_spec._internals`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed._shard.sharding_spec.api`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.distributed.remote_device`**: [api.py_docs.md](./api.py_docs.md)
- **`torch.utils`**: [api.py_docs.md](./api.py_docs.md)
- **`typing`**: [api.py_docs.md](./api.py_docs.md)
- **`typing_extensions`**: [api.py_docs.md](./api.py_docs.md)
- **`warnings`**: [api.py_docs.md](./api.py_docs.md)
- **`weakref`**: [api.py_docs.md](./api.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/distributed/_shard/sharded_tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/_shard/sharded_tensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`docs/torch/distributed/_shard/sharded_tensor`):

- [`metadata.py_docs.md_docs.md`](./metadata.py_docs.md_docs.md)
- [`logger.py_kw.md_docs.md`](./logger.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`reshard.py_docs.md_docs.md`](./reshard.py_docs.md_docs.md)
- [`metadata.py_kw.md_docs.md`](./metadata.py_kw.md_docs.md)
- [`shard.py_docs.md_docs.md`](./shard.py_docs.md_docs.md)
- [`shard.py_kw.md_docs.md`](./shard.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`logger.py_docs.md_docs.md`](./logger.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `api.py_kw.md_docs.md`
- **Keyword Index**: `api.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

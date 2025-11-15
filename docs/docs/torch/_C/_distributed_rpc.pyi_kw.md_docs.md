# Documentation: `docs/torch/_C/_distributed_rpc.pyi_kw.md`

## File Metadata

- **Path**: `docs/torch/_C/_distributed_rpc.pyi_kw.md`
- **Size**: 7,652 bytes (7.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_C/_distributed_rpc.pyi`

## File Information

- **Original File**: [torch/_C/_distributed_rpc.pyi](../../../torch/_C/_distributed_rpc.pyi)
- **Documentation**: [`_distributed_rpc.pyi_docs.md`](./_distributed_rpc.pyi_docs.md)
- **Folder**: `torch/_C`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`PyRRef`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`RemoteProfilerManager`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`RpcAgent`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`RpcBackendOptions`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`TensorPipeAgent`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`WorkerInfo`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_TensorPipeRpcBackendOptionsBase`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)

### Functions

- **`__eq__`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`__init__`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_cleanup_python_rpc_handler`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_delete_all_user_and_unforked_owner_rrefs`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_deserialize`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_destroy_rref_context`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_disable_jit_rref_pickle`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_disable_server_process_global_profiler`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_enable_jit_rref_pickle`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_enable_server_process_global_profiler`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_get_backend_options`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_get_current_rpc_agent`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_get_device_map`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_get_future`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_get_profiling_future`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_get_type`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_invoke_remote_builtin`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_invoke_remote_python_udf`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_invoke_remote_torchscript`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_invoke_rpc_builtin`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_invoke_rpc_python_udf`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_invoke_rpc_torchscript`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_is_current_rpc_agent_set`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_reset_current_rpc_agent`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_rref_context_get_debug_info`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_serialize`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_set_and_start_rpc_agent`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_set_device_map`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_set_profiler_node_id`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_set_profiling_future`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_set_rpc_timeout`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`_update_group_membership`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`confirmed_by_owner`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`enable_gil_profiling`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`get_debug_info`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`get_metrics`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`get_rpc_timeout`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`get_worker_info`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`get_worker_infos`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`id`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`is_owner`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`is_static_group`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`join`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`local_value`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`name`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`owner`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`owner_name`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`remote`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`rpc_async`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`rpc_sync`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`set_current_profiling_key`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`shutdown`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`store`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`sync`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`to_here`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)

### Imports

- **`Any`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`Future`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`ProfilerConfig`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`ProfilerEvent`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`Store`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`datetime`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`timedelta`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`torch`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`torch._C`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`torch._C._autograd`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`torch._C._distributed_c10d`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`torch._C._profiler`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)
- **`typing`**: [_distributed_rpc.pyi_docs.md](./_distributed_rpc.pyi_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_C`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_C`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/_C`):

- [`_nvtx.pyi_docs.md_docs.md`](./_nvtx.pyi_docs.md_docs.md)
- [`_aoti.pyi_docs.md_docs.md`](./_aoti.pyi_docs.md_docs.md)
- [`_cpu.pyi_docs.md_docs.md`](./_cpu.pyi_docs.md_docs.md)
- [`_lazy_ts_backend.pyi_docs.md_docs.md`](./_lazy_ts_backend.pyi_docs.md_docs.md)
- [`_distributed_c10d.pyi_kw.md_docs.md`](./_distributed_c10d.pyi_kw.md_docs.md)
- [`_profiler.pyi_docs.md_docs.md`](./_profiler.pyi_docs.md_docs.md)
- [`_functionalization.pyi_kw.md_docs.md`](./_functionalization.pyi_kw.md_docs.md)
- [`_distributed.pyi_docs.md_docs.md`](./_distributed.pyi_docs.md_docs.md)
- [`_itt.pyi_docs.md_docs.md`](./_itt.pyi_docs.md_docs.md)
- [`build.bzl_kw.md_docs.md`](./build.bzl_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_distributed_rpc.pyi_kw.md_docs.md`
- **Keyword Index**: `_distributed_rpc.pyi_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Keyword Index: `torch/cuda/memory.py`

## File Information

- **Original File**: [torch/cuda/memory.py](../../../torch/cuda/memory.py)
- **Documentation**: [`memory.py_docs.md`](./memory.py_docs.md)
- **Folder**: `torch/cuda`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Block`**: [memory.py_docs.md](./memory.py_docs.md)
- **`CUDAPluggableAllocator`**: [memory.py_docs.md](./memory.py_docs.md)
- **`Frame`**: [memory.py_docs.md](./memory.py_docs.md)
- **`MemPool`**: [memory.py_docs.md](./memory.py_docs.md)
- **`Segment`**: [memory.py_docs.md](./memory.py_docs.md)
- **`Snapshot`**: [memory.py_docs.md](./memory.py_docs.md)
- **`TraceEntry`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_Block`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_CUDAAllocator`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_Frame`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_Segment`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_Snapshot`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_TraceEntry`**: [memory.py_docs.md](./memory.py_docs.md)

### Functions

- **`__init__`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_augment_frames`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_augment_memory_snapshot_stack_traces`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_dump_snapshot`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_format_count`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_format_size`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_free_mutex`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_get_current_allocator`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_get_memory_metadata`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_host_allocator`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_record_memory_history`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_record_memory_history_impl`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_record_memory_history_legacy`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_recurse_add_to_result`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_save_memory_usage`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_save_segment_usage`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_set_allocator_settings`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_set_memory_metadata`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_snapshot`**: [memory.py_docs.md](./memory.py_docs.md)
- **`allocator`**: [memory.py_docs.md](./memory.py_docs.md)
- **`caching_allocator_alloc`**: [memory.py_docs.md](./memory.py_docs.md)
- **`caching_allocator_delete`**: [memory.py_docs.md](./memory.py_docs.md)
- **`caching_allocator_enable`**: [memory.py_docs.md](./memory.py_docs.md)
- **`change_current_allocator`**: [memory.py_docs.md](./memory.py_docs.md)
- **`empty_cache`**: [memory.py_docs.md](./memory.py_docs.md)
- **`get_allocator_backend`**: [memory.py_docs.md](./memory.py_docs.md)
- **`get_per_process_memory_fraction`**: [memory.py_docs.md](./memory.py_docs.md)
- **`host_memory_stats`**: [memory.py_docs.md](./memory.py_docs.md)
- **`host_memory_stats_as_nested_dict`**: [memory.py_docs.md](./memory.py_docs.md)
- **`id`**: [memory.py_docs.md](./memory.py_docs.md)
- **`list_gpu_processes`**: [memory.py_docs.md](./memory.py_docs.md)
- **`max_memory_allocated`**: [memory.py_docs.md](./memory.py_docs.md)
- **`max_memory_cached`**: [memory.py_docs.md](./memory.py_docs.md)
- **`max_memory_reserved`**: [memory.py_docs.md](./memory.py_docs.md)
- **`mem_get_info`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_allocated`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_cached`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_reserved`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_snapshot`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_stats`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_stats_as_nested_dict`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory_summary`**: [memory.py_docs.md](./memory.py_docs.md)
- **`reset_accumulated_host_memory_stats`**: [memory.py_docs.md](./memory.py_docs.md)
- **`reset_accumulated_memory_stats`**: [memory.py_docs.md](./memory.py_docs.md)
- **`reset_max_memory_allocated`**: [memory.py_docs.md](./memory.py_docs.md)
- **`reset_max_memory_cached`**: [memory.py_docs.md](./memory.py_docs.md)
- **`reset_peak_host_memory_stats`**: [memory.py_docs.md](./memory.py_docs.md)
- **`reset_peak_memory_stats`**: [memory.py_docs.md](./memory.py_docs.md)
- **`set_per_process_memory_fraction`**: [memory.py_docs.md](./memory.py_docs.md)
- **`snapshot`**: [memory.py_docs.md](./memory.py_docs.md)
- **`use_count`**: [memory.py_docs.md](./memory.py_docs.md)
- **`use_mem_pool`**: [memory.py_docs.md](./memory.py_docs.md)

### Imports

- **`.`**: [memory.py_docs.md](./memory.py_docs.md)
- **`._memory_viz`**: [memory.py_docs.md](./memory.py_docs.md)
- **`Any`**: [memory.py_docs.md](./memory.py_docs.md)
- **`Device`**: [memory.py_docs.md](./memory.py_docs.md)
- **`FX_GRAPH_MODULE_FILE_PREFIX`**: [memory.py_docs.md](./memory.py_docs.md)
- **`NVMLError_DriverNotLoaded`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_C`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_FX_METADATA_REGISTRY`**: [memory.py_docs.md](./memory.py_docs.md)
- **`_dummy_type`**: [memory.py_docs.md](./memory.py_docs.md)
- **`amdsmi`**: [memory.py_docs.md](./memory.py_docs.md)
- **`collections`**: [memory.py_docs.md](./memory.py_docs.md)
- **`contextlib`**: [memory.py_docs.md](./memory.py_docs.md)
- **`ctypes`**: [memory.py_docs.md](./memory.py_docs.md)
- **`deprecated`**: [memory.py_docs.md](./memory.py_docs.md)
- **`inspect`**: [memory.py_docs.md](./memory.py_docs.md)
- **`memory`**: [memory.py_docs.md](./memory.py_docs.md)
- **`os`**: [memory.py_docs.md](./memory.py_docs.md)
- **`pickle`**: [memory.py_docs.md](./memory.py_docs.md)
- **`pynvml`**: [memory.py_docs.md](./memory.py_docs.md)
- **`re`**: [memory.py_docs.md](./memory.py_docs.md)
- **`signature`**: [memory.py_docs.md](./memory.py_docs.md)
- **`sys`**: [memory.py_docs.md](./memory.py_docs.md)
- **`torch`**: [memory.py_docs.md](./memory.py_docs.md)
- **`torch._C`**: [memory.py_docs.md](./memory.py_docs.md)
- **`torch._utils`**: [memory.py_docs.md](./memory.py_docs.md)
- **`torch.fx.graph_module`**: [memory.py_docs.md](./memory.py_docs.md)
- **`torch.fx.traceback`**: [memory.py_docs.md](./memory.py_docs.md)
- **`torch.types`**: [memory.py_docs.md](./memory.py_docs.md)
- **`typing`**: [memory.py_docs.md](./memory.py_docs.md)
- **`typing_extensions`**: [memory.py_docs.md](./memory.py_docs.md)
- **`warnings`**: [memory.py_docs.md](./memory.py_docs.md)


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

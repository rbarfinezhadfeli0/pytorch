# Keyword Index: `torch/profiler/profiler.py`

## File Information

- **Original File**: [torch/profiler/profiler.py](../../../torch/profiler/profiler.py)
- **Documentation**: [`profiler.py_docs.md`](./profiler.py_docs.md)
- **Folder**: `torch/profiler`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ExecutionTraceObserver`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`ProfilerAction`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_ITraceObserver`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_KinetoProfile`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_NumpyEncoder`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`profile`**: [profiler.py_docs.md](./profiler.py_docs.md)

### Functions

- **`__del__`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`__enter__`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`__exit__`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`__init__`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_default_schedule_fn`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_get_distributed_info`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_memory_profile`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_record_pg_config`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_save_gz_file`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_save_triton_kernels`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_stats`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_trace_ready`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_transit_action`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`add_metadata`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`add_metadata_json`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`build_execution_trace_obs_from_env`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`cleanup`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`default`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`events`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`export_chrome_trace`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`export_memory_timeline`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`export_stacks`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`get_output_file_path`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`get_resources_dir`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`get_resources_dir_for_et_path`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`get_temp_uncompressed_file`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`get_trace_id`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`handler_fn`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`is_registered`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`is_running`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`key_averages`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`prepare_trace`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`preset_metadata_json`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`register_callback`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`schedule`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`schedule_fn`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`set_custom_trace_id_callback`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`set_extra_resource_collection`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`start`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`start_trace`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`step`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`stop`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`stop_trace`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`supported_activities`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`tensorboard_trace_handler`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`toggle_collection_dynamic`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`trace_handler`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`unregister_callback`**: [profiler.py_docs.md](./profiler.py_docs.md)

### Imports

- **`ABC`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`Any`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`Callable`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`Enum`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`MemoryProfile`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`PyCodeCache`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`Self`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`TorchVersion`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`_get_privateuse1_backend_name`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`abc`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`collections.abc`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`enum`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`functools`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`gzip`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`is_fbcode`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`json`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`kineto_available`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`numpy`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`os`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`partial`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`profiler_allow_cudagraph_cupti_lazy_reinit_cuda12`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`shutil`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`socket`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`tempfile`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`time`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch._C`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch._C._profiler`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch._environment`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch._inductor.codecache`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch._inductor.config`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch._utils_internal`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch.autograd`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch.autograd.profiler`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch.distributed`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch.profiler._memory_profiler`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`torch.torch_version`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`typing`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`typing_extensions`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`warn`**: [profiler.py_docs.md](./profiler.py_docs.md)
- **`warnings`**: [profiler.py_docs.md](./profiler.py_docs.md)


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

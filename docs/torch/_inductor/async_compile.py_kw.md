# Keyword Index: `torch/_inductor/async_compile.py`

## File Information

- **Original File**: [torch/_inductor/async_compile.py](../../../torch/_inductor/async_compile.py)
- **Documentation**: [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AsyncCompile`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`CompiledTritonKernels`**: [async_compile.py_docs.md](./async_compile.py_docs.md)

### Functions

- **`__init__`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_add_triton_kernel_info`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_compile_end`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_compile_start`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_get_ready`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_wait_futures`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`after_fork`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`cache_clear`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`caching_device_properties`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`cpp`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`cpp_pybinding`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`cuda`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`cutedsl`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`get`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`get_compile_threads`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`get_result`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`halide`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`key`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`maybe_warm_pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`multi_kernel`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`pallas`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`pre_fork_setup`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`process_pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`reload_kernel_in_parent`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`remove_future`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`rocm`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`save`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`shutdown_compile_workers`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`size_hint_multi_kernel`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`submit`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`task`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`triton`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`use_process_pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`wait`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`wait_pool_ready`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`wakeup`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`warm_pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)

### Imports

- **`Any`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`BrokenProcessPool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`CachingAutotuner`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`Callable`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`Future`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`HAS_TRITON`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`HalideMeta`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`MAIN_SUFFIX`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`MultiKernelCall`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`OrderedSet`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`SizeHintMultiKernelCall`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`V`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_Faketqdm`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`__future__`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`_async_compile_initializer`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`annotations`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`atexit`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`clear_on_fresh_cache`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`collections.abc`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`concurrent.futures`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`concurrent.futures.process`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`config`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`functools`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`get_registered_device_interfaces`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`has_triton_package`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`json`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`log_triton_builds`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`logging`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`multiprocessing`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`os`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`partial`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`re`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`sys`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`time`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._dynamo.device_interface`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._dynamo.utils`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.codecache`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.codegen.cutedsl.cutedsl_kernel`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.codegen.multi_kernel`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.codegen.pallas`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.compile_worker.subproc_pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.compile_worker.tracked_process_pool`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.compile_worker.utils`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.runtime.compile_tasks`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.runtime.hints`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.runtime.triton_compat`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.runtime.triton_heuristics`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.utils`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._inductor.virtualized`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch._utils_internal`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch.hub`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch.utils._ordered_set`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`torch.utils._triton`**: [async_compile.py_docs.md](./async_compile.py_docs.md)
- **`typing`**: [async_compile.py_docs.md](./async_compile.py_docs.md)


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

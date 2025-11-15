# Keyword Index: `torch/distributed/_symmetric_memory/_nvshmem_triton.py`

## File Information

- **Original File**: [torch/distributed/_symmetric_memory/_nvshmem_triton.py](../../../../torch/distributed/_symmetric_memory/_nvshmem_triton.py)
- **Documentation**: [`_nvshmem_triton.py_docs.md`](./_nvshmem_triton.py_docs.md)
- **Folder**: `torch/distributed/_symmetric_memory`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GridCallableWithExtern`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`NvshmemKernelRegistry`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`NvshmemLibFinder`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`that`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`to`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)

### Functions

- **`__init__`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`_log_triton_kernel`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`_nvshmem_init_hook`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`alltoall`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`alltoallmem_block_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`barrier_all`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`broadcast`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`broadcastmem_block_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`deregister`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`enable_triton`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`fence`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`find_device_library`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`foo`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`get`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`get_nbi`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`getmem_block_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`getmem_nbi_block_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`has`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`my_pe`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`n_pes`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`on_exit`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`put`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`putmem_block_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`putmem_signal_block`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`putmem_signal_block_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`quiet`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`reduce`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`reduce_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`register`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`requires_nvshmem`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`run`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`signal_op`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`signal_wait_until`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`signal_wait_until_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`sync_all`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`wait_until`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`wait_until_extern_wrapper`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)

### Imports

- **`Any`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`JITFunction`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`_nvshmemx_cumodule_init`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`atexit`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`core`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`has_triton`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`logging`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`os`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`subprocess`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`sysconfig`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`tempfile`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`torch`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`torch._C._distributed_c10d`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`torch.distributed`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`torch.utils._triton`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`triton`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`triton.language`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`triton.runtime.jit`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)
- **`typing`**: [_nvshmem_triton.py_docs.md](./_nvshmem_triton.py_docs.md)


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

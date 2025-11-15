# Documentation: `docs/source/cuda_environment_variables.rst`

## File Metadata

- **Path**: `docs/source/cuda_environment_variables.rst`
- **Size**: 3,376 bytes (3.30 KB)
- **Type**: Source File (.rst)
- **Extension**: `.rst`

## File Purpose

This file is part of the **documentation**.

## Original Source

```
.. _cuda_environment_variables:

CUDA Environment Variables
==========================
For more information on CUDA runtime environment variables, see `CUDA Environment Variables <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_.

**PyTorch Environment Variables**

.. list-table::
  :header-rows: 1

  * - Variable
    - Description
  * - ``PYTORCH_NO_CUDA_MEMORY_CACHING``
    - If set to ``1``, disables caching of memory allocations in CUDA. This can be useful for debugging.
  * - ``PYTORCH_CUDA_ALLOC_CONF``
    - For a more in depth explanation of this environment variable, see :ref:`cuda-memory-management`.
  * - ``PYTORCH_NVML_BASED_CUDA_CHECK``
    - If set to ``1``, before importing PyTorch modules that check if CUDA is available, PyTorch will use NVML to check if the CUDA driver is functional instead of using the CUDA runtime. This can be helpful if forked processes fail with a CUDA initialization error.
  * - ``TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT``
    - The cache limit for the cuDNN v8 API. This is used to limit the memory used by the cuDNN v8 API. The default value is 10000, which roughly corresponds to 2GiB assuming 200KiB per ExecutionPlan. Set to ``0`` for no limit or a negative value for no caching.
  * - ``TORCH_CUDNN_V8_API_DISABLED``
    - If set to ``1``, disables the cuDNN v8 API. And will fall back to the cuDNN v7 API.
  * - ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``
    - If set to ``1``, forces TF32 enablement, overrides ``set_float32_matmul_precision`` setting.
  * - ``TORCH_NCCL_USE_COMM_NONBLOCKING``
    - If set to ``1``, enables non-blocking error handling in NCCL.
  * - ``TORCH_NCCL_AVOID_RECORD_STREAMS``
    - If set to ``0``, enables fallback to record streams-based synchronization behavior in NCCL.
  * - ``TORCH_CUDNN_V8_API_DEBUG``
    - If set to ``1``, sanity check whether cuDNN V8 is being used.

**CUDA Runtime and Libraries Environment Variables**

.. list-table::
  :header-rows: 1

  * - Variable
    - Description
  * - ``CUDA_VISIBLE_DEVICES``
    - Comma-separated list of GPU device IDs that should be made available to CUDA runtime. If set to ``-1``, no GPUs are made available.
  * - ``CUDA_LAUNCH_BLOCKING``
    - If set to ``1``, makes CUDA calls synchronous. This can be useful for debugging.
  * - ``CUBLAS_WORKSPACE_CONFIG``
    - This environment variable is used to set the workspace configuration for cuBLAS per allocation. The format is ``:[SIZE]:[COUNT]``.
      As an example, the default workspace size per allocation is ``CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8`` which specifies a total size of ``2 * 4096 + 8 * 16 KiB``.
      To force cuBLAS to avoid using workspaces, set ``CUBLAS_WORKSPACE_CONFIG=:0:0``.
  * - ``CUDNN_CONV_WSCAP_DBG``
    - Similar to ``CUBLAS_WORKSPACE_CONFIG``, this environment variable is used to set the workspace configuration for cuDNN per allocation.
  * - ``CUBLASLT_WORKSPACE_SIZE``
    - Similar to ``CUBLAS_WORKSPACE_CONFIG``, this environment variable is used to set the workspace size for cuBLASLT.
  * - ``CUDNN_ERRATA_JSON_FILE``
    - Can be set to a file path for an errata filter that can be passed to cuDNN to avoid specific engine configs, used primarily for debugging or to hardcode autotuning.
  * - ``NVIDIA_TF32_OVERRIDE``
    - If set to ``0``, disables TF32 globally across all kernels, overriding all PyTorch settings.

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `cuda_environment_variables.rst_docs.md`
- **Keyword Index**: `cuda_environment_variables.rst_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

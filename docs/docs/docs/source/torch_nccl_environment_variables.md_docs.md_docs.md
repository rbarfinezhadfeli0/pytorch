# Documentation: `docs/docs/source/torch_nccl_environment_variables.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/torch_nccl_environment_variables.md_docs.md`
- **Size**: 5,671 bytes (5.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/torch_nccl_environment_variables.md`

## File Metadata

- **Path**: `docs/source/torch_nccl_environment_variables.md`
- **Size**: 3,157 bytes (3.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
(_torch_nccl_environment_variables)=
# PYTORCH ProcessGroupNCCL Environment Variables

For more information on the environment variables, see [ProcessGroupNCCL Environment Variables](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp).

```{list-table}
:header-rows: 1

* - **Variable**
  - **Description**
* - ``TORCH_NCCL_ASYNC_ERROR_HANDLING``
  - Control how we perform Async Error Handling with NCCL when an exception is observed in watchdog. If set to 0, no handling of asynchronous NCCL errors. If set to 1, aborting NCCL communicator and tearing down process upon error. If set to 2, only abort NCCL communicator and if set to 3, tearing down process without aborting NCCL communicator. By default, it is set to 3.
* - ``TORCH_NCCL_HIGH_PRIORITY``
  - Control whether to use high priority stream for the NCCL communicator.
* - ``TORCH_NCCL_BLOCKING_WAIT``
  - Control whether or not wait() is blocking or non-blocking.
* - ``TORCH_NCCL_DUMP_ON_TIMEOUT``
  - Control whether dumping debug info on watchdog timeout or exception is detected. This variable must be set together with TORCH_NCCL_TRACE_BUFFER_SIZE larger than 0.
* - ``TORCH_NCCL_DESYNC_DEBUG``
  - Control whether Desync Debug is enabled. This is helpful in figuring out the culprit rank of collective desync.
* - ``TORCH_NCCL_ENABLE_TIMING``
  - If set to ``1``, enable recording start-events for all ProcessGroupNCCL collectives, and compute accurate collective timing per-collective.
* - ``TORCH_NCCL_ENABLE_MONITORING``
  - If set to ``1``,enable monitoring thread which aborts the process when the ProcessGroupNCCL Watchdog thread gets stuck and no heartbeat is detected after TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling CUDA/NCCL APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged time than necessary tying up cluster resources.
* - ``TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC``
  - Control the watchdog heartbeat timeout period after which the monitoring thread will abort the process.
* - ``TORCH_NCCL_TRACE_BUFFER_SIZE``
  - The maximum number of events we store in the flight recorder's ring buffer. One event could be the start or end of a collective, for example. Set to 0 to disable the tracebuffer and debugging info dump.
* - ``TORCH_NCCL_TRACE_CPP_STACK``
  - Whether to collect cpp stack traces for flight recorder. Default value is False.
* - ``TORCH_NCCL_COORD_CHECK_MILSEC``
  - Control the interval inside the monitoring thread to check the coordinated signal from other ranks, e.g. to dump the debugging information. Default value is 1000 ms.
* - ``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``
  - Control how much extra time we will wait for dumping the debugging info before we exit and throws timeout exception.
* - ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``
  - The file into which the debugging info would be dumped.
* - ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE``
  - The pipe file to trigger debugging dump manually, write anything into the pipe would trigger the dump.
* - ``TORCH_NCCL_NAN_CHECK``
  - Control whether to enable NAN check for the input, Error would be thrown if NAN is detected.
```

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
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `torch_nccl_environment_variables.md_docs.md`
- **Keyword Index**: `torch_nccl_environment_variables.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/docs/source`):

- [`distributions.md_docs.md_docs.md`](./distributions.md_docs.md_docs.md)
- [`distributed.optim.md_docs.md_docs.md`](./distributed.optim.md_docs.md_docs.md)
- [`torch.compiler_dynamic_shapes.md_kw.md_docs.md`](./torch.compiler_dynamic_shapes.md_kw.md_docs.md)
- [`tensor_attributes.rst_docs.md_docs.md`](./tensor_attributes.rst_docs.md_docs.md)
- [`tensor_attributes.rst_kw.md_docs.md`](./tensor_attributes.rst_kw.md_docs.md)
- [`torch.compiler_dynamo_overview.md_docs.md_docs.md`](./torch.compiler_dynamo_overview.md_docs.md_docs.md)
- [`mtia.memory.md_kw.md_docs.md`](./mtia.memory.md_kw.md_docs.md)
- [`nn.attention.varlen.md_kw.md_docs.md`](./nn.attention.varlen.md_kw.md_docs.md)
- [`cpu.rst_kw.md_docs.md`](./cpu.rst_kw.md_docs.md)
- [`torch.compiler_faq.md_docs.md_docs.md`](./torch.compiler_faq.md_docs.md_docs.md)


## Cross-References

- **File Documentation**: `torch_nccl_environment_variables.md_docs.md_docs.md`
- **Keyword Index**: `torch_nccl_environment_variables.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

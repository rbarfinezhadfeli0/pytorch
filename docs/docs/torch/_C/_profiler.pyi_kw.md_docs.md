# Documentation: `docs/torch/_C/_profiler.pyi_kw.md`

## File Metadata

- **Path**: `docs/torch/_C/_profiler.pyi_kw.md`
- **Size**: 5,595 bytes (5.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_C/_profiler.pyi`

## File Information

- **Original File**: [torch/_C/_profiler.pyi](../../../torch/_C/_profiler.pyi)
- **Documentation**: [`_profiler.pyi_docs.md`](./_profiler.pyi_docs.md)
- **Folder**: `torch/_C`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ActiveProfilerType`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`CapturedTraceback`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`ProfilerActivity`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`ProfilerConfig`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`ProfilerState`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`RecordScope`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_EventType`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExperimentalConfig`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_Allocation`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_Backend`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_Kineto`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_OutOfMemory`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_PyCCall`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_PyCall`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ExtraFields_TorchOp`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_NNModuleInfo`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_OptimizerInfo`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_ProfilerEvent`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_PyFrameState`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_RecordFunctionFast`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_TensorMetadata`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)

### Functions

- **`__enter__`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`__exit__`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`__init__`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_add_execution_trace_observer`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_disable_execution_trace_observer`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_enable_execution_trace_observer`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_remove_execution_trace_observer`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_set_cuda_sync_enabled_val`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_set_fwd_bwd_enabled_val`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`_set_record_concrete_inputs_enabled_val`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`allocation_id`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`caller`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`callsite`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`cls_name`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`cls_ptr`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`correlation_id`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`device`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`dtype`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`duration_time_ns`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`end_time_ns`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`file_name`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`gather_traceback`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`id`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`inputs`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`layout`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`module`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`name`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`optimizer`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`parameters`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`parent`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`scope`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`self_ptr`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`sizes`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`strides`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`symbolize_tracebacks`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`tag`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`typed`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)

### Imports

- **`Enum`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`Literal`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`device`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`enum`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`torch._C`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)
- **`typing`**: [_profiler.pyi_docs.md](./_profiler.pyi_docs.md)


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

- **File Documentation**: `_profiler.pyi_kw.md_docs.md`
- **Keyword Index**: `_profiler.pyi_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

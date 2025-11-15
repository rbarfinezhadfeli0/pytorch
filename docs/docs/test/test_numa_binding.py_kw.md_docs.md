# Documentation: `docs/test/test_numa_binding.py_kw.md`

## File Metadata

- **Path**: `docs/test/test_numa_binding.py_kw.md`
- **Size**: 7,659 bytes (7.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/test_numa_binding.py`

## File Information

- **Original File**: [test/test_numa_binding.py](../../test/test_numa_binding.py)
- **Documentation**: [`test_numa_binding.py_docs.md`](./test_numa_binding.py_docs.md)
- **Folder**: `test`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MockDeviceProperties`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`NumaBindingTest`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`from`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)

### Functions

- **`_add_mock_hardware`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_get_corresponding_pci_numa_node_file_path`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_device_count`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_file_contents`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_get_device_properties`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_is_available`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_listdir`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_open`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_sched_getaffinity`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_mock_signpost_event`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_start_processes_for_callable_entrypoint_and_get_sched_setaffinity_cpus`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_start_processes_for_str_entrypoint_and_get_command_args`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`dummy_fn`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`mock_sched_getaffinity_impl`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`mock_sched_setaffinity`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`mock_sched_setaffinity_impl`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`setUp`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`tearDown`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_bind_all_threads_in_current_process_to_logical_cpus`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_binds_to_node_0_if_node_stored_as_minus_one`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_callable_entrypoint_basic`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_core_complex_numa_binding_with_extra_l3`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_core_complex_numa_binding_with_fewer_l3_than_gpu`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_core_complex_prefers_caches_with_more_cpus`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_core_complex_tiebreak_prefers_lower_cache_key`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_default_numa_binding`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_exclusive_numa_binding`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_exclusive_raises_if_too_few_physical_cores`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_explicit_numa_options_overrides_default`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_fallback`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_fallback_if_numactl_not_available`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_get_range_str_from_ints`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_get_set_of_int_from_ranges_str`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_no_numa_binding_if_numa_options_not_provided`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_node_numa_binding`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_nproc_must_equal_cuda_device_count_to_use_default_numa_options`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_raises_if_binding_to_empty_set`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_socket_numa_binding_with_multiple_numa_per_socket`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`test_socket_numa_binding_with_single_numa_per_socket`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)

### Imports

- **`Any`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`DefaultLogsSpecs`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`LaunchConfig`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`__future__`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`_wrap`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`annotations`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`dataclass`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`dataclasses`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`json`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`mock_open`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`run_tests`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`signpost_event`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`skipUnless`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`sys`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`to`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch._utils_internal`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch.distributed.elastic.multiprocessing`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch.distributed.elastic.multiprocessing.api`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch.distributed.launcher.api`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch.multiprocessing`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch.numa.binding`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`typing`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`unittest`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)
- **`unittest.mock`**: [test_numa_binding.py_docs.md](./test_numa_binding.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/test_numa_binding.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_numa_binding.py_kw.md_docs.md`
- **Keyword Index**: `test_numa_binding.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

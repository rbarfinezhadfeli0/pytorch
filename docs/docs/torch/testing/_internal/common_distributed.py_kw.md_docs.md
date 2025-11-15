# Documentation: `docs/torch/testing/_internal/common_distributed.py_kw.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/common_distributed.py_kw.md`
- **Size**: 15,167 bytes (14.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/testing/_internal/common_distributed.py`

## File Information

- **Original File**: [torch/testing/_internal/common_distributed.py](../../../../torch/testing/_internal/common_distributed.py)
- **Documentation**: [`common_distributed.py_docs.md`](./common_distributed.py_docs.md)
- **Folder**: `torch/testing/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DistributedTestBase`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`DynamoDistributedMultiProcTestCase`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`DynamoDistributedSingleProcTestCase`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`Event`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`MultiProcContinuousTest`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`MultiProcessTestCase`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`MultiThreadedTestCase`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`SaveForwardInputsModel`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`SaveForwardInputsModule`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`TestSkip`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`class`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`cls`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`for`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`from`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`hits`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`method`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`self`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`should`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`timeout`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`variables`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)

### Functions

- **`__init__`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_check_return_codes`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_current_test_name`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_dynamo_dist_per_rank_init`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_event_listener`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_get_timedout_process_traceback`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_init_pg`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_join_processes`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_join_threads`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_maybe_handle_skip_if_lt_x_gpu`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_run`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_run_test_given_id`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_run_test_method_with_multi_threads`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_should_stop_test_suite`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_spawn_processes`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_spawn_threads`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_start_processes`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_worker_loop`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_worker_run_main_wait`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`assertEqualOnRank`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`assertNotEqualOnRank`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`at_least_x_gpu`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`backend`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`backend_str`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`captured_output`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`cleanup_temp_dir`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`compute_sum`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`create_device`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`create_pg`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`create_tcp_store`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`decorator`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`destroy_pg_upon_exit`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`device_type`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`evaluate_platform_supports_symm_mem`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`forward`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`generate`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`get_required_world_size`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`get_timeout`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`has_efa`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`import_transformers_or_skip`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`init_multigpu_helper`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`initialize_temp_directories`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`is_master`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`join_or_run`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`nccl_skip_if_lt_x_gpu`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`opts`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`perThreadSetUp`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`perThreadTearDown`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`rank_to_device`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`require_n_gpus_for_nccl_backend`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_accelerator_dist_backend`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_ddp_rank`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_gloo`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_mpi`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_multicast_support`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_nccl`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_nccl_shrink`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_nccl_version`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_ucc`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`requires_world_size`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`run_subtests`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`run_test`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`run_test_with_threaded_pg`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`setUp`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`setUpClass`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`simple_sparse_reduce_tests`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_lt_x_gpu`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_no_gpu`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_odd_worldsize`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_rocm_arch_multiprocess`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_rocm_multiprocess`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_rocm_ver_lessthan_multiprocess`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_small_worldsize`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`skip_if_win32`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`sm_is_or_higher_than`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`spawn_threads_and_init_comms`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`tearDown`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`tearDownClass`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`test_something`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`tp_transports`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`verify_ddp_error_logged`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`with_dist_debug_levels`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`with_nccl_blocking_wait`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`worker`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`world_is_valid`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`world_size`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`wrapper`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)

### Imports

- **`Any`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`AutoModelForMaskedLM`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`Callable`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`DeviceType`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`Enum`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`StringIO`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`_SymmetricMemory`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`collections.abc`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`common_utils`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`contextlib`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`contextmanager`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`dataclass`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`dataclasses`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`datetime`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`enum`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`faulthandler`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`functools`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`io`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`itertools`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`logging`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`multiprocessing`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`operator`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`os`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`partial`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`patch`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`queue`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`subprocess`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`sys`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`tempfile`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`threading`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`time`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`timedelta`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch._C._autograd`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch._C._distributed_c10d`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch._dynamo.test_case`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch._logging._internal`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch.cuda.nccl`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch.distributed`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch.nn`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch.testing._internal`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch.testing._internal.common_utils`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`torch.testing._internal.distributed.multi_threaded_pg`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`trace_log`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`traceback`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`transformers`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`types`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`typing`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`unittest`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)
- **`unittest.mock`**: [common_distributed.py_docs.md](./common_distributed.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/common_distributed.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common_distributed.py_kw.md_docs.md`
- **Keyword Index**: `common_distributed.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

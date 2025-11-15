# Documentation: `docs/test/distributed/_composable/fsdp/test_fully_shard_compile.py_kw.md`

## File Metadata

- **Path**: `docs/test/distributed/_composable/fsdp/test_fully_shard_compile.py_kw.md`
- **Size**: 12,417 bytes (12.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/distributed/_composable/fsdp/test_fully_shard_compile.py`

## File Information

- **Original File**: [test/distributed/_composable/fsdp/test_fully_shard_compile.py](../../../../../test/distributed/_composable/fsdp/test_fully_shard_compile.py)
- **Documentation**: [`test_fully_shard_compile.py_docs.md`](./test_fully_shard_compile.py_docs.md)
- **Folder**: `test/distributed/_composable/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Mod`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`TestFullyShardCompile`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`TestFullyShardCompileCompute`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`TestModule`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`TestSubmodule`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`is`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`requires`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)

### Functions

- **`__init__`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_assert_no_aliased_unsharded_params_in_graph_inputs`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_check_count`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_check_fsdp_copy_and_resize_ops_count_in_graph`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_check_fsdp_ops_in_snodes`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_count_op_in_graph`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_create_nested_fully_shard_factory_fns`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_create_simple_mlp_factory_fns`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_create_transformer_factory_fns`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_decide_global_ordering_of_comms_with_checks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_get_resize_count_in_fx_graph`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_is_bwd_fx_graph`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_is_fallback_op_in_snodes`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_is_fwd_graph`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_maybe_add_graph_break_to_sdpa`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_maybe_run_decide_global_ordering_of_comms_with_checks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_reinplace_all_gather_with_optional_checks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_remove_fsdp2_unsharded_param_graph_input_usage_with_optional_checks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_run_with_checks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_sdpa_with_graph_break`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_test_disable_compiling_hooks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_test_nested_fully_shard_backend_inductor_fullgraph_True`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_test_traceable_fsdp`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`_test_transformer_backend_inductor_fullgraph_True`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`add_one_out`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`call`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`f`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`forward`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`fwd_bwd`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`inductor_code_check_fsdp_all_gather`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`inductor_code_check_fsdp_reduce_scatter`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`inductor_code_check_no_compute_op`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`input_creation_fn`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`model_init_fn`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`patched_trace_rules_check`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`run_iters`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`skipTestForOldSm`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_compiled`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_compiled_autograd_ctx`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_disable_compiling_hooks`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_dynamo_recompiles_on_fsdp_layers`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_dynamo_trace_use_training_state`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_eager`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_nested_fully_shard_backend_aot_eager`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_nested_fully_shard_backend_aot_eager_decomp_partition`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_nested_fully_shard_backend_inductor_fullgraph_False`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_nested_fully_shard_backend_inductor_fullgraph_True`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_nested_fully_shard_backend_inductor_fullgraph_True_graph_partition`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_simple_mlp_fullgraph_backend_aot_eager`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_simple_mlp_fullgraph_backend_aot_eager_decomp_partition`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_simple_mlp_fullgraph_backend_inductor`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_trace_fsdp_copy_`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_transformer_backend_aot_eager`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_transformer_backend_aot_eager_decomp_partition`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_transformer_backend_inductor_fullgraph_False`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_transformer_backend_inductor_fullgraph_True`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`test_transformer_backend_inductor_fullgraph_True_graph_partition`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)

### Imports

- **`FSDPParamGroup`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`FSDPTest`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`FileCheck`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`HAS_GPU`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`TrainingState`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`collections`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`comms`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`contextlib`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`copy`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`counters`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`defaultdict`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`functools`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`init_device_mesh`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`is_fallback_op`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`itertools`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`logging`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`mock`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`nn`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`run_tests`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch._dynamo.testing`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch._dynamo.utils`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch._inductor`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch._inductor.utils`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_common`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_param_group`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.nn.functional`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.testing`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)
- **`unittest`**: [test_fully_shard_compile.py_docs.md](./test_fully_shard_compile.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/test/distributed/_composable/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_composable/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/_composable/fsdp/test_fully_shard_compile.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_composable/fsdp`):

- [`test_fully_shard_clip_grad_norm_.py_docs.md_docs.md`](./test_fully_shard_clip_grad_norm_.py_docs.md_docs.md)
- [`test_fully_shard_autograd.py_kw.md_docs.md`](./test_fully_shard_autograd.py_kw.md_docs.md)
- [`test_fully_shard_ignore_params.py_kw.md_docs.md`](./test_fully_shard_ignore_params.py_kw.md_docs.md)
- [`test_fully_shard_comm.py_docs.md_docs.md`](./test_fully_shard_comm.py_docs.md_docs.md)
- [`test_fully_shard_state.py_docs.md_docs.md`](./test_fully_shard_state.py_docs.md_docs.md)
- [`test_fully_shard_ignore_params.py_docs.md_docs.md`](./test_fully_shard_ignore_params.py_docs.md_docs.md)
- [`test_fully_shard_clip_grad_norm_.py_kw.md_docs.md`](./test_fully_shard_clip_grad_norm_.py_kw.md_docs.md)
- [`test_fully_shard_state.py_kw.md_docs.md`](./test_fully_shard_state.py_kw.md_docs.md)
- [`test_fully_shard_mixed_precision.py_docs.md_docs.md`](./test_fully_shard_mixed_precision.py_docs.md_docs.md)
- [`test_fully_shard_state_dict.py_kw.md_docs.md`](./test_fully_shard_state_dict.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fully_shard_compile.py_kw.md_docs.md`
- **Keyword Index**: `test_fully_shard_compile.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

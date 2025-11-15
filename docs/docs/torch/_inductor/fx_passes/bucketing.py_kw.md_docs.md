# Documentation: `docs/torch/_inductor/fx_passes/bucketing.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/bucketing.py_kw.md`
- **Size**: 6,098 bytes (5.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/bucketing.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/bucketing.py](../../../../torch/_inductor/fx_passes/bucketing.py)
- **Documentation**: [`bucketing.py_docs.md`](./bucketing.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_ag_group_key`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_ag_group_key_multidtype`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_ar_group_key`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_insert_fn_trace_before_node`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_pre_bucket_all_gather`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_pre_bucket_all_gather_fake`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_pre_bucket_reduce_scatter`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_pre_bucket_reduce_scatter_fake`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_rs_group_key`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_trace`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`all_gather_merge_fn_to_trace`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`all_gather_merge_fn_to_trace_custom_ops`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`all_gather_merge_fn_to_trace_functional`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`all_reduce_merge_fn_to_trace`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_all_gather`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_all_gather_by_mb`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_all_reduce`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_all_reduce_by_mb`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_cap_mb_by_bucket_idx_default`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_key`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_reduce_scatter`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`bucket_reduce_scatter_by_mb`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`collect_node_descendants`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`create_trace_args`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`greedy_bucket_collective_by_mb`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`is_all_gather_into_tensor`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`is_all_reduce_tensor`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`is_reduce_scatter_tensor`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`is_wait_tensor`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`is_wait_tensor_from_all_gather_into_tensor`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`merge_all_gather`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`merge_all_gather_bucket`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`merge_all_reduce_bucket`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`merge_reduce_scatter`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`merge_reduce_scatter_bucket`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`pick_bucket_dtype`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`process_collective_bucket`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`reduce_scatter_merge_fn_to_trace`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`reduce_scatter_merge_fn_to_trace_custom_ops`**: [bucketing.py_docs.md](./bucketing.py_docs.md)

### Imports

- **`Any`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`Callable`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`OrderedSet`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`_resolve_process_group`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`collections`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`collections.abc`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`defaultdict`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`detect_fake_mode`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`dynamo_timed`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`enable_python_dispatcher`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`logging`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`make_fx`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`operator`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch._dispatch.python`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch._dynamo.utils`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch._inductor.fx_passes.bucketing`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch._logging`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch.distributed`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch.utils._ordered_set`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`torch.utils._pytree`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`trace_structured`**: [bucketing.py_docs.md](./bucketing.py_docs.md)
- **`typing`**: [bucketing.py_docs.md](./bucketing.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor/fx_passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/fx_passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`docs/torch/_inductor/fx_passes`):

- [`dedupe_symint_uses.py_kw.md_docs.md`](./dedupe_symint_uses.py_kw.md_docs.md)
- [`overlap_preserving_bucketer.py_kw.md_docs.md`](./overlap_preserving_bucketer.py_kw.md_docs.md)
- [`pre_grad.py_docs.md_docs.md`](./pre_grad.py_docs.md_docs.md)
- [`b2b_gemm.py_docs.md_docs.md`](./b2b_gemm.py_docs.md_docs.md)
- [`freezing_patterns.py_kw.md_docs.md`](./freezing_patterns.py_kw.md_docs.md)
- [`fsdp.py_docs.md_docs.md`](./fsdp.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`replace_random.py_kw.md_docs.md`](./replace_random.py_kw.md_docs.md)
- [`joint_graph.py_kw.md_docs.md`](./joint_graph.py_kw.md_docs.md)
- [`numeric_utils.py_docs.md_docs.md`](./numeric_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `bucketing.py_kw.md_docs.md`
- **Keyword Index**: `bucketing.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/_inductor/fx_passes/overlap_scheduling.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/overlap_scheduling.py_kw.md`
- **Size**: 9,750 bytes (9.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/overlap_scheduling.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/overlap_scheduling.py](../../../../torch/_inductor/fx_passes/overlap_scheduling.py)
- **Documentation**: [`overlap_scheduling.py_docs.md`](./overlap_scheduling.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`OverlapScheduler`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`class`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`from`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)

### Functions

- **`__init__`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_add_effect_tokens_for_overlap`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_align_compute_nodes_runtime_estimations_across_all_distributed_ranks`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_bucket_collectives`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_calculate_compute_node_domination_index`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_collect_node_ancestors`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_compute_score`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_find_schedulable_path`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_force_oldest_wait`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_get_oldest_wait`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_handle_collective_start`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_handle_compute`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_handle_other`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_handle_wait`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_identify_collectives`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_log_collective_benchmarks`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_reorder_graph`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_schedule`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_schedule_collectives_for_overlap`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_schedule_path_to_collective`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_should_force_wait_for_memory`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_wait_is_hidden`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`benchmark_node`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`benchmark_node_with_cache_key`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`compute_potential_hidden_collectives`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`compute_potential_hidden_nodes`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`compute_potential_hidden_waits`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`could_be_hidden`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`estimate_collective_time`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`estimate_fx_collective_size`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`gb_to_bytes`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`get_benchmark_cache`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`get_cached_node_time`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`get_collective_do_bench`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`get_custom_estimation`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`get_group_name`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`get_hint`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`in_overlappable_collective_unary_chain`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`is_cheap_fn`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`is_compute_node`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`is_exposed`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`off_compute_path`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`reorder_graph`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`run`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`schedule_overlap_bucketing`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`set_cached_node_time`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`should_assume_bucketed`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`to_real`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)

### Imports

- **`..pattern_matcher`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`Any`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`Callable`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`Counter`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`OrderedSet`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_disable_current_modes`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`_get_default_group`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`bucket_key`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`collections`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`collections.abc`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`counters`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`dataclass`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`dataclasses`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`functools`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`heapq`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`is_wait_tensor`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`itertools`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`logging`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`normalize_function`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`rand_strided`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`stable_topological_sort`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`sys`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._dynamo.testing`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._dynamo.utils`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.bucketing`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.control_dependencies`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.memory_estimator`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.node_runtime_estimation`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.overlap_preserving_bucketer`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._logging`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch.distributed`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch.fx`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch.fx.operator_schemas`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch.utils._ordered_set`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`torch.utils._python_dispatch`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`trace_structured`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`typing`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)
- **`unset_fake_temporarily`**: [overlap_scheduling.py_docs.md](./overlap_scheduling.py_docs.md)


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

- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `overlap_scheduling.py_kw.md_docs.md`
- **Keyword Index**: `overlap_scheduling.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

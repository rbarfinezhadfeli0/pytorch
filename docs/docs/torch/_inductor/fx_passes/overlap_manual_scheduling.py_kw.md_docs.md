# Documentation: `docs/torch/_inductor/fx_passes/overlap_manual_scheduling.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/fx_passes/overlap_manual_scheduling.py_kw.md`
- **Size**: 4,688 bytes (4.58 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/fx_passes/overlap_manual_scheduling.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/overlap_manual_scheduling.py](../../../../torch/_inductor/fx_passes/overlap_manual_scheduling.py)
- **Documentation**: [`overlap_manual_scheduling.py_docs.md`](./overlap_manual_scheduling.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ManualOverlapPreservingBucketer`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`ManualOverlapScheduler`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)

### Functions

- **`__init__`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_bucket_group`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_check_recursive_dep`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_collect_node_users`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_identify_collectives`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_manual_bucket_collectives`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_manual_reorder_graph`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_obtain_nodes_in_subgraph`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_schedule`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`manual_bucket_collectives`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`manual_overlap_bucketing`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`run`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)

### Imports

- **`.graph_view`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`Any`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`Counter`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`OrderedSet`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`__future__`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`_stable_topological_sort`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`annotations`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`collections`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`get_subgraph_by_path`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`heapq`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch._dynamo.graph_deduplication`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.bucketing`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.control_dependencies`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.overlap_preserving_bucketer`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch._inductor.fx_passes.overlap_scheduling`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch.fx`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`torch.utils._ordered_set`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)
- **`typing`**: [overlap_manual_scheduling.py_docs.md](./overlap_manual_scheduling.py_docs.md)


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

- **File Documentation**: `overlap_manual_scheduling.py_kw.md_docs.md`
- **Keyword Index**: `overlap_manual_scheduling.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Keyword Index: `torch/_inductor/fx_passes/overlap_preserving_bucketer.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/overlap_preserving_bucketer.py](../../../../torch/_inductor/fx_passes/overlap_preserving_bucketer.py)
- **Documentation**: [`overlap_preserving_bucketer.py_docs.md`](./overlap_preserving_bucketer.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`OverlapPreservingBucketer`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`class`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`from`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)

### Functions

- **`__call__`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`__init__`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_add_hiding_interval_constraints`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_ancestor_dep`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_apply_bucket`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_can_add_to_bucket`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_find_buckets`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_get_intervals`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_has_ancestor_conflicts`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_populate_node_to_event`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_preserve_dependencies_with_tokens`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_preserves_hiding_intervals`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_try_timeline_position`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`bucket_collectives`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`build_timeline`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`build_timelines`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`enclosed_interval`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`get_pos`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`get_wait`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`insert_between`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`is_collective_or_wait`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`is_compute`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`is_start`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`is_wait`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`remove_from_event`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`restore_to_event`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`unlink`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)

### Imports

- **`Any`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`AugmentedGraphHelper`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`OrderedSet`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`_stable_topological_sort`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`collections`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`counters`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`dataclass`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`dataclasses`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`defaultdict`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`logging`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch._dynamo.graph_deduplication`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch._dynamo.utils`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch._inductor.augmented_graph_helper`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch._inductor.fx_passes.bucketing`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch._inductor.fx_passes.control_dependencies`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch._inductor.fx_passes.overlap_scheduling`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch.fx`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`torch.utils._ordered_set`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)
- **`typing`**: [overlap_preserving_bucketer.py_docs.md](./overlap_preserving_bucketer.py_docs.md)


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

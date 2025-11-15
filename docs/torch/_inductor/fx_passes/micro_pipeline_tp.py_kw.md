# Keyword Index: `torch/_inductor/fx_passes/micro_pipeline_tp.py`

## File Information

- **Original File**: [torch/_inductor/fx_passes/micro_pipeline_tp.py](../../../../torch/_inductor/fx_passes/micro_pipeline_tp.py)
- **Documentation**: [`micro_pipeline_tp.py_docs.md`](./micro_pipeline_tp.py_docs.md)
- **Folder**: `torch/_inductor/fx_passes`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`class`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)

### Functions

- **`__post_init__`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_compute_mm_arithmetic_intensity`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_filter_nodes_by_target`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_filter_out_scaled_matmul`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_find_ancestors`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_find_consumer_matmuls`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_find_producer_matmul`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_find_reshape_mm_reshape`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_get_collective_to_overlappable_nodes`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_get_node_to_ancestors`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_get_tensor`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_get_unexposed_collectives`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_insert_fused_all_gather_matmul`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_insert_fused_matmul_reduce_scatter`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_is_backward`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_is_compute_intensive`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_is_last_dim`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_scatter_dim_after_reshape`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`_update_save_for_backward`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`erase`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`find_all_gather_patterns`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`find_reduce_scatter_patterns`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`from_match`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`fuse_all_gather_matmul`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`fuse_matmul_reduce_scatter`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`get_arg`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`is_collective`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`make_all_gather_split_pattern`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`make_cat_pattern`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`make_zero_dim_all_gather_pattern`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`micro_pipeline_tp_pass`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`reduce_scatter_template`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`replace_with`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)

### Imports

- **`..`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`..pattern_matcher`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`Any`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`OrderedSet`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`collections`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`config`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`dataclass`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`dataclasses`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`defaultdict`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`logging`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`math`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`operator`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`prod`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`torch`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`torch.distributed._symmetric_memory`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`torch.utils._ordered_set`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)
- **`typing`**: [micro_pipeline_tp.py_docs.md](./micro_pipeline_tp.py_docs.md)


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

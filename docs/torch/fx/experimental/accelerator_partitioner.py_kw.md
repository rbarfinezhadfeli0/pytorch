# Keyword Index: `torch/fx/experimental/accelerator_partitioner.py`

## File Information

- **Original File**: [torch/fx/experimental/accelerator_partitioner.py](../../../../torch/fx/experimental/accelerator_partitioner.py)
- **Documentation**: [`accelerator_partitioner.py_docs.md`](./accelerator_partitioner.py_docs.md)
- **Folder**: `torch/fx/experimental`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DAG`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`DAGNode`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`PartitionResult`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`Partitioner`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`contains`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`helps`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`is`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`maintains`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)

### Functions

- **`__init__`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`__str__`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`aot_based_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`calculate_extra_mem_bytes_needed_for`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`calculate_mem_bytes_needed`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`check_dependency`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`combine_partitions_based_on_size`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`combine_two_partitions`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`cost_aware_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`create_node`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`create_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`create_single_node_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`do_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`dump_dag`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`find_device_based_on_size`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`find_device_for`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`find_partition_to_combine_based_on_size`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`find_single_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`get_bfs_level_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`get_device_partition_stats`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`get_device_to_partitions_mapping`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`get_logical_id_to_device`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`get_node_to_partition_mapping`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`is_embedding_node`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`kl_based_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`partition_graph`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`reorganize_partitions`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`reset_partition_device`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`reset_partition_in_sparse_nn`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`saturate_host`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`search_combination`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`set_parents_and_children`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`size_based_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`sparse_nn_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`swap_node_to_partition`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`swap_nodes`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`try_combining_partitions`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`try_swap_nodes`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)

### Imports

- **`GraphModule`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`NamedTuple`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`collections`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`deque`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`get_size_of_all_nodes`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`map_arg`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`operator`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`split_module`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`torch`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`torch.fx.experimental.partitioner_utils`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`torch.fx.graph_module`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`torch.fx.node`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`torch.fx.passes.graph_manipulation`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`torch.fx.passes.split_module`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)
- **`typing`**: [accelerator_partitioner.py_docs.md](./accelerator_partitioner.py_docs.md)


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

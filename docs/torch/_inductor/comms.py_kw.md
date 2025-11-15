# Keyword Index: `torch/_inductor/comms.py`

## File Information

- **Original File**: [torch/_inductor/comms.py](../../../torch/_inductor/comms.py)
- **Documentation**: [`comms.py_docs.md`](./comms.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Runnable`**: [comms.py_docs.md](./comms.py_docs.md)
- **`class`**: [comms.py_docs.md](./comms.py_docs.md)
- **`from`**: [comms.py_docs.md](./comms.py_docs.md)

### Functions

- **`__init__`**: [comms.py_docs.md](./comms.py_docs.md)
- **`__lt__`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_calculate_potential_peak_memory_reorder`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_calculate_potential_peak_memory_sink_waits`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_create_group_node`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_find_buffers_with_changed_last_use`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_find_buffers_with_changed_last_use_sink_waits`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_format_and_log_reordering_stats`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_format_and_log_sink_waits_stats`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_group_name`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_group_names`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_group_nodes_from_linked_list`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_initialize_double_linked_list`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_initialize_memory_tracking`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_is_fake_dep`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_is_node_groupable_for_reorder`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_is_node_groupable_for_sink_waits`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_op_runtime_estimate_mult`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_perform_double_linked_list_swap`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_perform_double_linked_list_swap_sink_waits`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_reorder_communication_preserving_peak_memory_internal`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_schedule_for_comm`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_sink_waits_iterative_internal`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_temp_group_visit_leaves`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_update_memory_tracking_after_swap_reorder`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_update_memory_tracking_after_swap_sink_waits`**: [comms.py_docs.md](./comms.py_docs.md)
- **`accumulate_time`**: [comms.py_docs.md](./comms.py_docs.md)
- **`align_runtime_estimations_across_all_distributed_ranks`**: [comms.py_docs.md](./comms.py_docs.md)
- **`check_resize_pattern`**: [comms.py_docs.md](./comms.py_docs.md)
- **`coll_exposed_communication_time`**: [comms.py_docs.md](./comms.py_docs.md)
- **`contains_async_collective`**: [comms.py_docs.md](./comms.py_docs.md)
- **`contains_gemm_like`**: [comms.py_docs.md](./comms.py_docs.md)
- **`decide_global_ordering_of_comms`**: [comms.py_docs.md](./comms.py_docs.md)
- **`enforce_comm_ordering_for_fsdp`**: [comms.py_docs.md](./comms.py_docs.md)
- **`estimate_op_runtime`**: [comms.py_docs.md](./comms.py_docs.md)
- **`get_op_idx`**: [comms.py_docs.md](./comms.py_docs.md)
- **`get_overlapping_candidate`**: [comms.py_docs.md](./comms.py_docs.md)
- **`improvement`**: [comms.py_docs.md](./comms.py_docs.md)
- **`is_allowed_mutation`**: [comms.py_docs.md](./comms.py_docs.md)
- **`is_async_collective`**: [comms.py_docs.md](./comms.py_docs.md)
- **`is_corresponding_collective_wait`**: [comms.py_docs.md](./comms.py_docs.md)
- **`is_gemm_like`**: [comms.py_docs.md](./comms.py_docs.md)
- **`is_node_mutating_unsharded_param_or_its_alias`**: [comms.py_docs.md](./comms.py_docs.md)
- **`node_summary`**: [comms.py_docs.md](./comms.py_docs.md)
- **`raise_comms`**: [comms.py_docs.md](./comms.py_docs.md)
- **`reinplace_all_gather`**: [comms.py_docs.md](./comms.py_docs.md)
- **`reinplace_fsdp_all_gather`**: [comms.py_docs.md](./comms.py_docs.md)
- **`remove_fsdp2_unsharded_param_graph_input_usage`**: [comms.py_docs.md](./comms.py_docs.md)
- **`remove_unused_getitem`**: [comms.py_docs.md](./comms.py_docs.md)
- **`reorder_communication_preserving_peak_memory`**: [comms.py_docs.md](./comms.py_docs.md)
- **`reorder_compute_and_comm_for_overlap`**: [comms.py_docs.md](./comms.py_docs.md)
- **`reorder_compute_for_overlap`**: [comms.py_docs.md](./comms.py_docs.md)
- **`repl`**: [comms.py_docs.md](./comms.py_docs.md)
- **`schedule`**: [comms.py_docs.md](./comms.py_docs.md)
- **`schedule_collective_for_overlap`**: [comms.py_docs.md](./comms.py_docs.md)
- **`sink_waits`**: [comms.py_docs.md](./comms.py_docs.md)
- **`sink_waits_iterative`**: [comms.py_docs.md](./comms.py_docs.md)
- **`step_log`**: [comms.py_docs.md](./comms.py_docs.md)
- **`visualize_overlap`**: [comms.py_docs.md](./comms.py_docs.md)
- **`wait_exposed_communication_time`**: [comms.py_docs.md](./comms.py_docs.md)

### Imports

- **`.`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.comms_debug`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.dependencies`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.ir`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.memory`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.pattern_matcher`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.utils`**: [comms.py_docs.md](./comms.py_docs.md)
- **`.virtualized`**: [comms.py_docs.md](./comms.py_docs.md)
- **`Any`**: [comms.py_docs.md](./comms.py_docs.md)
- **`BaseSchedulerNode`**: [comms.py_docs.md](./comms.py_docs.md)
- **`GroupedSchedulerNode`**: [comms.py_docs.md](./comms.py_docs.md)
- **`IRNode`**: [comms.py_docs.md](./comms.py_docs.md)
- **`OrderedSet`**: [comms.py_docs.md](./comms.py_docs.md)
- **`StorageWeakRef`**: [comms.py_docs.md](./comms.py_docs.md)
- **`V`**: [comms.py_docs.md](./comms.py_docs.md)
- **`WeakDep`**: [comms.py_docs.md](./comms.py_docs.md)
- **`__future__`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_debug_iterative_memory_recompute`**: [comms.py_docs.md](./comms.py_docs.md)
- **`_get_default_group`**: [comms.py_docs.md](./comms.py_docs.md)
- **`annotations`**: [comms.py_docs.md](./comms.py_docs.md)
- **`collections`**: [comms.py_docs.md](./comms.py_docs.md)
- **`config`**: [comms.py_docs.md](./comms.py_docs.md)
- **`dataclass`**: [comms.py_docs.md](./comms.py_docs.md)
- **`dataclasses`**: [comms.py_docs.md](./comms.py_docs.md)
- **`defaultdict`**: [comms.py_docs.md](./comms.py_docs.md)
- **`heapq`**: [comms.py_docs.md](./comms.py_docs.md)
- **`importlib`**: [comms.py_docs.md](./comms.py_docs.md)
- **`itertools`**: [comms.py_docs.md](./comms.py_docs.md)
- **`logging`**: [comms.py_docs.md](./comms.py_docs.md)
- **`operator`**: [comms.py_docs.md](./comms.py_docs.md)
- **`scheduler`**: [comms.py_docs.md](./comms.py_docs.md)
- **`sys`**: [comms.py_docs.md](./comms.py_docs.md)
- **`tabulate`**: [comms.py_docs.md](./comms.py_docs.md)
- **`time`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch._inductor.scheduler`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch._logging`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch.distributed`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch.distributed.distributed_c10d`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_collectives`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch.multiprocessing.reductions`**: [comms.py_docs.md](./comms.py_docs.md)
- **`torch.utils._ordered_set`**: [comms.py_docs.md](./comms.py_docs.md)
- **`trace_structured`**: [comms.py_docs.md](./comms.py_docs.md)
- **`typing`**: [comms.py_docs.md](./comms.py_docs.md)


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

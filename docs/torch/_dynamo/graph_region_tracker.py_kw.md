# Keyword Index: `torch/_dynamo/graph_region_tracker.py`

## File Information

- **Original File**: [torch/_dynamo/graph_region_tracker.py](../../../torch/_dynamo/graph_region_tracker.py)
- **Documentation**: [`graph_region_tracker.py_docs.md`](./graph_region_tracker.py_docs.md)
- **Folder**: `torch/_dynamo`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BackwardBfsArgIter`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`GraphRegionTracker`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`InputPickler`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`NodeHashException`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`RegionWrapper`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`which`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)

### Functions

- **`__init__`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`__str__`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_append`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_extract_args`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_extract_tensor_metadata_for_node_hash`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_hash_node`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_is_identical`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_normalize_args`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_populate_recursive_ancestor_map`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_sort_with_ref_region`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`add`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`add_children`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`add_node_mutation`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`create`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`debug_log`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`dumps`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`fully_expand_region_group`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`get_global_state_key`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`get_identical_regions`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`next`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`next_candidate`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`peek`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`track_node`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`track_node_mutations`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`will_inclusion_create_cycle`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)

### Imports

- **`.graph_utils`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`.symbolic_convert`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`Any`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`Callable`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`FakeTensor`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`InstructionTranslatorBase`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`OrderedSet`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`__future__`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_get_flat_args_unique`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`_ident`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`annotations`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`collections`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`collections.abc`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`copyreg`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`dataclasses`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`defaultdict`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`fields`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`io`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`logging`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`math`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`operator`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`pickle`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`sha256_hash`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`torch._inductor.codecache`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`torch._logging`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`torch.fx`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`torch.utils._ordered_set`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`torch.utils._pytree`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`tree_flatten`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)
- **`typing`**: [graph_region_tracker.py_docs.md](./graph_region_tracker.py_docs.md)


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

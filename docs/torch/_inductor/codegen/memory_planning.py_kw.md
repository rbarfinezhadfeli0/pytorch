# Keyword Index: `torch/_inductor/codegen/memory_planning.py`

## File Information

- **Original File**: [torch/_inductor/codegen/memory_planning.py](../../../../torch/_inductor/codegen/memory_planning.py)
- **Documentation**: [`memory_planning.py_docs.md`](./memory_planning.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AllocationTreeNode`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`BufferGroup`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`ClearCacheOnAllocateMixin`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`LiveRanges`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`MemorySplitProtocol`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`class`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`for`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)

### Functions

- **`__eq__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`__hash__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`__init__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`__len__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`__post_init__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`__repr__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`_allocate`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`allocate`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`allocate_at_end`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`allocate_groups`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`allocate_output`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`begin`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`clear_cache`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`codegen`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`codegen_alloc_from_pool`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`codegen_create`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`codegen_destroy`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`compute_buffer_groups`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`compute_live_ranges`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`contains`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`convert_to_pool_lines`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`create`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`device`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`drop_removed_buffers`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`end`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`finalize`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`get_earliest_available`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`get_live_ranges`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`get_pools`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`get_size_hint`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`get_symbolic_size`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`is_empty`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`join`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`make_allocation`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`mark_allocated`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`mark_first_last_usage`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`node`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`overlaps`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`plan`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`pprint`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`sym_nbytes`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`update_restrict_live_range`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`update_usage`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)

### Imports

- **`..`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`..utils`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`..virtualized`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`.wrapper`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`Any`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`Iterable`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`OrderedSet`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`V`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`__future__`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`_align`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`annotations`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`collections`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`collections.abc`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`config`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`dataclasses`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`free_unbacked_symbols`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`itertools`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`pprint`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`sympy`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`torch`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`torch.utils._ordered_set`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)
- **`typing`**: [memory_planning.py_docs.md](./memory_planning.py_docs.md)


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

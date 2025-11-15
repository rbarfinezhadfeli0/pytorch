# Keyword Index: `torch/_inductor/sizevars.py`

## File Information

- **Original File**: [torch/_inductor/sizevars.py](../../../torch/_inductor/sizevars.py)
- **Documentation**: [`sizevars.py_docs.md`](./sizevars.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CanonicalExprFinder`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`SimplifyIndexing`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`SizeVarAllocator`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`is`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`that`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`works`**: [sizevars.py_docs.md](./sizevars.py_docs.md)

### Functions

- **`__init__`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_build_canonical_expr_mapping`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_check_args`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_choose`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_get_unbacked_replacements`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_join_dimensions_cached`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_lru_cache`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_simplify_loops_impl`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_simplify_with_ranges`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_stride_vars`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`_sub_unbacked_exprs`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`atomically_apply_size_hint`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`can_merge_dims`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`check`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`check_bounds`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`check_equals`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`check_equals_and_simplify`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`check_leq`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`check_lt`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`choose_leader`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`combine_modular_indexing_pairs`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`evaluate_expr`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`evaluate_max`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`evaluate_min`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`expand_floor_div`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`expect_true`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`find`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`find_expr`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`free_symbols`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`guard_int`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`guard_int_seq`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`guard_or_false`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`guard_or_true`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`index_expr`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`is_size_one_or_false`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`join_dimensions`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`load`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`lookup_precomputed_size`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`make_simplify_loops_cache`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`make_simplify_with_ranges_cache`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`make_stride_vars_cache`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`offset_var`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`prune`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`reindex`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`remove_precomputed_replacements`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`remove_zero_terms`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`simplify`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`simplify_loops`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`simplify_with_ranges`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`size_hint`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`size_hint_or_throw`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`size_hints`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`size_hints_or_throw`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_equals`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_geq`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_gt`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_leq`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_list_equals`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_lt`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_multiple_of`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_power_of_2`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`statically_known_true`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`store`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`store_reduction`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`stride_hints`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`stride_order`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`stride_vars`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`symbolic_hint`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`union`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`union_expr`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`visit_indexing_div`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`visit_modular_indexing`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`wrapper`**: [sizevars.py_docs.md](./sizevars.py_docs.md)

### Imports

- **`.runtime.runtime_utils`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`.utils`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`.virtualized`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`Any`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`Callable`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`Expr`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`FloorDiv`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`OrderedSet`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`V`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`bound_sympy`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`collections`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`collections.abc`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`defaultdict`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`functools`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`is_power_of_2`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`itertools`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`logging`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`symbol_is_type`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`sympy`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`torch.fx.experimental._config`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`torch.utils._ordered_set`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`torch.utils._sympy.functions`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`torch.utils._sympy.symbol`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [sizevars.py_docs.md](./sizevars.py_docs.md)
- **`typing`**: [sizevars.py_docs.md](./sizevars.py_docs.md)


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

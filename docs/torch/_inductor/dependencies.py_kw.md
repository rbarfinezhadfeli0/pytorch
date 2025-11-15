# Keyword Index: `torch/_inductor/dependencies.py`

## File Information

- **Original File**: [torch/_inductor/dependencies.py](../../../torch/_inductor/dependencies.py)
- **Documentation**: [`dependencies.py_docs.md`](./dependencies.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Dep`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`FreeSymbolsOpsHandler`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`IndexExprDep`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`MemoryDep`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`RecordLoadStore`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`StarDep`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`WeakDep`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`_RecordLoadStoreInner`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`class`**: [dependencies.py_docs.md](./dependencies.py_docs.md)

### Functions

- **`__init__`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`__repr__`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`_default`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`_normalize`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`add_var`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`bucketize`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`buffer_names`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`canonicalization_prefix`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`canonicalize`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`decide_loop_order_to_match`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`drop_unused_symbols`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`extract_free_symbols`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`extract_input_node_reduction_ranges`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`extract_loop_body_with_args`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`extract_read_writes`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`frexp`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`get_free_symbol_uses`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`get_numel`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`get_offset`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`has_unbacked_symbols`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`index`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`index_expr`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`index_vars_no_squeeze`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`index_vars_squeeze`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`indirect_indexing`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`is_contiguous`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`is_indirect`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`is_scalar`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`load`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`load_seed`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`masked`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`merge`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`merge_list`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`normalize`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`normalize_with_stride_order`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`num_vars`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`numbytes_hint`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`ranges`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`reads_and_writes`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`reduction`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`remove_reads`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`rename`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`scan`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`simplify_with_ranges`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`sort`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`store`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`store_reduction`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`stride1_for_last_dim`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`var_builder`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`with_read`**: [dependencies.py_docs.md](./dependencies.py_docs.md)

### Imports

- **`..utils._sympy.symbol`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`.codegen.common`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`.ir`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`.loop_body`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`.ops_handler`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`.utils`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`.virtualized`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`Any`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`Callable`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`ComputedBuffer`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`DefaultHandler`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`FlexibleLayout`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`LoopBody`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`MemoryUsageType`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`OrderedSet`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`ReductionType`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`Self`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`SqueezeView`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`abc`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`collections.abc`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`dataclasses`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`free_symbols`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`from`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`get_free_symbols`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`here`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`index_prevent_reordering`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`ir`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`itertools`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`logging`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`make_symbol`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`patch`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`re`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`sympy`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`torch`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`torch._inductor`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`torch._inductor.utils`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`torch.utils._ordered_set`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`typing`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`typing_extensions`**: [dependencies.py_docs.md](./dependencies.py_docs.md)
- **`unittest.mock`**: [dependencies.py_docs.md](./dependencies.py_docs.md)


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

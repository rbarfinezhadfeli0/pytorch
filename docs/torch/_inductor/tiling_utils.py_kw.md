# Keyword Index: `torch/_inductor/tiling_utils.py`

## File Information

- **Original File**: [torch/_inductor/tiling_utils.py](../../../torch/_inductor/tiling_utils.py)
- **Documentation**: [`tiling_utils.py_docs.md`](./tiling_utils.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CoalesceVarAnalysis`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`FusedNormalizedReadsWrites`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`NodeSplitGetter`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`VarTiling`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`that`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)

### Functions

- **`__init__`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`_solve_simple_expr`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`analyze_memory_coalescing`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`apply_var_mapping`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`extract_normalized_read_writes`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`find_broadcast_var`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`find_coalesced_var`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`get_hint`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`get_node_splits`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`get_pw_red_splits`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`get_score`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`indexing_div_rep`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`remove_identity`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`solve_for_tiling`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`solve_for_zero`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`try_get_buf_size`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`try_split`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)

### Imports

- **`.virtualized`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`Callable`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`CantSplit`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`Counter`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`FloorDiv`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`FusedSchedulerNode`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`Identity`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`Literal`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`OrderedSet`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`V`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`collections`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`collections.abc`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`config`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`dataclasses`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`index_vars_no_squeeze`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`itertools`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`symbol_is_type`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`sympy`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`sympy_product`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch._inductor`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch._inductor.codegen.simd`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch._inductor.dependencies`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch._inductor.scheduler`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch._inductor.utils`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch.utils._ordered_set`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch.utils._sympy.functions`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch.utils._sympy.solve`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`torch.utils._sympy.symbol`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`try_solve`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)
- **`typing`**: [tiling_utils.py_docs.md](./tiling_utils.py_docs.md)


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

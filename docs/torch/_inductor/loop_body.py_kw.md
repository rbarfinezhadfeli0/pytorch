# Keyword Index: `torch/_inductor/loop_body.py`

## File Information

- **Original File**: [torch/_inductor/loop_body.py](../../../torch/_inductor/loop_body.py)
- **Documentation**: [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CaptureIndexing`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`CountOps`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`InterpreterShim`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`LightTracer`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`LoopBody`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`LoopBodyBlock`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`MemoryEntry`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`MemoryUsageType`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`into`**: [loop_body.py_docs.md](./loop_body.py_docs.md)

### Functions

- **`__call__`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`__init__`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`_add_index`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`_default`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`_dummy_gm`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`_init_with_copy`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`_init_with_tracing`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`_simplify`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`add_index_expr`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`add_indirect`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`add_submodule`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`bind_masked_shim`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`bind_scan_shim`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`bind_set_indirect_shim`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`bounds`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`bucketize`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`check_bounds`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`clone`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`contains_only_ops`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`debug_str`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`expand_dimension_for_pointwise_node`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`extract_pw_from_reduction`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`forward`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`frexp`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_all_read_expr`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_all_write_expr`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_index`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_nodes`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_original_num_rdims`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_read_expr`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_read_exprs`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_write_expr`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`get_write_exprs`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`has_op`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`index_expr`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`indexing_from_args`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`indirect_indexing`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`is_memory_copy`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`load`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`load_seed`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`masked`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`merge_loops`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`new_body`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`output`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`reduction`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`reorder_iter_loops`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`replace_indirect`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`run`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`run_node`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`scan`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`set_indirect`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`shim`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`sort`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`store`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`store_reduction`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`vars`**: [loop_body.py_docs.md](./loop_body.py_docs.md)

### Imports

- **`.`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.bounds`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.codegen.common`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.index_propagation`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.ir`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.ops_handler`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.utils`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`.virtualized`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`Any`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`BoundVars`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`Callable`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`DefaultHandler`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`IndexPropagation`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`Scope`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`SymT`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`__future__`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`annotations`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`auto`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`collections`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`collections.abc`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`config`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`enum`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`functools`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`identity`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`index_prevent_reordering`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`itertools`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`ops`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`re`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`same_reorder`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`sympy`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`to`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`torch._dynamo.utils`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`torch.fx`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`torch.fx.proxy`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`torch.utils._sympy.symbol`**: [loop_body.py_docs.md](./loop_body.py_docs.md)
- **`typing`**: [loop_body.py_docs.md](./loop_body.py_docs.md)


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

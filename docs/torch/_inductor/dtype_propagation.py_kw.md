# Keyword Index: `torch/_inductor/dtype_propagation.py`

## File Information

- **Original File**: [torch/_inductor/dtype_propagation.py](../../../torch/_inductor/dtype_propagation.py)
- **Documentation**: [`dtype_propagation.py_docs.md`](./dtype_propagation.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DTypeVar`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`DtypePropagationOpsHandler`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`_typecheck_DtypePropagation`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)

### Functions

- **`__init__`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`__new__`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`bucketize`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`ceil_to_int`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`check_bounds`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`constant`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`construct_input`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`device_assert_async`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`dot`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`dtype`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`floor`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`floor_to_int`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`floordiv`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`fmod`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`frexp`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`gelu`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`get_promoted_dtype`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`halide_clamp`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`identity`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`index_expr`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`indirect_indexing`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`inline_asm_elementwise`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`int_truediv`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`load`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`load_seed`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`lshift`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`masked`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`mod`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`mul`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`op_dtype_rule`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`output`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`partial_accumulate`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`placeholder`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`pow`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`promote_types`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`rand`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`randint64`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`randn`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`reduction`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`return_dtype`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`round`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`round_to_int`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`rshift`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`scan`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`sort`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`store`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`store_reduction`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`to_dtype`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`to_dtype_bitcast`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`truediv`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`trunc`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`trunc_to_int`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`truncdiv`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`where`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)

### Imports

- **`.loop_body`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`.ops_handler`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`.utils`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`.virtualized`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`Any`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`Callable`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`ELEMENTWISE_TYPE_PROMOTION_KIND`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`LoopBodyBlock`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`OP_NAMES`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`OpsValue`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`OrderedSet`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`collections.abc`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`functools`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`sympy`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`torch`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`torch._prims_common`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`torch.utils._ordered_set`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`typing`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)
- **`upcast_compute_type`**: [dtype_propagation.py_docs.md](./dtype_propagation.py_docs.md)


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

# Keyword Index: `torch/_inductor/codegen/cpp_utils.py`

## File Information

- **Original File**: [torch/_inductor/codegen/cpp_utils.py](../../../../torch/_inductor/codegen/cpp_utils.py)
- **Documentation**: [`cpp_utils.py_docs.md`](./cpp_utils.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CppCSEVariable`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`CppPrinter`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`LocalBufferContext`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`LocalizeBufferHandler`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`creates`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)

### Functions

- **`__enter__`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`__exit__`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`__init__`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`__repr__`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`_check_supported_and_same_indexes`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`_get_dtype_from_loopbodies`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`_get_indexes_of_template_buf_read`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`_get_loop_body`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`_set_dependent_itervars`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`_template_fusion_supported`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`add_local_buffer`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`cexpr_index`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`codegen_rand`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`create_epilogue_with_attr`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`depends_on`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`doprint`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`get_dtype`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`get_gemm_template_output_and_compute_dtype`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`get_promote_dtype`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`hardsigmoid_float`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`inner`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`inner_fn`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`input`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`load`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`localize`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`localize_function`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`localize_nodes`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`may_unify_binary_op_mask_type`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`output`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`parenthesize`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`promote_arg`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`promote_args`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`rewrite_index_for_function`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`rewrite_index_for_nodes`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`store`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`store_reduction`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`template_fusion_with_epilogues_supported`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`unify_mask_base_type`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`update_on_args`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`value_to_cpp`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`wrap_inner_fn_for_node`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)

### Imports

- **`..`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`..dependencies`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`..loop_body`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`..scheduler`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`..shape_propagation`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`..utils`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`..virtualized`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`.common`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`Any`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`BaseSchedulerNode`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`BlockShapeType`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`CSEVariable`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`Callable`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`CppPrinter`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`Dep`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`IndentedBuffer`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`LoopBody`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`OrderedSet`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`ValueRanges`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`collections`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`collections.abc`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`contextlib`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`dataclasses`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`functools`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`ir`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`is_integer_dtype`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`math`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`namedtuple`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`ops`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`patch`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`symbol_is_type`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`sympy`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`sys`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`torch`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`torch._prims_common`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`torch.utils._ordered_set`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`torch.utils._sympy.printers`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`torch.utils._sympy.symbol`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`typing`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)
- **`unittest.mock`**: [cpp_utils.py_docs.md](./cpp_utils.py_docs.md)


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

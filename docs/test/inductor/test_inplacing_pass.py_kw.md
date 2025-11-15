# Keyword Index: `test/inductor/test_inplacing_pass.py`

## File Information

- **Original File**: [test/inductor/test_inplacing_pass.py](../../../test/inductor/test_inplacing_pass.py)
- **Documentation**: [`test_inplacing_pass.py_docs.md`](./test_inplacing_pass.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MySin`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`TestReinplacingPassCorrectness`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)

### Functions

- **`_test`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`backward`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`boo`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`f`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`fn`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`forward`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`get_not_inplaced_count`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`miss_inplaced_bytes`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`mutate_op`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`num_reinplacing_failures`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`setUp`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`sin`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`sin_cos`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`sin_kernel`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`sin_triton`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_counters_functionalize_old`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_counters_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_dont_modify_input`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_dont_modify_live`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_dont_modify_view_of_live`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_generalized_scatter`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_lists_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_lists_old_functionalize`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_multi_output_intermediate`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_multiple_intermediate`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_multiple_mutations`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_partitioner_recomputes_factory`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_should_modify_inner`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_should_modify_input`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_view_inplaced2_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_view_inplaced_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_views_not_inplaced2_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_views_not_inplaced3_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`test_views_not_inplaced_functionalize_v2`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)

### Imports

- **`GPU_TYPE`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`ReinplaceCounters`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`Tensor`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`functorch`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`logs_to_string`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`make_fx`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`reinplace_inplaceable_ops_core`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`run_tests`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch._dynamo.utils`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch._higher_order_ops.auto_functionalize`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch._inductor.config`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch._inductor.fx_passes.reinplace`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch._inductor.test_case`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`torch.testing._internal.logging_utils`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`triton`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)
- **`triton.language`**: [test_inplacing_pass.py_docs.md](./test_inplacing_pass.py_docs.md)


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

# Keyword Index: `test/inductor/test_group_batch_fusion.py`

## File Information

- **Original File**: [test/inductor/test_group_batch_fusion.py](../../../test/inductor/test_group_batch_fusion.py)
- **Documentation**: [`test_group_batch_fusion.py_docs.md`](./test_group_batch_fusion.py_docs.md)
- **Folder**: `test/inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MyModule`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`MyModule2`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`MyModule3`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`MyModule4`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`MyModule5`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestBMMFusionModule`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestDropout`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestFindIndependentSubsetGreedy`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestGroupBatchFusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestHighwaySelfGating`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestMathOps`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestPoitwiseOps`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestPoitwiseOpsPostGrad`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`TestPostGradBatchLinearFusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)

### Functions

- **`__init__`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`build_graph`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`compare_dict_tensors`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`compare_gradients`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`compare_parameters`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`compare_pred`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`forward`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_batch_dropout_pre_grad_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_batch_layer_norm_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_batch_linear_lhs_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_batch_linear_post_grad_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_batch_linear_pre_grad_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_find_independent_subset_greedy`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_find_independent_subset_greedy_fuse`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_gate_fusion_post_grad`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_group_linear_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_group_linear_fusion_different_shapes`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_math_op_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_pointwise_op_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`test_pointwise_op_fusion_post_grad`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`verify`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)

### Imports

- **`GPU_TYPE`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`collections`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`counters`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`run_tests`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`torch`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`torch._dynamo.utils`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`torch._inductor`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`torch._inductor.fx_passes.group_batch_fusion`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`torch._inductor.test_case`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`torch.testing._internal.inductor_utils`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)
- **`unittest`**: [test_group_batch_fusion.py_docs.md](./test_group_batch_fusion.py_docs.md)


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

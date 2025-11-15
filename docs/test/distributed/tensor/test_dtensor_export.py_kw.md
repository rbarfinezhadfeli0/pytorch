# Keyword Index: `test/distributed/tensor/test_dtensor_export.py`

## File Information

- **Original File**: [test/distributed/tensor/test_dtensor_export.py](../../../../test/distributed/tensor/test_dtensor_export.py)
- **Documentation**: [`test_dtensor_export.py_docs.md`](./test_dtensor_export.py_docs.md)
- **Folder**: `test/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Bar`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`DTensorExportTest`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`EinsumModel`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`FlexAttentionModel`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`Foo`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`SimpleModel`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`SimpleModelAnnotated`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`SimpleModelDynamicShapes`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)

### Functions

- **`__init__`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`_count_op`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`_run_test`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`aot_export_joint_with_descriptors_alone`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`causal_mask`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`fn`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`forward`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`graph_capture_and_aot_export_joint_with_descriptors_v2`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`has_tag`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`marked_nodes`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`nest_fn`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`setUp`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`strict_export_and_aot_export_joint_with_descriptors`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`tearDown`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_annotate_aot_export_joint_with_descriptors_alone`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_dtensor_data_dependent_index_and_slice`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_dynamic_shapes`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_einsum_dtensor_export`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_export_parallelize_module_with_dtensor_input`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_flex_attention_dtensor_export`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_strict_export_parallelize_module_with_dtensor_input`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`test_union_typed_annotation`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`unmarked_nodes`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)

### Imports

- **`DTensor`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`DTensorSpec`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`FakeStore`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`MLPModule`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`aot_export_joint_with_descriptors`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`contextlib`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`distribute_tensor`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`dynamo_graph_capture_for_export`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`init_device_mesh`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`min_cut_rematerialization_partition`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`register_pytree_node`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch._dynamo.functional_export`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch._functorch.aot_autograd`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch._functorch.partitioners`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch._guards`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.distributed`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.distributed.tensor`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.distributed.tensor._api`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.fx.traceback`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.nn.attention.flex_attention`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.testing._internal.distributed.fake_pg`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`torch.utils._pytree`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`tracing`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)
- **`unittest`**: [test_dtensor_export.py_docs.md](./test_dtensor_export.py_docs.md)


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

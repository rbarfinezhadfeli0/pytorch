# Keyword Index: `torch/distributed/tensor/experimental/_tp_transform.py`

## File Information

- **Original File**: [torch/distributed/tensor/experimental/_tp_transform.py](../../../../../torch/distributed/tensor/experimental/_tp_transform.py)
- **Documentation**: [`_tp_transform.py_docs.md`](./_tp_transform.py_docs.md)
- **Folder**: `torch/distributed/tensor/experimental`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`_TensorParallelTransformPass`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)

### Functions

- **`__init__`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_clean_up_graph_metadata`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_create_placement_strategy`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_generate_default_output_sharding`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_generate_parameter_and_buffer_placements`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_get_input_node_fqn`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_get_input_node_specs`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_get_op_schema`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_get_output_spec_from_output_sharding`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_insert_reshard_gm`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_mark_sharding`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_mark_tensor_parallel_shardings`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_partition_val`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_partitioner`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_populate_tensor_meta`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_shard_state_dict`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`call`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`create_output_spec`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`reshard_fn`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`tensor_parallel_transformation`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`update_arg_spec`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)

### Imports

- **`Any`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`ColwiseParallel`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`DTensorSpec`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`DeviceMesh`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`ExportGraphSignature`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`ExportedProgram`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`FakeTensor`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`GraphModule`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`Node`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`PassBase`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`Placement`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`Sequence`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_extract_tensor_metadata`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`_pytree`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`collections.abc`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`copy`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`make_fx`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`operator`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`redistribute_local_tensor`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.distributed.tensor`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.distributed.tensor._op_schema`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.distributed.tensor._redistribute`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.distributed.tensor.parallel.style`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.export`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.export.exported_program`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.fx`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.fx.node`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.fx.passes.infra.pass_base`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.fx.passes.shape_prop`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`torch.utils`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)
- **`typing`**: [_tp_transform.py_docs.md](./_tp_transform.py_docs.md)


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

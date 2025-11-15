# Keyword Index: `torch/distributed/tensor/_op_schema.py`

## File Information

- **Original File**: [torch/distributed/tensor/_op_schema.py](../../../../torch/distributed/tensor/_op_schema.py)
- **Documentation**: [`_op_schema.py_docs.md`](./_op_schema.py_docs.md)
- **Folder**: `torch/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Args`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`OpStrategy`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`StrategyType`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`TODO`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`TupleStrategy`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`class`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`from`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`that`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`type`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)

### Functions

- **`__eq__`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`__hash__`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`__init__`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`__post_init__`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`__repr__`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`__str__`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`_inplace_rewrap_schema_suggestion`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`_pretty_print_spec`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`_rebuild_tensor_from_dtensor_meta`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`_recompute_comparison_key`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`arg_type_tensor_or_tensor_list_like`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`args_spec`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`args_strategy`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`child_mesh`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`childs`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`gen_fake_args`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`gen_fake_kwargs`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`get_mesh_from_args`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`input_spec`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`is_inplace_op`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`is_out_variant_op`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`is_view_op`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`max_num_shards`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`mesh`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`mesh_shape`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`ndim`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`output_spec`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`return_type_list_tensor_like`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`return_type_tensor`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`return_type_tuple_tensor_like`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`shape`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)

### Imports

- **`Any`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`DTensorSpec`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`DeviceMesh`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`OpOverload`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`Placement`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`Sequence`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`cached_property`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`collections.abc`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`dataclass`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`dataclasses`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`deprecated`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`functools`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch._C`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch._ops`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch.distributed.device_mesh`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch.utils._cxx_pytree`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`torch.utils._pytree`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`typing`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)
- **`typing_extensions`**: [_op_schema.py_docs.md](./_op_schema.py_docs.md)


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

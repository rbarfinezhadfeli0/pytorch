# Keyword Index: `torch/distributed/tensor/_ops/_math_ops.py`

## File Information

- **Original File**: [torch/distributed/tensor/_ops/_math_ops.py](../../../../../torch/distributed/tensor/_ops/_math_ops.py)
- **Documentation**: [`_math_ops.py_docs.md`](./_math_ops.py_docs.md)
- **Folder**: `torch/distributed/tensor/_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`NormReduction`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`Reduction`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_NormPartial`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`from`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)

### Functions

- **`__eq__`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`__hash__`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`__init__`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_add_target_input_spec`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_common_norm_backward_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_common_norm_forward_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_infer_reduce_dims_map`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_infer_reduction_dims`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_partition_value`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_post_reduce_transform`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_pre_reduce_transform`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_reduce_shard_value`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_reduce_value`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_replicate_dims_start_at`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`_skip_dim`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`common_reduction_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`cumsum_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`foreach_norm_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`fused_rms_norm_bwd_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`fused_rms_norm_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`get_placement_from_reduction_op`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`histc_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`layer_norm_bwd_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`layer_norm_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`linalg_replicate_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`linear_reduction_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`logsumexp_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`map_placements_after_reduction`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`nll_loss_backward_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`nll_loss_forward_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`replicate_reduction_dims`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`softmax_backward_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`softmax_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`sort_default_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`sort_stable_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`sort_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`topk_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`var_reduction_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`vector_norm_strategy`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)

### Imports

- **`DTensorSpec`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`DeviceMesh`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`Enum`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`Sequence`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`cast`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`collections.abc`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`dataclass`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`dataclasses`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`enum`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`math`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`normalize_to_torch_size`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch.distributed.device_mesh`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch.distributed.tensor._op_schema`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch.distributed.tensor._ops.utils`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch.distributed.tensor._utils`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`torch.distributed.tensor.placement_types`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)
- **`typing`**: [_math_ops.py_docs.md](./_math_ops.py_docs.md)


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

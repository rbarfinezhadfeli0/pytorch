# Keyword Index: `test/distributed/tensor/test_math_ops.py`

## File Information

- **Original File**: [test/distributed/tensor/test_math_ops.py](../../../../test/distributed/tensor/test_math_ops.py)
- **Documentation**: [`test_math_ops.py_docs.md`](./test_math_ops.py_docs.md)
- **Folder**: `test/distributed/tensor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DistMathOpsTest`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`LnTpBlock`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`SubTest`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)

### Functions

- **`__init__`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`_check_module`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`_replicate_fn`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`apply_rotary_emb`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`forward`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`linear_op_reductions`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_conj_complex_dtensor`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_cumsum`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_foreach_add_different_mesh`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_foreach_norm`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_foreach_norm_different_mesh`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_foreach_norm_partial`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_histc`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_layer_norm_bwd`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_layer_norm_bwd_req_grad`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_layer_norm_fwd`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_linalg_eigh`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_linear_op_reductions`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_logsumexp`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_matching_partial_reduction_ops`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_mean`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_nll_loss_and_cross_entropy`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_partial_reduction_ops`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_rotary_embedding_complex_ops`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_shard0_svd`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_shard_math_ops`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_softmax_fwd`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_softmax_with_bwd`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_topk`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_upsampling`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_vector_norm`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`test_vector_norm_partial`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`valid_filter`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)

### Imports

- **`CommDebugMode`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`NamedTuple`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`TensorMeta`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`copy`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`init_device_mesh`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`is_tensor_partial`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`itertools`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`pformat`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`pprint`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`run_tests`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed.tensor`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed.tensor._dtensor_spec`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed.tensor._ops.utils`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed.tensor.debug`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)
- **`typing`**: [test_math_ops.py_docs.md](./test_math_ops.py_docs.md)


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

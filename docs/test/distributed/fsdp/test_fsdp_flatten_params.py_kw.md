# Keyword Index: `test/distributed/fsdp/test_fsdp_flatten_params.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_flatten_params.py](../../../../test/distributed/fsdp/test_fsdp_flatten_params.py)
- **Documentation**: [`test_fsdp_flatten_params.py_docs.md`](./test_fsdp_flatten_params.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`EmbeddingModel`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`EmptyModule`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`TestFlattenParams`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_get_default_config`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_get_empty_module`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_get_output`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_get_pnorm_after_step`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_get_shared_params_transformer`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_get_transformer`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_flat_param_shard_metadata`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_flatten_nothing`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_numel`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_numel_with_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_numel_without_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_output`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_output_with_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_output_without_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_partial_flattening`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`_test_pnorm_after_step_with_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`forward`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`get_input`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_empty_module`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_flat_param_shard_metadata_aligned_full_precision`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_flat_param_shard_metadata_aligned_mixed_precision`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_flat_param_shard_metadata_unaligned`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_flat_param_shard_metadata_with_memory_format`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_flatten_nothing`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_numel_with_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_numel_without_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_output_with_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_output_without_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_partial_flattening`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_pnorm_after_step_with_shared_params`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`test_writeback_orig_params_no_shard`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`world_size`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)

### Imports

- **`FSDPTest`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`FullyShardedDataParallel`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`distributed`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`sys`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch.distributed.fsdp._flat_param`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch.nn`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_flatten_params.py_docs.md](./test_fsdp_flatten_params.py_docs.md)


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

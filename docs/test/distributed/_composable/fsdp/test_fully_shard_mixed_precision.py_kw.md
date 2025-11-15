# Keyword Index: `test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py`

## File Information

- **Original File**: [test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py](../../../../../test/distributed/_composable/fsdp/test_fully_shard_mixed_precision.py)
- **Documentation**: [`test_fully_shard_mixed_precision.py_docs.md`](./test_fully_shard_mixed_precision.py_docs.md)
- **Folder**: `test/distributed/_composable/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Model`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`TestFullyShardMixedPrecisionCasts`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`TestFullyShardMixedPrecisionTraining`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`ToyModel`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`ToyModule`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`class`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)

### Functions

- **`__init__`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_get_use_shard_placement_fn_vals_for_bf16_reduce`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_init_models_and_optims`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_shard_placement_fn`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_test_compute_dtype`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_test_grad_acc_with_reduce_dtype`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_test_norm_modules`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_test_reduce_dtype_bf16_reduce`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_test_reduce_dtype_fp32_reduce`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`_test_submodules_with_external_inputs`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`assert_fn`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`forward`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`inner`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_clamp_reduce_dtype`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_compute_dtype`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_dataclass_input`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_float16_on_one_submodule`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_grad_acc_with_reduce_dtype`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_norm_modules_bf16`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_norm_modules_fp16`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_reduce_dtype`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`test_submodules_with_external_inputs`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`world_size`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)

### Imports

- **`Optional`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`Shard`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`copy`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`dataclasses`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`fully_shard`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`functools`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.distributed`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.distributed._functional_collectives`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_collectives`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.distributed.tensor`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.nn`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)
- **`typing`**: [test_fully_shard_mixed_precision.py_docs.md](./test_fully_shard_mixed_precision.py_docs.md)


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

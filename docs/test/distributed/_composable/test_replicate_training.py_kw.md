# Keyword Index: `test/distributed/_composable/test_replicate_training.py`

## File Information

- **Original File**: [test/distributed/_composable/test_replicate_training.py](../../../../test/distributed/_composable/test_replicate_training.py)
- **Documentation**: [`test_replicate_training.py_docs.md`](./test_replicate_training.py_docs.md)
- **Folder**: `test/distributed/_composable`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Model`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`MultiForwardModule`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`ParamlessModule`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicate1DTrainingCore`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateCastAfterInit`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateCustomForwardMethod`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateForwardInputs`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateGradientAccumulation`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateRegisteredParams`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateSharedParams`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateTPTraining`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`TestReplicateTrainingCompose`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`VisionTransformer`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)

### Functions

- **`__init__`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_assert_dtensor_params`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_assert_same_params`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_assert_tensor_params`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_1f1b_microbatching`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_gradient_accumulation`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_multi_forward_module`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_replicate_tp`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_train_parity_multi_group`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_train_parity_single_group`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_train_parity_with_activation_checkpointing`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`_test_train_shared_params`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`delayed_all_gather`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`delayed_reduce_scatter`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`forward`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`forward_features`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`init_global_mesh`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`set_backward_flags`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`set_grad_sync_flag`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`step_post_hook`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_1f1b_microbatching`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_explicit_prefetching`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_gradient_accumulation`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_multi_forward_module`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_non_root_forward_backward`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_param_registration_after_backward`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_param_registration_after_forward`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_post_optim_event`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_register_fsdp_forward_method`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_replicate_tp`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_root_move_forward_input_to_device`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_to_float64_after_init`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_train_parity_multi_group_cpu_offload_eager`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_train_parity_multi_groups`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_train_parity_single_group`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_train_parity_with_activation_checkpointing`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`test_train_parity_with_shared_params`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`world_size`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)

### Imports

- **`CommDebugMode`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`DTensor`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`DeviceMesh`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`Iterable`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`Union`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`checkpoint`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`collections`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`collections.abc`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`contextlib`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`copy`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`defaultdict`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`functools`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`get_devtype`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`itertools`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`replicate`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed._composable`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed._composable.replicate_with_fsdp`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed.algorithms._checkpoint.checkpoint_wrapper`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed.device_mesh`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed.fsdp`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed.tensor`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed.tensor.debug`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.nn`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`torch.testing._internal.distributed._tensor.common_dtensor`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`typing`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)
- **`unittest`**: [test_replicate_training.py_docs.md](./test_replicate_training.py_docs.md)


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

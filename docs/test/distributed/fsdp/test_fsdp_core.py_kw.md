# Keyword Index: `test/distributed/fsdp/test_fsdp_core.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_core.py](../../../../test/distributed/fsdp/test_fsdp_core.py)
- **Documentation**: [`test_fsdp_core.py_docs.md`](./test_fsdp_core.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`TestAutograd`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`TestHooks`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`TestNoGrad`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`TestParamInit`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`TestParityWithDDP`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)

### Functions

- **`_dummy_ddp_fn`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_get_device_init_modes`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_get_subtest_config`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_patch_use_unsharded_views`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_register_pre_backward_hooks_with_count`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_test_pre_backward_hook_registration`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_test_unshard_params_as_tensors`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_use_unsharded_views_assert_as_tensors`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_delayed_optim_step`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_delayed_reduce_scatter`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_mixture_of_experts`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_mixture_of_experts_with_delay_before_free`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_nested_always_wrap_model`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_nested_wrapped_model`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_nested_wrapped_model_single_iteration_mixed_precision`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_param_change_after_init`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_pre_backward_hook_registration`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_pre_backward_hook_registration_after_state_dict`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_register_functions_called`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_transformer`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_transformer_no_grad`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`test_unshard_params_as_tensors`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)

### Imports

- **`Any`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`CPUOffload`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`Callable`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`FlatParamHandle`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`ModuleWrapPolicy`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`_p_assert`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`collections.abc`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`contextlib`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`functools`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`instantiate_device_type_tests`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`itertools`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`mock`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`sys`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.distributed`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.distributed.fsdp._flat_param`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.distributed.fsdp.fully_sharded_data_parallel`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.distributed.utils`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.nn`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.testing._internal.common_device_type`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`typing`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)
- **`unittest`**: [test_fsdp_core.py_docs.md](./test_fsdp_core.py_docs.md)


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

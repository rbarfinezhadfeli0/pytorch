# Keyword Index: `torch/testing/_internal/distributed/rpc/jit/rpc_test.py`

## File Information

- **Original File**: [torch/testing/_internal/distributed/rpc/jit/rpc_test.py](../../../../../../../torch/testing/_internal/distributed/rpc/jit/rpc_test.py)
- **Documentation**: [`rpc_test.py_docs.md`](./rpc_test.py_docs.md)
- **Folder**: `torch/testing/_internal/distributed/rpc/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FutureTypingTest`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`JitRpcOpTest`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`JitRpcTest`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`LocalRRefTest`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`MyModuleInterface`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`MyScriptClass`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`MyScriptModule`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`MyScriptModuleWithRRefs`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`RRefAPITest`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`RRefTypingTest`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`and`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)

### Functions

- **`__init__`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`_create_rref`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`assorted_types_args_kwargs`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`async_add`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`async_wrong_decorator_order`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`async_wrong_type`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`call_fork_with_profiling`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`call_rpc_torchscript_with_record_function`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`call_rpc_with_profiling`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`callback`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`construct_my_script_module`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`custom_func`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`forward`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`future_return_to_python`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`future_wait_in_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`get_value`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`list_create`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`load_script_module_with_pickled_rref`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`my_script_module_init`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`no_arg`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`nonexisting_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`one_arg`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`owner_create_rref_my_script_class`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`owner_create_rref_my_script_module`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`python_function`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`python_return_future`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`raise_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`record_function_on_caller_rpc_async`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`return_rref`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`return_value`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rpc_async_call_remote_nonexisting_torchscript_in_torchscript`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rpc_async_call_remote_py_function_in_torchscript`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rpc_async_call_remote_raising_torchscript_in_torchscript`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rpc_return_rref`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_isinstance`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_list_mutate`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_local_value`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_python_annotation`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_script_annotation`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_tensor_is_owner`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`rref_to_here`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`run_ref_script_module`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`save_rref`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_add`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_add_ones`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_add_ones_with_record_function`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_check_rref_confirmed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_fork_wait_throw`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_fork_wait_udf`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_raise_func`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call_with_assorted_types`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call_with_less_args`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call_with_more_args`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call_with_unexpected_kwarg`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call_without_args_kwargs_passed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_async_call_without_kwargs_passed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_remote_call`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rpc_sync_call`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rref_get_value_my_script_class`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_rref_run_forward_my_script_module`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`script_use_future`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`sleep`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_add_done_callback`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_all_kwargs_are_populated_by_defaults`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_args_and_kwargs_contain_different_types`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_args_kwargs_are_neither_passed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_function_remote`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_function_remote_multi`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_function_simple`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_function_wrong_decorator_order`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_function_wrong_return_type`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_function_wrong_return_type_remote`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_script_throw`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_async_script_udf`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_call_fork_in_jit_with_profiling`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_call_python_function_remotely_from_script_not_supported`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_call_rpc_with_profiling`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_call_script_function_that_not_exists_remotely_from_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_call_script_function_that_raises_remotely_from_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_callback_chain`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_callback_simple`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_callback_with_exception`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_create_local_script_class_rref_in_py`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_create_local_script_module_rref_in_py`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_create_script_module_on_remote`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_future_passed_between_python_and_jit`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_future_python_annotation`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_kwargs_not_passed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_less_than_needed_args_are_specified`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_load_script_module_with_pickled_rref`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_local_rref_local_value`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_more_than_needed_args_are_specified`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_my_script_module_with_rrefs`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_no_kwargs_are_populated_by_defaults`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_record_function_jit_end_callbacks_with_fork`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_record_function_on_caller_rpc_async`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_remote_script_module`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_remote_script_throw`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_remote_script_udf`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_return_local_script_class_rref_in_py_and_use_in_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_return_local_script_module_rref_in_py_and_use_in_script`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rpc_async_jit_profiled`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rpc_torchscript_record_function`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rref_as_arg_and_return`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rref_is_owner`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rref_jit_pickle_not_supported`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rref_list_mutate`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rref_local_value`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_rref_python_annotation`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_some_kwargs_are_populated_by_defaults`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_torchscript_function`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_torchscript_function_exception`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_torchscript_functions_not_supported`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_unexepected_kwarg_is_specified`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_user_rrefs_confirmed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`test_user_rrefs_confirmed_remote`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`two_args_two_kwargs`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`use_rref_on_owner`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)

### Imports

- **`Any`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`Future`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`RRef`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`TemporaryFileName`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`Tensor`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`_build_rpc_profiling_key`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`io`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`profile`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`record_function`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`time`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.autograd.profiler`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.autograd.profiler_legacy`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.distributed`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.distributed.rpc`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.distributed.rpc.api`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.distributed.rpc.internal`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.futures`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.testing._internal.common_utils`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.testing._internal.dist_utils`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`torch.testing._internal.distributed.rpc.rpc_agent_test_fixture`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)
- **`typing`**: [rpc_test.py_docs.md](./rpc_test.py_docs.md)


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

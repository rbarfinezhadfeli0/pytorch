# Keyword Index: `torch/testing/_internal/common_fsdp.py`

## File Information

- **Original File**: [torch/testing/_internal/common_fsdp.py](../../../../torch/testing/_internal/common_fsdp.py)
- **Documentation**: [`common_fsdp.py_docs.md`](./common_fsdp.py_docs.md)
- **Folder**: `torch/testing/_internal`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AlwaysWrapNestedWrappedModule`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`DEVICEInitMode`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`DoubleLinear`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`DummyDDP`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`DummyProcessGroup`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`FSDPInitMode`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`FSDPTest`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`FSDPTestModel`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`FSDPTestMultiThread`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`FullyShardMode`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`MLP`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`MLPStack`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`MixtureOfExperts`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`ModuleWithDelay`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`NestedLinear`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`NestedWrappedModule`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`NestedWrappedModuleWithDelay`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`NonUniformReqGradNWM`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`SkipModel`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`SkipModule`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`TransformerWithSharedParams`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`replicated_module`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`that`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`to`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`wraps`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)

### Functions

- **`__init__`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_assert_module_states`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_broadcast_state_dict`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_check_backward_prefetch`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_check_cpu_offload`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_check_forward_prefetch`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_delayed_reduce_scatter`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_delayed_reshard`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_get_state_dict`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_maybe_wrap`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_maybe_wrap_fsdp`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_move_to_device`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_run`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_set_nonuniform_req_grad`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_test_fsdp_parity`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_train_for_several_steps`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`_zero_model`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`allreduce`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`check_sharded_parity`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`compiled_fsdp_test`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`decorator`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`destroy_pg_upon_exit`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`forward`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`fully_shard_with_compiled_compute`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`get_devtype`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`get_full_params`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`get_future`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`get_ignored_modules`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`get_input`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`get_loss`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`init`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`init_method`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`parallelize`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_all_gather`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_all_reduce`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_foreach_all_gather`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_foreach_reduce`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_post_backward`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_reduce_scatter`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_register_post_backward_hook_backward`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_reshard`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`patch_unshard`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`perThreadSetUp`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`perThreadTearDown`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`process_group`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`rank`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`reduce_scatter_with_assert`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`reset_parameters`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`run_backward`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`run_subtests`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`setUp`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`size`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`subtest_name`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`world_size`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`wrapper`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)

### Imports

- **`...`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`ABC`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`Any`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`Callable`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`DeviceMesh`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`DistributedDataParallel`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`NO_RESHARD_AFTER_FORWARD_STRATEGIES`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`ShardedGradScaler`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`TrainingState`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`TransformerDecoderLayer`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`abc`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`always_wrap_policy`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`auto`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`checkpoint`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`collections.abc`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`contextlib`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`copy`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`deepcopy`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`distribute_tensor`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`enum`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`fully_shard`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`functools`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`has_triton`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`mock`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`nullcontext`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`os`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`re`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`sys`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`time`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch..._reshard`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed._composable`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.device_mesh`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp._common_utils`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp._fully_shard._fsdp_param_group`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp._init_utils`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp.fully_sharded_data_parallel`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp.sharded_grad_scaler`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.tensor`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.distributed.tensor.parallel`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.nn`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.nn.functional`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.nn.parallel.distributed`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.testing._internal.common_utils`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`torch.utils._triton`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`typing`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`unittest`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`warnings`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)
- **`wraps`**: [common_fsdp.py_docs.md](./common_fsdp.py_docs.md)


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

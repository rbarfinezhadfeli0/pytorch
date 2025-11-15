# Keyword Index: `test/distributed/fsdp/test_wrap.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_wrap.py](../../../../test/distributed/fsdp/test_wrap.py)
- **Documentation**: [`test_wrap.py_docs.md`](./test_wrap.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BatchNormNet`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`LoraAttention`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`LoraDecoder`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`LoraMLP`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`LoraModel`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`MyModel`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`MyModule`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`Nested`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`NestedSequentialModel`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`TestAutoWrap`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`TestFSDPWrap`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`TestWrapUtils`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`WrapMethod`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`ZeroArguModel`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)

### Functions

- **`__init__`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_get_already_wrapped_fsdp`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_get_linear`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_test_custom_policy`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_test_frozen_params`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_test_transformer_wrapping`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_test_validate_frozen_params`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`forward`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`get_model`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`lambda_fn`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`lambda_fn_nonuniform`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`lambda_fn_uniform`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`never_wrap_policy`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`setUp`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_always_wrap`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_always_wrap_with_ignored_modules`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_api`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_preset_exclude_wrap`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_preset_exclude_wrap_include_children`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_preset_force_leaf`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_preset_force_leaf_custom`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_smoke_test`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_auto_wrap_with_ignored_modules`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_bn_always_wrapped_individually`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_custom_policy`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_error_already_wrapped`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_frozen_params`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_main_wrap_api`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_module_wrap_policy`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_module_wrap_policy_callable`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_transformer_auto_wrap_policy`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_validate_frozen_params`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_wrap`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_wrap_batchnorm_individually`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_wrap_disabled_outside_context`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_wrap_override_defaults`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`test_zero_argument`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`verify_model`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`verify_model_all_wrapped`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`wrap_bn_container`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)

### Imports

- **`Callable`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`TEST_MULTIGPU`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`TransformerDecoderLayer`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`Union`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_BatchNorm`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`_validate_frozen_params`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`auto`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`collections.abc`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`enum`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`functools`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`itertools`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`os`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`tempfile`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.distributed.fsdp._wrap_utils`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.distributed.fsdp.fully_sharded_data_parallel`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.nn`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.nn.functional`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.nn.modules.batchnorm`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`typing`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)
- **`unittest`**: [test_wrap.py_docs.md](./test_wrap.py_docs.md)


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

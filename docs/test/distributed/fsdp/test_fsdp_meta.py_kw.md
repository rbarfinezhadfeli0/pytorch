# Keyword Index: `test/distributed/fsdp/test_fsdp_meta.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_meta.py](../../../../test/distributed/fsdp/test_fsdp_meta.py)
- **Documentation**: [`test_fsdp_meta.py_docs.md`](./test_fsdp_meta.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FakeLinear`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`Model`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`MyBuffer`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`MyLinear`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`MyModel`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`NestedModel`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`TestFSDPWithMetaDevice`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_compare_fsdp`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_init_with_reset_params`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_init_with_torchdistX`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_module_init_fn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_param_init_fn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_reset_params_if_meta`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_test_bad_arg`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_test_nested_model_with_meta_device`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`_test_simple_model_with_meta_device`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`check_fn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`forward`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`meta_module_fn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`process_group`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`reset_parameters`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_bad_arg_meta`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_bad_arg_torchdistx`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_meta_device_with_mixed_precision`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_nested_model_with_meta_device_default_init`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_nested_model_with_meta_device_reset_params`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_nested_model_with_torchdistX_default_init`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_nested_model_with_torchdistX_init_fn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_simple_model_with_meta_device_default_init`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_simple_model_with_meta_device_reset_params`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_simple_model_with_torchdistX_default_init`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`test_simple_model_with_torchdistX_init_fn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`world_size`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)

### Imports

- **`FSDPTest`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`FullyShardedDataParallel`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`Union`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`deferred_init`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`itertools`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`sys`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.distributed`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.nn`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`torchdistx`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)
- **`typing`**: [test_fsdp_meta.py_docs.md](./test_fsdp_meta.py_docs.md)


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

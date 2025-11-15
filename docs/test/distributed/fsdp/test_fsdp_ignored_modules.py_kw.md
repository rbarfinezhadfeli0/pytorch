# Keyword Index: `test/distributed/fsdp/test_fsdp_ignored_modules.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_ignored_modules.py](../../../../test/distributed/fsdp/test_fsdp_ignored_modules.py)
- **Documentation**: [`test_fsdp_ignored_modules.py_docs.md`](./test_fsdp_ignored_modules.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`IgnoredModule`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`Model`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`ModelWithIgnoredModules`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`TestFSDPIgnoredModules`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`_test_diff_ignored_modules_across_ranks`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`_test_ignored_modules_nested`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`_test_ignored_modules_transformer`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`_test_ignored_states_auto_wrap`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`_test_ignored_states_check`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`_train_model`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`forward`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`get_input`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`get_loss`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`run_backward`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_diff_ignored_modules_across_ranks`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_ignored_modules_invalid`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_ignored_modules_nested`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_ignored_modules_not_under_wrapped_root`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_ignored_modules_transformer`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_ignored_states_auto_wrap`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`test_ignored_states_check`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`world_size`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)

### Imports

- **`FullyShardedDataParallel`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`ModuleWrapPolicy`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`distributed`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`functools`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`math`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`skip_if_lt_x_gpu`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`sys`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.distributed.fsdp._traversal_utils`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.nn`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_ignored_modules.py_docs.md](./test_fsdp_ignored_modules.py_docs.md)


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

# Documentation: `docs/test/distributed/fsdp/test_fsdp_mixed_precision.py_kw.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_mixed_precision.py_kw.md`
- **Size**: 10,399 bytes (10.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Keyword Index: `test/distributed/fsdp/test_fsdp_mixed_precision.py`

## File Information

- **Original File**: [test/distributed/fsdp/test_fsdp_mixed_precision.py](../../../../test/distributed/fsdp/test_fsdp_mixed_precision.py)
- **Documentation**: [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- **Folder**: `test/distributed/fsdp`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`BatchNormNet`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`IgnoredModule`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`LinearMixedPrecision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`ModelWithIgnoredModule`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`MyModel`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`NonLearnableConv`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TestFSDPDifferentSubmodulePrecision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TestFSDPMixedPrecision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TestFSDPMixedPrecisionIgnoredModules`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TestFSDPMixedPrecisionSharded`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TestFSDPMixedPrecisionUnsharded`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TestFSDPTrainEval`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`ToyModel`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`ToyModule`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`TransformerWithEMA`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)

### Functions

- **`__init__`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_get_simple_model`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_get_simple_nested_model`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_get_subtest_config`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_get_underlying_module`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_reduce_scatter_validate_mp`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_run_test_mixed_precision_e2e`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_test_grads_reduced_precision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_test_input_grads_with_param_mixed_precision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_test_mixed_precision_embedding_table`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_test_train_ema_eval_flow`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_validate_mp_shard_freed`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_validate_no_mp_shard`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`forward`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`never_wrap_policy`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`patch_reduce_scatter`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_buffer_dtype_no_root_handle`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_eval_root_cast_inputs`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_float16_on_one_submodule`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_float16_on_one_submodule_skip_inputs`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_float16_on_one_submodule_skip_inputs_error`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_full_precision_in_eval`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_full_precision_in_eval_buffers`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_full_precision_in_eval_comm`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_grads_reduced_precision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_input_grads_with_param_mixed_precision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mixed_precision_e2e_full_shard`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mixed_precision_no_reshard_after_forward`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mixed_precision_resnet`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mixed_precision_with_ignored_module`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mp_batchnorm`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mp_embedding_default`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mp_embedding_only_params_and_bufs`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mp_embedding_params_and_reduce_diff`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_mp_embedding_reduce`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_submodules_with_different_precisions`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_submodules_with_different_precisions_error`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_submodules_with_external_inputs`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`test_train_ema_eval_flow`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`verify_eval_buffer_dtype`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`world_size`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)

### Imports

- **`Any`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`AveragedModel`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`ModuleWrapPolicy`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`ShardedGradScaler`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`_BatchNorm`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`contextlib`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`distributed`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`functools`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`itertools`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`os`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`partial`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`product`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`sys`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.cuda.nccl`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.distributed.fsdp`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.distributed.fsdp.sharded_grad_scaler`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.distributed.fsdp.wrap`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.nn`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.nn.functional`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.nn.modules.batchnorm`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.optim.swa_utils`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.testing._internal.common_distributed`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.testing._internal.common_fsdp`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`torchvision`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)
- **`typing`**: [test_fsdp_mixed_precision.py_docs.md](./test_fsdp_mixed_precision.py_docs.md)


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

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/fsdp/test_fsdp_mixed_precision.py_kw.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/fsdp`):

- [`test_fsdp_grad_acc.py_docs.md_docs.md`](./test_fsdp_grad_acc.py_docs.md_docs.md)
- [`test_fsdp_ignored_modules.py_kw.md_docs.md`](./test_fsdp_ignored_modules.py_kw.md_docs.md)
- [`test_fsdp_meta.py_kw.md_docs.md`](./test_fsdp_meta.py_kw.md_docs.md)
- [`test_fsdp_apply.py_docs.md_docs.md`](./test_fsdp_apply.py_docs.md_docs.md)
- [`test_fsdp_tp_integration.py_kw.md_docs.md`](./test_fsdp_tp_integration.py_kw.md_docs.md)
- [`test_fsdp_fx.py_docs.md_docs.md`](./test_fsdp_fx.py_docs.md_docs.md)
- [`test_fsdp_memory.py_kw.md_docs.md`](./test_fsdp_memory.py_kw.md_docs.md)
- [`test_fsdp_apply.py_kw.md_docs.md`](./test_fsdp_apply.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_docs.md_docs.md`](./test_fsdp_tp_integration.py_docs.md_docs.md)
- [`test_fsdp_multiple_forward.py_kw.md_docs.md`](./test_fsdp_multiple_forward.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_mixed_precision.py_kw.md_docs.md`
- **Keyword Index**: `test_fsdp_mixed_precision.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

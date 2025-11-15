# Keyword Index: `test/quantization/core/test_workflow_module.py`

## File Information

- **Original File**: [test/quantization/core/test_workflow_module.py](../../../../test/quantization/core/test_workflow_module.py)
- **Documentation**: [`test_workflow_module.py_docs.md`](./test_workflow_module.py_docs.md)
- **Folder**: `test/quantization/core`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`Model`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestDistributed`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestFakeQuantize`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestFusedModuleScriptable`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestFusedObsFakeQuantModule`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestHistogramObserver`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestObserver`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`TestRecordHistogramObserver`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_ReferenceHistogramObserver`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)

### Functions

- **`__init__`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_compute_quantization_error`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_get_buffer_ids`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_get_norm`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_get_ref_params`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_non_linear_param_search`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`forward`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_compare_fused_obs_fq_oss_module`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_default_fused_qat_config`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_device_affinity`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_dynamic_quant_observer`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_dynamic_quant_observer_matching_choose_qparams`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_embedding_bag_qat_config`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_embedding_qat_config`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fake_quant_preserves_buffers`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fq_module_per_channel`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fq_serializable_per_channel`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fused_mod_per_channel`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fused_mod_reduce_range`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fused_moving_avg_obs_fake_quant`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fused_obs_fq_module`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fused_obs_fq_moving_avg_module`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_fx_qat_convbn_fused_jit_scriptable`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_against_reference`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_consistent_buffer_shape`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_correct_numel`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_extreme_inputs`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_handle_OOM_due_to_close_min_max_value`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_handle_close_to_infinity`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_ignore_infinity`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_one_sided`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_same_inputs`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_save_load_state_dict`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_single_inputs`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_histogram_observer_update_within_range_succeeds`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_observer_qparams_respects_device_affinity`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_observer_scriptable`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_observers_preserve_buffers`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_per_channel_observers`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_per_channel_observers_load_state_dict`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_per_tensor_observers`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_qat_convbn_fused_jit_scriptable`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_qat_convbn_fused_syncbn_replacement`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_qat_data_parallel`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_quant_min_max_override`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_record_observer`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_save_load_state_dict_script`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_state_dict_respects_device_affinity`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_syncbn_preserves_qconfig`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`test_zero_numel`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)

### Imports

- **`TEST_CUDA`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`_get_observer_dict`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`copy`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`given`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`hypothesis`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`io`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`itertools`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`math`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`numpy`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`skipIfTorchDynamo`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.ao.quantization`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.ao.quantization.quantize`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.nn`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.testing._internal.common_cuda`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.testing._internal.common_quantized`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`torch.testing._internal.hypothesis_utils`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)
- **`unittest`**: [test_workflow_module.py_docs.md](./test_workflow_module.py_docs.md)


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

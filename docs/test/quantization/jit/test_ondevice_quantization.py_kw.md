# Keyword Index: `test/quantization/jit/test_ondevice_quantization.py`

## File Information

- **Original File**: [test/quantization/jit/test_ondevice_quantization.py](../../../../test/quantization/jit/test_ondevice_quantization.py)
- **Documentation**: [`test_ondevice_quantization.py_docs.md`](./test_ondevice_quantization.py_docs.md)
- **Folder**: `test/quantization/jit`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MyConvLinearModule`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`OnDevicePTQUtils`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`TestOnDeviceDynamicPTQFinalize`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`TestOnDeviceDynamicPTQInsertObservers`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`TestOnDeviceDynamicPTQInsertQuantDequant`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`myMod`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)

### Functions

- **`__init__`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_against_ref_dynamic_ptq`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_device_side_api`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_num_and_type_of_observers`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_observer_method`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_quant_dequant_and_calc_qparams`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_quantize_forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_quantize_forward_runs`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_quantized_forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_serdes_and_device_side_api_helper`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_check_serialization_deserialization`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_observer_is_weight_only`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_calculate_qparams`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_no_linear_unpack`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_no_observer_forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_packed_params`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_quant_dequant_nodes`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_quantized_forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_validate_setattr_fp_weights`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`find_observer_modules`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`get_example_inputs`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`get_linear_packed_param_fp_weight`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`insert_observers`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`is_calculate_qparam`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`is_per_channel_quantized_packed_param`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`is_value_type_observer`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`ptq_dynamic_quantize`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_against_offdevice_dynamic_ptq`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_device_side_api`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_num_observers`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_num_quant_dequant_nodes`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_observe_method`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_quantize_forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_quantize_forward_runs`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_quantized_forward`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_serialization_deserialization`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`test_weight_only_observers`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)

### Imports

- **`FileCheck`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`TestCase`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`_load_for_lite_interpreter`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`bundled_inputs`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`default_dynamic_qconfig`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`io`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch._C`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.ao.quantization`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.ao.quantization.quantize_jit`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.jit.mobile`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.testing`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)
- **`torch.utils`**: [test_ondevice_quantization.py_docs.md](./test_ondevice_quantization.py_docs.md)


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

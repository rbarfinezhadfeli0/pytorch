# Keyword Index: `torch/ao/quantization/quantization_mappings.py`

## File Information

- **Original File**: [torch/ao/quantization/quantization_mappings.py](../../../../torch/ao/quantization/quantization_mappings.py)
- **Documentation**: [`quantization_mappings.py_docs.md`](./quantization_mappings.py_docs.md)
- **Folder**: `torch/ao/quantization`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`DEFAULT_MODULE_TO_ACT_POST_PROCESS`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`corresponding`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`def`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`is`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`types`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)

### Functions

- **`_get_special_act_post_process`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`_has_special_act_post_process`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_compare_output_module_list`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_dynamic_quant_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_dynamic_sparse_quant_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_float_to_quantized_operator_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_qat_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_qconfig_propagation_list`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_static_quant_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_static_quant_reference_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_default_static_sparse_quant_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_dynamic_quant_module_class`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_embedding_qat_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_embedding_static_quant_module_mappings`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_quantized_operator`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_static_quant_module_class`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`no_observer_set`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)

### Imports

- **`Any`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`Callable`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`DeQuantStub`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`collections.abc`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`copy`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`get_combined_dict`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`nn`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`the`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.intrinsic`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.intrinsic.qat`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.intrinsic.quantized`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.intrinsic.quantized.dynamic`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.qat`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.qat.dynamic`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.quantized`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.quantized.dynamic`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.quantized.reference`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.nn.sparse`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.quantization.fake_quantize`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.quantization.stubs`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.ao.quantization.utils`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.nn.functional`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`torch.nn.utils.parametrize`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`type_before_parametrizations`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)
- **`typing`**: [quantization_mappings.py_docs.md](./quantization_mappings.py_docs.md)


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

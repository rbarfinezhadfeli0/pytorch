# Keyword Index: `torch/ao/quantization/quantizer/xnnpack_quantizer.py`

## File Information

- **Original File**: [torch/ao/quantization/quantizer/xnnpack_quantizer.py](../../../../../torch/ao/quantization/quantizer/xnnpack_quantizer.py)
- **Documentation**: [`xnnpack_quantizer.py_docs.md`](./xnnpack_quantizer.py_docs.md)
- **Folder**: `torch/ao/quantization/quantizer`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`XNNPACKQuantizer`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)

### Functions

- **`__init__`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_annotate_all_dynamic_patterns`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_annotate_all_static_patterns`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_annotate_for_dynamic_quantization_config`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_annotate_for_static_quantization_config`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_dynamo_graph`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_linear_patterns`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_module_type_filter`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_not_module_type_or_name_filter`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_supported_config_and_operators`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_supported_symmetric_config_and_operators`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_supported_symmetric_quantized_operators`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`annotate`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`get_supported_operator_for_quantization_config`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`get_supported_operators`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`get_supported_quantization_configs`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`get_symmetric_quantization_config`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`linear_op`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`module_type_filter`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`not_module_type_or_name_filter`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`set_global`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`set_module_name`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`set_module_type`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`set_operator_type`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`transform_for_annotation`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`validate`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)

### Imports

- **`Any`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`Callable`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`Node`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`QuantizationSpec`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_ObserverOrFakeQuantizeConstructor`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`__future__`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`_get_module_name_filter`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`annotations`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`collections.abc`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`compatibility`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`copy`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`functools`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch._dynamo`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.ao.quantization.fake_quantize`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.ao.quantization.observer`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.ao.quantization.qconfig`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.ao.quantization.quantizer`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.ao.quantization.quantizer.utils`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.ao.quantization.quantizer.xnnpack_quantizer_utils`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.fx`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.fx._compatibility`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`torch.nn.functional`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`typing`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)
- **`typing_extensions`**: [xnnpack_quantizer.py_docs.md](./xnnpack_quantizer.py_docs.md)


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

# Keyword Index: `torch/ao/quantization/fx/_decomposed.py`

## File Information

- **Original File**: [torch/ao/quantization/fx/_decomposed.py](../../../../../torch/ao/quantization/fx/_decomposed.py)
- **Documentation**: [`_decomposed.py_docs.md`](./_decomposed.py_docs.md)
- **Folder**: `torch/ao/quantization/fx`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`FakeQuantPerChannel`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)

### Functions

- **`_choose_qparams_per_token_asymmetric_impl`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`_per_token_quant_qparam_dim_check`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`_permute_to_axis_zero`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`_quant_min_max_bounds_check`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`backward`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_per_token`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_per_token_asymmetric`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_per_token_asymmetric_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_per_token_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_symmetric_tensor`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_symmetric_tensor_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_tensor`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`choose_qparams_tensor_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`convert_element_type`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`convert_element_type_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_channel`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_channel_group`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_channel_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_tensor`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_tensor_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_tensor_tensor`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_tensor_tensor2`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_tensor_tensor2_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_tensor_tensor_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_token`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`dequantize_per_token_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`fake_quant_per_channel`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`fake_quant_per_channel_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`forward`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_channel`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_channel_group`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_channel_group_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_channel_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_tensor`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_tensor_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_tensor_tensor`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_tensor_tensor2`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_tensor_tensor2_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_tensor_tensor_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_token`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`quantize_per_token_meta`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)

### Imports

- **`_unsqueeze_multiple`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`determine_qparams`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`impl`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`math`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`torch`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`torch._refs`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`torch.ao.quantization.utils`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)
- **`torch.library`**: [_decomposed.py_docs.md](./_decomposed.py_docs.md)


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

# Keyword Index: `test/quantization/bc/test_backward_compatibility.py`

## File Information

- **Original File**: [test/quantization/bc/test_backward_compatibility.py](../../../../test/quantization/bc/test_backward_compatibility.py)
- **Documentation**: [`test_backward_compatibility.py_docs.md`](./test_backward_compatibility.py_docs.md)
- **Folder**: `test/quantization/bc`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`LSTMModule`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`Model`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`TestSerialization`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)

### Functions

- **`__init__`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_do_quant_transforms`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_eval_fn`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_get_get_attr_target_strings`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_test_obs`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_test_op`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_test_op_graph`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`_test_package`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`forward`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`get_filenames`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`remove_prefix`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_graph`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_graph_v2`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_graph_v3`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_nobias`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_nobias_graph`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_nobias_graph_v2`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_nobias_graph_v3`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv2d_relu`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv3d`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_conv3d_relu`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_default_qat_qconfig`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_linear`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_linear_dynamic`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_linear_relu`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_linear_relu_package_quantization_transforms`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_lstm`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_per_channel_observer`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`test_per_tensor_observer`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)

### Imports

- **`GraphModule`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`MinMaxObserver`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`os`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`skipIfNoFBGEMM`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`sys`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.ao.nn.intrinsic.quantized`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.ao.nn.quantized`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.ao.nn.quantized.dynamic`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.ao.quantization`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.ao.quantization.quantize_fx`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.fx`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.nn`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.testing._internal.common_quantization`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.testing._internal.common_quantized`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.testing._internal.common_utils`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`torch.testing._internal.quantization_torch_package_models`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)
- **`unittest`**: [test_backward_compatibility.py_docs.md](./test_backward_compatibility.py_docs.md)


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

# Keyword Index: `torch/ao/quantization/quantize_fx.py`

## File Information

- **Original File**: [torch/ao/quantization/quantize_fx.py](../../../../torch/ao/quantization/quantize_fx.py)
- **Documentation**: [`quantize_fx.py_docs.md`](./quantize_fx.py_docs.md)
- **Folder**: `torch/ao/quantization`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`M`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`Submodule`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)

### Functions

- **`__init__`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_attach_meta_to_node_if_not_exist`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_check_is_graph_module`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_convert_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_convert_standalone_module_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_convert_to_reference_decomposed_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_fuse_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_prepare_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_prepare_standalone_module_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_swap_ff_with_fxff`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`attach_preserved_attrs_to_model`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`calibrate`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`convert_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`convert_to_reference_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`forward`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`fuse_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`prepare_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`prepare_qat_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`train_loop`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)

### Imports

- **`.backend_config`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.convert`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.custom_config`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.fuse`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.graph_module`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.prepare`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.tracer`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.fx.utils`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.qconfig_mapping`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`.utils`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`Any`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`BackendConfig`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`ConvertCustomConfig`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`DEPRECATION_WARNING`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`GraphModule`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`ObservedGraphModule`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`QConfigMapping`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`QuantizationTracer`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`_USER_PRESERVED_ATTRIBUTES_KEY`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`convert`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`copy`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`fuse`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`fuse_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`get_default_qat_qconfig_mapping`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`get_default_qconfig_mapping`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`prepare`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`prepare_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`prepare_qat_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`torch`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`torch.ao.quantization`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`torch.ao.quantization.quantize_fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`torch.fx`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`torch.fx.graph_module`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`typing`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`typing_extensions`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)
- **`warnings`**: [quantize_fx.py_docs.md](./quantize_fx.py_docs.md)


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

# Keyword Index: `torch/onnx/_internal/exporter/_onnx_program.py`

## File Information

- **Original File**: [torch/onnx/_internal/exporter/_onnx_program.py](../../../../../torch/onnx/_internal/exporter/_onnx_program.py)
- **Documentation**: [`_onnx_program.py_docs.md`](./_onnx_program.py_docs.md)
- **Folder**: `torch/onnx/_internal/exporter`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ONNXProgram`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`to`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)

### Functions

- **`__call__`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`__init__`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`__repr__`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_convert_complex_to_real_representation`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_count_initializer_size`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_create_value_mapping`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_flatten_inputs`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_from_numpy_array`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_from_ort_value`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_ort_session_initializer`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_process_args`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_remove_none_from_inputs`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_rename_dynamic_axes`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_set_graph_outputs`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_to_numpy_array`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_to_ort_value`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`apply_weights`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`call_reference`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`compute_values`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`initialize_inference_session`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`model_proto`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`optimize`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`release`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`save`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)

### Imports

- **`Any`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`Callable`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`__future__`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_core`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_dynamic_shapes`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`_pytree`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`annotations`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`collections.abc`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`contextlib`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`copy`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`gc`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`import`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`logging`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`ml_dtypes`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`module`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`numpy`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`onnx.reference`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`onnxruntime`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`os`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`tempfile`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`textwrap`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`torch`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`torch.onnx._internal._lazy_import`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`torch.onnx._internal.exporter`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`torch.utils`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`typing`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)
- **`warnings`**: [_onnx_program.py_docs.md](./_onnx_program.py_docs.md)


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

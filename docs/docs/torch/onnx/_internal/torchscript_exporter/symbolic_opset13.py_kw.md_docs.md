# Documentation: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset13.py_kw.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/torchscript_exporter/symbolic_opset13.py_kw.md`
- **Size**: 4,663 bytes (4.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Keyword Index: `torch/onnx/_internal/torchscript_exporter/symbolic_opset13.py`

## File Information

- **Original File**: [torch/onnx/_internal/torchscript_exporter/symbolic_opset13.py](../../../../../torch/onnx/_internal/torchscript_exporter/symbolic_opset13.py)
- **Documentation**: [`symbolic_opset13.py_docs.md`](./symbolic_opset13.py_docs.md)
- **Folder**: `torch/onnx/_internal/torchscript_exporter`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_reduce_op_symbolic`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`_reduce_with_dtype`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`diagonal`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`fake_quantize_per_channel_affine`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`fake_quantize_per_tensor_affine`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`frobenius_norm`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`log_softmax`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`nonzero_numpy`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv1d`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv1d_relu`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv2d`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv2d_relu`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv3d`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv3d_relu`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv_transpose1d`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv_transpose2d`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_conv_transpose3d`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_linear`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`quantized_linear_relu`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`reduce`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`reduce_dim`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`reduce_nodim`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`repeat_interleave`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`softmax`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`split`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`split_with_sizes`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`symbolic`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`tensor_split`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`tile`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`unbind`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`unflatten`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`unsafe_chunk`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`unsafe_split`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`unsafe_split_with_sizes`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`where`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)

### Imports

- **`_constants`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`functools`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`torch`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`torch._C._onnx`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`torch.onnx`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)
- **`torch.onnx._internal.torchscript_exporter`**: [symbolic_opset13.py_docs.md](./symbolic_opset13.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/torchscript_exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/torchscript_exporter`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/onnx/_internal/torchscript_exporter`):

- [`symbolic_opset14.py_docs.md_docs.md`](./symbolic_opset14.py_docs.md_docs.md)
- [`symbolic_opset18.py_kw.md_docs.md`](./symbolic_opset18.py_kw.md_docs.md)
- [`_experimental.py_kw.md_docs.md`](./_experimental.py_kw.md_docs.md)
- [`onnx_proto_utils.py_docs.md_docs.md`](./onnx_proto_utils.py_docs.md_docs.md)
- [`symbolic_opset12.py_docs.md_docs.md`](./symbolic_opset12.py_docs.md_docs.md)
- [`symbolic_opset16.py_docs.md_docs.md`](./symbolic_opset16.py_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`symbolic_helper.py_kw.md_docs.md`](./symbolic_helper.py_kw.md_docs.md)
- [`symbolic_opset8.py_docs.md_docs.md`](./symbolic_opset8.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `symbolic_opset13.py_kw.md_docs.md`
- **Keyword Index**: `symbolic_opset13.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/csrc/jit/tensorexpr/external_functions.cpp_kw.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/external_functions.cpp_kw.md`
- **Size**: 8,034 bytes (7.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/csrc/jit/tensorexpr/external_functions.cpp`

## File Information

- **Original File**: [torch/csrc/jit/tensorexpr/external_functions.cpp](../../../../../torch/csrc/jit/tensorexpr/external_functions.cpp)
- **Documentation**: [`external_functions.cpp_docs.md`](./external_functions.cpp_docs.md)
- **Folder**: `torch/csrc/jit/tensorexpr`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`deduce_memory_format`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`from_blob_quantized`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_adaptive_avg_pool2d`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_addmm`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_conv1d`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_conv1d_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_conv2d`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_dequantize`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_dequantize_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_embedding`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_max_red`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_max_red_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_mean`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantize_per_tensor`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantize_per_tensor_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_add`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_cat`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_conv1d`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_conv1d_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_conv2d`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_conv2d_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_conv2d_relu`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_conv2d_relu_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_linear`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_linear_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_linear_relu`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_mul`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_mul_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_mul_scalar`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_mul_scalar_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_relu`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_sigmoid`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_quantized_sigmoid_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_triangular_solve`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_upsample_nearest2d`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_aten_upsample_nearest2d_out`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_mkldnn_prepacked_conv_run`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_prepacked_conv2d_clamp_run`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`nnc_prepacked_linear_clamp_run`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`quantized_add`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`quantized_cat`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`quantized_mul`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`quantized_mul_scalar`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)

### Includes

- **`ATen/ATen.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/Functions.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/Parallel.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/mkldnn/OpContext.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/quantized/PackedParams.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/quantized/cpu/BinaryOps.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/quantized/cpu/QuantUtils.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/quantized/cpu/QuantizedOps.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/quantized/cpu/conv_serialization.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/native/xnnpack/OpContext.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`ATen/quantized/QTensorImpl.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`c10/core/TensorImpl.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`c10/core/TensorOptions.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`c10/util/ArrayRef.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`c10/util/irange.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`torch/csrc/jit/serialization/import_source.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`torch/csrc/jit/serialization/pickle.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/exceptions.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/external_functions.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`torch/csrc/jit/tensorexpr/external_functions_registry.h`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`utility`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)

### Namespaces

- **`at`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)
- **`torch`**: [external_functions.cpp_docs.md](./external_functions.cpp_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/tensorexpr`):

- [`loopnest.h_kw.md_docs.md`](./loopnest.h_kw.md_docs.md)
- [`expr.h_docs.md_docs.md`](./expr.h_docs.md_docs.md)
- [`block_codegen.h_kw.md_docs.md`](./block_codegen.h_kw.md_docs.md)
- [`ir_cloner.cpp_kw.md_docs.md`](./ir_cloner.cpp_kw.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`tensorexpr_init.h_docs.md_docs.md`](./tensorexpr_init.h_docs.md_docs.md)
- [`lowerings.cpp_kw.md_docs.md`](./lowerings.cpp_kw.md_docs.md)
- [`graph_opt.h_kw.md_docs.md`](./graph_opt.h_kw.md_docs.md)
- [`eval.h_kw.md_docs.md`](./eval.h_kw.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `external_functions.cpp_kw.md_docs.md`
- **Keyword Index**: `external_functions.cpp_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

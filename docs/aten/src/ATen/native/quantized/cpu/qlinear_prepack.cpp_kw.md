# Keyword Index: `aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp](../../../../../../../aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp)
- **Documentation**: [`qlinear_prepack.cpp_docs.md`](./qlinear_prepack.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`QLinearPackWeightFp16`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`QLinearPackWeightFp16Legacy`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`QLinearPackWeightFp16Onednn`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`QLinearPackWeightInt8`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`QLinearPackWeightInt8Legacy`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`QLinearPackWeightInt8Onednn`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)

### Functions

- **`_saturate_weight_to_fp16`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`_wrapped_linear_prepack`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`_wrapped_linear_prepack_meta`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`_wrapped_quantized_linear_prepacked`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`_wrapped_quantized_linear_prepacked_meta`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`calc_col_offsets_transpose`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`if`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`pack_weight_to_fp16_onednn_tensor`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`pack_weight_to_onednn_tensor`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/Functions.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/cpp_custom_type_hack.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/mkldnn/MKLDNNCommon.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/PackedParams.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/cpu/ACLUtils.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/cpu/OnednnUtils.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/cpu/QnnpackUtils.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/cpu/QuantUtils.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/cpu/fbgemm_utils.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/cpu/init_qnnpack.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/native/quantized/library.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/_saturate_weight_to_fp16.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/_saturate_weight_to_fp16_native.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/_wrapped_linear_prepack_native.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/_wrapped_quantized_linear_prepacked_native.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/dequantize.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/empty.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/quantize_per_tensor.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`ATen/quantized/Quantizer.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`algorithm`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`c10/util/irange.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`torch/custom_class.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`torch/library.h`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`utility`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`vector`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)

### Namespaces

- **`at`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)
- **`c10`**: [qlinear_prepack.cpp_docs.md](./qlinear_prepack.cpp_docs.md)


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

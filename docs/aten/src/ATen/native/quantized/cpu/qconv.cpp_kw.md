# Keyword Index: `aten/src/ATen/native/quantized/cpu/qconv.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/qconv.cpp](../../../../../../../aten/src/ATen/native/quantized/cpu/qconv.cpp)
- **Documentation**: [`qconv.cpp_docs.md`](./qconv.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`QConv1dInt8`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`QConvAddInt8`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`QConvInt8`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`QConvInt8ForBC`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)

### Functions

- **`ConvDimChecks`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`_fp8_convolution_onednn_ref`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`_quantized_convolution_onednn`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`can_use_xnnp`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`compute_deconv_shape`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`constexpr`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`if`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`run`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/Functions.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/Parallel.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/SmallVector.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/core/List.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/mkldnn/MKLDNNCommon.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/ConvUtils.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/PackedParams.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/cpu/OnednnUtils.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/cpu/QnnpackUtils.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/cpu/QuantUtils.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/cpu/XnnpackUtils.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/cpu/fbgemm_utils.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/native/quantized/cpu/qconv.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/_empty_affine_quantized.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/_empty_affine_quantized_native.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/_empty_per_channel_affine_quantized_native.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/convolution.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/empty.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/hardswish.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/hardtanh.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/leaky_relu.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/linear.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/quantize_per_channel_native.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/quantize_per_tensor_native.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/relu.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/sigmoid.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/tanh.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`ATen/quantized/Quantizer.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`algorithm`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`c10/util/irange.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`caffe2/utils/threadpool/pthreadpool-cpp.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`cmath`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`string`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`torch/library.h`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`vector`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)

### Namespaces

- **`at`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)
- **`template`**: [qconv.cpp_docs.md](./qconv.cpp_docs.md)


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

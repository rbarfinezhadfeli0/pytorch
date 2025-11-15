# Keyword Index: `aten/src/ATen/native/vulkan/ops/Convolution.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/vulkan/ops/Convolution.cpp](../../../../../../../aten/src/ATen/native/vulkan/ops/Convolution.cpp)
- **Documentation**: [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/vulkan/ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Block`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`Params`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`QParams`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)

### Functions

- **`available`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`bias_valid`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`conv2d_clamp_run`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`convolution`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`convolution1d`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`determine_method`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`get_shader`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`if`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`is_depthwise`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`is_pointwise`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`pack_biases`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`pack_weights`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`pack_weights_using_width_packing`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`rearrange_bias`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`rearrange_weights_2d`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`rearrange_weights_dw`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`record_op`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`record_quantized_op`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`run_conv1d_context`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`run_conv1d_context_impl`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`run_conv2d_context`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`run_conv2d_context_impl`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`run_qconv2d_context`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`run_tconv2d_context`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`usable`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`weight_valid`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/Functions.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/ConvUtils.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/utils/ParamUtils.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/vulkan/api/Utils.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/vulkan/impl/Packing.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/vulkan/ops/Common.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/vulkan/ops/Convolution.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/vulkan/ops/Copy.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/native/vulkan/ops/Utils.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/ops/dequantize.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/ops/pad.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/ops/permute.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/ops/quantize_per_tensor.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`c10/util/irange.h`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)

### Namespaces

- **`api`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`at`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`conv1d`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`conv2d`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`namespace`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`native`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`ops`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)
- **`vulkan`**: [Convolution.cpp_docs.md](./Convolution.cpp_docs.md)


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

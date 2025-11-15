# Keyword Index: `aten/src/ATen/native/quantized/cpu/qlinear.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/qlinear.cpp](../../../../../../../aten/src/ATen/native/quantized/cpu/qlinear.cpp)
- **Documentation**: [`qlinear.cpp_docs.md`](./qlinear.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`QLinearInt8`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`QLinearInt8FusedQDQ`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`QLinearLeakyReluInt8`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`QLinearOnednn`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`QLinearTanhInt8`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)

### Functions

- **`_weight_int4pack_mm_cpu_tensor`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`can_use_xnnp`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`fp8_qlinear_onednn_ref`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`if`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`linear_int8_with_onednn_weight`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`run`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`run_pointwise`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`run_pointwise_binary`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`values`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)

### Includes

- **`ATen/Context.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/Functions.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/Parallel.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/core/List.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/mkldnn/MKLDNNCommon.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/PackedParams.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/ACLUtils.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/OnednnUtils.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/QnnpackUtils.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/QuantUtils.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/XnnpackUtils.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/fbgemm_utils.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/cpu/qlinear.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/native/quantized/library.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/_empty_affine_quantized.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/_empty_affine_quantized_native.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/_weight_int4pack_mm_for_cpu.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/empty.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/gelu.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/hardswish.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/hardtanh.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/leaky_relu.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/linear.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/quantize_per_channel_native.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/quantize_per_tensor_native.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/relu.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/sigmoid.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/tanh.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`ATen/ops/zeros.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`algorithm`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`c10/util/irange.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`caffe2/utils/threadpool/pthreadpool-cpp.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`string`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)
- **`torch/library.h`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)

### Namespaces

- **`at`**: [qlinear.cpp_docs.md](./qlinear.cpp_docs.md)


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

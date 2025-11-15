# Keyword Index: `aten/src/ATen/native/Activation.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/Activation.cpp](../../../../../aten/src/ATen/native/Activation.cpp)
- **Documentation**: [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- **Folder**: `aten/src/ATen/native`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_prelu_kernel`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`_rrelu_with_noise_train`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`celu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`hardswish`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`hardswish_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`hardtanh`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`hardtanh_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`infinitely_differentiable_gelu_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`log_sigmoid`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`log_sigmoid_backward_cpu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`log_sigmoid_backward_cuda`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`math_mish_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`math_silu_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`mish_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`prelu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`relu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`relu6`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`rrelu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`rrelu_with_noise_backward`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`rrelu_with_noise_cpu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`selu`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`softshrink_check`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`use_mkldnn`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/Functions.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/OpMathType.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/Parallel.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ScalarOps.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/TensorIterator.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/core/DistributionsHelper.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/native/Activation.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/native/mkldnn/MKLDNNCommon.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/native/mkldnn/Utils.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/native/xnnpack/Engine.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/_prelu_kernel.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/_prelu_kernel_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/_prelu_kernel_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/celu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/clamp.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/clamp_min.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/elu.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/elu_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/elu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/empty.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/empty_like.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/gelu_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/gelu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardshrink_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardshrink_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardsigmoid_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardsigmoid_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardswish_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardswish_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardtanh.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardtanh_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/hardtanh_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/infinitely_differentiable_gelu_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/leaky_relu.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/leaky_relu_backward.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/leaky_relu_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/leaky_relu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/log_sigmoid_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/log_sigmoid_forward.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/log_sigmoid_forward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/log_sigmoid_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/mish_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/mish_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/prelu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/relu6_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/relu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/rrelu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/rrelu_with_noise.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/rrelu_with_noise_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/rrelu_with_noise_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/selu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/sigmoid.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/silu_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/silu_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/softplus.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/softplus_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/softplus_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/softshrink_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/softshrink_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/tanh.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/threshold_backward_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`ATen/ops/threshold_native.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`c10/core/ScalarType.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`c10/util/irange.h`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)
- **`utility`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)

### Namespaces

- **`at`**: [Activation.cpp_docs.md](./Activation.cpp_docs.md)


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

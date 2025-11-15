# Keyword Index: `aten/src/ATen/native/RNN.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/RNN.cpp](../../../../../aten/src/ATen/native/RNN.cpp)
- **Documentation**: [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- **Folder**: `aten/src/ATen/native`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`BidirLayerT`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`Cell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`CellParams`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`CellParamsBase`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`FullBidirectionalLayer`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`FullLayer`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`GRUCell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`LSTMCell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`Layer`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`LayerOutput`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`LayerT`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`PackedBidirectionalLayer`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`PackedLayer`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`PackedSequence`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`QRNNCellParamsWrapper`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`QuantizedCellParams`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`QuantizedCellParamsDynamic`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`QuantizedCellParamsFP16`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ReversedPackedLayer`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`SimpleCell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`in`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`of`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`only`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`relu_f`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`so`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`tanh_f`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)

### Functions

- **`_use_cudnn_rnn_flatten_weight`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`check_rnn_cell_forward_hidden`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`check_rnn_cell_forward_input`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`dropout`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`gru_cell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`hidden_as_output`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`hidden_concat`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`hidden_slice`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`linear_hh`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`linear_ih`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`matmul_hh`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`matmul_hr`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`matmul_ih`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`name`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`prepare_quantized_hx`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`rnn_relu_cell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`rnn_tanh_cell`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`use_cudnn`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`use_miopen`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`use_mkldnn`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)

### Includes

- **`ATen/Config.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/Context.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/Functions.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/NativeFunctions.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/TensorOperators.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/core/List.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/mps/MPSDevice.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/native/RNN.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/native/mkldnn/Utils.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/native/quantized/PackedParams.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/native/quantized/cpu/QnnpackUtils.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/native/quantized/cpu/fbgemm_utils.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/native/quantized/library.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_lstm_mps.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_differentiable_gru_cell_backward_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_differentiable_lstm_cell_backward_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_fused_gru_cell.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_fused_lstm_cell.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_fused_lstm_cell_backward.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_fused_lstm_cell_backward_impl.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_thnn_fused_lstm_cell_backward_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/_use_cudnn_rnn_flatten_weight_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/cat.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/cudnn_is_acceptable.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/dropout.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/fbgemm_linear_int8_weight_fp32_activation.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/fbgemm_linear_quantize_weight_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/fbgemm_pack_quantized_matrix_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/gru_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/gru_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/linear.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/lstm_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/lstm_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/matmul.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/quantized_gru_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/quantized_lstm_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/quantized_rnn_relu_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/quantized_rnn_tanh_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/relu.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/rnn_relu_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/rnn_relu_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/rnn_tanh_cell_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/rnn_tanh_native.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/sigmoid_backward.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/stack.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/tanh.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/tanh_backward.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/zeros_like.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`ATen/ops/zeros_like_ops.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`c10/core/GradMode.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`c10/macros/Macros.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`c10/util/irange.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`torch/custom_class.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`torch/library.h`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`utility`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)

### Namespaces

- **`at`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)
- **`bool`**: [RNN.cpp_docs.md](./RNN.cpp_docs.md)


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

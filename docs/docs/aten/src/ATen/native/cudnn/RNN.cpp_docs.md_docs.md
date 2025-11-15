# Documentation: `docs/aten/src/ATen/native/cudnn/RNN.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cudnn/RNN.cpp_docs.md`
- **Size**: 53,248 bytes (52.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cudnn/RNN.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cudnn/RNN.cpp`
- **Size**: 88,540 bytes (86.46 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/MatrixRef.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/RNN.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cudnn_init_dropout_state.h>
#include <ATen/ops/_cudnn_init_dropout_state_native.h>
#include <ATen/ops/_cudnn_rnn.h>
#include <ATen/ops/_cudnn_rnn_backward_native.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_native.h>
#include <ATen/ops/_cudnn_rnn_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// See Note [ATen preprocessor philosophy]

Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    bool fn_bidirectional) {
  TORCH_CHECK(
      false, "_cudnn_rnn_flatten_weight: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const std::optional<Tensor>& weight_buf_r_opt,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const std::optional<Tensor>& fn_dropout_state_opt) {
  TORCH_CHECK(false, "_cudnn_rnn: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _cudnn_rnn_backward(
    const Tensor& input,
    TensorList weight,
    int64_t weight_stride0,
    const Tensor& weight_buf,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    const Tensor& output,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    IntArrayRef batch_sizes,
    const std::optional<Tensor>& dropout_state_opt,
    const Tensor& reserve,
    std::array<bool, 4> output_mask) {
  TORCH_CHECK(
      false, "_cudnn_rnn_backward: ATen not compiled with cuDNN support");
}

Tensor _cudnn_init_dropout_state(
    double dropout,
    bool train,
    int64_t dropout_seed,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(
      false, "_cudnn_init_dropout_state: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/RNNUtils.h>

namespace at {
namespace native {

namespace {
// DropoutDescriptor

struct DropoutDescriptorParams {
  bool train;
  double dropout;
  Tensor dropout_state;
  DropoutDescriptorParams() = default;
  void set(bool train_, double dropout_, Tensor dropout_state_) {
    train = train_;
    dropout = dropout_;
    dropout_state = dropout_state_;
  }
  DropoutDescriptor descriptor(cudnnHandle_t handle) const {
    auto dropout_p = train ? dropout : 0;
    DropoutDescriptor dropout_desc;
    if (dropout_p == 0) {
      dropout_desc.set_no_dropout(handle);
    } else {
      dropout_desc.set(handle, dropout_p, dropout_state);
    }
    return dropout_desc;
  }
};

// RNNDescriptor

struct RNNDescriptorParams {
#ifdef USE_CUDNN_RNN_V8_API
  int64_t input_size;
  bool packed;
#endif
  int64_t hidden_size;
  int64_t proj_size;
  int64_t num_layers;
  cudnnDirectionMode_t bidirectional;
  cudnnRNNMode_t mode;
  cudnnDataType_t datatype;
  cudnnDataType_t input_datatype;
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;

  int64_t num_directions() const {
    return bidirectional ? 2 : 1;
  }

  void set_mode(int64_t fn_mode) {
    switch (fn_mode) {
      case CUDNN_RNN_RELU:
        mode = CUDNN_RNN_RELU;
        break;
      case CUDNN_RNN_TANH:
        mode = CUDNN_RNN_TANH;
        break;
      case CUDNN_LSTM:
        mode = CUDNN_LSTM;
        break;
      case CUDNN_GRU:
        mode = CUDNN_GRU;
        break;
      default: {
        std::ostringstream oss;
        oss << "unrecognized cuDNN RNN mode " << fn_mode;
        TORCH_CHECK(false, oss.str());
      }
    }
  }

  void set_bidirectional(bool fn_bidirectional) {
    bidirectional =
        fn_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  }

  void set_algo(cudnnRNNAlgo_t algo) {
    this->algo = algo;
  }

#ifndef USE_CUDNN_RNN_V8_API
  void set(
      int64_t mode,
      int64_t hidden_size,
      int64_t proj_size,
      int64_t num_layers,
      bool bidirectional,
      cudnnDataType_t datatype,
      cudnnDataType_t input_datatype){
#else
  void set(
      int64_t mode,
      int64_t input_size,
      bool packed,
      int64_t hidden_size,
      int64_t proj_size,
      int64_t num_layers,
      bool bidirectional,
      cudnnDataType_t datatype,
      cudnnDataType_t input_datatype) {
#endif
      this -> set_mode(mode);
#ifdef USE_CUDNN_RNN_V8_API
  this->input_size = input_size;
  this->packed = packed;
#endif
  this->hidden_size = hidden_size;
  this->proj_size = proj_size;
  this->num_layers = num_layers;
  this->set_bidirectional(bidirectional);
  this->datatype = datatype;
  this->input_datatype = input_datatype;
}

RNNDescriptor
descriptor(cudnnHandle_t handle, DropoutDescriptor&& dropout_desc) const {
  RNNDescriptor rnn_desc;
#ifndef USE_CUDNN_RNN_V8_API
  rnn_desc.set(
      handle,
      hidden_size,
      proj_size,
      num_layers,
      std::move(dropout_desc),
      input_mode,
      bidirectional,
      mode,
      datatype,
      input_datatype,
      algo,
      at::globalContext().allowTF32CuDNN(at::Float32Op::RNN));
#else
    rnn_desc.set(
        handle,
        input_size,
        packed,
        hidden_size,
        proj_size,
        num_layers,
        std::move(dropout_desc),
        input_mode,
        bidirectional,
        mode,
        datatype,
        input_datatype,
        algo,
        at::globalContext().allowTF32CuDNN(at::Float32Op::RNN));
#endif
  return rnn_desc;
}

// In some cases, a use of RNNDescriptor does not rely on the
// DropoutDescriptor.  In this case, we fake up a no-dropout
// descriptor to make the RNN descriptor initialization go through.
// This is used by _cudnn_rnn_flatten_weight, which needs an
// RNNDescriptor for get_parameters(), but does not actually need
// a fully initialized dropout descriptor.  This lets us avoid
// having to pass the dropout state to flatten, which has no business
// knowing what the dropout state is.
RNNDescriptor descriptor(cudnnHandle_t handle) const {
  DropoutDescriptor dropout_desc;
  dropout_desc.set_no_dropout(handle);
  return descriptor(handle, std::move(dropout_desc));
}
}; // namespace

// TensorDescriptor list
#ifndef USE_CUDNN_RNN_V8_API
std::vector<TensorDescriptor> rnn_descriptor_sequence(
    const Tensor& tensor,
    IntArrayRef batch_sizes) {
  std::vector<TensorDescriptor> descriptors(batch_sizes.size());
  size_t i = 0;
  // To be mutated in the loop
  auto batch_tensor_size = tensor.sizes().vec();
  for (auto batch_size : batch_sizes) {
    batch_tensor_size[0] = batch_size;
    // NB: cuDNN RNN API does not support 2d descriptors, so we
    // must pad it out to 3d.
    descriptors[i].set(
        getCudnnDataType(tensor), batch_tensor_size, tensor.strides(), 3);
    i++;
  }
  return descriptors;
}

std::vector<TensorDescriptor> rnn_descriptor(const Tensor& tensor, int64_t N) {
  std::vector<TensorDescriptor> descriptors(N);
  for (const auto i : c10::irange(N)) {
    descriptors[i].set(tensor, 5);
  }
  return descriptors;
}
#else
auto rnn_descriptor_sequence(
    const Tensor& tensor,
    uint32_t batch_size,
    const IntArrayRef batch_sizes,
    uint32_t seq_len,
    uint32_t vector_size) { // packed case
  RNNDataDescriptor r;
  std::vector<int> seqLengthArray(batch_size, 1);
  // cuDNN wants the sequence lengths for a packed batch as if they
  // were unpacked, e.g., for the
  // Sequence 1: ABCD
  // Sequence 2: EF
  // Sequence 3: G
  // case below, this would be [4, 2, 1] (has length == mini_batch)
  // TODO(eqy): There's probably a smarter way to do this than O(SN)
  for (auto it = batch_sizes.begin(); it != batch_sizes.end(); it++) {
    // everyone starts at sequence length 1 so we skip an iteration
    if (it == batch_sizes.begin()) {
      continue;
    }
    for (const auto idx : c10::irange(*it)) {
      seqLengthArray[idx]++;
    }
  }
  r.set(
      tensor,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      seq_len,
      batch_size,
      vector_size,
      seqLengthArray.data());
  return r;
}

auto rnn_descriptor(
    const Tensor& tensor,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t vector_size) {
  RNNDataDescriptor r;
  // NB: Looks like even if batch_first is true here we always want
  // SEQ_MAJOR_UNPACKED, because the input appears to be transposed if it is
  // barch-major
  const auto layout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
  std::vector<int32_t> seqLengthArray(batch_size, seq_len);
  r.set(
      tensor, layout, seq_len, batch_size, vector_size, seqLengthArray.data());
  return r;
}
#endif

// The best way to understand the meaning of the values stored in
// this struct is to consider each of the possible ways our
// input can be structured.
//
// Suppose you want to run RNN on the following variable
// length inputs:
//
//    Sequence 1: ABCD
//    Sequence 2: EF
//    Sequence 3: G
//
// (Let _ be padding when we have non-packed representations.)
//
// # Packed input (batch_sizes is non-empty)
//
//  input_size
// +------+                    +
// | A    |                    |
// | E    | mini_batch =       |
// | G    | batch_sizes[0] = 3 |
// +------+                    |
// | B    |                    | batch_sizes_sum = 7
// | F    | batch_sizes[1] = 2 |
// +------+                    |
// | C    | batch_sizes[2] = 1 |
// +------+                    |
// | D    | batch_sizes[3] = 1 |
// +------+                    +
//
//              (seq_length = 4)
//
//    input.size() = batch_sizes_sum x input_size
//
// # Unpacked input (batch_first = false)
//
//  mini_batch = 3
// +-------+
// | A E G |
// | B F _ | seq_length = 4
// | C _ _ |
// | D _ _ |
// +-------+
//    ...    input_size
// +-------+
//
//    input.size() = seq_length x mini_batch x input_size
//
// # Unpacked input (batch_first = true)
//
//  seq_length = 4
// +---------+
// | A B C D |
// | E F _ _ | mini_batch = 3
// | G _ _ _ |
// +---------+
//     ...     input_size
// +---------+
//
//    input.size() = mini_batch x seq_length x input_size
//
struct TensorDescriptorListParams {
  IntArrayRef batch_sizes;
  int64_t seq_length;
  int64_t mini_batch;
  // NB: this is not input.size(), which is an IntArrayRef; instead, this
  // size of the inner-most dimension.  In NL applications, this is usually
  // the size of the embedding.  You can also think of this as the size
  // of the "channel" dimension (at risk of confusing vision researchers :)
  int64_t input_size;
  // Only valid when !is_input_packed
  int64_t batch_sizes_sum; // == sum(batch_sizes)

  [[nodiscard]] bool is_input_packed() const {
    return !batch_sizes.empty();
  }

  void set(
      IntArrayRef input_sizes,
      IntArrayRef batch_sizes_,
      bool batch_first) {
    batch_sizes = batch_sizes_;
    if (is_input_packed()) {
      seq_length = batch_sizes.size();
      mini_batch = batch_sizes[0];
      // NB: When input is packed, the mini_batch size is NOT the size
      // of the outer dimension
      batch_sizes_sum = input_sizes[0];
      input_size = input_sizes[1];
    } else {
      if (batch_first) {
        seq_length = input_sizes[1];
        mini_batch = input_sizes[0];
      } else {
        seq_length = input_sizes[0];
        mini_batch = input_sizes[1];
      }
      input_size = input_sizes[2];
      // TODO: Actually, would this make ASAN's job harder catching
      // an uninitialized access?
      batch_sizes_sum = -1; // something bogus in case we access it
    }
  }
#ifndef USE_CUDNN_RNN_V8_API
  // TODO: check x for consistency with input_size?
  std::vector<TensorDescriptor> descriptors(Tensor x) const {
    if (is_input_packed()) {
      return rnn_descriptor_sequence(x, batch_sizes);
    } else {
      return rnn_descriptor(x[0], seq_length);
    }
  }
#else
  auto descriptors(Tensor x) const {
    if (is_input_packed()) {
      return rnn_descriptor_sequence(
          x, mini_batch, batch_sizes, seq_length, x.size(-1));
    } else {
      return rnn_descriptor(x, mini_batch, seq_length, x.size(-1));
    }
  }
#endif
};

// Everything together

struct RNNParams {
  DropoutDescriptorParams dropout;
  RNNDescriptorParams rnn;
  TensorDescriptorListParams tensors;
};

// NB: Doesn't include the weight descriptor
struct RNNDescriptors {
  RNNDescriptor rnn_desc;
  // NB: this won't actually lay out the tensor descriptor pointers
  // in the right way, so you'll have to preprocess them
#ifndef USE_CUDNN_RNN_V8_API
  std::vector<TensorDescriptor> x_descs;
  std::vector<TensorDescriptor> y_descs;
#else
  RNNDataDescriptor x_descs;
  RNNDataDescriptor y_descs;
#endif
  TensorDescriptor hx_desc;
  TensorDescriptor hy_desc;
  TensorDescriptor cx_desc;
  TensorDescriptor cy_desc;

  RNNDescriptors(
      const RNNParams& fn,
      cudnnHandle_t handle,
      Tensor x,
      Tensor y,
      Tensor hx,
      Tensor cx) {
    rnn_desc = fn.rnn.descriptor(handle, fn.dropout.descriptor(handle));
    x_descs = fn.tensors.descriptors(x);
    y_descs = fn.tensors.descriptors(y);
    hx_desc.set(hx, 5);
    hy_desc.set(hx, 5);
    if (cx.defined()) {
      cx_desc.set(cx, 5);
      cy_desc.set(cx, 5);
    }
  }

  // TODO: This is annoying, having to put the cudnnTensorDescriptor_t
  // in a contiguous array...
  std::vector<cudnnTensorDescriptor_t> get_descs(
      const std::vector<TensorDescriptor>& descs) {
    std::vector<cudnnTensorDescriptor_t> r;
    r.reserve(descs.size());
    for (auto& desc : descs) {
      r.emplace_back(desc.desc());
    }
    return r;
  }
#ifndef USE_CUDNN_RNN_V8_API
  std::vector<cudnnTensorDescriptor_t> get_x_descs() {
    return get_descs(x_descs);
  }

  std::vector<cudnnTensorDescriptor_t> get_y_descs() {
    return get_descs(y_descs);
  }
#endif
};

int64_t get_num_weights(
    cudnnHandle_t handle,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
#endif
    cudnnDataType_t datatype) {
  size_t weight_size;
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetRNNParamsSize(
      handle, rnn_desc.desc(), x_desc.desc(), &weight_size, datatype));
#else
  AT_CUDNN_CHECK(
      cudnnGetRNNWeightSpaceSize(handle, rnn_desc.desc(), &weight_size));
#endif
  auto elem_size = dataSize(datatype);
  TORCH_INTERNAL_ASSERT(
      weight_size % elem_size == 0,
      "cudnnGetRNNParamsSize returned nonsensical weight_size");
  return weight_size / elem_size;
}

int64_t _num_linear_layers(cudnnRNNMode_t mode) {
  switch (mode) {
    case CUDNN_LSTM:
      return 8;
    case CUDNN_GRU:
      return 6;
    case CUDNN_RNN_RELU:
      return 2;
    case CUDNN_RNN_TANH:
      return 2;
    default:
      TORCH_CHECK(false, "unknown cuDNN RNN mode ", mode);
  }
}

void add_projection_weights(
    cudnnHandle_t handle,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
    const FilterDescriptor& w_desc,
#endif
    const Tensor& weight_buf,
    int64_t layer,
    std::vector<Tensor>& params) {
  void* matrix_pointer = nullptr;
  // assuming it's LSTM which has 8 "linear layers" (i.e. 4 weights and 4
  // biases)
  int64_t linear_id = 8;
#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor lin_layer_mat_desc;
  AT_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
      /*handle=*/handle,
      /*rnnDesc=*/rnn_desc.desc(),
      /*layer=*/layer,
      /*xDesc=*/x_desc.desc(),
      /*wDesc=*/w_desc.desc(),
      /*w=*/weight_buf.data_ptr(),
      /*linLayerID=*/linear_id,
      /*linLayerMatDesc=*/lin_layer_mat_desc.mut_desc(),
      /*linLayerMat=*/&matrix_pointer));
#else
  TensorDescriptor lin_layer_mat_desc;
  AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
      /*handle=*/handle,
      /*rnnDesc=*/rnn_desc.desc(),
      /*layer=*/layer,
      /*wDesc=*/weight_buf.numel() * weight_buf.element_size(),
      /*w=*/weight_buf.data_ptr(),
      /*linLayerID=*/linear_id,
      /*linLayerMatDesc=*/lin_layer_mat_desc.mut_desc(),
      /*linLayerMat=*/&matrix_pointer,
      nullptr,
      nullptr));
#endif

  cudnnDataType_t data_type;
#ifndef USE_CUDNN_RNN_V8_API
  cudnnTensorFormat_t format;
#else
  int stride_dim_a[5];
#endif
  int nb_dims;
  constexpr int min_dim = 3;
  int filter_dim_a[min_dim];
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
      lin_layer_mat_desc.desc(),
      min_dim,
      &data_type,
      &format,
      &nb_dims,
      filter_dim_a));
#else
  AT_CUDNN_CHECK(cudnnGetTensorNdDescriptor(
      lin_layer_mat_desc.desc(),
      min_dim,
      &data_type,
      &nb_dims,
      filter_dim_a,
      stride_dim_a));
#endif

  TORCH_INTERNAL_ASSERT(
      nb_dims <= min_dim, "nb_dims = ", nb_dims, "; min_dim  = ", min_dim);
  auto elem_size = dataSize(getCudnnDataType(weight_buf));
  auto offset_bytes = static_cast<const char*>(matrix_pointer) -
      static_cast<const char*>(weight_buf.data_ptr());
  TORCH_INTERNAL_ASSERT(
      offset_bytes % elem_size == 0,
      "offset_bytes = ",
      offset_bytes,
      "; elem_size = ",
      elem_size);
  size_t offset = offset_bytes / elem_size;

  int mat_numel = c10::multiply_integers(filter_dim_a, filter_dim_a + nb_dims);
  // Generate a new parameter tensor which is a view into the weight_buf.
  std::initializer_list<int64_t> size = {mat_numel, 1};
  Tensor param = at::empty({0}, weight_buf.options())
                     .set_(weight_buf.storage(), offset, size);
  params.emplace_back(std::move(param));
}

/*
  Returns weight and bias tensors for each layer of the RNN. These tensors
  are views on the underlying weight buffer allocated by CuDNN.

  Note: for LSTM and GRU, which have multiple parameters of each type (4 and 3,
  respectively), these parameters are concatenated along the first dimension.
        These parameters are returned in a consistent order by CuDNN:
            (reset, forget, cell, output) for LSTM
            (reset, input, new) for GRU
  Args:
      fn: The RNN function object holding the RNN state
      handle: a CuDNN handle
      weight_buf: a 1D tensor containing the CuDNN-allocated weight (or
  grad_weight) buffer Returns: parameters: [(weight_ih, weight_hh, bias_ih,
  bias_hh)*], with length equal to the num_layers. This is represented as a pair
  of vector, and outer-dimension stride (NB: Can't return MatrixRef because we
  need to allocate the underlying tensor)
*/
std::pair<std::vector<Tensor>, size_t> // stride0
get_parameters(
    cudnnHandle_t handle,
    const RNNDescriptorParams& rnn,
    const RNNDescriptor& rnn_desc,
#ifndef USE_CUDNN_RNN_V8_API
    const TensorDescriptor& x_desc,
    const FilterDescriptor& w_desc,
#endif
    const Tensor& weight_buf,
    bool include_bias = true) {
#ifndef USE_CUDNN_RNN_V8_API
  auto cudnn_methods = {
      cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams};
#else
  auto cudnn_methods = {true, false};
#endif
  std::vector<Tensor> params;
  int64_t num_linear_layers = _num_linear_layers(rnn.mode);
  int64_t num_layers = rnn.num_directions() * rnn.num_layers;
  size_t cur_offset = 0;
  size_t global_layer_params_count = 0;
  for (const auto layer : c10::irange(num_layers)) {
    size_t layer_params_count = 0;
    for (auto cudnn_method : cudnn_methods) {
      for (const auto linear_id : c10::irange(num_linear_layers)) {
        void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
        FilterDescriptor lin_layer_mat_desc;
        AT_CUDNN_CHECK(cudnn_method(
            handle,
            rnn_desc.desc(),
            layer,
            x_desc.desc(),
            w_desc.desc(),
            weight_buf.data_ptr(),
            linear_id,
            lin_layer_mat_desc.mut_desc(),
            &matrix_pointer));
#else
        TensorDescriptor lin_layer_mat_desc;
        for (int stateless = 0; stateless < 100; stateless++) {
          if (cudnn_method) { // matrix
            AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
                handle,
                rnn_desc.desc(),
                layer,
                weight_buf.numel() * weight_buf.element_size(),
                weight_buf.data_ptr(),
                linear_id,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer,
                nullptr,
                nullptr));
          } else { // bias
            AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
                handle,
                rnn_desc.desc(),
                layer,
                weight_buf.numel() * weight_buf.element_size(),
                weight_buf.data_ptr(),
                linear_id,
                nullptr,
                nullptr,
                lin_layer_mat_desc.mut_desc(),
                &matrix_pointer));
          }
        }
#endif
        cudnnDataType_t data_type;
#ifndef USE_CUDNN_RNN_V8_API
        cudnnTensorFormat_t format;
#else
        int stride_dim_a[5];
#endif
        int nb_dims;
        constexpr int min_dim = 3;
        int filter_dim_a[min_dim];
#ifndef USE_CUDNN_RNN_V8_API
        AT_CUDNN_CHECK(cudnnGetFilterNdDescriptor(
            lin_layer_mat_desc.desc(),
            min_dim,
            &data_type,
            &format,
            &nb_dims,
            filter_dim_a));
#else
        AT_CUDNN_CHECK(cudnnGetTensorNdDescriptor(
            lin_layer_mat_desc.desc(),
            min_dim,
            &data_type,
            &nb_dims,
            filter_dim_a,
            stride_dim_a));
#endif

        TORCH_INTERNAL_ASSERT(
            nb_dims <= min_dim,
            "nb_dims = ",
            nb_dims,
            "; min_dim  = ",
            min_dim);
        auto elem_size = dataSize(getCudnnDataType(weight_buf));
        auto offset_bytes = static_cast<const char*>(matrix_pointer) -
            static_cast<const char*>(weight_buf.data_ptr());
        TORCH_INTERNAL_ASSERT(
            offset_bytes % elem_size == 0,
            "offset_bytes = ",
            offset_bytes,
            "; elem_size = ",
            elem_size);
        size_t offset = offset_bytes / elem_size;
        // for all the RNN types provided by CUDNN, all the ih weights
        // are the same size and are allocated in a contiguous chunk
        // (same for the hh weights, and the ih and hh biases).
        // Since we're storing all the weights in a single tensor anyway,
        // might as well merge the CUDNN ones into a single tensor as well
        int mat_numel =
            c10::multiply_integers(filter_dim_a, filter_dim_a + nb_dims);
        if (linear_id == 0 || linear_id == num_linear_layers / 2) {
          // We could also exclude bias params by restricting cudnn_methods to
          // just { cudnnGetRNNLinLayerMatrixParams } at the very top.  However,
          // to do so would throw off the cur_offset account, which is currently
          // a strict and informative check that all params are laid out the way
          // we think they are.  If include_bias is false, I'd rather keep full
          // cur_offset checks rather than save some CPU overhead by skipping
          // the cudnn_method = cudnnGetRNNLinLayerBiasParams iteration.
#ifndef USE_CUDNN_RNN_V8_API
          if (include_bias || cudnn_method != cudnnGetRNNLinLayerBiasParams) {
#else
          if (include_bias || cudnn_method) {
#endif
            // Generate a new parameter tensor which is a view into the
            // weight_buf.
            std::initializer_list<int64_t> size = {
                mat_numel * num_linear_layers / 2, 1};
            Tensor param = at::empty({0}, weight_buf.options())
                               .set_(weight_buf.storage(), offset, size);
            params.emplace_back(std::move(param));
            layer_params_count++;
          }
        } else {
          TORCH_INTERNAL_ASSERT(
              cur_offset == offset,
              "cur_offset = ",
              cur_offset,
              "; offset = ",
              offset);
        }
        cur_offset = offset + mat_numel;
      }
    } // for cudnn_method
    if (rnn.proj_size != 0) {
#ifndef USE_CUDNN_RNN_V8_API
      add_projection_weights(
          handle, rnn_desc, x_desc, w_desc, weight_buf, layer, params);
#else
      add_projection_weights(handle, rnn_desc, weight_buf, layer, params);
#endif
      layer_params_count++;
    }

    if (layer == 0) {
      global_layer_params_count = layer_params_count;
    } else {
      TORCH_INTERNAL_ASSERT(
          global_layer_params_count == layer_params_count,
          "global_layer_params_count = ",
          global_layer_params_count,
          "; layer_params_count = ",
          layer_params_count);
    }
  } // for layer
  return std::make_pair(params, global_layer_params_count);
}

// This is a lightweight version of the method above used to quickly get the
// expected parameter offsets.
std::vector<void*> get_expected_data_ptrs(
    const Tensor& weight_buf,
    cudnnHandle_t handle,
    const RNNDescriptorParams& rnn,
    const RNNDescriptor& rnn_desc,
    const TensorDescriptor& x_desc,
    cudnnDataType_t datatype) {
#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  int64_t num_linear_layers = _num_linear_layers(rnn.mode);
  int64_t num_dir_layers = rnn.num_directions() * rnn.num_layers;
#ifndef USE_CUDNN_RNN_V8_API
  const auto cudnn_methods = {
      cudnnGetRNNLinLayerMatrixParams, cudnnGetRNNLinLayerBiasParams};
#else
  const auto cudnn_methods = {true, false};
#endif
  std::vector<void*> data_ptrs;
  if (rnn.proj_size != 0) {
    data_ptrs.reserve(num_dir_layers * (2 * 2 + 1));
  } else {
    data_ptrs.reserve(num_dir_layers * 2 * 2);
  }
  for (const auto layer : c10::irange(num_dir_layers)) {
    for (auto cudnn_method : cudnn_methods) {
      // This API returns a separate pointer for weight of every gate,
      // but we represent them as a single tensor, so we're only interested
      // in a very limited subset of possible values.
      const std::array<int64_t, 2> linear_offsets = {0, num_linear_layers / 2};
      for (int64_t linear_id : linear_offsets) {
        void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
        FilterDescriptor lin_layer_mat_desc;
        AT_CUDNN_CHECK(cudnn_method(
            handle,
            rnn_desc.desc(),
            layer,
            x_desc.desc(),
            w_desc.desc(),
            weight_buf.data_ptr(),
            linear_id,
            lin_layer_mat_desc.mut_desc(),
            &matrix_pointer));
#else
        TensorDescriptor lin_layer_mat_desc;
        if (cudnn_method) { // matrix
          AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
              handle,
              rnn_desc.desc(),
              layer,
              weight_buf.numel() * weight_buf.element_size(),
              weight_buf.data_ptr(),
              linear_id,
              lin_layer_mat_desc.mut_desc(),
              &matrix_pointer,
              nullptr,
              nullptr));
        } else { // bias
          AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
              handle,
              rnn_desc.desc(),
              layer,
              weight_buf.numel() * weight_buf.element_size(),
              weight_buf.data_ptr(),
              linear_id,
              nullptr,
              nullptr,
              lin_layer_mat_desc.mut_desc(),
              &matrix_pointer));
        }
#endif
        data_ptrs.push_back(matrix_pointer);
      }
    }
    if (rnn.proj_size != 0) {
      // assuming it's LSTM which has 8 "linear layers" (i.e. 4 weights and 4
      // biases)
      int64_t linear_id = 8;
      void* matrix_pointer;
#ifndef USE_CUDNN_RNN_V8_API
      FilterDescriptor lin_layer_mat_desc;
      AT_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
          handle,
          rnn_desc.desc(),
          layer,
          x_desc.desc(),
          w_desc.desc(),
          weight_buf.data_ptr(),
          linear_id,
          lin_layer_mat_desc.mut_desc(),
          &matrix_pointer));
#else
      TensorDescriptor lin_layer_mat_desc;

      AT_CUDNN_CHECK(cudnnGetRNNWeightParams(
          handle,
          rnn_desc.desc(),
          layer,
          weight_buf.numel() * weight_buf.element_size(),
          weight_buf.data_ptr(),
          linear_id,
          lin_layer_mat_desc.mut_desc(),
          &matrix_pointer,
          nullptr,
          nullptr));
#endif
      data_ptrs.push_back(matrix_pointer);
    }
  }
  return data_ptrs;
}

void _viewOrCopyOneParam(
    const Tensor& param_from,
    const Tensor& param_to,
    bool copy,
    bool allow_type_change = false) {
  // if copying, allow_type_change may be true or false.
  // if viewing, allow_type_change must be false.
  TORCH_INTERNAL_ASSERT(
      copy || !allow_type_change, "if viewing, type change is not allowed.");
  TORCH_INTERNAL_ASSERT(
      allow_type_change || (param_from.scalar_type() == param_to.scalar_type()),
      "parameter types mismatch");
  if (copy) {
    param_to.copy_(param_from.view_as(param_to));
  } else {
    param_from.resize_as_(param_to);
  }
}

void _viewOrCopyParams(
    MatrixRef<Tensor> params_from,
    MatrixRef<Tensor> params_to,
    bool copy,
    bool allow_type_change = false) {
  TORCH_INTERNAL_ASSERT(
      params_from.size(0) == params_to.size(0), "number of layers mismatch");
  for (const auto i : c10::irange(params_from.size(0))) {
    auto layer_params_from = params_from[i];
    auto layer_params_to = params_to[i];
    // NOTE: these lists have all weights before all biases, so if the layer
    // doesn't use biases, iteration will terminate once layer_params_from ends
    // and ignore them.

    // NOTE: there is an exception from the above statement. If LSTMs with
    // projections are used, weights layout will be w_ih, w_hh, b_ih, b_hh,
    // w_hr. So need to handle no-bias case specially, because will need to copy
    // 0->0, 1->1, 2->4. This case can be uniquely identified by checking if
    // number of defined parameters for each layer is 3.
    if (layer_params_from.size() == 3 && layer_params_to.size() != 3) {
      _viewOrCopyOneParam(
          layer_params_from[0], layer_params_to[0], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[1], layer_params_to[1], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[2], layer_params_to[4], copy, allow_type_change);
      continue;
    }
    if (layer_params_to.size() == 3 && layer_params_from.size() != 3) {
      _viewOrCopyOneParam(
          layer_params_from[0], layer_params_to[0], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[1], layer_params_to[1], copy, allow_type_change);
      _viewOrCopyOneParam(
          layer_params_from[4], layer_params_to[2], copy, allow_type_change);
      continue;
    }
    for (auto a = layer_params_from.begin(), b = layer_params_to.begin();
         a != layer_params_from.end() && b != layer_params_to.end();
         ++a, ++b) {
      _viewOrCopyOneParam(*a, *b, copy, allow_type_change);
    }
  }
}

void _copyParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
  _viewOrCopyParams(params_from, params_to, true);
}

void _viewParams(MatrixRef<Tensor> params_from, MatrixRef<Tensor> params_to) {
  _viewOrCopyParams(params_from, params_to, false);
}

std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, tensors.input_size};
  } else {
    return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
  }
}

std::vector<int64_t> _hidden_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  if (rnn.proj_size != 0) {
    return {
        rnn.num_layers * rnn.num_directions(),
        tensors.mini_batch,
        rnn.proj_size};
  } else {
    return {
        rnn.num_layers * rnn.num_directions(),
        tensors.mini_batch,
        rnn.hidden_size};
  }
}

std::vector<int64_t> _cell_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  return {
      rnn.num_layers * rnn.num_directions(),
      tensors.mini_batch,
      rnn.hidden_size};
}

std::vector<int64_t> _output_size(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  auto out_size = rnn.hidden_size;
  if (rnn.proj_size != 0) {
    out_size = rnn.proj_size;
  }
  if (tensors.is_input_packed()) {
    return {tensors.batch_sizes_sum, out_size * rnn.num_directions()};
  } else {
    return {
        tensors.seq_length,
        tensors.mini_batch,
        out_size * rnn.num_directions()};
  }
}

inline bool use_persist_common_heuristics(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  return rnn.num_layers == 1 && rnn.hidden_size <= 1024 &&
      rnn.num_directions() == 1 && rnn.hidden_size % 128 == 0 &&
      tensors.input_size % 128 == 0;
}

inline bool use_persist_device_heuristics(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors) {
  auto bsize = tensors.mini_batch;
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major == 7) {
    if (prop->minor == 5) {
      // Excludes Turing from using persistent rnn.
      return false;
    } else {
      // technically, batch size should be multiple of 8, but there are quite a
      // few multiple-of-8 batchsizes that give bad perf, weed them out
      return ((bsize % 16 == 0 && bsize != 80 && bsize != 112) || bsize == 8) &&
          ((tensors.seq_length >= 40 && bsize <= 128) ||
           (tensors.seq_length >= 20 && bsize <= 96) ||
           (tensors.seq_length >= 10 && bsize <= 32));
    }
  } else if (prop->major >= 8 && prop->multiProcessorCount >= 98) {
    // SM count check excludes A30 (similar issue to A40)
    if (prop->minor == 6) {
      // Excludes sm_86 GPU devices from using persistent rnn.
      // This is because there are some edge cases that will throw exceptions
      // with cudnn 8.0.5 on Nvidia A40 GPU.
      return false;
    }
    // Based on tests by Vasily Volkov and xwang233.  Vasily only tried bsize <=
    // 128, so conservatively enable persistence for bsize <= 128 only.
    // TODO:  Run more tests for bsize > 128.
    if (rnn.mode == CUDNN_GRU) {
      // Persistent GRU performance is flakier than other RNN types.  Exclude
      // them for now.
      // TODO:  Write a more refined GRU heuristic.
      return false;
    } else if (rnn.mode == CUDNN_LSTM) {
      // Persistent LSTMs are comparable to or better than non-persistent for
      // bsize <= 128.
      return (bsize % 8 == 0) && (bsize <= 128);
    } else {
      // Persistent RNN_RELU and TANH show poor performance when bsize >= 96 AND
      // hidden size >= 896.
      return (bsize % 8 == 0) && (bsize <= 128) &&
          (bsize < 96 || rnn.hidden_size < 896);
    }
  } else {
    return false;
  }
}

inline bool use_rnn_persist_small_h(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors,
    bool forward) {
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major < 6)
    return false;

  if (forward) {
    if (rnn.mode == CUDNN_RNN_RELU || rnn.mode == CUDNN_RNN_TANH) {
      return rnn.hidden_size <= 384;
    }
    if (rnn.mode == CUDNN_LSTM || rnn.mode == CUDNN_GRU) {
      return rnn.hidden_size <= 192;
    }
  } else /* backward */ {
    if (rnn.mode == CUDNN_RNN_RELU || rnn.mode == CUDNN_RNN_TANH) {
      return rnn.hidden_size <= 256;
    }
    if (rnn.mode == CUDNN_LSTM || rnn.mode == CUDNN_GRU) {
      return rnn.hidden_size <= 128;
    }
  }

  return false;
}

cudnnRNNAlgo_t get_algo(
    const RNNDescriptorParams& rnn,
    const TensorDescriptorListParams& tensors,
    const Tensor input,
    bool forward) {
  // LSTM with projections only works with standard algorithm
  if (rnn.proj_size != 0) {
    return CUDNN_RNN_ALGO_STANDARD;
  }

  // Persistent algos typically don't work for packed inputs with sequence
  // lengths that vary across batch elements, and will return
  // CUDNN_STATUS_NOT_SUPPORTED if attempted. See
  // https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/developer-guide/index.html#features-of-rnn-functions
  if (!tensors.is_input_packed()) {
    auto cudnnDataType = getCudnnDataType(input);
    if (cudnnDataType != CUDNN_DATA_DOUBLE) {
      if (use_rnn_persist_small_h(rnn, tensors, forward)) {
        return CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H;
      }
    }
    if (cudnnDataType == CUDNN_DATA_HALF) {
      if (use_persist_common_heuristics(rnn, tensors) &&
          use_persist_device_heuristics(rnn, tensors)) {
        return CUDNN_RNN_ALGO_PERSIST_STATIC;
      }
    }
  }

  return CUDNN_RNN_ALGO_STANDARD;
}

cudnnDataType_t promote_rnn_math_type(cudnnDataType_t dtype) {
  if (dtype == CUDNN_DATA_HALF || dtype == CUDNN_DATA_BFLOAT16) {
    return CUDNN_DATA_FLOAT;
  }
  return dtype;
}

int64_t _cudnn_rnn_flatten_weight_prologue(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    bool bidirectional,
    const cudnnDataType_t flat_buf_datatype,
    const cudnnHandle_t& handle,
    RNNDescriptorParams& rnn,
    RNNDescriptor& rnn_desc,
    const TensorGeometry& x_geom,
    TensorDescriptor& x_desc) {
  // flat_buf_datatype is accepted as a separate argument (rather than extracted
  // from flat_buf_options) because to extract flat_buf_datatype from
  // flat_buf_options, we'd need to say auto flat_buf_datatype =
  // getCudnnDataTypeFromScalarType(typeMetaToScalarType(options.dtype()));
  // typeMetaToScalarType is a surprisingly nontrivial function.  We should
  // avoid it if we can.
  TORCH_CHECK(
      !weight_arr.empty(),
      "copy_weights_to_flat_buf_views: cannot flatten empty weight list");

  rnn.set(
      mode,
#ifdef USE_CUDNN_RNN_V8_API
      input_size,
      false, // eqy: bogus as we do not know if the input is packed here
             // but it should not affect the weights (what are are interested
             // in)
#endif
      hidden_size,
      proj_size,
      num_layers,
      bidirectional,
      promote_rnn_math_type(flat_buf_datatype),
      flat_buf_datatype);

  rnn_desc = rnn.descriptor(handle);

  // Why do we pad to 5 dims here (and elsewhere)?
  // https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-892/api/index.html#cudnnRNNForwardTraining
  // expects descriptors padded to 3 dimensions.
  x_desc.set(flat_buf_datatype, x_geom.sizes(), x_geom.strides(), 5);

#ifndef USE_CUDNN_RNN_V8_API
  return get_num_weights(handle, rnn_desc, x_desc, flat_buf_datatype);
#else
  return get_num_weights(handle, rnn_desc, flat_buf_datatype);
#endif
}

} // namespace native

// Utilities exposed in RNNUtils.h
namespace cudnn_rnn {

TORCH_CUDA_CPP_API std::tuple<Tensor, std::vector<Tensor>>
copy_weights_to_flat_buf_views(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    bool bidirectional,
    const cudnnDataType_t flat_buf_datatype,
    const TensorOptions& flat_buf_options,
    bool set_orig_weights_to_flat_buf,
    bool allow_type_change /*=false*/,
    bool include_bias /*=true*/) {
  TORCH_CHECK(!weight_arr.empty(), "empty weight list");
  auto handle = getCudnnHandle();
  RNNDescriptorParams rnn;
  RNNDescriptor rnn_desc;
  TensorDescriptor x_desc;
  TensorGeometry x_geom({1, input_size});
  auto num_weights = _cudnn_rnn_flatten_weight_prologue(
      weight_arr,
      weight_stride0,
      input_size,
      mode,
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      bidirectional,
      flat_buf_datatype,
      handle,
      rnn,
      rnn_desc,
      x_geom,
      x_desc);
  auto weight_buf = at::zeros(num_weights, flat_buf_options);

#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
  w_desc.set(weight_buf, 3);
#endif

  // Slice off views into weight_buf
  auto [params_arr, params_stride0] = get_parameters(
#ifndef USE_CUDNN_RNN_V8_API
      handle, rnn, rnn_desc, x_desc, w_desc, weight_buf, include_bias);
#else
      handle, rnn, rnn_desc, weight_buf, include_bias);
#endif
  MatrixRef<Tensor> weight{weight_arr, static_cast<size_t>(weight_stride0)},
      params{params_arr, params_stride0};

  // Copy weights
  _viewOrCopyParams(weight, params, /*copy=*/true, allow_type_change);
  if (set_orig_weights_to_flat_buf) {
    // Update the storage
    for (const auto i : c10::irange(weight.size(0))) {
      // There is a special case for LSTM with projections and no bias,
      // where weight copy is done in 0->0, 1->1, 2->4 layout
      if (weight[i].size() == 3 && params[i].size() == 5) {
        weight[i][0].set_(params[i][0].view_as(weight[i][0]));
        weight[i][1].set_(params[i][1].view_as(weight[i][1]));
        weight[i][2].set_(params[i][4].view_as(weight[i][2]));
      } else {
        for (auto orig_param_it = weight[i].begin(),
                  new_param_it = params[i].begin();
             orig_param_it != weight[i].end() &&
             new_param_it != params[i].end();
             orig_param_it++, new_param_it++) {
          auto orig_param = *orig_param_it, new_param = *new_param_it;
          orig_param.set_(new_param.view_as(orig_param));
        }
      }
    }
  }

  return std::make_tuple(weight_buf, params_arr);
}

} // namespace cudnn_rnn

using namespace cudnn_rnn;

// NB: does inplace update into TensorList
// It would be a relatively simple matter to refactor this into multiple
// functions, only one of which does an inplace update, but we leave this
// for future work
Tensor _cudnn_rnn_flatten_weight(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    bool fn_bidirectional) {
  TORCH_CHECK(!weight_arr.empty(), "empty weight list");
  // returns flat weight_buf
  return std::get<0>(copy_weights_to_flat_buf_views(
      weight_arr,
      weight_stride0,
      input_size,
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      batch_first,
      fn_bidirectional,
      /*flat_buf_datatype=*/getCudnnDataType(weight_arr[0]),
      /*flat_buf_options=*/weight_arr[0].options(),
      /*set_orig_weights_to_flat_buf=*/true));
}

Tensor _cudnn_rnn_flatten_weight_meta(
    TensorList weight_arr,
    int64_t weight_stride0,
    c10::SymInt input_size,
    int64_t mode,
    c10::SymInt hidden_size,
    c10::SymInt proj_size,
    int64_t num_layers,
    bool batch_first,
    bool bidirectional) {
  TORCH_CHECK(!weight_arr.empty(), "empty weight list");
  auto handle = getCudnnHandle();
  RNNDescriptorParams rnn;
  RNNDescriptor rnn_desc;
  TensorDescriptor x_desc;
  TensorGeometry x_geom({1, input_size});
  auto num_weights = _cudnn_rnn_flatten_weight_prologue(
      weight_arr,
      weight_stride0,
      input_size.guard_int(__FILE__, __LINE__),
      mode,
      hidden_size.guard_int(__FILE__, __LINE__),
      proj_size.guard_int(__FILE__, __LINE__),
      num_layers,
      batch_first,
      bidirectional,
      getCudnnDataType(weight_arr[0]),
      handle,
      rnn,
      rnn_desc,
      x_geom,
      x_desc);

  return at::zeros_symint({num_weights}, weight_arr[0].options());
}

const char* WEIGHT_FORMAT_WARN =
    "RNN module weights are not part of single contiguous "
    "chunk of memory. This means they need to be compacted "
    "at every call, possibly greatly increasing memory usage. "
    "To compact weights again call flatten_parameters().";

// NB: when fn_batch_sizes is empty, that means no batch sizes was specified
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _cudnn_rnn(
    const Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const std::optional<Tensor>& weight_buf_r_opt,
    const Tensor& hx,
    const std::optional<Tensor>& cx_opt,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t fn_proj_size,
    int64_t fn_num_layers,
    bool batch_first,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    IntArrayRef fn_batch_sizes,
    const std::optional<Tensor>& fn_dropout_state_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_buf_r_maybe_owned =
      at::borrow_from_optional_tensor(weight_buf_r_opt);
  const Tensor& weight_buf_r = *weight_buf_r_maybe_owned;
  const Tensor& cx = cx_opt.value_or(Tensor());
  const Tensor& fn_dropout_state = fn_dropout_state_opt.value_or(Tensor());

  check_attributes(input_r, weight, {hx, cx}, /*check_dtype=*/true);
  auto input = input_r;
  auto weight_buf = weight_buf_r;
  if (!weight_buf.defined()) {
    TORCH_WARN(WEIGHT_FORMAT_WARN);
  }
  if (fn_dropout_state.defined()) {
    auto input_arg = TensorArg(input, "input", 1);
    auto dropout_state_arg = TensorArg(fn_dropout_state, "dropout_states", 15);
    checkSameGPU("cudnn_rnn", input_arg, dropout_state_arg);
  }
  RNNParams fn;
  auto datatype = getCudnnDataType(input);
#ifndef USE_CUDNN_RNN_V8_API
  fn.rnn.set(
      fn_mode,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#else
  auto input_size = input_r.size(-1);
  auto packed = !fn_batch_sizes.empty();
  fn.rnn.set(
      fn_mode,
      input_size,
      packed,
      fn_hidden_size,
      fn_proj_size,
      fn_num_layers,
      fn_bidirectional,
      promote_rnn_math_type(datatype),
      datatype);
#endif
  fn.dropout.set(fn_train, fn_dropout, fn_dropout_state);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  // TODO: Set device to input

  if (fn.rnn.mode != CUDNN_LSTM) {
    TORCH_CHECK(!cx.defined(), "rnn: illegal defined cx for non-LSTM RNN");
  }

  // TODO: can batch_first be a wrapper around this function?
  auto is_input_packed = !fn.tensors.batch_sizes.empty();
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto cell_size = _cell_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  TORCH_CHECK(hx.is_contiguous(), "rnn: hx is not contiguous");
  TORCH_CHECK(!cx.defined() || cx.is_contiguous(), "rnn: cx is not contiguous");

  auto x = input.contiguous();
  auto output = at::empty(output_size, input.options());
  auto hy = at::empty(hidden_size, hx.options());
  Tensor cy;
  if (cx.defined()) {
    cy = at::empty(cell_size, cx.options());
  } else {
    cy = at::empty(
        {0}, hx.options()); // NB: Not allowed to return undefined tensors
  }
  auto y = output;

  auto handle = getCudnnHandle();
  cudnnRNNAlgo_t algo = get_algo(fn.rnn, fn.tensors, input, true);
  fn.rnn.set_algo(algo);
  RNNDescriptors descs(fn, handle, x, y, hx, cx);

#ifndef USE_CUDNN_RNN_V8_API
  FilterDescriptor w_desc;
#endif
  if (!weight_buf.defined()) {
#ifndef USE_CUDNN_RNN_V8_API
    auto num_weights =
        get_num_weights(handle, descs.rnn_desc, descs.x_descs[0], datatype);
#else
    auto num_weights = get_num_weights(handle, descs.rnn_desc, datatype);
#endif
    weight_buf = at::empty(num_weights, x.options());
#ifndef USE_CUDNN_RNN_V8_API
    w_desc.set(weight_buf, 3);
#endif
    weight_buf.zero_();
#ifndef USE_CUDNN_RNN_V8_API
    auto [params, params_stride0] = get_parameters(
        handle, fn.rnn, descs.rnn_desc, descs.x_descs[0], w_desc, weight_buf);
#else
    auto [params, params_stride0] =
        get_parameters(handle, fn.rnn, descs.rnn_desc, weight_buf);
#endif
    _copyParams(
        MatrixRef<Tensor>{weight, static_cast<size_t>(weight_stride0)},
        MatrixRef<Tensor>{params, params_stride0});
  } else {
#ifndef USE_CUDNN_RNN_V8_API
    w_desc.set(weight_buf, 3);
#endif
  }

  TORCH_CHECK(
      !cx.defined() || cx.sizes().equals(cell_size),
      "Expected cell size ",
      IntArrayRef{cell_size},
      ", got ",
      cx.sizes());
  size_t workspace_size;
#ifndef USE_CUDNN_RNN_V8_API
  auto x_descs_arr = descs.get_x_descs();
  auto y_descs_arr = descs.get_y_descs();
#else
  auto& x_descs_arr = descs.x_descs;
  auto& y_descs_arr = descs.y_descs;
#endif
#ifndef USE_CUDNN_RNN_V8_API
  AT_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
      handle,
      descs.rnn_desc.desc(),
      fn.tensors
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cudnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/aten/src/ATen/native/cudnn`):

- [`BatchNorm.cpp_kw.md_docs.md`](./BatchNorm.cpp_kw.md_docs.md)
- [`GridSampler.cpp_kw.md_docs.md`](./GridSampler.cpp_kw.md_docs.md)
- [`ConvShared.cpp_docs.md_docs.md`](./ConvShared.cpp_docs.md_docs.md)
- [`MHA.cpp_kw.md_docs.md`](./MHA.cpp_kw.md_docs.md)
- [`AffineGridGenerator.cpp_docs.md_docs.md`](./AffineGridGenerator.cpp_docs.md_docs.md)
- [`Conv_v8.cpp_kw.md_docs.md`](./Conv_v8.cpp_kw.md_docs.md)
- [`Conv_v7.cpp_kw.md_docs.md`](./Conv_v7.cpp_kw.md_docs.md)
- [`AffineGridGenerator.cpp_kw.md_docs.md`](./AffineGridGenerator.cpp_kw.md_docs.md)
- [`BatchNorm.h_kw.md_docs.md`](./BatchNorm.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `RNN.cpp_docs.md_docs.md`
- **Keyword Index**: `RNN.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

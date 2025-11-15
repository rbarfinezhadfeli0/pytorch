# Documentation: `aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_dynamic.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/ao_sparse/quantized/cpu/qlinear_dynamic.cpp`
- **Size**: 6,563 bytes (6.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <c10/util/accumulate.h>

#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/library.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/empty.h>
#endif

namespace ao::sparse {

#ifdef USE_PYTORCH_QNNPACK
template <>
at::Tensor PackedLinearWeightQnnp::apply_dynamic_impl<true>(
    const at::Tensor& input) {
  TORCH_INTERNAL_ASSERT(
      false,
      "Sparse quantized dynamic linear with fused relu is not yet "
      "supported on qnnpack backend.");
  return at::Tensor();
}

template <>
at::Tensor PackedLinearWeightQnnp::apply_dynamic_impl<false>(
    const at::Tensor& input) {
  TORCH_CHECK(
      input.dim() >= 2,
      "quantized_sparse_linear(): Input tensor rank should be >= 2");

  const auto rows_input = c10::multiply_integers(input.sizes().begin(), input.sizes().end() - 1);
  const auto cols_input = input.size(input.dim() - 1);
  TORCH_CHECK(
      cols_input == input_channels_,
      "quantized_sparse_linear: Input tensor's last and weight tensor's"
      " second dimension must match.");

  // On empty input, no output data will be generated,
  // so use arbitrary qparams.
  float x_min = 0;
  float x_max = 0;
  // Otherwise...
  if (input.numel() > 0) {
    x_min = input.min().item<float>();
    x_max = input.max().item<float>();
  }

  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/255);

  // Quantize input
  at::Tensor q_input = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);

  auto q_input_contig = q_input.contiguous();
  if (sparse_linear_op_ == nullptr) {
    // We calculate requant scale here as the vector holding the requant scale
    // is owned by this module. The pointer is then passed to qnnpack backend.
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales_, q_input_contig.q_scale(), 1.f, requantization_scales_);
    input_scale_ = q_input_contig.q_scale();
    pytorch_qnnp_operator_t sparse_linear_op{nullptr};
    pytorch_qnnp_status status =
        pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
            input_channels_,
            output_channels_,
            q_input_contig.q_zero_point(),
            w_zero_points_.data(),
            bcsr_matrix_->col_indices_data_ptr(),
            bcsr_matrix_->row_values_data_ptr(),
            bcsr_matrix_->values.data(),
            bcsr_matrix_->row_block_size, /* out_features_block_size */
            bcsr_matrix_->col_block_size, /* in_features_block_size */
            bcsr_matrix_->indices_dtype,
            0, /* output zero point: not used */
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max(),
            0, /* flags */
            requantization_scales_.data(),
            true, /* use prepacking kernel */
            &sparse_linear_op);
    TORCH_CHECK(
        status == pytorch_qnnp_status_success,
        "Failed to create sparse linear operator on"
        " qnnpack backend.");
    sparse_linear_op_ =
        std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>(
            sparse_linear_op);
  }

  // Input on next iteration can be different, thus resulting in
  // different input scale. This will require us to recalculate requantization
  // scales.
  if (input_scale_ != q_input_contig.q_scale()) {
    generate_requantization_scales(
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        w_scales_, q_input_contig.q_scale(), 1.f, requantization_scales_);
  }
  // Update input related quantization params in the operator.
  sparse_linear_op_->dynamic_conv_quantization_params.input_zero_point =
      q_input_contig.q_zero_point();
  sparse_linear_op_->dynamic_conv_quantization_params.multipliers =
      requantization_scales_.data();

  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = output_channels_;

  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));

  pytorch_qnnp_status status =
      pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
          sparse_linear_op_.get(),
          rows_input, /* batch size */
          reinterpret_cast<uint8_t*>(q_input_contig.data_ptr<c10::quint8>()),
          cols_input, /* num input channels */
          bias_.data_ptr<float>(),
          output.data_ptr<float>(),
          output_channels_);
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "Failed to setup sparse linear operator on"
      " qnnpack backend.");

  status = pytorch_qnnp_run_operator(
      sparse_linear_op_.get(), caffe2::pthreadpool_());
  TORCH_CHECK(
      status == pytorch_qnnp_status_success,
      "Failed to run sparse linear operator on"
      " qnnpack backend.");

  return output;
}

at::Tensor PackedLinearWeightQnnp::apply_dynamic(
    const at::Tensor& input) {
  return apply_dynamic_impl<false>(input);
}

at::Tensor PackedLinearWeightQnnp::apply_dynamic_relu(
    const at::Tensor& input) {
  return apply_dynamic_impl<true>(input);
}

#endif // USE_PYTORCH_QNNPACK

namespace {

template <bool ReluFused>
class QLinearDynamicInt8 final {
 public:
  static at::Tensor run(
      const at::Tensor& input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    auto& ctx = at::globalContext();
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      if (ReluFused) {
        return packed_weight->apply_dynamic_relu(input);
      } else {
        return packed_weight->apply_dynamic(input);
      }
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation ao::sparse::qlinear_dynamic",
        toString(ctx.qEngine()));
  }
};

TORCH_LIBRARY_IMPL(sparse, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_dynamic"),
      TORCH_FN(QLinearDynamicInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::qlinear_relu_dynamic"),
      TORCH_FN(QLinearDynamicInt8<true>::run));
}

} // namespace
} // namespace ao::sparse

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ao`

**Classes/Structs**: `QLinearDynamicInt8`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/ao_sparse/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Parallel.h`
- `torch/custom_class.h`
- `torch/library.h`
- `c10/util/accumulate.h`
- `ATen/native/quantized/cpu/QuantUtils.h`
- `ATen/native/quantized/library.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `ATen/native/ao_sparse/quantized/cpu/packed_params.h`
- `ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h`
- `ATen/Functions.h`
- `ATen/ops/quantize_per_tensor.h`
- `ATen/ops/empty.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`aten/src/ATen/native/ao_sparse/quantized/cpu`):

- [`packed_params.h_docs.md`](./packed_params.h_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`qlinear_serialize.cpp_docs.md`](./qlinear_serialize.cpp_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`fbgemm_utils.cpp_docs.md`](./fbgemm_utils.cpp_docs.md)
- [`qlinear_deserialize.cpp_docs.md`](./qlinear_deserialize.cpp_docs.md)
- [`qlinear_prepack.cpp_docs.md`](./qlinear_prepack.cpp_docs.md)
- [`qnnpack_utils.h_docs.md`](./qnnpack_utils.h_docs.md)
- [`qlinear.cpp_docs.md`](./qlinear.cpp_docs.md)


## Cross-References

- **File Documentation**: `qlinear_dynamic.cpp_docs.md`
- **Keyword Index**: `qlinear_dynamic.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

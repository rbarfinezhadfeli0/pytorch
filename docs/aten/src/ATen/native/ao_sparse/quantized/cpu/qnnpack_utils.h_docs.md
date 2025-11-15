# Documentation: `aten/src/ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/ao_sparse/quantized/cpu/qnnpack_utils.h`
- **Size**: 3,242 bytes (3.17 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/QScheme.h>

#ifdef USE_PYTORCH_QNNPACK
// TODO: Refacto QnnpackUtils.h so as to separate code
// needed for quantized op from the generic qnnpack specific
// quantization utilities.
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <pack_block_sparse.h>

namespace ao::sparse {

struct TORCH_API PackedLinearWeightQnnp : public LinearPackedParamsBase {
  PackedLinearWeightQnnp(const at::Tensor& weight, const std::optional<at::Tensor>& bias, const int64_t out_features_block_size /* block sparsity size across output_features */, const int64_t in_features_block_size /* block sparsity size across input_features */);
  explicit PackedLinearWeightQnnp(const BCSRSerializationType& serialized);
  std::optional<at::Tensor> orig_bias_;
  // Separate copy of bias exist so that we can fill in zeros when
  // optional bias does not exist. This is to compy with qnnpack operator that
  // expects bias to be present.
  // In case bias is present bias_ is just a reference to orig_bias_
  at::Tensor bias_;
  c10::QScheme q_scheme_;
  double input_scale_{};
  std::unique_ptr<qnnpack::BCSRMatrix> bcsr_matrix_;
  at::Tensor w_scales_;
  std::vector<uint8_t> w_zero_points_;
  std::vector<float> requantization_scales_;
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      sparse_linear_op_{nullptr};
  int64_t output_channels_;
  int64_t input_channels_;
  // Deserialized Tensors are stored to maintain the lifetime of underlying
  // BCSR data.
  // These are left empty if PackedLinearWeightQnnp is created via prepacking
  // rather than deserializing.
  at::Tensor deserialized_bcsr_row_block_indices_;
  at::Tensor deserialized_bcsr_col_block_indices_;
  at::Tensor deserialized_bcsr_weight_values_;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    TORCH_CHECK(
        false, "Static quantized sparse linear unimplemented on QNNPACK");
  }
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override {
    TORCH_CHECK(
        false, "Static quantized sparse linear unimplemented on QNNPACK");
  }

  at::Tensor apply_dynamic(const at::Tensor& input) override;
  at::Tensor apply_dynamic_relu(const at::Tensor& input) override;

  LinearPackedSerializationType unpack() override;

  BCSRSerializationType serialize() override;

  static c10::intrusive_ptr<LinearPackedParamsBase> deserialize(
      const BCSRSerializationType& serialized);

  std::optional<at::Tensor> bias() override {
    return orig_bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      const at::Tensor& weight,
      const std::optional<at::Tensor>& bias,
      const int64_t out_features_block_size,
      const int64_t in_features_block_size);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(const at::Tensor& input);
};

} // namespace ao::sparse

#endif // USE_PYTORCH_QNNPACK

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ao`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/ao_sparse/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `c10/core/QScheme.h`
- `ATen/native/ao_sparse/quantized/cpu/packed_params.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `pack_block_sparse.h`


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

Files in the same folder (`aten/src/ATen/native/ao_sparse/quantized/cpu`):

- [`packed_params.h_docs.md`](./packed_params.h_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`qlinear_serialize.cpp_docs.md`](./qlinear_serialize.cpp_docs.md)
- [`qlinear_dynamic.cpp_docs.md`](./qlinear_dynamic.cpp_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`fbgemm_utils.cpp_docs.md`](./fbgemm_utils.cpp_docs.md)
- [`qlinear_deserialize.cpp_docs.md`](./qlinear_deserialize.cpp_docs.md)
- [`qlinear_prepack.cpp_docs.md`](./qlinear_prepack.cpp_docs.md)
- [`qlinear.cpp_docs.md`](./qlinear.cpp_docs.md)


## Cross-References

- **File Documentation**: `qnnpack_utils.h_docs.md`
- **Keyword Index**: `qnnpack_utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

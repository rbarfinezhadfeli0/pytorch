# Documentation: `aten/src/ATen/native/transformers/transformer.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/transformers/transformer.cpp`
- **Size**: 4,099 bytes (4.00 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>

#include <torch/library.h>

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_native_multi_head_attention.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_native.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/layer_norm.h>
#endif

namespace at::native {

namespace {
Tensor linear_for_ffn(
    const Tensor& bias,
    const Tensor& mat1,
    const Tensor& mat2,
    std::optional<bool> use_gelu) {
  if (mat1.is_nested()) {
    return NestedTensor_times_Tensor_plus_Tensor_addmm(
        bias, mat1, mat2.t(), 1, 1, use_gelu);
  }

  auto mat1_ = mat1.view({mat1.sizes()[0] * mat1.sizes()[1], mat1.sizes()[2]});
  Tensor result;
  if (use_gelu.has_value()) {
    result = at::_addmm_activation(bias, mat1_, mat2.t(), 1, 1, *use_gelu);
  } else {
    result = at::addmm(bias, mat1_, mat2.t());
  }
  return result.view({mat1.sizes()[0], mat1.sizes()[1], -1});
}

Tensor ffn(
    const Tensor& input,
    const Tensor& w1,
    const Tensor& b1,
    const Tensor& w2,
    const Tensor& b2,
    bool use_gelu,
    bool add_norm) {
  TORCH_CHECK(add_norm == false, "TODO add_norm to be supported in FFN");
  TORCH_CHECK(input.dim() == 3, "batched input size should be 3");
  TORCH_CHECK(w1.dim() == 2, "2d weights expected");
  TORCH_CHECK(w2.dim() == 2, "2d weights expected");
  Tensor res = linear_for_ffn(b1, input, w1, use_gelu);
  res = linear_for_ffn(b2, res, w2, std::nullopt);
  return res;
}

Tensor norm(
    const Tensor& input,
    const int64_t embed_dim,
    const double eps,
    const Tensor& weight,
    const Tensor& bias,
    const bool use_nested_tensor) {
  return at::layer_norm(input, {embed_dim}, weight, bias, eps, true);
}

} // namespace

Tensor transformer_encoder_layer_forward(
    const Tensor& src,
    const int64_t embed_dim,
    const int64_t num_heads,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const bool use_gelu,
    const bool norm_first,
    const double layer_norm_eps,
    const Tensor& layer_norm_weight_1,
    const Tensor& layer_norm_bias_1,
    const Tensor& layer_norm_weight_2,
    const Tensor& layer_norm_bias_2,
    const Tensor& ffn_weight_1,
    const Tensor& ffn_bias_1,
    const Tensor& ffn_weight_2,
    const Tensor& ffn_bias_2,
    const std::optional<Tensor>& mask,
    const std::optional<int64_t> mask_type) {
  {
    const Tensor& check_for_empty = src.is_nested() ? get_nested_tensor_impl(src)->get_buffer() : src;
    if (check_for_empty.numel() == 0) {
      return src.is_nested()
        ? at::detail::make_tensor<NestedTensorImpl>(check_for_empty, get_nested_tensor_impl(src)->get_nested_sizes())
        : src.clone();
    }
  }
  const bool use_nested_tensor = src.is_nested();
  Tensor x = src;
  if (norm_first) {
    x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_1, layer_norm_bias_1, use_nested_tensor);
  }
  x = std::get<0>(at::_native_multi_head_attention(
      x,
      x,
      x,
      embed_dim,
      num_heads,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      mask,
      false /* need_weights */,
      true /* average_attn_weights */,
      mask_type));

  x.add_(src);
  if (!norm_first) {
    x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_1, layer_norm_bias_1, use_nested_tensor);
  }


  auto pre_ffn_res = x;

  if (norm_first) {
    x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_2, layer_norm_bias_2, use_nested_tensor);
  }
  x = ffn(
      x,
      ffn_weight_1,
      ffn_bias_1,
      ffn_weight_2,
      ffn_bias_2,
      use_gelu,
      /* add_norm* */ false);
  x.add_(pre_ffn_res);
  if (!norm_first) {
    x = norm(x, embed_dim, layer_norm_eps, layer_norm_weight_2, layer_norm_bias_2, use_nested_tensor);
  }
  return x;
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/transformers`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/NestedTensorImpl.h`
- `torch/library.h`
- `ATen/native/nested/NestedTensorTransformerFunctions.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_addmm_activation.h`
- `ATen/ops/_native_multi_head_attention.h`
- `ATen/ops/_transformer_encoder_layer_fwd_native.h`
- `ATen/ops/addmm.h`
- `ATen/ops/layer_norm.h`


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

Files in the same folder (`aten/src/ATen/native/transformers`):

- [`sdp_utils.h_docs.md`](./sdp_utils.h_docs.md)
- [`sdp_utils_cpp.h_docs.md`](./sdp_utils_cpp.h_docs.md)
- [`sdp_utils_cpp.cpp_docs.md`](./sdp_utils_cpp.cpp_docs.md)
- [`attention.cpp_docs.md`](./attention.cpp_docs.md)
- [`attention.h_docs.md`](./attention.h_docs.md)


## Cross-References

- **File Documentation**: `transformer.cpp_docs.md`
- **Keyword Index**: `transformer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

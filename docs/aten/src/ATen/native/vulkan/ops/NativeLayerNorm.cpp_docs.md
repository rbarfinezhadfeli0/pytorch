# Documentation: `aten/src/ATen/native/vulkan/ops/NativeLayerNorm.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/NativeLayerNorm.cpp`
- **Size**: 3,810 bytes (3.72 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void _check_layer_norm_inputs(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight /* optional */,
    const std::optional<Tensor>& bias /* optional */) {
  const auto normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight->defined() || weight->sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight->sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias->defined() || bias->sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias->sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.sizes().size();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    TORCH_CHECK(false, ss.str());
  }
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {
  _check_layer_norm_inputs(input_arg, normalized_shape, weight_opt, bias_opt);

  TORCH_CHECK(
      input_arg.dim() >= 2 && input_arg.dim() <= 4,
      "Vulkan layernorm expects input of 2d, 3d or 4d!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layernorm expects weight and bias arguments");

  const Tensor weight =
      weight_opt->is_vulkan() ? *weight_opt : weight_opt->vulkan();

  const Tensor bias = bias_opt->is_vulkan() ? *bias_opt : bias_opt->vulkan();

  std::vector<int64_t> dims_to_reduce;
  for (const auto i : c10::irange(normalized_shape.size())) {
    dims_to_reduce.push_back(input_arg.dim() - i - 1);
  }
  IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  bool mean_keep_dim = true;
  bool var_keep_dim = true;
  auto mean = input.mean(dims_to_reduce_ref, mean_keep_dim);

  // in order to avoid recomputation of mean, we manually compute var as below
  // instead of invoking the var operator.
  auto input_minus_mean = input.sub(mean);
  auto var = input_minus_mean.mul(input_minus_mean)
                 .mean(dims_to_reduce_ref, var_keep_dim);
  auto std_inv = var.add(eps).pow(-0.5f);

  // use the formular in this page to compute layer_norm:
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  auto layernorm = input_minus_mean.mul(std_inv).mul(weight).add(bias);
  std::tuple<Tensor, Tensor, Tensor> output =
      std::make_tuple(layernorm, mean, std_inv);
  return output;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_layer_norm"),
      TORCH_FN(native_layer_norm));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `native`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/ops/Common.h`
- `torch/library.h`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`aten/src/ATen/native/vulkan/ops`):

- [`Convert.h_docs.md`](./Convert.h_docs.md)
- [`Batchnorm.cpp_docs.md`](./Batchnorm.cpp_docs.md)
- [`Slice.cpp_docs.md`](./Slice.cpp_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)
- [`Shape.cpp_docs.md`](./Shape.cpp_docs.md)
- [`Mean.cpp_docs.md`](./Mean.cpp_docs.md)
- [`UnaryOp.cpp_docs.md`](./UnaryOp.cpp_docs.md)
- [`Permute.cpp_docs.md`](./Permute.cpp_docs.md)
- [`Unsqueeze.cpp_docs.md`](./Unsqueeze.cpp_docs.md)
- [`Stack.cpp_docs.md`](./Stack.cpp_docs.md)


## Cross-References

- **File Documentation**: `NativeLayerNorm.cpp_docs.md`
- **Keyword Index**: `NativeLayerNorm.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

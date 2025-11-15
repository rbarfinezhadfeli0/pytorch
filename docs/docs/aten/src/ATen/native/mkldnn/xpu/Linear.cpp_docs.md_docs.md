# Documentation: `docs/aten/src/ATen/native/mkldnn/xpu/Linear.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/mkldnn/xpu/Linear.cpp_docs.md`
- **Size**: 5,811 bytes (5.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/mkldnn/xpu/Linear.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/xpu/Linear.cpp`
- **Size**: 3,547 bytes (3.46 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/DeviceGuard.h>
#include <torch/library.h>

#include <FusionUtils.h>

namespace at::native::xpu {

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
collapse_in_out_dim(at::Tensor input, int64_t dim, at::Tensor weight) {
  // dim collapse, e.g. [B, M, K] -> [BM, K]
  std::vector<int64_t> input_reshaped_size = (dim == 2)
      ? std::vector<int64_t>(input.size(0), input.size(1))
      : std::vector<int64_t>{
            input.numel() / (input.size(input.dim() - 1)),
            input.size(input.dim() - 1)};
  // [B, M, K] -> [B, M]
  std::vector<int64_t> output_size(
      input.sizes().begin(), input.sizes().end() - 1);
  // [B, M, N]
  output_size.push_back(weight.size(0));

  // [BM, N]
  std::vector<int64_t> output_reshaped_size{
      input_reshaped_size[0], weight.size(0)};
  return {input_reshaped_size, output_size, output_reshaped_size};
}

Tensor linear_pointwise(
    const Tensor& input_t, // [M, K] or [B, M, K]
    const Tensor& weight_t, // [N, K]
    const std::optional<Tensor>& bias_opt,
    std::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  onednn::Attr att;
  const OptionalDeviceGuard device_guard(device_of(input_t));
  att = construct_unary_attr(att, attr, scalars, algorithm);
  auto input = input_t.contiguous();

  const int64_t dim = input.dim();

  auto [input_reshaped_size, output_size, output_reshaped_size] =
      collapse_in_out_dim(input, dim, weight_t);
  Tensor output = at::empty(output_size, input.options());
  Tensor input_reshaped = input;
  if (dim != 2) {
    output = output.reshape(output_reshaped_size);
    input_reshaped = input_reshaped.reshape(input_reshaped_size);
  }

  auto bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
  at::native::onednn::matmul(
      output, input_reshaped, weight_t, bias, /*m2_trans*/ false, att);

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

Tensor linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view binary_attr) {
  const OptionalDeviceGuard device_guard(device_of(input_t));
  onednn::Attr attr;
  attr = construct_binary_attr<true>(attr, binary_attr, other_t);
  auto input = input_t.contiguous();

  const int64_t dim = input.dim();

  // dim collapse
  auto [input_reshaped_size, output_size, output_reshaped_size] =
      collapse_in_out_dim(input, dim, weight_t);
  Tensor output = at::empty(output_size, input.options());
  Tensor input_reshaped = input;

  if (dim != 2) {
    // input [m, k], weight [n, k], output [m, n]
    output = output.reshape(output_reshaped_size);
    input_reshaped = input_reshaped.reshape(input_reshaped_size);
  } else {
    TORCH_CHECK(
        output.dim() == other_t.dim(),
        "linear_binary_run expects the dimension of output and other tensor to be the same");
  }

  auto bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
  at::native::onednn::matmul(
      output, input_reshaped, weight_t, bias, /*m2_trans*/ false, attr);

  if (dim != 2) {
    output = output.reshape(output_size);
  }
  return output;
}

TORCH_LIBRARY_IMPL(mkldnn, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
      TORCH_FN(linear_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
      TORCH_FN(linear_pointwise_binary));
}

} // namespace at::native::xpu

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/DeviceGuard.h`
- `torch/library.h`
- `FusionUtils.h`


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

Files in the same folder (`aten/src/ATen/native/mkldnn/xpu`):

- [`Attention.cpp_docs.md`](./Attention.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`qlinear.h_docs.md`](./qlinear.h_docs.md)
- [`qconv.cpp_docs.md`](./qconv.cpp_docs.md)
- [`ScaledBlas.cpp_docs.md`](./ScaledBlas.cpp_docs.md)
- [`qconv.h_docs.md`](./qconv.h_docs.md)
- [`FusionUtils.cpp_docs.md`](./FusionUtils.cpp_docs.md)
- [`FusionUtils.h_docs.md`](./FusionUtils.h_docs.md)
- [`Conv.cpp_docs.md`](./Conv.cpp_docs.md)


## Cross-References

- **File Documentation**: `Linear.cpp_docs.md`
- **Keyword Index**: `Linear.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/mkldnn/xpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/mkldnn/xpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/mkldnn/xpu`):

- [`FusionUtils.cpp_kw.md_docs.md`](./FusionUtils.cpp_kw.md_docs.md)
- [`qconv.cpp_docs.md_docs.md`](./qconv.cpp_docs.md_docs.md)
- [`Conv.h_kw.md_docs.md`](./Conv.h_kw.md_docs.md)
- [`qconv.h_docs.md_docs.md`](./qconv.h_docs.md_docs.md)
- [`ScaledBlas.cpp_docs.md_docs.md`](./ScaledBlas.cpp_docs.md_docs.md)
- [`Attention.cpp_kw.md_docs.md`](./Attention.cpp_kw.md_docs.md)
- [`FusionUtils.h_docs.md_docs.md`](./FusionUtils.h_docs.md_docs.md)
- [`Blas.cpp_docs.md_docs.md`](./Blas.cpp_docs.md_docs.md)
- [`Conv.cpp_docs.md_docs.md`](./Conv.cpp_docs.md_docs.md)
- [`qconv.cpp_kw.md_docs.md`](./qconv.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Linear.cpp_docs.md_docs.md`
- **Keyword Index**: `Linear.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/ConvolutionTBC.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/ConvolutionTBC.cpp_docs.md`
- **Size**: 6,938 bytes (6.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/ConvolutionTBC.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/ConvolutionTBC.cpp`
- **Size**: 4,357 bytes (4.25 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/conv_tbc_backward_native.h>
#include <ATen/ops/conv_tbc_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

Tensor conv_tbc(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad) {
  TORCH_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  TORCH_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1-D");

  auto input_size = self.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  auto real_pad = (olen - ilen + kw - 1) / 2;

  // Make sure shapes are correct.
  // Input = (time, batch, in_channels)
  // Weight = (kernel_width, in_channels, out_channels)
  // Bias = (out_channels)
  TORCH_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  TORCH_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  // input * weights + bias -> output_features
  Tensor output = at::empty({
    olen,
    input_size[1],
    weight_size[2],
  }, self.options());
  output.copy_(bias.expand(output.sizes()));
  for (const auto k : c10::irange(kw)) {
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    long t = std::min(ilen + real_pad - k, olen) - oShift;
    // Note: gemm assumes column-major matrices
    // input    is l*m (row-major)
    // weight   is m*r (row-major)
    // output   is l*r (row-major)
    if (t > 0) {
      auto W = weight[k];
      auto I = self.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      O.addmm_(I, W);
    }
  }
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv_tbc_backward(const Tensor& dOutput, const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t pad) {
  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  auto outputPlanes = weight_size[2];
  auto kw = weight.sizes()[0];
  auto olen = input_size[0] - kw + 1 + pad * 2;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int real_pad = (olen - ilen + kw - 1) / 2;

  Tensor dInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // dOutput * T(weight) -> dInput
    if (t > 0) {
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      dI.addmm_(dO, weight[k].t());
    }
  }

  Tensor dWeight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  for (int k = 0; k < kw; k++) {
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // T(input) * dOutput -> dWeight
    if (t > 0) {
      auto dW = dWeight[k];
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      dW.addmm_(I, dO);
    }
  }

  Tensor dBias = at::zeros_like(bias, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto tmp = dOutput.sum(0, false);
  dBias.copy_(tmp.sum(0));

  return std::make_tuple(dInput, dWeight, dBias);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `c10/util/irange.h`
- `tuple`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/conv_tbc_backward_native.h`
- `ATen/ops/conv_tbc_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/zeros_like.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `ConvolutionTBC.cpp_docs.md`
- **Keyword Index**: `ConvolutionTBC.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ConvolutionTBC.cpp_docs.md_docs.md`
- **Keyword Index**: `ConvolutionTBC.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

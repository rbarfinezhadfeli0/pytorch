# Documentation: `docs/aten/src/ATen/native/AdaptiveAveragePooling.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/AdaptiveAveragePooling.cpp_docs.md`
- **Size**: 8,010 bytes (7.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/AdaptiveAveragePooling.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/AdaptiveAveragePooling.cpp`
- **Size**: 5,212 bytes (5.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d.h>
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/adaptive_avg_pool2d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d.h>
#endif


namespace at::native {

namespace {

  void adaptive_avg_pool2d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
    int64_t ndim = input.dim();
    TORCH_CHECK((ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ", input.sizes());
    for (const auto i : {-2, -1}) {
      TORCH_CHECK(input.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i + ndim, " being "
        "empty");
    }

    TORCH_CHECK(input.dtype() == output.dtype(),
      "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

    int64_t channels  = input.size(-3);
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];

    if (ndim == 3) {
      output.resize_({channels, output_height, output_width});
    } else {
      int64_t nbatch = input.size(0);
      output.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());
    }

    if (output.numel() == 0) {
      return;
    }

    adaptive_avg_pool2d_kernel(kCPU, output, input, output_size);
  }

  Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input)
  {
    adaptive_pool_empty_output_check(grad_output, "adaptive_avg_pool2d_backward");
    int64_t ndim = grad_output.dim();
    TORCH_CHECK(input.dim() == ndim,
      __func__, ": Expected dimensions ", input.dim(), " for `grad_output` but got dimensions ", ndim);
    TORCH_CHECK((ndim == 3 || ndim == 4),
      __func__, ": Expected 3D or 4D tensor, but got ", input.sizes());
    TORCH_CHECK(input.dtype() == grad_output.dtype(),
      __func__, ": Expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());
    TORCH_CHECK(input.dtype() == grad_input.dtype(),
      __func__, ": Expected dtype ", input.dtype(), " for `grad_input` but got dtype ", grad_input.dtype());

    grad_input.resize_(input.sizes(), input.suggest_memory_format());
    grad_input.zero_();

    adaptive_avg_pool2d_backward_kernel(kCPU, grad_input, grad_output);
    return grad_input;
  }

} // namespace

  Tensor& adaptive_avg_pool2d_out_cpu(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output)
  {
    adaptive_avg_pool2d_out_cpu_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d_cpu(
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    auto output = at::empty({0}, input.options());
    adaptive_avg_pool2d_out_cpu_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d_symint(at::Tensor const& input, SymIntArrayRef output_size) {
    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
    TORCH_CHECK(
        (output_size[0] >= 0 && output_size[1] >= 0),
        "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 ",
        "but received {", output_size[0], ", ", output_size[1], "}");

    if (input.is_mkldnn()) {
      return at::mkldnn_adaptive_avg_pool2d(input, C10_AS_INTARRAYREF_SLOW(output_size));
    }

    if (!input.is_quantized() && output_size[0] == 1 && output_size[1] == 1) {
      // in this case, adaptive pooling is just computing mean over hw
      // dimensions, which can be done more efficiently
      #if defined(C10_MOBILE) && defined(USE_XNNPACK)
      if (xnnpack::use_global_average_pool(input)) {
        return xnnpack::global_average_pool(input);
      }
      #endif

      Tensor out = input.mean({-1, -2}, /* keepdim = */ true);
      if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
        // assert ndim == 4, since ndim = 3 doesn't give channels_last
        const auto n = input.sym_size(0);
        const auto c = input.sym_size(1);
        out.as_strided__symint({n, c, 1, 1}, {c, 1, c, c});
      }
      return out;
    } else {
      return _adaptive_avg_pool2d_symint(input, output_size);
    }
  }

  Tensor adaptive_avg_pool2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input)
  {
    auto grad_input = at::empty({0}, input.options());
    adaptive_avg_pool2d_backward_out_cpu_template(
      grad_input, grad_output, input);
    return grad_input;
  }

DEFINE_DISPATCH(adaptive_avg_pool2d_kernel);
DEFINE_DISPATCH(adaptive_avg_pool2d_backward_kernel);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/native/AdaptivePooling.h`
- `ATen/native/xnnpack/Engine.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_adaptive_avg_pool2d.h`
- `ATen/ops/_adaptive_avg_pool2d_backward_native.h`
- `ATen/ops/_adaptive_avg_pool2d_native.h`
- `ATen/ops/adaptive_avg_pool2d_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/mkldnn_adaptive_avg_pool2d.h`


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

- **File Documentation**: `AdaptiveAveragePooling.cpp_docs.md`
- **Keyword Index**: `AdaptiveAveragePooling.cpp_kw.md`
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

- **File Documentation**: `AdaptiveAveragePooling.cpp_docs.md_docs.md`
- **Keyword Index**: `AdaptiveAveragePooling.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

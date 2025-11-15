# Documentation: `docs/aten/src/ATen/native/quantized/cpu/AveragePool3d.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/AveragePool3d.cpp_docs.md`
- **Size**: 8,392 bytes (8.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/AveragePool3d.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/AveragePool3d.cpp`
- **Size**: 5,636 bytes (5.50 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pool.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/avg_pool3d_native.h>
#endif

#include <vector>

namespace at::native {

DEFINE_DISPATCH(qavg_pool3d_nhwc_stub);

namespace {

inline std::tuple<int, int, int> get_kernel(IntArrayRef kernel_size) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);
  return std::make_tuple(kW, kH, kD);
}

inline std::tuple<int, int, int> get_stride(IntArrayRef stride, int kW, int kH, int kD) {
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty()
      ? kH
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);
  return std::make_tuple(dW, dH, dD);
}

inline std::tuple<int, int, int> get_padding(IntArrayRef padding) {
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);
  return std::make_tuple(padW, padH, padD);
}

std::vector<int64_t> get_output_shape(
    const Tensor& input_,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool ceil_mode) {
  const int64_t nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
  const int64_t nInputPlane = input_.size(-4);
  const int64_t inputDepth = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);
  const int64_t outputDepth =
      pooling_output_shape<int64_t>(inputDepth, kD, padD, dD, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  if (input_.ndimension() == 4) {
    return {nInputPlane, outputDepth, outputHeight, outputWidth};
  }
  return {nbatch, nInputPlane, outputDepth, outputHeight, outputWidth};
}

template <typename scalar_t>
Tensor q_avg_pool3d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  auto [kW, kH, kD] = get_kernel(kernel_size);
  auto [dW, dH, dD] = get_stride(stride, kW, kH, kD);
  auto [padW, padH, padD] = get_padding(padding);

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nInputPlane = input.size(-4);
  const int64_t inputDepth = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  auto output_shape =
      get_output_shape(input, kW, kH, kD, dW, dH, dD, padW, padH, padD, ceil_mode);
  const int64_t outputDepth = output_shape[output_shape.size() - 3];
  const int64_t outputHeight = output_shape[output_shape.size() - 2];
  const int64_t outputWidth = output_shape[output_shape.size() - 1];

  auto input_nhwc = input.contiguous(MemoryFormat::ChannelsLast3d);

  auto output = at::_empty_affine_quantized(
      output_shape,
      input_nhwc.options().memory_format(input_nhwc.suggest_memory_format()),
      input_nhwc.q_scale(),
      input_nhwc.q_zero_point(),
      std::nullopt);
  // fast path for channel last: qavg_pool_2d_nhwc_stub
  qavg_pool3d_nhwc_stub(
      input_nhwc.device().type(),
      input_nhwc,
      output,
      nbatch,
      nInputPlane,
      inputWidth,
      inputHeight,
      inputDepth,
      outputWidth,
      outputHeight,
      outputDepth,
      kW,
      kH,
      kD,
      dW,
      dH,
      dD,
      padW,
      padH,
      padD,
      count_include_pad,
      divisor_override);
  return output;
}

} // namespace

Tensor avg_pool3d_quantized_cpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  Tensor output;
  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "avg_pool3d_quantized_cpu", [&]() {
    output = q_avg_pool3d<scalar_t>(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  });
  return output;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/native/Pool.h`
- `ATen/native/quantized/cpu/init_qnnpack.h`
- `ATen/native/quantized/cpu/QnnpackUtils.h`
- `ATen/native/quantized/cpu/QuantizedOps.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_affine_quantized.h`
- `ATen/ops/avg_pool3d_native.h`
- `vector`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `AveragePool3d.cpp_docs.md`
- **Keyword Index**: `AveragePool3d.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu`):

- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`init_qnnpack.cpp_docs.md_docs.md`](./init_qnnpack.cpp_docs.md_docs.md)
- [`qelu.cpp_kw.md_docs.md`](./qelu.cpp_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)
- [`qclamp.cpp_docs.md_docs.md`](./qclamp.cpp_docs.md_docs.md)
- [`qembeddingbag_prepack.h_docs.md_docs.md`](./qembeddingbag_prepack.h_docs.md_docs.md)
- [`qdropout.cpp_docs.md_docs.md`](./qdropout.cpp_docs.md_docs.md)
- [`qelu.cpp_docs.md_docs.md`](./qelu.cpp_docs.md_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md_docs.md`](./qembeddingbag_unpack.cpp_docs.md_docs.md)
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `AveragePool3d.cpp_docs.md_docs.md`
- **Keyword Index**: `AveragePool3d.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

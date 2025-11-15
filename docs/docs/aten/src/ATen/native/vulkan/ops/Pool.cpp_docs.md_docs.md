# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Pool.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Pool.cpp_docs.md`
- **Size**: 10,867 bytes (10.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Pool.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Pool.cpp`
- **Size**: 8,480 bytes (8.28 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/Pool.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor adaptive_avg_pool2d(
    const at::Tensor& self_arg,
    const IntArrayRef output_size) {
  TORCH_CHECK(
      self_arg.dim() == 4,
      "Vulkan adaptive_avg_pool2d expects 4-dimensional input!");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      {
          self_arg.size(Layout::Activation4D::batch),
          self_arg.size(Layout::Activation4D::channels),
          output_size[Layout::Activation4D::batch],
          output_size[Layout::Activation4D::channels],
      },
      v_self.dtype(),
  };

  const uvec3 v_output_size = v_output.extents();
  const uvec3 v_self_size = v_self.extents();

  const vec2 stride{
      static_cast<float>(v_self_size.data[0u]) / v_output_size.data[0u],
      static_cast<float>(v_self_size.data[1u]) / v_output_size.data[1u],
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    vec2 kernel;
    vec2 stride;
  } block{
      v_output.extents(),
      0u,
      {
          v_self_size.data[0u] -
              (v_output_size.data[0u] - 1u) * stride.data[0u],
          v_self_size.data[1u] -
              (v_output_size.data[1u] - 1u) * stride.data[1u],
      },
      stride,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(adaptive_avg_pool2d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor pool2d(
    const Tensor& self_arg,
    const IntArrayRef kernel_arg,
    IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool ceil_mode,
    const api::ShaderInfo& shader_descriptor) {
  if (stride_arg.empty()) {
    stride_arg = kernel_arg;
  }

  TORCH_CHECK(!kernel_arg.empty(), "Kernel size cannot be empty!");
  TORCH_CHECK(!stride_arg.empty(), "Stride cannot be empty!");
  TORCH_CHECK(!padding_arg.empty(), "Padding cannot be empty!");

  static const auto normalize = [](const IntArrayRef parameter) {
    return std::array<int64_t, 2>{
        parameter[0],
        (2 == parameter.size()) ? parameter[1] : parameter[0],
    };
  };

  const auto input_size = self_arg.sizes();
  const auto kernel = normalize(kernel_arg);
  const auto stride = normalize(stride_arg);
  const auto padding = normalize(padding_arg);
  const auto dilation = normalize(dilation_arg);

  const int64_t output_height = pooling_output_shape(
      input_size[Layout::Activation4D::height],
      kernel[Layout::Parameter::height],
      padding[Layout::Parameter::height],
      stride[Layout::Parameter::height],
      dilation[Layout::Parameter::height],
      ceil_mode);

  const int64_t output_width = pooling_output_shape(
      input_size[Layout::Activation4D::width],
      kernel[Layout::Parameter::width],
      padding[Layout::Parameter::width],
      stride[Layout::Parameter::width],
      dilation[Layout::Parameter::width],
      ceil_mode);

  pool2d_shape_check(
      self_arg,
      kernel[Layout::Parameter::height],
      kernel[Layout::Parameter::width],
      stride[Layout::Parameter::height],
      stride[Layout::Parameter::width],
      padding[Layout::Parameter::height],
      padding[Layout::Parameter::width],
      dilation[Layout::Parameter::height],
      dilation[Layout::Parameter::width],
      input_size[Layout::Activation4D::channels],
      input_size[Layout::Activation4D::height],
      input_size[Layout::Activation4D::width],
      output_height,
      output_width,
      self_arg.suggest_memory_format());

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      {
          input_size[Layout::Activation4D::batch],
          input_size[Layout::Activation4D::channels],
          output_height,
          output_width,
      },
      v_self.dtype(),
  };
  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  api::UniformParamsBuffer params;
  const struct Block final {
    uvec3 extents;
    int32_t range;
    ivec4 kernel;
    ivec2 stride;
    ivec2 padding;
    ivec2 dilation;
  } block{
      v_output.extents(),
      safe_downcast<int32_t>(
          kernel[Layout::Parameter::width] * kernel[Layout::Parameter::height]),
      {
          safe_downcast<int32_t>(kernel[Layout::Parameter::width]),
          safe_downcast<int32_t>(kernel[Layout::Parameter::height]),
          safe_downcast<int32_t>(self_arg.size(Layout::Activation4D::width)),
          safe_downcast<int32_t>(self_arg.size(Layout::Activation4D::height)),
      },
      {
          safe_downcast<int32_t>(stride[Layout::Parameter::width]),
          safe_downcast<int32_t>(stride[Layout::Parameter::height]),
      },
      {
          safe_downcast<int32_t>(padding[Layout::Parameter::width]),
          safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      },
      {
          safe_downcast<int32_t>(dilation[Layout::Parameter::width]),
          safe_downcast<int32_t>(dilation[Layout::Parameter::height]),
      },
  };
  params = api::UniformParamsBuffer(context, block);

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor avg_pool2d(
    const Tensor& self_arg,
    const IntArrayRef kernel_arg,
    IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const bool ceil_mode,
    const bool /* count_include_pad */,
    const std::optional<int64_t> /* divisor_override */) {
  return pool2d(
      self_arg,
      kernel_arg,
      stride_arg,
      padding_arg,
      {1, 1},
      ceil_mode,
      VK_KERNEL(avg_pool2d));
}

Tensor max_pool2d(
    const Tensor& self_arg,
    const IntArrayRef kernel_arg,
    IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool ceil_mode) {
  if (self_arg.scalar_type() == kQUInt8) {
    return pool2d(
        self_arg,
        kernel_arg,
        stride_arg,
        padding_arg,
        dilation_arg,
        ceil_mode,
        VK_KERNEL(quantized_max_pool2d_quint8));
  } else if (self_arg.scalar_type() == kQInt8) {
    return pool2d(
        self_arg,
        kernel_arg,
        stride_arg,
        padding_arg,
        dilation_arg,
        ceil_mode,
        VK_KERNEL(quantized_max_pool2d_qint8));
  } else {
    return pool2d(
        self_arg,
        kernel_arg,
        stride_arg,
        padding_arg,
        dilation_arg,
        ceil_mode,
        VK_KERNEL(max_pool2d));
  }
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_adaptive_avg_pool2d"),
      TORCH_FN(adaptive_avg_pool2d));
  m.impl(TORCH_SELECTIVE_NAME("aten::avg_pool2d"), TORCH_FN(avg_pool2d));
  m.impl(TORCH_SELECTIVE_NAME("aten::max_pool2d"), TORCH_FN(max_pool2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `native`, `at`

**Classes/Structs**: `Block`, `Block`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/Pool.h`
- `ATen/native/vulkan/ops/Common.h`
- `torch/library.h`


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

- **File Documentation**: `Pool.cpp_docs.md`
- **Keyword Index**: `Pool.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/vulkan/ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/vulkan/ops`):

- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`Select.cpp_docs.md_docs.md`](./Select.cpp_docs.md_docs.md)
- [`Batchnorm.h_docs.md_docs.md`](./Batchnorm.h_docs.md_docs.md)
- [`Lstm.cpp_kw.md_docs.md`](./Lstm.cpp_kw.md_docs.md)
- [`Concat.cpp_kw.md_docs.md`](./Concat.cpp_kw.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Zero.cpp_kw.md_docs.md`](./Zero.cpp_kw.md_docs.md)
- [`Gru.h_kw.md_docs.md`](./Gru.h_kw.md_docs.md)
- [`Repeat.cpp_kw.md_docs.md`](./Repeat.cpp_kw.md_docs.md)
- [`Register.cpp_docs.md_docs.md`](./Register.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Pool.cpp_docs.md_docs.md`
- **Keyword Index**: `Pool.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

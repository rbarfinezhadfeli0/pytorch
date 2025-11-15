# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Upsample.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Upsample.cpp_docs.md`
- **Size**: 8,055 bytes (7.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Upsample.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Upsample.cpp`
- **Size**: 5,601 bytes (5.47 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/UpSample.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
using namespace api::utils;

static Tensor upsample_nearest2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
      "Invalid input!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const auto v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {
          v_input_sizes[Layout::Activation4D::batch],
          v_input_sizes[Layout::Activation4D::channels],
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  if (v_input.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_input.get_scale());
    v_output.set_zero_point(v_input.get_zero_point());
  }

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
    ivec2 iextents;
    vec2 scale;
  } block{
      v_output.extents(),
      0u,
      {
          safe_downcast<int32_t>(
              input_arg.size(Layout::Activation4D::width) - 1),
          safe_downcast<int32_t>(
              input_arg.size(Layout::Activation4D::height) - 1),
      },
      {
          compute_scales_value<float>(
              scales_w,
              v_input_sizes[Layout::Activation4D::width],
              output_sizes[Layout::Parameter::width]),
          compute_scales_value<float>(
              scales_h,
              v_input_sizes[Layout::Activation4D::height],
              output_sizes[Layout::Parameter::height]),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      v_input.is_quantized() ? VK_KERNEL(quantized_upsample_nearest2d)
                             : VK_KERNEL(upsample_nearest2d),
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
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

static Tensor upsample_bilinear2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
      "Invalid input!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      {
          get_dim<Dim4D::Batch>(v_input),
          get_dim<Dim4D::Channel>(v_input),
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  const api::utils::uvec3 output_extents = v_output.extents();
  const struct Block final {
    uvec3 oextents;
    uint32_t padding;
    ivec2 iextents;
    vec2 scale;
  } block{
      v_output.extents(), // oextents
      0u, // padding
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(input_arg) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(input_arg) - 1),
      }, // iextents
      {
          compute_scales_value<float>(
              scales_w,
              get_dim<Dim4D::Width>(input_arg),
              get_dim<Dim4D::Width>(v_output)),
          compute_scales_value<float>(
              scales_h,
              get_dim<Dim4D::Height>(input_arg),
              get_dim<Dim4D::Height>(v_output)),
      }, // scale
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  api::ShaderInfo shader_desc;
  if (align_corners) {
    shader_desc = VK_KERNEL(upsample_bilinear2d_align_true);
  } else {
    shader_desc = VK_KERNEL(upsample_bilinear2d_align_false);
  }
  context->submit_compute_job(
      // shader descriptor
      shader_desc,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      output_extents,
      // local work group size
      adaptive_work_group_size(output_extents),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d"),
      TORCH_FN(upsample_nearest2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d"),
      TORCH_FN(upsample_bilinear2d));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

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

- `ATen/native/UpSample.h`
- `ATen/native/vulkan/ops/Common.h`
- `ATen/native/vulkan/ops/QuantizedFunctions.h`
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

- **File Documentation**: `Upsample.cpp_docs.md`
- **Keyword Index**: `Upsample.cpp_kw.md`
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

- **File Documentation**: `Upsample.cpp_docs.md_docs.md`
- **Keyword Index**: `Upsample.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

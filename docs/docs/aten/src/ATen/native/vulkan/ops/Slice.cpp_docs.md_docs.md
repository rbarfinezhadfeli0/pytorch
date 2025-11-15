# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Slice.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Slice.cpp_docs.md`
- **Size**: 10,784 bytes (10.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Slice.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Slice.cpp`
- **Size**: 8,442 bytes (8.24 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor slice_4d(
    const Tensor& input_arg,
    const int64_t dim,
    const int64_t start,
    const int64_t end,
    const int64_t step,
    const uvec4& in_tsize,
    const uvec4& out_tsize,
    vTensor& v_output) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  uint32_t out_channels = out_tsize.data[1u];
  uint32_t in_channels = in_tsize.data[1u];

  uint32_t out_c_aligned = api::utils::align_up(out_channels, 4u);
  uint32_t in_c_aligned = api::utils::align_up(in_channels, 4u);

  const struct Block final {
    ivec3 size; // output texture size
    int32_t fill_0; // dummy
    ivec3 isize; // input texture size
    int32_t fill_1; // dummy
    uvec4 tensor_size; // output tensor size
    uvec4 itensor_size; // input tensor size
    uvec4 args; // input arguments (dim, start, end, step)
    uvec2 c_info; // tensor channels aligned to 4
  } block{
      api::utils::make_ivec3(v_output.extents()),
      0,
      api::utils::make_ivec3(v_self.extents()),
      0,
      out_tsize,
      in_tsize,
      {safe_downcast<uint32_t>(dim),
       safe_downcast<uint32_t>(start),
       safe_downcast<uint32_t>(end),
       safe_downcast<uint32_t>(step)},
      {out_c_aligned, in_c_aligned},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(slice_4d),
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

Tensor slice_width(
    const Tensor& input_arg,
    const int64_t start,
    const int64_t end,
    const int64_t step,
    vTensor& v_output) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  uvec3 src_offset{};
  uvec3 dst_offset{};

  if (step == 1) {
    src_offset.data[0u] = start;

    uvec3 copy_extents{
        safe_downcast<uint32_t>(end - start),
        v_self.extents().data[1u],
        v_self.extents().data[2u]};

    api::PipelineBarrier pipeline_barrier{};

    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // images
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // copy details
        copy_extents,
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);
  } else {
    uvec3 copy_extents{
        1u, v_self.extents().data[1u], v_self.extents().data[2u]};

    const auto x_max = v_self.extents().data[0u];

    for (int64_t x = start, x_new = 0; x < end; x += step, ++x_new) {
      if (x >= x_max) { // out of range
        continue;
      }

      src_offset.data[0u] = x;
      dst_offset.data[0u] = x_new;

      api::PipelineBarrier pipeline_barrier{};

      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // pipeline barrier
          pipeline_barrier,
          // images
          v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // copy details
          copy_extents,
          src_offset,
          dst_offset,
          // fence handle
          VK_NULL_HANDLE);
    }
  }

  return convert(v_output);
}

Tensor slice_height(
    const Tensor& input_arg,
    const int64_t start,
    const int64_t end,
    const int64_t step,
    vTensor& v_output) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  uvec3 src_offset{};
  uvec3 dst_offset{};

  if (step == 1) {
    src_offset.data[1u] = start;

    uvec3 copy_extents{
        v_self.extents().data[0u],
        safe_downcast<uint32_t>(end - start),
        v_self.extents().data[2u]};

    api::PipelineBarrier pipeline_barrier{};

    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // images
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // copy details
        copy_extents,
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);
  } else {
    uvec3 copy_extents{
        v_self.extents().data[0u], 1u, v_self.extents().data[2u]};

    const auto y_max = v_self.extents().data[1u];
    for (int64_t y = start, y_new = 0; y < end; y += step, ++y_new) {
      if (y >= y_max) { // out of range
        continue;
      }
      src_offset.data[1u] = y;
      dst_offset.data[1u] = y_new;

      api::PipelineBarrier pipeline_barrier{};

      context->submit_copy<api::VulkanImage, api::VulkanImage>(
          // pipeline barrier
          pipeline_barrier,
          // images
          v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          // copy details
          copy_extents,
          src_offset,
          dst_offset,
          // fence handle
          VK_NULL_HANDLE);
    }
  }

  return convert(v_output);
}

Tensor slice(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    const int64_t step) {
  TORCH_CHECK(step > 0, "slice step must be positive");
  auto nDims = safe_downcast<uint32_t>(self.dim());
  dim = maybe_wrap_dim(dim, nDims);
  DimVector newSizes(self.sizes().begin(), self.sizes().end());

  // handle optional parameters
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  // INT64_MAX stands for default value.
  if (start_val == INT64_MAX) {
    start_val = 0;
  }
  if (start_val < 0) {
    start_val += newSizes[dim];
  }
  if (end_val < 0) {
    end_val += newSizes[dim];
  }
  if (start_val < 0) {
    start_val = 0;
  } else if (start_val >= newSizes[dim]) {
    start_val = newSizes[dim];
  }
  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val >= newSizes[dim]) {
    end_val = newSizes[dim];
  }

  auto len = end_val - start_val;
  newSizes[dim] = (len + step - 1) / step; // round-up

  // generalize into 4D tensor
  uvec4 in_tsize{1u, 1u, 1u, 1u}, out_tsize{1u, 1u, 1u, 1u};
  for (const auto i : c10::irange(nDims)) {
    in_tsize.data[(4u - nDims) + i] = self.sizes()[i];
    out_tsize.data[(4u - nDims) + i] = newSizes[i];
  }
  dim += 4 - nDims;

  IntArrayRef output_sizes(newSizes);
  vTensor v_output{
      api::context(), output_sizes.vec(), convert_dtype(self.scalar_type())};

  if (dim == 3) {
    slice_width(self, start_val, end_val, step, v_output);
  } else if (dim == 2) {
    slice_height(self, start_val, end_val, step, v_output);
  } else {
    slice_4d(
        self, dim, start_val, end_val, step, in_tsize, out_tsize, v_output);
  }

  auto result = convert(v_output);
  namedinference::propagate_names(result, self);
  return result;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::slice.Tensor"), TORCH_FN(slice));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `native`, `at`

**Classes/Structs**: `Block`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/NamedTensorUtils.h`
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
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)
- [`Shape.cpp_docs.md`](./Shape.cpp_docs.md)
- [`Mean.cpp_docs.md`](./Mean.cpp_docs.md)
- [`UnaryOp.cpp_docs.md`](./UnaryOp.cpp_docs.md)
- [`Permute.cpp_docs.md`](./Permute.cpp_docs.md)
- [`Unsqueeze.cpp_docs.md`](./Unsqueeze.cpp_docs.md)
- [`Stack.cpp_docs.md`](./Stack.cpp_docs.md)


## Cross-References

- **File Documentation**: `Slice.cpp_docs.md`
- **Keyword Index**: `Slice.cpp_kw.md`
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

- **File Documentation**: `Slice.cpp_docs.md_docs.md`
- **Keyword Index**: `Slice.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Select.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Select.cpp_docs.md`
- **Size**: 15,611 bytes (15.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Select.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Select.cpp`
- **Size**: 13,192 bytes (12.88 KB)
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
Tensor select_batch_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[1], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (c, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, c / 4)[c % 4]
  */
  const struct Block final {
    ivec2 batch_info;
  } block{
      {static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index)}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_batch_4d),
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

Tensor select_depth_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_depth_3d),
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

Tensor select_depth_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, h, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, h, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 depth_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_depth_4d),
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

Tensor select_height_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[2]},
      v_input.dtype(),
  };
  // Input tensor is a (c, h, w)
  // Output tensor is a (c, w)
  // In shader, the input texture's coordinate is (w, h, c)
  // In shader, the output texture's coordinate is (w, c, 1)
  uint32_t w = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];
  uint32_t z = 1;
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(w),
       static_cast<int32_t>(c),
       static_cast<int32_t>(z),
       static_cast<int32_t>(index)}};

  // Encoding of c-channel is packed into texel, hence we only call ceil(c/4)
  // times to minimize invocation and read.
  // For the last dimension, it is the selected height. Shader will do a direct
  // lookup based on block.index.
  uvec3 global_workgroup_size{w, api::utils::div_up(c, 4u), z};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_height_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_size,
      // local work group size
      adaptive_work_group_size(global_workgroup_size),
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

Tensor select_height_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[3]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, c, w)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (w, c, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 height_info;
  } block{
      {static_cast<int32_t>(v_input_sizes[0]),
       static_cast<int32_t>(
           std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
       static_cast<int32_t>(index),
       0}};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_height_4d),
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

Tensor select_width_3d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1]},
      v_input.dtype(),
  };

  const struct Block final {
    ivec4 width_info;
  } block{
      {static_cast<int32_t>(v_output.extents().data[0u]),
       static_cast<int32_t>(v_output.extents().data[1u]),
       static_cast<int32_t>(v_output.extents().data[2u]),
       static_cast<int32_t>(index)}};

  // Input tensor is a (c, h, w)
  // Output tensor is a (c, h)
  // In shader, the input texture's coordinate is (w, h, c)
  // In shader, the output texture's coordinate is (h, c, 1)
  uint32_t h = v_output.extents().data[0u];
  uint32_t c = v_output.extents().data[1u];

  // Encoding of c-channel is packed into texel, hence we only call ceil(c/4)
  // times to minimize invocation and read.
  // For the last dimension, it is the selected width. Shader will do a direct
  // lookup based on block.index.
  uvec3 global_workgroup_size{h, api::utils::div_up(c, 4u), 1};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_width_3d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_size,
      // local work group size
      adaptive_work_group_size(global_workgroup_size),
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

Tensor select_width_4d(const Tensor& input_arg, uint32_t index) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {v_input_sizes[0], v_input_sizes[1], v_input_sizes[2]},
      v_input.dtype(),
  };
  /*
  Input tensor: (n, c, h, w)
  Output tensor: (n, c, h)
  Input texture coor: (w, h, texels_per_batch * n + c / 4)[c % 4]
    where texels_per_batch = ceil(number_of_channels / 4)
  Output texture coor: (h, c, n / 4)[n % 4]
  */
  const struct Block final {
    ivec4 width_info;
  } block{
      static_cast<int32_t>(v_input_sizes[0]),
      static_cast<int32_t>(std::ceil(static_cast<float>(v_input_sizes[1]) / 4)),
      static_cast<int32_t>(index),
      0};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(select_width_4d),
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

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
  TORCH_CHECK(
      self.dim() == 3 || self.dim() == 4,
      "Vulkan select only supports 3d and 4d tensors!");

  const int64_t size = self.size(dim);

  if (index < -size || index >= size) {
    TORCH_CHECK_INDEX(
        false,
        "select(): index ",
        index,
        " out of range for tensor of size ",
        self.sizes(),
        " at dimension ",
        dim);
  }
  if (index < 0) {
    index += size;
  }
  if (self.dim() == 3) {
    if (dim == 0) {
      return select_depth_3d(self, index);
    } else if (dim == 1) {
      return select_height_3d(self, index);
    } else {
      return select_width_3d(self, index);
    }
  } else { // self.dim() == 4
    if (dim == 0) {
      return select_batch_4d(self, index);
    } else if (dim == 1) {
      return select_depth_4d(self, index);
    } else if (dim == 2) {
      return select_height_4d(self, index);
    } else {
      return select_width_4d(self, index);
    }
  }
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::select.int"), TORCH_FN(select));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 58 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `native`, `at`

**Classes/Structs**: `Block`, `Block`, `Block`, `Block`, `Block`, `Block`, `Block`


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

- **File Documentation**: `Select.cpp_docs.md`
- **Keyword Index**: `Select.cpp_kw.md`
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
- [`Batchnorm.h_docs.md_docs.md`](./Batchnorm.h_docs.md_docs.md)
- [`Lstm.cpp_kw.md_docs.md`](./Lstm.cpp_kw.md_docs.md)
- [`Concat.cpp_kw.md_docs.md`](./Concat.cpp_kw.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Zero.cpp_kw.md_docs.md`](./Zero.cpp_kw.md_docs.md)
- [`Gru.h_kw.md_docs.md`](./Gru.h_kw.md_docs.md)
- [`Repeat.cpp_kw.md_docs.md`](./Repeat.cpp_kw.md_docs.md)
- [`Register.cpp_docs.md_docs.md`](./Register.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Select.cpp_docs.md_docs.md`
- **Keyword Index**: `Select.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

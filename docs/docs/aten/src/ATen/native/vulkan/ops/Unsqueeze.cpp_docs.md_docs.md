# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Unsqueeze.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Unsqueeze.cpp_docs.md`
- **Size**: 6,153 bytes (6.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Unsqueeze.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Unsqueeze.cpp`
- **Size**: 3,788 bytes (3.70 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct Block final {
  ivec2 info;
};

Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_CHECK(
      self.dim() <= 3,
      "Vulkan unsqueeze only supports up to 3d tensors as input!");
  TORCH_CHECK(
      dim >= -self.dim() - 1 && dim <= self.dim(),
      "Vulkan unsqueeze dimension out of range expected to be in range of [",
      -self.dim() - 1,
      ",",
      self.dim(),
      "], but got ",
      dim);

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // Create the output texture. For unsqueeze, add a dimension.
  std::vector<int64_t> output_size = v_input.sizes();
  if (dim < 0) {
    dim += (self.dim() + 1);
  }
  output_size.insert(output_size.begin() + dim, 1);
  // Create the output texture
  vTensor v_output{
      context,
      output_size,
      convert_dtype(self.scalar_type()),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Total number of work items is equal to the size of the output texture
  uvec3 global_size = v_output.extents();
  // Adaptively determine local work group size, will usually be {4, 4, 4}
  uvec3 local_size = adaptive_work_group_size(global_size);

  // When unsqueezing in the 0th dimension, only the metadata changes.
  // So we can perform a copy.
  if (dim == 0) {
    const vTensor& v_self = convert(self);
    uvec3 src_offset{};
    uvec3 dst_offset{};
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
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);
    return convert(v_output);
  }

  else {
    int channel_index = 1; // Channel dimension in a 3D tensor
    // Shift dim and channel_index for 1D, 2D tensors
    if (self.dim() < 3) {
      dim += (3 - self.dim());
      channel_index = 0;
    }

    // Create the params buffer
    struct Block block{{
        // Dimension to unsqueeze
        static_cast<int32_t>(dim),
        // Keep track of the channel in Image3D
        static_cast<int32_t>(
            std::ceil(static_cast<float>(output_size[channel_index]) / 4)),
    }};

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(unsqueeze),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        global_size,
        // local work group size
        local_size,
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
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::unsqueeze"), TORCH_FN(unsqueeze));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

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

- `ATen/native/vulkan/ops/Common.h`
- `ATen/native/vulkan/ops/Utils.h`
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
- [`Stack.cpp_docs.md`](./Stack.cpp_docs.md)


## Cross-References

- **File Documentation**: `Unsqueeze.cpp_docs.md`
- **Keyword Index**: `Unsqueeze.cpp_kw.md`
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

- **File Documentation**: `Unsqueeze.cpp_docs.md_docs.md`
- **Keyword Index**: `Unsqueeze.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

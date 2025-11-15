# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Concat.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Concat.cpp_docs.md`
- **Size**: 11,933 bytes (11.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Concat.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Concat.cpp`
- **Size**: 9,538 bytes (9.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/vulkan/ops/Common.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

namespace {
inline int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}
} // namespace

Tensor cat_batch(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

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
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);

    // Increment by the number of texels in the depth dimension
    dst_offset.data[2u] += v_self.extents().data[2u];
  }

  return convert(v_output);
}

Tensor cat_feature(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  // Determine the channels of the output tensor
  uint32_t ch_total = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_total += get_dim<Dim4D::Channel>(tensor);
  }

  // Running counter of the number of channels already appended.
  uint32_t ch_current = 0;
  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    // Determine the number of channel texels that will be modified by
    // appending this input tensor
    uint32_t start_ch4 = ch_current / 4;

    uint32_t end_ch4 =
        api::utils::div_up(ch_current + get_dim<Dim4D::Channel>(v_self), 4u);

    uint32_t ch4_range = end_ch4 - start_ch4;
    uint32_t nc4_range = ch4_range * get_dim<Dim4D::Batch>(v_self);

    const struct Block final {
      ivec3 outExtents;
      int32_t fill0;
      ivec3 inExtents;
      int32_t fill1;
      uvec2 outChInfo;
      uvec2 inChInfo;
      uvec4 appendedChInfo;
    } block{
        api::utils::make_ivec3(v_output.extents()),
        0,
        api::utils::make_ivec3(v_self.extents()),
        0,
        {
            ch_total,
            api::utils::div_up(ch_total, 4u),
        },
        {
            get_dim<Dim4D::Channel>(v_self),
            api::utils::align_up(get_dim<Dim4D::Channel>(v_self), 4u),
        },
        {
            ch_current,
            start_ch4,
            ch4_range,
            0u,
        },
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(cat_feature),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            get_dim<Dim4D::Width>(v_output),
            get_dim<Dim4D::Height>(v_output),
            nc4_range,
        },
        // local work group size
        adaptive_work_group_size(v_self.extents()),
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());

    ch_current += get_dim<Dim4D::Channel>(v_self);
  }

  return convert(v_output);
}

Tensor cat_feature_mult4ch(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  int64_t depth_size_allprior = 0;
  int64_t ch_interval = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_interval += get_dim<Dim4D::Channel>(tensor);
  }
  const int64_t depth_interval = ch_interval / 4;

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor_arg : tensors) {
    const Tensor tensor =
        tensor_arg.is_vulkan() ? tensor_arg : tensor_arg.vulkan();
    const vTensor& v_self = convert(tensor);

    const uint32_t depth_slice =
        safe_downcast<uint32_t>(get_dim<Dim4D::Channel>(tensor) / 4);

    uvec3 copy_extents{
        v_self.extents().data[0u], v_self.extents().data[1u], depth_slice};

    for (const auto b : c10::irange(get_dim<Dim4D::Batch>(tensor))) {
      src_offset.data[2u] = safe_downcast<uint32_t>(depth_slice * b);
      dst_offset.data[2u] =
          depth_size_allprior + safe_downcast<uint32_t>(depth_interval * b);

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

    depth_size_allprior += depth_slice;
  }

  return convert(v_output);
}

Tensor cat_width(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  // TORCH_CHECK(false, "Vulkan cat not implemented for width dimension!");
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

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
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);

    // Increment by width
    dst_offset.data[0u] += v_self.extents().data[0u];
  }

  return convert(v_output);
}

Tensor cat_height(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

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
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);

    // Increment by height
    dst_offset.data[1u] += v_self.extents().data[1u];
  }

  return convert(v_output);
}

Tensor cat(const at::ITensorListRef& tensors, const int64_t in_dim) {
  TORCH_CHECK(!tensors.empty(), "Vulkan cat expects at least one tensor");
  auto materialized = tensors.materialize();
  TORCH_INTERNAL_ASSERT(!materialized.empty(), "Accessing empty array");
  const at::Tensor& tensor = materialized[0];
  auto ndim = safe_downcast<uint32_t>(tensor.dim());
  const int64_t dim = normalize_dim(in_dim, ndim);
  int64_t cat_dim_size = 0;
  bool is_mult4ch = true;

  for (const at::Tensor& t : materialized) {
    TORCH_INTERNAL_ASSERT(
        t.dim() <= 4,
        "Vulkan cat expects inputs to have at most 4 dimensions, but got ",
        t.dim(),
        "d");

    if (ndim < 3 || get_dim<Dim4D::Channel>(t) % 4 != 0) {
      is_mult4ch = false;
    }

    for (const auto d : c10::irange(ndim)) {
      if (d == dim) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensor.size(d),
          "Vulkan cat inputs must have matching sizes except concatenated dimension");
    }
    cat_dim_size += t.size(dim);
  }

  auto result_size = tensor.sizes().vec();
  TORCH_INTERNAL_ASSERT(!result_size.empty(), "Accessing empty array");
  result_size[dim] = cat_dim_size;

  vTensor v_output{
      api::context(), result_size, convert_dtype(tensor.scalar_type())};

  if (dim == ndim - 1) {
    return cat_width(materialized, v_output);
  }
  if (dim == ndim - 2) {
    return cat_height(materialized, v_output);
  } else if (dim == ndim - 3) {
    if (is_mult4ch) {
      return cat_feature_mult4ch(materialized, v_output);
    }
    return cat_feature(materialized, v_output);
  }
  return cat_batch(materialized, v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::cat"), TORCH_FN(cat));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `Tensor`, `native`, `at`

**Classes/Structs**: `Block`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/ops/Common.h`
- `c10/util/irange.h`
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

- **File Documentation**: `Concat.cpp_docs.md`
- **Keyword Index**: `Concat.cpp_kw.md`
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

- **File Documentation**: `Concat.cpp_docs.md_docs.md`
- **Keyword Index**: `Concat.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

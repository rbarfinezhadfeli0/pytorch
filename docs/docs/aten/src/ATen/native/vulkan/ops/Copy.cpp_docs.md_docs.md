# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Copy.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Copy.cpp_docs.md`
- **Size**: 13,054 bytes (12.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Copy.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Copy.cpp`
- **Size**: 10,572 bytes (10.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/vulkan/Context.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

//
// Utility functions for memcpy
//

void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping) {
  if (src.dtype() == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping);
  } else if (src.dtype() == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping);
  } else if (src.dtype() == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping);
  } else if (src.dtype() == c10::kBool) {
    memcpy_to_mapping_uint8(src, dst_mapping);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kHalf, or at::Float but got ",
        src.dtype());
  }
}

void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst) {
  if (dst.dtype() == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst);
  } else if (dst.dtype() == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst);
  } else if (dst.dtype() == c10::kBool) {
    memcpy_from_mapping_bool(src_mapping, dst);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kHalf or at::Float but got ",
        dst.dtype());
  }
}

//
// CPU <-> GPU copy implementations (these functions use Transfer commands)
//

void transfer_cpu_to_vulkan(const Tensor& src, vTensor& v_dst) {
  api::Context* const context = api::context();

  // Convert to dtype corresponding to the image format of the texture to
  // ensure that byte alignment is consistent when copying. In some cases
  // a 16 bit format will be used for at::kFloat.
  Tensor src_nc4hw =
      utils::nchw_to_nc4hw(src).to(convert_dtype(v_dst.texture_dtype()));

  api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());
  // Copy data into the staging buffer
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    memcpy_to_mapping(src_nc4hw, mapping);
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
}

void transfer_vulkan_to_cpu(vTensor& v_src, Tensor& dst) {
  api::Context* const context = api::context();

  // Temporary tensor to receive copied NC4HW data
  at::Tensor dst_tmp = utils::create_staging_tensor(v_src);

  api::StorageBuffer staging(context, v_src.texture_dtype(), v_src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to comment in submit_compute_job. When syncing with the GPU, the
    // context must not allow other threads to record dispatches into it between
    // between calling vkQueueSubmit and flushing the context. Therefore,
    // cmd_mutex_ must be manually managed by the calling thread.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    api::PipelineBarrier pipeline_barrier{};
    utils::copy_vtensor_to_buffer(
        v_src, staging.buffer(), pipeline_barrier, fence.get_submit_handle());

    fence.wait();

    context->flush();
    // cmd_mutex_ will be released when exiting this scope.
  }

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    memcpy_from_mapping(mapping, dst_tmp);
  }

  context->fences().return_fence(fence);

  dst = utils::nc4hw_to_nchw(dst_tmp, v_src.sizes())
            .to(convert_dtype(v_src.dtype()));
}

static void transfer_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanImage, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // images
      src.image(pipeline_barrier, api::PipelineStage::TRANSFER),
      dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

//
// CPU <-> GPU copy implementations (these functions use compute shaders)
//

void pack_cpu_to_vulkan(const Tensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  // Ensure that src is contiguous in its memory format
  Tensor src_contig = src.contiguous(src.suggest_memory_format());

  // Note that the float data type has been enforced for the storage buffer
  // below. The reason for this is that the nchw_to_image and image_to_nchw
  // shaders which perform the transfer to/from an image texture expect a buffer
  // of floats as input. GLSL/Vulkan does not natively support 16 bit arithmetic
  // types, so for now storage buffers created for compute shaders must define
  // floats as their base data type.
  api::StorageBuffer staging(context, api::kFloat, dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    // If the dtype() of src is at::kHalf, then first convert it to 32 bit
    // float. This is required since the nchw_to_image shader uses a float
    // buffer as input (note that at::kFloat is used to create the StorageBuffer
    // above).
    if (src.dtype() == at::kHalf) {
      memcpy_to_mapping(src_contig.to(at::kFloat), mapping);
    } else {
      memcpy_to_mapping(src_contig, mapping);
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  TORCH_CHECK(
      !src.is_quantized(),
      "Copy of vulkan quantized tensors to cpu is currently disabled!");
  api::Context* const context = api::context();

  // Refer to the comment in pack_cpu_to_vulkan for why at::kFloat is specified
  // for the storage buffer below.
  api::StorageBuffer staging(context, api::kFloat, src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to comment in submit_compute_job. When syncing with the GPU, the
    // context must not allow other threads to record dispatches into it between
    // between calling vkQueueSubmit and flushing the context. Therefore,
    // cmd_mutex_ must be manually managed by the calling thread.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    bool submitted_to_gpu = utils::pack_vtensor_to_staging(
        src, staging.buffer(), fence.get_submit_handle());

    // Only wait on the fence if work was actually submitted to the GPU.
    // Otherwise, it will hang indefinitely.
    if (submitted_to_gpu) {
      fence.wait();
    }

    context->flush();
    // cmd_mutex_ will be released when exiting this scope.
  }

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    // If the dtype() of dst is at::kHalf, then copy the data into a float
    // version of it first, similar to pack_cpu_to_vulkan().
    if (dst.dtype() == at::kHalf) {
      Tensor dst_float = dst.to(at::kFloat);
      memcpy_from_mapping(mapping, dst_float);
      dst = dst_float.to(at::kHalf);
    } else {
      memcpy_from_mapping(mapping, dst);
    }
  }

  context->fences().return_fence(fence);
}

//
// Copy op implementations
//

Tensor& copy_(Tensor& dst, const Tensor& src) {
  // Check that sizes are equal
  TORCH_CHECK(
      dst.sizes() == src.sizes(), "Vulkan copy_: Tensor sizes are mismatched!");

  // X -> Vulkan
  if (at::kVulkan == dst.device().type()) {
    vTensor& v_self = convert(dst);

    // Vulkan -> Vulkan
    if (at::kVulkan == src.device().type()) {
      vTensor& v_src = convert(src);
      transfer_vulkan_to_vulkan(v_src, v_self);
    }
    // CPU -> Vulkan
    else {
      pack_cpu_to_vulkan(src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    vTensor& v_src = convert(src);

    // Vulkan -> CPU
    if (dst.device().is_cpu()) {
      pack_vulkan_to_cpu(v_src, dst);
    } else {
      TORCH_CHECK(false, "Unsupported!");
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid code path taken! Either the source or the destination tensor "
        "was expected to be Vulkan a tensor!  Incorrect dispatch?");
  }

  return dst;
}

vTensor to_vulkan(at::Tensor& src, const api::StorageType storage_type) {
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan(): input tensor must be a CPU tensor!")

  vTensor v_ret{
      api::context(),
      src.sizes().vec(),
      convert_dtype(src.scalar_type()),
      storage_type,
      get_gpu_memory_layout(storage_type, src.suggest_memory_format()),
  };

  ops::pack_cpu_to_vulkan(src, v_ret);

  return v_ret;
}

at::Tensor from_vulkan(vTensor& v_src) {
  at::TensorOptions opt(at::kCPU);
  opt = opt.dtype(convert_dtype(v_src.dtype()));

  c10::MemoryFormat v_src_memory_format;

  switch (v_src.gpu_memory_layout()) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      v_src_memory_format = c10::MemoryFormat::Contiguous;
      break;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      v_src_memory_format = c10::MemoryFormat::ChannelsLast;
      break;
    default:
      TORCH_CHECK(false, "No corresponding memory format");
  }

  at::Tensor ret = at::empty(v_src.sizes(), opt).to(v_src_memory_format);
  ops::pack_vulkan_to_cpu(v_src, ret);
  return ret;
}

//
// VulkanImpl
//

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return api::available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 43 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `native`, `at`

**Classes/Structs**: `VulkanImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/native/vulkan/ops/Copy.h`
- `ATen/native/vulkan/ops/Utils.h`
- `ATen/vulkan/Context.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `Copy.cpp_docs.md`
- **Keyword Index**: `Copy.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `Copy.cpp_docs.md_docs.md`
- **Keyword Index**: `Copy.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

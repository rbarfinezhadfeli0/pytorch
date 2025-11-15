# Documentation: `aten/src/ATen/native/vulkan/api/Command.h`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/api/Command.h`
- **Size**: 4,335 bytes (4.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class CommandBuffer final {
 public:
  explicit CommandBuffer(VkCommandBuffer, const VkCommandBufferUsageFlags);

  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  CommandBuffer(CommandBuffer&&) noexcept;
  CommandBuffer& operator=(CommandBuffer&&) noexcept;

  ~CommandBuffer() = default;

  // The lifecycle of a command buffer is as follows:
  enum State {
    INVALID, // Used to indicate the command buffer is moved from
    NEW, // Set during constructor
    RECORDING, // Set during call to begin(), dispatch(), and
               // copy_*_to_*()
    PIPELINE_BOUND, // Set during call to  bind_pipeline()
    DESCRIPTORS_BOUND, // Set during call to bind_descriptors()
    BARRIERS_INSERTED, // Set during call to insert_barrier()
    READY, //  Set during call to end()
    SUBMITTED, // Set during call to get_submit_handle()
  };

  struct Bound {
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    utils::uvec3 local_workgroup_size;
    VkDescriptorSet descriptors;

    explicit Bound()
        : pipeline{VK_NULL_HANDLE},
          pipeline_layout{VK_NULL_HANDLE},
          local_workgroup_size{0u, 0u, 0u},
          descriptors{VK_NULL_HANDLE} {}

    inline void reset() {
      pipeline = VK_NULL_HANDLE;
      pipeline_layout = VK_NULL_HANDLE;
      local_workgroup_size = {0u, 0u, 0u};
      descriptors = VK_NULL_HANDLE;
    }
  };

 private:
  VkCommandBuffer handle_;
  VkCommandBufferUsageFlags flags_;
  State state_;
  Bound bound_;

 public:
  inline bool is_reusable() {
    return !(flags_ & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  }

  inline void invalidate() {
    handle_ = VK_NULL_HANDLE;
    bound_.reset();
  }

  void begin();
  void end();

  void bind_pipeline(VkPipeline, VkPipelineLayout, const utils::uvec3);
  void bind_descriptors(VkDescriptorSet);

  void insert_barrier(PipelineBarrier& pipeline_barrier);
  void dispatch(const utils::uvec3&);

  void copy_buffer_to_buffer(
      const api::VulkanBuffer&,
      const api::VulkanBuffer&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void copy_texture_to_texture(
      const api::VulkanImage&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void copy_texture_to_buffer(
      const api::VulkanImage&,
      const api::VulkanBuffer&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void copy_buffer_to_texture(
      const api::VulkanBuffer&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void write_timestamp(VkQueryPool, const uint32_t) const;
  void reset_querypool(VkQueryPool, const uint32_t, const uint32_t) const;

  VkCommandBuffer get_submit_handle(const bool final_use = false);

  inline operator bool() const {
    return VK_NULL_HANDLE != handle_;
  }
};

struct CommandPoolConfig final {
  uint32_t cmdPoolInitialSize;
  uint32_t cmdPoolBatchSize;
};

class CommandPool final {
 public:
  explicit CommandPool(VkDevice, const uint32_t, const CommandPoolConfig&);

  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;

  CommandPool(CommandPool&&) = delete;
  CommandPool& operator=(CommandPool&&) = delete;

  ~CommandPool();

 private:
  VkDevice device_;
  uint32_t queue_family_idx_;
  VkCommandPool pool_;
  CommandPoolConfig config_;
  // New Buffers
  std::mutex mutex_;
  std::vector<VkCommandBuffer> buffers_;
  size_t in_use_;

 public:
  CommandBuffer get_new_cmd(bool reusable = false);

  void flush();

 private:
  void allocate_new_batch(const uint32_t);
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 29 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `api`, `native`, `at`

**Classes/Structs**: `CommandBuffer`, `Bound`, `CommandPoolConfig`, `CommandPool`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/api`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/api/vk_api.h`
- `ATen/native/vulkan/api/Descriptor.h`
- `ATen/native/vulkan/api/Pipeline.h`
- `ATen/native/vulkan/api/Resource.h`
- `ATen/native/vulkan/api/Shader.h`
- `ATen/native/vulkan/api/Utils.h`


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

Files in the same folder (`aten/src/ATen/native/vulkan/api`):

- [`Shader.h_docs.md`](./Shader.h_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`Pipeline.h_docs.md`](./Pipeline.h_docs.md)
- [`Adapter.h_docs.md`](./Adapter.h_docs.md)
- [`Adapter.cpp_docs.md`](./Adapter.cpp_docs.md)
- [`Types.h_docs.md`](./Types.h_docs.md)
- [`QueryPool.h_docs.md`](./QueryPool.h_docs.md)
- [`Allocator.h_docs.md`](./Allocator.h_docs.md)
- [`Command.cpp_docs.md`](./Command.cpp_docs.md)
- [`Descriptor.h_docs.md`](./Descriptor.h_docs.md)


## Cross-References

- **File Documentation**: `Command.h_docs.md`
- **Keyword Index**: `Command.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/native/vulkan/api/Descriptor.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/api/Descriptor.h_docs.md`
- **Size**: 5,794 bytes (5.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/api/Descriptor.h`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/api/Descriptor.h`
- **Size**: 3,313 bytes (3.24 KB)
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

#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>

#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class DescriptorSet final {
 public:
  explicit DescriptorSet(VkDevice, VkDescriptorSet, ShaderLayout::Signature);

  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;

  DescriptorSet(DescriptorSet&&) noexcept;
  DescriptorSet& operator=(DescriptorSet&&) noexcept;

  ~DescriptorSet() = default;

  struct ResourceBinding final {
    uint32_t binding_idx;
    VkDescriptorType descriptor_type;
    bool is_image;

    union {
      VkDescriptorBufferInfo buffer_info;
      VkDescriptorImageInfo image_info;
    } resource_info;
  };

 private:
  VkDevice device_;
  VkDescriptorSet handle_;
  ShaderLayout::Signature shader_layout_signature_;
  std::vector<ResourceBinding> bindings_;

 public:
  DescriptorSet& bind(const uint32_t, const VulkanBuffer&);
  DescriptorSet& bind(const uint32_t, const VulkanImage&);

  VkDescriptorSet get_bind_handle() const;

 private:
  void add_binding(const ResourceBinding& resource);
};

class DescriptorSetPile final {
 public:
  DescriptorSetPile(
      const uint32_t,
      VkDescriptorSetLayout,
      VkDevice,
      VkDescriptorPool);

  DescriptorSetPile(const DescriptorSetPile&) = delete;
  DescriptorSetPile& operator=(const DescriptorSetPile&) = delete;

  DescriptorSetPile(DescriptorSetPile&&) = default;
  DescriptorSetPile& operator=(DescriptorSetPile&&) = default;

  ~DescriptorSetPile() = default;

 private:
  uint32_t pile_size_;
  VkDescriptorSetLayout set_layout_;
  VkDevice device_;
  VkDescriptorPool pool_;
  std::vector<VkDescriptorSet> descriptors_;
  size_t in_use_;

 public:
  VkDescriptorSet get_descriptor_set();

 private:
  void allocate_new_batch();
};

struct DescriptorPoolConfig final {
  // Overall Pool capacity
  uint32_t descriptorPoolMaxSets;
  // DescriptorCounts by type
  uint32_t descriptorUniformBufferCount;
  uint32_t descriptorStorageBufferCount;
  uint32_t descriptorCombinedSamplerCount;
  uint32_t descriptorStorageImageCount;
  // Pile size for pre-allocating descriptor sets
  uint32_t descriptorPileSizes;
};

class DescriptorPool final {
 public:
  explicit DescriptorPool(VkDevice, const DescriptorPoolConfig&);

  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  DescriptorPool(DescriptorPool&&) = delete;
  DescriptorPool& operator=(DescriptorPool&&) = delete;

  ~DescriptorPool();

 private:
  VkDevice device_;
  VkDescriptorPool pool_;
  DescriptorPoolConfig config_;
  // New Descriptors
  std::mutex mutex_;
  std::unordered_map<VkDescriptorSetLayout, DescriptorSetPile> piles_;

 public:
  operator bool() const {
    return (pool_ != VK_NULL_HANDLE);
  }

  void init(const DescriptorPoolConfig& config);

  DescriptorSet get_descriptor_set(
      VkDescriptorSetLayout handle,
      const ShaderLayout::Signature& signature);

  void flush();
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `api`, `native`, `at`

**Classes/Structs**: `DescriptorSet`, `ResourceBinding`, `DescriptorSetPile`, `DescriptorPoolConfig`, `DescriptorPool`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/api`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/api/vk_api.h`
- `ATen/native/vulkan/api/Resource.h`
- `ATen/native/vulkan/api/Shader.h`
- `unordered_map`


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


## Cross-References

- **File Documentation**: `Descriptor.h_docs.md`
- **Keyword Index**: `Descriptor.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/vulkan/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/vulkan/api`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/vulkan/api`):

- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`Command.h_docs.md_docs.md`](./Command.h_docs.md_docs.md)
- [`Command.h_kw.md_docs.md`](./Command.h_kw.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`Exception.h_docs.md_docs.md`](./Exception.h_docs.md_docs.md)
- [`Tensor.h_docs.md_docs.md`](./Tensor.h_docs.md_docs.md)
- [`StringUtil.h_kw.md_docs.md`](./StringUtil.h_kw.md_docs.md)
- [`vk_api.h_docs.md_docs.md`](./vk_api.h_docs.md_docs.md)
- [`Pipeline.cpp_kw.md_docs.md`](./Pipeline.cpp_kw.md_docs.md)
- [`Descriptor.h_kw.md_docs.md`](./Descriptor.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Descriptor.h_docs.md_docs.md`
- **Keyword Index**: `Descriptor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

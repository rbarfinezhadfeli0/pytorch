# Documentation: `aten/src/ATen/native/vulkan/api/Descriptor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/api/Descriptor.cpp`
- **Size**: 7,799 bytes (7.62 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <algorithm>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// DescriptorSet
//

DescriptorSet::DescriptorSet(
    VkDevice device,
    VkDescriptorSet handle,
    ShaderLayout::Signature shader_layout_signature)
    : device_(device),
      handle_(handle),
      shader_layout_signature_(std::move(shader_layout_signature)),
      bindings_{} {}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_),
      shader_layout_signature_(std::move(other.shader_layout_signature_)),
      bindings_(std::move(other.bindings_)) {
  other.handle_ = VK_NULL_HANDLE;
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  shader_layout_signature_ = std::move(other.shader_layout_signature_);
  bindings_ = std::move(other.bindings_);

  other.handle_ = VK_NULL_HANDLE;

  return *this;
}

DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanBuffer& buffer) {
  VK_CHECK_COND(
      buffer.has_memory(),
      "Buffer must be bound to memory for it to be usable");

  DescriptorSet::ResourceBinding binder{};
  binder.binding_idx = idx; // binding_idx
  binder.descriptor_type = shader_layout_signature_[idx]; // descriptor_type
  binder.is_image = false; // is_image
  binder.resource_info.buffer_info.buffer = buffer.handle(); // buffer
  binder.resource_info.buffer_info.offset = buffer.mem_offset(); // offset
  binder.resource_info.buffer_info.range = buffer.mem_range(); // range
  add_binding(binder);

  return *this;
}

DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanImage& image) {
  VK_CHECK_COND(
      image.has_memory(), "Image must be bound to memory for it to be usable");

  VkImageLayout binding_layout = image.layout();
  if (shader_layout_signature_[idx] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
    binding_layout = VK_IMAGE_LAYOUT_GENERAL;
  }

  DescriptorSet::ResourceBinding binder{};
  binder.binding_idx = idx; // binding_idx
  binder.descriptor_type = shader_layout_signature_[idx]; // descriptor_type
  binder.is_image = true; // is_image
  binder.resource_info.image_info.sampler = image.sampler(); // buffer
  binder.resource_info.image_info.imageView = image.image_view(); // imageView
  binder.resource_info.image_info.imageLayout = binding_layout; // imageLayout
  add_binding(binder);

  return *this;
}

VkDescriptorSet DescriptorSet::get_bind_handle() const {
  std::vector<VkWriteDescriptorSet> write_descriptor_sets;

  for (const ResourceBinding& binding : bindings_) {
    VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
        nullptr, // pNext
        handle_, // dstSet
        binding.binding_idx, // dstBinding
        0u, // dstArrayElement
        1u, // descriptorCount
        binding.descriptor_type, // descriptorType
        nullptr, // pImageInfo
        nullptr, // pBufferInfo
        nullptr, // pTexelBufferView
    };

    if (binding.is_image) {
      write.pImageInfo = &binding.resource_info.image_info;
    } else {
      write.pBufferInfo = &binding.resource_info.buffer_info;
    }

    write_descriptor_sets.emplace_back(write);
  }

  vkUpdateDescriptorSets(
      device_,
      write_descriptor_sets.size(),
      write_descriptor_sets.data(),
      0u,
      nullptr);

  VkDescriptorSet ret = handle_;

  return ret;
}

void DescriptorSet::add_binding(const ResourceBinding& binding) {
  const auto bindings_itr = std::find_if(
      bindings_.begin(),
      bindings_.end(),
      [binding_idx = binding.binding_idx](const ResourceBinding& other) {
        return other.binding_idx == binding_idx;
      });

  if (bindings_.end() == bindings_itr) {
    bindings_.emplace_back(binding);
  } else {
    *bindings_itr = binding;
  }
}

//
// DescriptorSetPile
//

DescriptorSetPile::DescriptorSetPile(
    const uint32_t pile_size,
    VkDescriptorSetLayout descriptor_set_layout,
    VkDevice device,
    VkDescriptorPool descriptor_pool)
    : pile_size_{pile_size},
      set_layout_{descriptor_set_layout},
      device_{device},
      pool_{descriptor_pool},
      descriptors_{},
      in_use_(0u) {
  descriptors_.resize(pile_size_);
  allocate_new_batch();
}

VkDescriptorSet DescriptorSetPile::get_descriptor_set() {
  // No-ops if there are descriptor sets available
  allocate_new_batch();

  VkDescriptorSet handle = descriptors_[in_use_];
  descriptors_[in_use_] = VK_NULL_HANDLE;

  in_use_++;
  return handle;
}

void DescriptorSetPile::allocate_new_batch() {
  // No-ops if there are still descriptor sets available
  if (in_use_ < descriptors_.size() &&
      descriptors_[in_use_] != VK_NULL_HANDLE) {
    return;
  }

  std::vector<VkDescriptorSetLayout> layouts(descriptors_.size());
  fill(layouts.begin(), layouts.end(), set_layout_);

  const VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // descriptorPool
      utils::safe_downcast<uint32_t>(layouts.size()), // descriptorSetCount
      layouts.data(), // pSetLayouts
  };

  VK_CHECK(
      vkAllocateDescriptorSets(device_, &allocate_info, descriptors_.data()));

  in_use_ = 0u;
}

//
// DescriptorPool
//

DescriptorPool::DescriptorPool(
    VkDevice device,
    const DescriptorPoolConfig& config)
    : device_(device),
      pool_(VK_NULL_HANDLE),
      config_(config),
      mutex_{},
      piles_{} {
  if (config.descriptorPoolMaxSets > 0) {
    init(config);
  }
}

DescriptorPool::~DescriptorPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  vkDestroyDescriptorPool(device_, pool_, nullptr);
}

void DescriptorPool::init(const DescriptorPoolConfig& config) {
  VK_CHECK_COND(
      pool_ == VK_NULL_HANDLE,
      "Trying to init a DescriptorPool that has already been created!");

  config_ = config;

  std::vector<VkDescriptorPoolSize> type_sizes{
      {
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          config_.descriptorUniformBufferCount,
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          config_.descriptorStorageBufferCount,
      },
      {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          config_.descriptorCombinedSamplerCount,
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          config_.descriptorStorageBufferCount,
      },
  };

  const VkDescriptorPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      config_.descriptorPoolMaxSets, // maxSets
      static_cast<uint32_t>(type_sizes.size()), // poolSizeCounts
      type_sizes.data(), // pPoolSizes
  };

  VK_CHECK(vkCreateDescriptorPool(device_, &create_info, nullptr, &pool_));
}

DescriptorSet DescriptorPool::get_descriptor_set(
    VkDescriptorSetLayout set_layout,
    const ShaderLayout::Signature& signature) {
  VK_CHECK_COND(
      pool_ != VK_NULL_HANDLE, "DescriptorPool has not yet been initialized!");

  auto it = piles_.find(set_layout);
  if (piles_.cend() == it) {
    it = piles_
             .insert({
                 set_layout,
                 DescriptorSetPile(
                     config_.descriptorPileSizes, set_layout, device_, pool_),
             })
             .first;
  }

  VkDescriptorSet handle = it->second.get_descriptor_set();

  return DescriptorSet(device_, handle, signature);
}

void DescriptorPool::flush() {
  if (pool_ != VK_NULL_HANDLE) {
    VK_CHECK(vkResetDescriptorPool(device_, pool_, 0u));
    piles_.clear();
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `api`, `native`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/api`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/api/Descriptor.h`
- `ATen/native/vulkan/api/Utils.h`
- `algorithm`
- `utility`


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

- **File Documentation**: `Descriptor.cpp_docs.md`
- **Keyword Index**: `Descriptor.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

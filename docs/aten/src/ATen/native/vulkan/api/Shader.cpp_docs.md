# Documentation: `aten/src/ATen/native/vulkan/api/Shader.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/api/Shader.cpp`
- **Size**: 5,568 bytes (5.44 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <utility>

#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// ShaderInfo
//

ShaderInfo::ShaderInfo()
    : src_code{
          nullptr,
          0u,
      } {}

ShaderInfo::ShaderInfo(
    std::string name,
    const uint32_t* const spirv_bin,
    const uint32_t size,
    std::vector<VkDescriptorType>  layout)
    : src_code{
          spirv_bin,
          size,
      },
      kernel_name{std::move(name)},
      kernel_layout{std::move(layout)} {}

ShaderInfo::ShaderInfo(
    std::string name,
    const uint32_t* const spirv_bin,
    const uint32_t size,
    std::vector<VkDescriptorType>  layout,
    const std::vector<uint32_t>& tile_size,
    const StorageType bias_storage_type,
    const StorageType weight_storage_type)
    : src_code{
          spirv_bin,
          size,
      },
      kernel_name{std::move(name)},
      kernel_layout{std::move(layout)},
      tile_size(tile_size),
      bias_storage_type(bias_storage_type),
      weight_storage_type(weight_storage_type) {
  for (uint64_t i = 0; i < tile_size.size(); ++i) {
    out_tile_size.data[i] = tile_size[i];
  }
}

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2) {
  return (
      _1.src_code.bin == _2.src_code.bin &&
      _1.src_code.size == _2.src_code.size);
}

//
// ShaderLayout
//

ShaderLayout::ShaderLayout(
    VkDevice device,
    const ShaderLayout::Signature& signature)
    : device_(device), handle_{VK_NULL_HANDLE} {
  std::vector<VkDescriptorSetLayoutBinding> bindings;

  uint32_t binding_num = 0u;
  for (const VkDescriptorType type : signature) {
    bindings.push_back({
        binding_num++, // binding
        type, // descriptorType
        1u, // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
        nullptr, // pImmutableSamplers
    });
  }

  const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(bindings.size()), // bindingCount
      bindings.data(), // pBindings
  };

  VK_CHECK(vkCreateDescriptorSetLayout(
      device_, &descriptor_set_layout_create_info, nullptr, &handle_));
}

ShaderLayout::ShaderLayout(ShaderLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderLayout::~ShaderLayout() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDescriptorSetLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkDescriptorSetLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderModule
//

ShaderModule::ShaderModule(VkDevice device, const ShaderInfo& source)
    : device_(device), handle_{VK_NULL_HANDLE} {
  const uint32_t* code = source.src_code.bin;
  uint32_t size = source.src_code.size;

  const VkShaderModuleCreateInfo shader_module_create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // codeSize
      code, // pCode
  };

  VK_CHECK(vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &handle_));
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderModule::~ShaderModule() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyShaderModule(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkShaderModule tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderLayoutCache
//

ShaderLayoutCache::ShaderLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderLayoutCache::ShaderLayoutCache(ShaderLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

ShaderLayoutCache::~ShaderLayoutCache() {
  purge();
}

VkDescriptorSetLayout ShaderLayoutCache::retrieve(
    const ShaderLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, ShaderLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// ShaderCache
//

ShaderCache::ShaderCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderCache::ShaderCache(ShaderCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

ShaderCache::~ShaderCache() {
  purge();
}

VkShaderModule ShaderCache::retrieve(const ShaderCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, ShaderCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderCache::purge() {
  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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

- `utility`
- `ATen/native/vulkan/api/Shader.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `Shader.cpp_docs.md`
- **Keyword Index**: `Shader.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

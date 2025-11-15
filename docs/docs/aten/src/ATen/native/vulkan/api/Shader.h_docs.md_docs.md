# Documentation: `docs/aten/src/ATen/native/vulkan/api/Shader.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/api/Shader.h_docs.md`
- **Size**: 7,709 bytes (7.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/api/Shader.h`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/api/Shader.h`
- **Size**: 5,110 bytes (4.99 KB)
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

#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <mutex>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class ShaderLayout final {
 public:
  using Signature = std::vector<VkDescriptorType>;

  explicit ShaderLayout(VkDevice, const Signature&);

  ShaderLayout(const ShaderLayout&) = delete;
  ShaderLayout& operator=(const ShaderLayout&) = delete;

  ShaderLayout(ShaderLayout&&) noexcept;
  ShaderLayout& operator=(ShaderLayout&&) = delete;

  ~ShaderLayout();

 private:
  VkDevice device_;
  VkDescriptorSetLayout handle_;

 public:
  VkDescriptorSetLayout handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept;
};

struct ShaderInfo final {
  struct {
    const uint32_t* bin;
    uint32_t size;
  } src_code;

  std::string kernel_name{""};
  ShaderLayout::Signature kernel_layout{};

  // Shader Metadata
  utils::uvec3 out_tile_size{1u, 1u, 1u};

  std::vector<uint32_t> tile_size;
  StorageType bias_storage_type{StorageType::UNKNOWN};
  StorageType weight_storage_type{StorageType::UNKNOWN};

  explicit ShaderInfo();
  explicit ShaderInfo(std::string, const char*);
  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      std::vector<VkDescriptorType>);
  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      std::vector<VkDescriptorType>,
      const std::vector<uint32_t>& tile_size,
      const StorageType bias_storage_type,
      const StorageType weight_storage_type);
};

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2);

class ShaderModule final {
 public:
  explicit ShaderModule(VkDevice device, const ShaderInfo& source);

  ShaderModule(const ShaderModule&) = delete;
  ShaderModule& operator=(const ShaderModule&) = delete;

  ShaderModule(ShaderModule&&) noexcept;
  ShaderModule& operator=(ShaderModule&&) = delete;

  ~ShaderModule();

 private:
  VkDevice device_;
  VkShaderModule handle_;

 public:
  inline VkShaderModule handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept;
};

class ShaderLayoutCache final {
 public:
  explicit ShaderLayoutCache(VkDevice device);

  ShaderLayoutCache(const ShaderLayoutCache&) = delete;
  ShaderLayoutCache& operator=(const ShaderLayoutCache&) = delete;

  ShaderLayoutCache(ShaderLayoutCache&&) noexcept;
  ShaderLayoutCache& operator=(ShaderLayoutCache&&) = delete;

  ~ShaderLayoutCache();

  using Key = ShaderLayout::Signature;
  using Value = ShaderLayout;

  struct Hasher {
    inline size_t operator()(const ShaderLayout::Signature& signature) const {
      size_t hashed = 0u;

      for (const VkDescriptorType type : signature) {
        hashed =
            utils::hash_combine(hashed, std::hash<VkDescriptorType>()(type));
      }

      return hashed;
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkDescriptorSetLayout retrieve(const Key&);
  void purge();
};

class ShaderCache final {
 public:
  explicit ShaderCache(VkDevice device);

  ShaderCache(const ShaderCache&) = delete;
  ShaderCache& operator=(const ShaderCache&) = delete;

  ShaderCache(ShaderCache&&) noexcept;
  ShaderCache& operator=(ShaderCache&&) = delete;

  ~ShaderCache();

  using Key = ShaderInfo;
  using Value = ShaderModule;

  struct Hasher {
    inline size_t operator()(const ShaderInfo& source) const {
      size_t seed = 0;
      seed = utils::hash_combine(
          seed, std::hash<const uint32_t*>()(source.src_code.bin));
      seed = utils::hash_combine(
          seed, std::hash<uint32_t>()(source.src_code.size));

      return seed;
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkShaderModule retrieve(const Key&);
  void purge();
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

inline bool operator==(
    const VkDescriptorSetLayoutBinding& _1,
    const VkDescriptorSetLayoutBinding& _2) {
  return (
      _1.binding == _2.binding && _1.descriptorType == _2.descriptorType &&
      _1.descriptorCount == _2.descriptorCount &&
      _1.stageFlags == _2.stageFlags &&
      _1.pImmutableSamplers == _2.pImmutableSamplers);
}

#endif /* USE_VULKAN_API */

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `api`, `native`, `at`

**Classes/Structs**: `ShaderLayout`, `ShaderInfo`, `ShaderModule`, `ShaderLayoutCache`, `Hasher`, `ShaderCache`, `Hasher`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/api`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/api/vk_api.h`
- `ATen/native/vulkan/api/Types.h`
- `ATen/native/vulkan/api/Utils.h`
- `mutex`
- `unordered_map`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `Shader.h_docs.md`
- **Keyword Index**: `Shader.h_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `Shader.h_docs.md_docs.md`
- **Keyword Index**: `Shader.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

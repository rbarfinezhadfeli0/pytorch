# Documentation: `torch/csrc/jit/serialization/storage_context.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/storage_context.h`
- **Size**: 2,487 bytes (2.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ivalue.h>

namespace torch::jit {

// Used in torch.package and TorchScript serialization to coordinate
// sharing of storages between models. Also used to create deterministic
// naming for storages.
class TORCH_API SerializationStorageContext {
 public:
  explicit SerializationStorageContext() = default;
  SerializationStorageContext operator=(const SerializationStorageContext&) =
      delete;
  SerializationStorageContext(const SerializationStorageContext&) = delete;

  uint64_t getOrAddStorage(const c10::Storage& storage) {
    if (!hasStorage(storage)) {
      uint64_t size = storage_id_map_.size();
      storage_id_map_[storage] = size;
    }
    return storage_id_map_[storage];
  }

  bool hasStorage(const c10::Storage& storage) {
    return storage_id_map_.find(storage) != storage_id_map_.end();
  }

  ~SerializationStorageContext() = default;

 private:
  class StorageSerializationHash {
   public:
    size_t operator()(const c10::Storage& storage) const {
      return std::hash<void*>()(
          reinterpret_cast<void*>(storage.unsafeGetStorageImpl()));
    }
  };

  class StorageSerializationEqual {
   public:
    bool operator()(const c10::Storage& lhs, const c10::Storage& rhs) const {
      return lhs.unsafeGetStorageImpl() == rhs.unsafeGetStorageImpl();
    }
  };

  std::unordered_map<
      c10::Storage,
      uint64_t,
      StorageSerializationHash,
      StorageSerializationEqual>
      storage_id_map_;
};

// Used in torch.package and TorchScript deserialization to coordinate
// sharing of storages between models.
class TORCH_API DeserializationStorageContext {
 public:
  explicit DeserializationStorageContext() = default;
  DeserializationStorageContext operator=(
      const DeserializationStorageContext&) = delete;
  DeserializationStorageContext(const DeserializationStorageContext&) = delete;

  void addStorage(std::string name, c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(!hasStorage(name));
    name_storage_map_.emplace(std::move(name), std::move(storage));
  }

  bool hasStorage(const std::string& name) {
    return name_storage_map_.find(name) != name_storage_map_.end();
  }

  c10::Storage getStorage(const std::string& name) {
    TORCH_INTERNAL_ASSERT(hasStorage(name));
    return name_storage_map_.find(name)->second;
  }
  ~DeserializationStorageContext() = default;

 private:
  std::unordered_map<std::string, c10::Storage> name_storage_map_;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `StorageSerializationHash`, `StorageSerializationEqual`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/csrc/jit/serialization`):

- [`import_read.h_docs.md`](./import_read.h_docs.md)
- [`unpickler.h_docs.md`](./unpickler.h_docs.md)
- [`import_export_functions.h_docs.md`](./import_export_functions.h_docs.md)
- [`import.h_docs.md`](./import.h_docs.md)
- [`pickle.cpp_docs.md`](./pickle.cpp_docs.md)
- [`source_range_serialization_impl.h_docs.md`](./source_range_serialization_impl.h_docs.md)
- [`mobile_bytecode_generated.h_docs.md`](./mobile_bytecode_generated.h_docs.md)
- [`import_export_helpers.cpp_docs.md`](./import_export_helpers.cpp_docs.md)
- [`import_export_constants.h_docs.md`](./import_export_constants.h_docs.md)
- [`source_range_serialization.h_docs.md`](./source_range_serialization.h_docs.md)


## Cross-References

- **File Documentation**: `storage_context.h_docs.md`
- **Keyword Index**: `storage_context.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

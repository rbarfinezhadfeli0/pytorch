# Documentation: `docs/torch/csrc/jit/serialization/flatbuffer_serializer.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/serialization/flatbuffer_serializer.h_docs.md`
- **Size**: 5,810 bytes (5.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/serialization/flatbuffer_serializer.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/flatbuffer_serializer.h`
- **Size**: 3,043 bytes (2.97 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/mobile/module.h>

/**
 * Defines the public API for serializing mobile modules to flatbuffer.
 * Note that this header must not include or depend on flatbuffer-defined
 * types, to avoid leaking those details to PyTorch clients.
 */

namespace torch::jit {

/// Maps file names to file contents.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

/**
 * Represents a span of data. Typically owned by a UniqueDetachedBuffer.
 */
class TORCH_API DetachedBuffer final {
 public:
  /// Creates a new DetachedBuffer with an optional data owner. This interface
  /// is provided to let users create objects of this type for testing.
  DetachedBuffer(void* data, size_t size, void* internal_data_owner = nullptr)
      : data_(data), size_(size), data_owner_(internal_data_owner) {}

  /// Returns a pointer to the data.
  [[nodiscard]] void* data() {
    return data_;
  }
  /// Returns a pointer to the data.
  [[nodiscard]] const void* data() const {
    return data_;
  }
  /// Returns the size of the data, in bytes.
  [[nodiscard]] size_t size() const {
    return size_;
  }

  /// Wrapper type that typically owns data_owner_.
  using UniqueDetachedBuffer =
      std::unique_ptr<DetachedBuffer, std::function<void(DetachedBuffer*)>>;

 private:
  /// Deletes the owner, if present, and the buf itself.
  /// Note: we could have provided a movable type with a destructor that did
  /// this work, but the unique wrapper was easier in practice.
  static void destroy(DetachedBuffer* buf);

  /// Provides access to destroy() for implementation and testing.
  friend struct DetachedBufferFriend;
  friend struct DetachedBufferTestingFriend;

  /// Pointer to the data. Not owned by this class.
  void* data_;
  /// The size of `data_`, in bytes.
  size_t size_;
  /// Opaque pointer to the underlying owner of `data_`. This class
  /// (DetachedBuffer) does not own the owner or the data. It will typically be
  /// owned by a UniqueDetachedBuffer that knows how to delete the owner along
  /// with this class.
  void* data_owner_;
};

TORCH_API void save_mobile_module(
    const mobile::Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files = ExtraFilesMap(),
    const ExtraFilesMap& jit_sources = ExtraFilesMap(),
    const std::vector<IValue>& jit_constants = {});

TORCH_API DetachedBuffer::UniqueDetachedBuffer save_mobile_module_to_bytes(
    const mobile::Module& module,
    const ExtraFilesMap& extra_files = ExtraFilesMap(),
    const ExtraFilesMap& jit_sources = ExtraFilesMap(),
    const std::vector<IValue>& jit_constants = {});

TORCH_API void save_mobile_module_to_func(
    const mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func);

// TODO(qihan): delete
TORCH_API bool register_flatbuffer_serializer();

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `DetachedBufferFriend`, `DetachedBufferTestingFriend`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `functional`
- `memory`
- `string`
- `unordered_map`
- `vector`
- `ATen/core/ivalue.h`
- `c10/macros/Macros.h`
- `torch/csrc/jit/mobile/module.h`


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

- **File Documentation**: `flatbuffer_serializer.h_docs.md`
- **Keyword Index**: `flatbuffer_serializer.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/serialization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/serialization`):

- [`pickler.h_docs.md_docs.md`](./pickler.h_docs.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`import_export_functions.h_docs.md_docs.md`](./import_export_functions.h_docs.md_docs.md)
- [`import_export_helpers.h_docs.md_docs.md`](./import_export_helpers.h_docs.md_docs.md)
- [`flatbuffer_serializer_jit.cpp_kw.md_docs.md`](./flatbuffer_serializer_jit.cpp_kw.md_docs.md)
- [`source_range_serialization.cpp_kw.md_docs.md`](./source_range_serialization.cpp_kw.md_docs.md)
- [`export.cpp_kw.md_docs.md`](./export.cpp_kw.md_docs.md)
- [`import_read.h_kw.md_docs.md`](./import_read.h_kw.md_docs.md)
- [`pickle.cpp_kw.md_docs.md`](./pickle.cpp_kw.md_docs.md)
- [`export_bytecode.cpp_docs.md_docs.md`](./export_bytecode.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `flatbuffer_serializer.h_docs.md_docs.md`
- **Keyword Index**: `flatbuffer_serializer.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

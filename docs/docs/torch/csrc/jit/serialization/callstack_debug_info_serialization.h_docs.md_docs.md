# Documentation: `docs/torch/csrc/jit/serialization/callstack_debug_info_serialization.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/serialization/callstack_debug_info_serialization.h_docs.md`
- **Size**: 5,533 bytes (5.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/serialization/callstack_debug_info_serialization.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/callstack_debug_info_serialization.h`
- **Size**: 2,604 bytes (2.54 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/scope.h>

#include <ATen/core/ivalue.h>

#include <vector>

#include <c10/util/flat_hash_map.h>

namespace c10 {
struct IValue;
}

namespace torch::jit {

class Pickler;
class InlinedCallStackSerializer {
 public:
  // Serialize InlinedCallStack as
  // SerializedInlinedCallStack =
  // [module_info, source range tag, SerializedInlinedCallStack]
  // module_info = [ClassType.qualifiedName, instance_name]
  // source_range_tag = unique source range id
  c10::IValue serialize(
      const InlinedCallStackPtr& cs_ptr,
      const SourceRangeTagMap& source_range_tags);

 private:
  // module_info = [ClassType.qualifiedName, instance_name]
  c10::IValue serialize_module_instance_info(
      const std::optional<ModuleInstanceInfo>& m);

  // This caches serialized inlined callstack ptr, since many
  // InlinedCallStackPtr can refer to the same one.
  ska::flat_hash_map<InlinedCallStackPtr, c10::IValue>
      serialized_inlined_callstack_;
  // This caches serialized module instance info.
  // There might be many nodes that are part of the same
  // parent, grandparent etc. module.
  ska::flat_hash_map<std::string, c10::IValue> serialized_module_instance_info_;
};

class TORCH_API CallStackDebugInfoPickler {
 public:
  CallStackDebugInfoPickler() = default;

  std::vector<char> pickle(
      const std::unordered_map<int64_t, DebugInfoTuple>& callstack_ptrs,
      const SourceRangeTagMap& source_range_tags);

 private:
  InlinedCallStackSerializer css_;
};

class InlinedCallStackDeserializer {
 public:
  InlinedCallStackPtr deserialize(
      const c10::IValue& iv,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  std::optional<ModuleInstanceInfo> deserialize_module_instance_info(
      const c10::IValue& iv,
      const std::shared_ptr<CompilationUnit>& cu);

  ska::
      flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, InlinedCallStackPtr>
          cached_inlined_callstacks_;
  ska::flat_hash_map<c10::intrusive_ptr<c10::ivalue::Tuple>, ModuleInstanceInfo>
      cached_module_instance_info_;
};

class TORCH_API CallStackDebugInfoUnpickler {
 public:
  ska::flat_hash_map<int64_t, DebugInfoTuple> unpickle(
      const at::DataPtr& data,
      size_t size,
      const ska::flat_hash_map<int64_t, SourceRange>& source_range_map,
      const std::shared_ptr<CompilationUnit>& cu);

 private:
  InlinedCallStackDeserializer csds_;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `c10`

**Classes/Structs**: `IValue`, `Pickler`, `InlinedCallStackSerializer`, `TORCH_API`, `InlinedCallStackDeserializer`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Allocator.h`
- `torch/csrc/jit/frontend/source_range.h`
- `torch/csrc/jit/ir/scope.h`
- `ATen/core/ivalue.h`
- `vector`
- `c10/util/flat_hash_map.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `callstack_debug_info_serialization.h_docs.md`
- **Keyword Index**: `callstack_debug_info_serialization.h_kw.md`
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

- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `callstack_debug_info_serialization.h_docs.md_docs.md`
- **Keyword Index**: `callstack_debug_info_serialization.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

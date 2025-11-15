# Documentation: `docs/torch/csrc/jit/mobile/flatbuffer_loader.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/flatbuffer_loader.h_docs.md`
- **Size**: 7,351 bytes (7.18 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/flatbuffer_loader.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/flatbuffer_loader.h`
- **Size**: 4,774 bytes (4.66 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/mobile/module.h>
#include <optional>

/**
 * Defines the public API for loading flatbuffer-serialized mobile modules.
 * Note that this header must not include or depend on flatbuffer-defined
 * types, to avoid leaking those details to PyTorch clients.
 */

namespace torch::jit {

/// All non-copied data pointers provided to `parse_and_initialize_*` functions
/// must be aligned to this boundary. Since the Module will point directly into
/// the data, this alignment is necessary to ensure that certain types/structs
/// are properly aligned.
constexpr size_t kFlatbufferDataAlignmentBytes = 16;

/// Maps file names to file contents.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

// On high level, to produce a Module from a file on disk, we need to go
// through the follow steps:
// 1. Read: Read the file from disk -> memory
// 2. Deserialize: Parse the bytes to produce some in memory manipulable
//    structure
// 3. Module initialization: Produce mobile::Module out of the structure
//    produced in 2.
// Under this context, the structure described in 2. is the flatbuffer-defined
// type mobile::serialization::Module. However, this step/type is not visible in
// the public API.

// Parse a mobile::Module from raw bytes.
//
// This function does steps 2+3 described above.
//
// Does not take ownership of `data`; if you want it to take ownership, see the
// shared_ptr overload of this function.
//
// If should_copy_tensor_memory is true, then the returned module will NOT have
// references to `data`, so `data` can be freed immediately.
//
// If should_copy_tensor_memory is false, then returned module will have tensors
// that points inside of `data`; the caller will need to make sure that `data`
// outlives the returned Module. Also, `data` must be aligned to
// kFlatbufferDataAlignmentBytes.
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t size, // of `data`, in bytes.
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr,
    bool should_copy_tensor_memory = false);

// Parse a mobile::Module from raw bytes.
//
// This function does steps 2+3 described above.
//
// The returned Module holds a reference to `data`, which must be aligned to
// kFlatbufferDataAlignmentBytes.
//
// If you do not want the Module to hold a reference to `data`, see the raw
// pointer overload of this function.
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    std::shared_ptr<char> data,
    size_t size, // of `data`, in bytes.
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

// Parse a mobile::Module from raw bytes, also returning JIT-related metadata.
//
// This is the same as parse_and_initialize_mobile_module() except that it also
// extracts JIT source files and constants. Can be used to construct a
// jit::Module.
TORCH_API mobile::Module parse_and_initialize_mobile_module_for_jit(
    void* data,
    size_t size, // of `data`, in bytes.
    ExtraFilesMap& jit_sources,
    std::vector<IValue>& jit_constants,
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

// Load a mobile::Module from a filepath.
//
// This function does steps 1+2+3 described above.
//
// We need to have this as a convenience because Python API will need to wrap
// this. C++ clients should use one of the versions of
// parse_and_initialize_mobile_module() so they can manage the raw data more
// directly.
TORCH_API mobile::Module load_mobile_module_from_file(
    const std::string& filename,
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

TORCH_API uint64_t get_bytecode_version(std::istream& in);
TORCH_API uint64_t get_bytecode_version(const std::string& filename);
TORCH_API uint64_t get_bytecode_version_from_bytes(char* flatbuffer_content);

TORCH_API mobile::ModuleInfo get_module_info_from_flatbuffer(
    char* flatbuffer_content);

// The methods below are less efficient because it need to read the stream in
// its entirety to a buffer
TORCH_API mobile::Module load_mobile_module_from_stream_with_copy(
    std::istream& in,
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

TORCH_API mobile::Module parse_flatbuffer_no_object(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device);

// no op, TODO(qihan) delete
TORCH_API bool register_flatbuffer_loader();

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `istream`
- `memory`
- `string`
- `unordered_map`
- `vector`
- `ATen/core/ivalue.h`
- `c10/core/Device.h`
- `c10/macros/Macros.h`
- `torch/csrc/jit/mobile/module.h`
- `optional`


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

Files in the same folder (`torch/csrc/jit/mobile`):

- [`register_ops_common_utils.cpp_docs.md`](./register_ops_common_utils.cpp_docs.md)
- [`import.h_docs.md`](./import.h_docs.md)
- [`prim_ops_registery.h_docs.md`](./prim_ops_registery.h_docs.md)
- [`profiler_edge.h_docs.md`](./profiler_edge.h_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`file_format.h_docs.md`](./file_format.h_docs.md)
- [`module.h_docs.md`](./module.h_docs.md)
- [`observer.h_docs.md`](./observer.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`flatbuffer_loader.cpp_docs.md`](./flatbuffer_loader.cpp_docs.md)


## Cross-References

- **File Documentation**: `flatbuffer_loader.h_docs.md`
- **Keyword Index**: `flatbuffer_loader.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/mobile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/mobile`):

- [`code.h_docs.md_docs.md`](./code.h_docs.md_docs.md)
- [`register_ops_common_utils.cpp_docs.md_docs.md`](./register_ops_common_utils.cpp_docs.md_docs.md)
- [`observer.h_kw.md_docs.md`](./observer.h_kw.md_docs.md)
- [`prim_ops_registery.cpp_kw.md_docs.md`](./prim_ops_registery.cpp_kw.md_docs.md)
- [`quantization.h_docs.md_docs.md`](./quantization.h_docs.md_docs.md)
- [`debug_info.cpp_kw.md_docs.md`](./debug_info.cpp_kw.md_docs.md)
- [`interpreter.cpp_kw.md_docs.md`](./interpreter.cpp_kw.md_docs.md)
- [`debug_info.h_docs.md_docs.md`](./debug_info.h_docs.md_docs.md)
- [`interpreter.cpp_docs.md_docs.md`](./interpreter.cpp_docs.md_docs.md)
- [`promoted_prim_ops.cpp_docs.md_docs.md`](./promoted_prim_ops.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `flatbuffer_loader.h_docs.md_docs.md`
- **Keyword Index**: `flatbuffer_loader.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

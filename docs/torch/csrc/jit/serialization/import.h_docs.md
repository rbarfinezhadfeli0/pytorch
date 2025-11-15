# Documentation: `torch/csrc/jit/serialization/import.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/import.h`
- **Size**: 4,831 bytes (4.72 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <istream>

namespace caffe2::serialize {
class ReadAdapterInterface;
} // namespace caffe2::serialize

namespace torch::jit {

class DeserializationStorageContext;

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true,
    bool restore_shapes = false);

// For reading unified serialization format from torch.Package
TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    std::shared_ptr<torch::jit::DeserializationStorageContext> storage_context,
    std::optional<at::Device> device,
    const std::string& ts_id /* torchscript identifier inside package */);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::istream& in,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true,
    bool restore_shapes = false);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::unique_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

TORCH_API Module import_ir_module(
    std::shared_ptr<CompilationUnit> cu,
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given `istream`.
///
/// The istream must contain a serialized `Module`, exported via
/// `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::istream& in,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module load(
    std::istream& in,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given `filename`.
///
/// The file stored at the location given in `filename` must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module load(
    const std::string& filename,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

/// Loads a serialized `Module` from the given shared_ptr `rai`.
///
/// The reader adapter, which is for customized input stream, must contain a
/// serialized `Module`, exported either via `ScriptModule.save()` in
/// Python or `torch::jit::ExportModule` in C++.
TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device = std::nullopt,
    bool load_debug_files = true);

TORCH_API Module load(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai,
    std::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    bool load_debug_files = true);

TORCH_API Module jitModuleFromSourceAndConstants(
    const IValue& ivalue,
    const ExtraFilesMap& source,
    const std::vector<IValue>& constants,
    int32_t version);

TORCH_API Module parse_and_initialize_jit_module(
    const std::shared_ptr<char>& data,
    size_t size,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

TORCH_API Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

TORCH_API Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    std::optional<at::Device> device = std::nullopt);

TORCH_API c10::intrusive_ptr<c10::ivalue::Object> ObjLoaderFunc(
    const at::StrongTypePtr& type,
    IValue input);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `torch`

**Classes/Structs**: `ReadAdapterInterface`, `DeserializationStorageContext`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `caffe2/serialize/inline_container.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/ir/ir.h`
- `istream`


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
- [`pickle.cpp_docs.md`](./pickle.cpp_docs.md)
- [`source_range_serialization_impl.h_docs.md`](./source_range_serialization_impl.h_docs.md)
- [`mobile_bytecode_generated.h_docs.md`](./mobile_bytecode_generated.h_docs.md)
- [`import_export_helpers.cpp_docs.md`](./import_export_helpers.cpp_docs.md)
- [`import_export_constants.h_docs.md`](./import_export_constants.h_docs.md)
- [`source_range_serialization.h_docs.md`](./source_range_serialization.h_docs.md)


## Cross-References

- **File Documentation**: `import.h_docs.md`
- **Keyword Index**: `import.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

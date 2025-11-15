# Documentation: `docs/torch/csrc/jit/serialization/import_source.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/serialization/import_source.h_docs.md`
- **Size**: 6,402 bytes (6.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/serialization/import_source.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/import_source.h`
- **Size**: 3,424 bytes (3.34 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ivalue_inl.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/custom_class.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace torch::jit {

using SourceLoader = std::function<std::shared_ptr<Source>(const std::string&)>;

struct SourceImporterImpl : public Resolver,
                            std::enable_shared_from_this<SourceImporterImpl> {
  SourceImporterImpl(
      std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::IValue>* constant_table,
      SourceLoader source_loader,
      size_t version);
  TypePtr findNamedType(const QualifiedName& name);
  Function* findFunction(const QualifiedName& name);
  void parseSourceIfNeeded(const std::string& qualifier);
  void LEGACY_import_methods(
      const Module& mod,
      const std::shared_ptr<Source>& src);

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override;
  TypePtr resolveType(const std::string& name, const SourceRange& loc) override;

 private:
  void importFunction(const std::string& qualifier, const Def& def);
  void importNamedType(const std::string& qualifier, const ClassDef& class_def);
  std::optional<Assign> attributeAssignmentSpecialHandlingHack(
      const QualifiedName& qualified_classname,
      const Assign& assign);
  void importClass(
      const QualifiedName& qualified_classname,
      const ClassDef& class_def,
      bool is_module);
  void importEnum(
      const QualifiedName& qualified_name,
      const ClassDef& enum_def);
  void importNamedTuple(
      const QualifiedName& qualified_name,
      const ClassDef& named_tuple_def);

  void parsePossibleVersionNumber(Lexer& L);

  void parseImports(Lexer& L);

  std::shared_ptr<CompilationUnit> cu_;
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
  SourceLoader source_loader_;
  std::optional<size_t> version_ = std::nullopt;
  std::unordered_set<std::string> loaded_sources_;
  // named types and functions loaded from a file but not yet defined because
  // their type has not been requested yet.
  std::unordered_map<QualifiedName, TreeRef> to_be_defined_;
};

// Given a directory of serialized TorchScript sources,
// This class allows the loading of individual named types in source.
// Resolves the dependencies between source files and parses
// the source files as necessary.

struct TORCH_API SourceImporter {
  SourceImporter(
      // The compilation unit that will own the imported source
      std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::IValue>* constant_table,
      SourceLoader loader,
      size_t version);

  TypePtr loadType(const QualifiedName& name) const;

  // Add the methods defined in `src` to the module `mod`, using SourceImporter
  // to resolve any classes via loadType
  void LEGACY_import_methods(
      const Module& mod,
      const std::shared_ptr<Source>& src);
  ~SourceImporter();

 private:
  std::shared_ptr<SourceImporterImpl> pImpl;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `SourceImporterImpl`, `allows`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue_inl.h`
- `ATen/core/qualified_name.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/parser.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/frontend/script_type_parser.h`
- `torch/csrc/jit/frontend/source_range.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/serialization/export.h`
- `torch/custom_class.h`
- `functional`
- `memory`
- `optional`
- `string`
- `vector`


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

- **File Documentation**: `import_source.h_docs.md`
- **Keyword Index**: `import_source.h_kw.md`
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

- **File Documentation**: `import_source.h_docs.md_docs.md`
- **Keyword Index**: `import_source.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

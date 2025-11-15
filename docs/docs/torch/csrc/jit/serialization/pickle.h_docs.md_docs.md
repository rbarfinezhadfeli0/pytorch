# Documentation: `docs/torch/csrc/jit/serialization/pickle.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/serialization/pickle.h_docs.md`
- **Size**: 7,421 bytes (7.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/serialization/pickle.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/pickle.h`
- **Size**: 4,690 bytes (4.58 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace torch::jit {

/// Pickle an IValue by calling a function to handle writing the data.
///
/// `writer` is a function that takes in a pointer to a chunk of memory and its
/// size and consumes it.
///
/// See `jit::pickle` for more details.
TORCH_API void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

/// Save a `torch::IValue` in a format compatible with Python's `pickle` module
///
/// If present, `tensor_table` is a pointer to a table in which tensors that
/// are contained within `ivalue` are stored, and the bytes returned by the
/// pickler will only include references to these tensors in the table. This can
/// be used to keep the binary blob size small.
/// If not provided, tensors are stored in the same byte stream as the pickle
/// data, similar to `torch.save()` in eager Python.
///
/// Pickled values can be loaded in Python and C++:
/// \rst
/// .. code-block:: cpp
///
///  torch::IValue float_value(2.3);
///
///  // TODO: when tensors are stored in the pickle, delete this
///  std::vector<at::Tensor> tensor_table;
///  auto data = torch::jit::pickle(float_value, &tensor_table);
///
///  std::vector<torch::IValue> ivalues =
///      torch::jit::unpickle(data.data(), data.size());
///
/// .. code-block:: python
///
///   values = torch.load('data.pkl')
///   print(values)
///
/// \endrst
TORCH_API std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

/// Save a `torch::IValue` in a format that can be loaded by both
/// `torch::pickle_load` in C++ and `torch.load` in Python.
TORCH_API std::vector<char> pickle_save(const IValue& ivalue);

/// Deserialize a `torch::IValue` from bytes produced by either
/// `torch::pickle_save` in C++ or `torch.save` in Python
TORCH_API IValue pickle_load(const std::vector<char>& data);

/// Deserialize a `torch::IValue` from bytes produced by either
/// `torch::pickle_save` in C++ or `torch.save` in Python with custom object.
TORCH_API IValue pickle_load_obj(std::string_view data);

/// `reader` is a function that takes in a size to read from some pickled
/// binary. `reader` should remember where it last read, and return
/// the number of bytes read.
/// See `torch::pickle` for details.
/// type_resolver is used to resolve any JIT type based on type str
TORCH_API IValue unpickle(
    std::function<size_t(char*, size_t)> reader,
    TypeResolver type_resolver,
    c10::ArrayRef<at::Tensor> tensor_table,
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser,
    ObjLoader obj_loader = nullptr);

/// Decode a chunk of memory containing pickled data into its `torch::IValue`s.
///
/// If any `torch::IValue`s in the pickled data are `Object`s, then a
/// `class_resolver` function must be provided.
///
/// See `torch::pickle` for details.
TORCH_API IValue unpickle(
    const char* data,
    size_t size,
    TypeResolver type_resolver = nullptr,
    c10::ArrayRef<at::Tensor> tensor_table = {},
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser);

/// Decode a chunk of memory containing pickled data into its `torch::IValue`s.
///
/// If any `torch::IValue`s in the pickled data are `Object`s, then a
/// `class_resolver` function must be provided.
///
/// See `torch::pickle` for details.
TORCH_API IValue unpickle(
    const char* data,
    size_t size,
    ObjLoader obj_loader,
    TypeResolver type_resolver = nullptr,
    c10::ArrayRef<at::Tensor> tensor_table = {},
    c10::TypePtr (*type_parser)(const std::string&) =
        Unpickler::defaultTypeParser);

#ifndef C10_MOBILE
class VectorReader : public caffe2::serialize::ReadAdapterInterface {
 public:
  VectorReader(std::vector<char> data) : data_(std::move(data)) {}

  size_t size() const override {
    return data_.size();
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what)
      const override;

 private:
  std::vector<char> data_;
};

class StringViewReader : public caffe2::serialize::ReadAdapterInterface {
 public:
  StringViewReader(std::string_view data) : data_(data) {}

  size_t size() const override {
    return data_.size();
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what)
      const override;

 private:
  std::string_view data_;
};
#endif
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `VectorReader`, `StringViewReader`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `c10/util/ArrayRef.h`
- `caffe2/serialize/inline_container.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/serialization/pickler.h`
- `torch/csrc/jit/serialization/unpickler.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `pickle.h_docs.md`
- **Keyword Index**: `pickle.h_kw.md`
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

- **File Documentation**: `pickle.h_docs.md_docs.md`
- **Keyword Index**: `pickle.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

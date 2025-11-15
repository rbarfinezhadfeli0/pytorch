# Documentation: `docs/torch/csrc/jit/serialization/pickler_helper.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/serialization/pickler_helper.h_docs.md`
- **Size**: 9,640 bytes (9.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/serialization/pickler_helper.h`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/pickler_helper.h`
- **Size**: 6,950 bytes (6.79 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <string>

#include <ATen/Utils.h>
#include <ATen/core/ivalue.h>

namespace torch::jit {

// See Python's pickletools.py for a detailed description of each of these codes
enum class PickleOpCode : char {
  MARK = '(',
  STOP = '.',
  POP = '0',
  POP_MARK = '1',
  DUP = '2',
  FLOAT = 'F',
  INT = 'I',
  BININT = 'J',
  BININT1 = 'K',
  LONG = 'L',
  BININT2 = 'M',
  NONE = 'N',
  PERSID = 'P',
  BINPERSID = 'Q',
  REDUCE = 'R',
  STRING = 'S',
  BINSTRING = 'T',
  SHORT_BINSTRING = 'U',
  // NB: Avoid using UNICODE as it is a macro in the Windows API
  UNICODE_ = 'V',
  BINUNICODE = 'X',
  APPEND = 'a',
  BUILD = 'b',
  GLOBAL = 'c',
  DICT = 'd',
  EMPTY_DICT = '}',
  APPENDS = 'e',
  GET = 'g',
  BINGET = 'h',
  INST = 'i',
  LONG_BINGET = 'j',
  LIST = 'l',
  EMPTY_LIST = ']',
  OBJ = 'o',
  PUT = 'p',
  BINPUT = 'q',
  LONG_BINPUT = 'r',
  SETITEM = 's',
  TUPLE = 't',
  EMPTY_TUPLE = ')',
  SETITEMS = 'u',
  BINFLOAT = 'G',

  // Protocol 2
  // NOLINTNEXTLINE(readability-redundant-inline-specifier)
  PROTO = char('\x80'),
  NEWOBJ = '\x81',
  EXT1 = '\x82',
  EXT2 = '\x83',
  EXT4 = '\x84',
  TUPLE1 = '\x85',
  TUPLE2 = '\x86',
  TUPLE3 = '\x87',
  NEWTRUE = '\x88',
  NEWFALSE = '\x89',
  LONG1 = '\x8a',
  LONG4 = '\x8b',

  // Protocol 3 (Python 3.x)
  BINBYTES = 'B',
  SHORT_BINBYTES = 'C',

  // Protocol 4
  // NOLINTNEXTLINE(readability-redundant-inline-specifier)
  SHORT_BINUNICODE = char('\x8c'),
  BINUNICODE8 = '\x8d',
  BINBYTES8 = '\x8e',
  EMPTY_SET = '\x8f',
  ADDITEMS = '\x90',
  FROZENSET = '\x91',
  NEWOBJ_EX = '\x92',
  STACK_GLOBAL = '\x93',
  MEMOIZE = '\x94',
  FRAME = '\x95'
};

struct WriteableTensorData {
  const char* data() const {
    return static_cast<const char*>(tensor_.storage().data());
  }
  size_t sizeInBytes() const {
    return size_;
  }
  size_t nbytes() const {
    return tensor_.storage().nbytes();
  }
  bool storageHasDeleter() const {
    return tensor_.storage().data_ptr().get_context() != nullptr;
  }

 private:
  friend TORCH_API WriteableTensorData
  getWriteableTensorData(const at::Tensor& tensor, bool to_cpu);
  at::Tensor tensor_;
  uint64_t size_;
};

// returns a (tensor, record_size) for a tensor, converting it to a CPU tensor
// if it was CUDA and to_cpu is True.
TORCH_API WriteableTensorData
getWriteableTensorData(const at::Tensor& tensor, bool to_cpu = true);

// if the cls has __getstate__/__setstate__
// assert they have the right schema and return true,
// otherwise return false
bool checkHasValidSetGetState(const c10::ClassType& cls);

// Declare BackendMeta serialization and deserialization function pointer types.
using BackendMetaPtr = std::function<
    void(const at::Tensor&, std::unordered_map<std::string, bool>&)>;

// A allowlist of device type, currently available is PrivateUse1
TORCH_API std::unordered_set<c10::DeviceType>& GetBackendMetaAllowlist();

// Dynamically obtain serialization function pairs
// that require the corresponding backend.
TORCH_API std::array<
    std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>,
    at::COMPILE_TIME_MAX_DEVICE_TYPES>&
GetBackendMetaSerialization();

// Return a map of Tensor Metadata which including BackendMetaData for
// serialization. For now, it only takes care of `conj` and `neg` bit.
inline std::unordered_map<std::string, bool> getTensorMetadata(
    const at::Tensor& t) {
  // We don't support serializing `ZeroTensor` as it is not public
  // facing yet.
  TORCH_CHECK(
      !t._is_zerotensor(),
      "ZeroTensor is not serializable,",
      " please file an issue if required.");
  std::unordered_map<std::string, bool> metadata{};

  // Only add meta-data if the value is not default.
  if (t.is_conj()) {
    metadata["conj"] = true;
  }
  if (t.is_neg()) {
    metadata["neg"] = true;
  }
  // Only add BackendMetaData for custom backend if the function pointer is
  // registered.
  int device_type = static_cast<int>(t.device().type());
  const auto& BackendMetaSerialization = GetBackendMetaSerialization();
  if (BackendMetaSerialization[device_type].has_value()) {
    // Pass the tensor and metadata map references as parameters to the custom
    // serialization function.
    BackendMetaPtr fptr = BackendMetaSerialization[device_type].value().first;
    fptr(t, metadata);
  }
  return metadata;
}

// set Tensor Metadata based on the map.
// Refer: getTensorMetadata
inline void setTensorMetadata(
    const at::Tensor& t,
    std::unordered_map<std::string, bool> metadata) {
  auto iter_end = metadata.end();
  auto iter_temp = metadata.find("conj");
  if (iter_temp != iter_end) {
    t._set_conj(true);
    metadata.erase(iter_temp);
  }
  iter_temp = metadata.find("neg");
  if (iter_temp != iter_end) {
    t._set_neg(true);
    metadata.erase(iter_temp);
  }
  // Only set BackendMetaData for custom backend if the function pointer is
  // registered.
  int device_type = static_cast<int>(t.device().type());
  const auto& BackendMetaSerialization = GetBackendMetaSerialization();
  if (BackendMetaSerialization[device_type].has_value()) {
    // Pass the tensor and metadata map references as parameters to the custom
    // deserialization function.
    BackendMetaPtr fptr = BackendMetaSerialization[device_type].value().second;
    fptr(t, metadata);
  }
}

// set Tensor metadata based on the map.
// NOTE: This overload is required by unpickler.cpp
inline void setTensorMetadata(
    const at::Tensor& t,
    const c10::Dict<c10::IValue, c10::IValue>& metadata_idict) {
  std::unordered_map<std::string, bool> metadata;
  for (auto& pair : metadata_idict) {
    auto key = *pair.key().toString();
    metadata[key] = pair.value().toBool();
  }
  setTensorMetadata(t, std::move(metadata));
}

// Register function pointer of Tensor BackendMetadata for serialization.
inline void TensorBackendMetaRegistry(
    c10::DeviceType t,
    const BackendMetaPtr& get_fptr,
    const BackendMetaPtr& set_fptr) {
  // allowlist verification
  // Only if the devicetype is in the allowlist,
  // we allow the serialization extension to be registered for backendmeta data.
  const auto& DeviceTypeAllowlist = GetBackendMetaAllowlist();
  TORCH_CHECK(
      DeviceTypeAllowlist.find(t) != DeviceTypeAllowlist.end(),
      "It is not allowed to register the serialization method ",
      "of backendMeta data for PrivateUse1. ",
      "If you have related serialization requirements, ",
      "please expand the allowlist");
  // Register function pointer
  int device_type = static_cast<int>(t);
  auto& BackendMetaSerialization = GetBackendMetaSerialization();
  TORCH_CHECK(
      !BackendMetaSerialization[device_type].has_value(),
      "The tensor BackendMeta serialization function pointer for ",
      t,
      " has been registered.");
  BackendMetaSerialization[device_type] =
      std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>(
          std::make_pair(get_fptr, set_fptr));
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `PickleOpCode`, `WriteableTensorData`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `ATen/Utils.h`
- `ATen/core/ivalue.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `pickler_helper.h_docs.md`
- **Keyword Index**: `pickler_helper.h_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `pickler_helper.h_docs.md_docs.md`
- **Keyword Index**: `pickler_helper.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

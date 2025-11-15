# Documentation: `torch/csrc/jit/serialization/import_read.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/import_read.cpp`
- **Size**: 2,592 bytes (2.53 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <utility>

namespace torch::jit {

IValue readArchiveAndTensors(
    const std::string& archive_name,
    const std::string& pickle_prefix,
    const std::string& tensor_prefix,
    std::optional<TypeResolver> type_resolver,
    std::optional<ObjLoader> obj_loader,
    std::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader,
    c10::TypePtr (*type_parser)(const std::string&),
    std::shared_ptr<DeserializationStorageContext> storage_context) {
  std::string picklename = pickle_prefix + archive_name + ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size = 0;
  std::tie(pickle_ptr, pickle_size) = stream_reader.getRecord(picklename);

  size_t bytes_read = 0;
  auto data = reinterpret_cast<const char*>(pickle_ptr.get());
  auto reader = [&](char* buffer, size_t len) -> size_t {
    if (bytes_read >= pickle_size) {
      return 0;
    }
    len = std::min(pickle_size - bytes_read, len);
    // Copy len bytes into buffer
    const char* start = data + bytes_read;
    std::memcpy(buffer, start, len);
    bytes_read += len;
    return len;
  };

  std::string tensor_dir_path =
      (!tensor_prefix.empty()) ? tensor_prefix : archive_name + "/";

  auto read_record = [&](const std::string& name) {
    std::string ss = tensor_dir_path + name;
    return std::get<0>(stream_reader.getRecord(ss));
  };

  Unpickler unpickler(
      reader,
      type_resolver ? std::move(*type_resolver) : nullptr,
      obj_loader ? std::move(*obj_loader) : nullptr,
      std::move(read_record),
      device,
      false,
      type_parser,
      std::move(storage_context));
  unpickler.set_version(stream_reader.version());
  return unpickler.parse_ivalue();
}

bool check_zip_file(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai) {
  std::array<uint8_t, 2> first_short{};
  static constexpr uint8_t first_slot = 0x80;
  static constexpr uint8_t second_slot = 0x02;
  rai->read(
      /*pos=*/0,
      /*buf=*/&first_short,
      /*n=*/2,
      /*what=*/"checking archive");

  // NB: zip files by spec can start with any data, so technically they might
  // start with 0x80 0x02, but in practice zip files start with a file entry
  // which begins with 0x04034b50. Furthermore, PyTorch will never produce zip
  // files that do not start with the file entry, so it is relatively safe to
  // perform this check.
  return !(first_short[0] == first_slot && first_short[1] == second_slot);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `caffe2/serialize/inline_container.h`
- `torch/csrc/jit/serialization/import_read.h`
- `utility`


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

- **File Documentation**: `import_read.cpp_docs.md`
- **Keyword Index**: `import_read.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

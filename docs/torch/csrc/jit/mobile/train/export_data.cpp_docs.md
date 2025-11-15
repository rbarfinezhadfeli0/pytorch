# Documentation: `torch/csrc/jit/mobile/train/export_data.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/train/export_data.cpp`
- **Size**: 4,712 bytes (4.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/mobile/train/export_data.h>

#include <torch/csrc/jit/mobile/import_export_common.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>

#include <string>
#include <vector>

namespace torch::jit {
namespace mobile {

namespace {

/**
 * Serializes an IValue using Pickle, and puts it in a file named "data.pkl"
 * in a ZIP wrapper.
 */
class IValuePickler final {
 public:
  explicit IValuePickler(const std::string& filename) : writer_(filename) {}

  explicit IValuePickler(
      const std::function<size_t(const void*, size_t)>& writer_func)
      : writer_(writer_func) {}

  void serialize(const IValue& object) {
    // Serialize just the data
    writeArchive("data", object);
  }

 private:
  void writeArchive(const std::string& archive_name, const IValue& value) {
    std::vector<char> data;
    // Vector to capture the run-time class types during pickling the IValues
    std::vector<c10::ClassTypePtr> memoizedClassTypes;
    Pickler data_pickle(
        [&](const char* buf, size_t size) {
          data.insert(data.end(), buf, buf + size);
        },
        nullptr,
        [&](const c10::ClassTypePtr& t) {
          return type_name_uniquer_.getUniqueName(t);
        },
        &memoizedClassTypes);
    data_pickle.protocol();
    data_pickle.pushIValue(value);
    data_pickle.stop();
    size_t i = 0;
    std::string prefix = archive_name + "/";
    for (const auto& td : data_pickle.tensorData()) {
      WriteableTensorData writable_td = getWriteableTensorData(td);
      std::string fname = prefix + std::to_string(i++);
      writer_.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
    }
    std::string fname = archive_name + ".pkl";
    writer_.writeRecord(fname, data.data(), data.size());
  }

  caffe2::serialize::PyTorchStreamWriter writer_;
  TypeNameUniquer type_name_uniquer_;
};

} // namespace

/**
 * Converts a map of named tensors to a c10::Dict.
 */
c10::Dict<std::string, at::Tensor> tensor_map_to_dict(
    const std::map<std::string, at::Tensor>& map) {
  c10::Dict<std::string, at::Tensor> dict;
  for (const auto& e : map) {
    dict.insert(e.first, e.second);
  }
  return dict;
}

/**
 * Returns a Module with a single attribute, with the attribute name specified
 * by #internal::kSavedParametersAttributeName, whose value is the provided
 * dict.
 */
mobile::Module tensor_dict_to_mobile(
    const c10::Dict<std::string, at::Tensor>& dict) {
  // Create an Object to back the Module, with an attribute to hold the dict.
  auto cu = std::make_shared<torch::jit::CompilationUnit>();
  // Note that the name doesn't really matter, but it must begin with
  // "__torch__." to be treated as a valid class when being imported.
  auto cls = c10::ClassType::create(
      "__torch__.SavedParameters", cu, /*is_module=*/true);
  cls->addAttribute(
      internal::kSavedParametersAttributeName,
      c10::DictType::create(dict.keyType(), dict.valueType()));
  auto object = c10::ivalue::Object::create(
      c10::StrongTypePtr(std::move(cu), std::move(cls)), /*numSlots=*/1);

  // Add the dict as an attribute.
  object->setAttr(internal::kSavedParametersAttributeName, dict);

  // Wrap the Object in a Module.
  auto mcu = std::make_shared<mobile::CompilationUnit>();
  return mobile::Module(object, mcu);
}

} // namespace mobile

void (*_save_mobile_module_to)(
    const mobile::Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func) = nullptr;

void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    std::ostream& out,
    bool use_flatbuffer) {
  auto dict = mobile::tensor_map_to_dict(map);

  auto write_func = [&out](const void* buf, size_t nbytes) -> size_t {
    out.write(
        static_cast<const char*>(buf), static_cast<std::streamsize>(nbytes));
    return !out ? 0 : nbytes;
  };

  if (use_flatbuffer) {
    save_mobile_module_to_func(mobile::tensor_dict_to_mobile(dict), write_func);
  } else {
    // For Pickle, we only serialize the dict itself.
    mobile::IValuePickler pickler(write_func);
    pickler.serialize(dict);
  }
}

void _save_parameters(
    const std::map<std::string, at::Tensor>& map,
    const std::string& filename,
    bool use_flatbuffer) {
  auto dict = mobile::tensor_map_to_dict(map);

  std::ofstream ifile(filename);
  _save_parameters(map, ifile, use_flatbuffer);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `mobile`, `torch`

**Classes/Structs**: `IValuePickler`, `types`, `when`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/train`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/mobile/train/export_data.h`
- `torch/csrc/jit/mobile/import_export_common.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/runtime/instruction.h`
- `torch/csrc/jit/serialization/flatbuffer_serializer.h`
- `torch/csrc/jit/serialization/pickler.h`
- `torch/csrc/jit/serialization/type_name_uniquer.h`
- `caffe2/serialize/inline_container.h`
- `ATen/core/ivalue.h`
- `ATen/core/jit_type.h`
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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/mobile/train`):

- [`sequential.h_docs.md`](./sequential.h_docs.md)
- [`sequential.cpp_docs.md`](./sequential.cpp_docs.md)
- [`export_data.h_docs.md`](./export_data.h_docs.md)
- [`random.cpp_docs.md`](./random.cpp_docs.md)
- [`random.h_docs.md`](./random.h_docs.md)


## Cross-References

- **File Documentation**: `export_data.cpp_docs.md`
- **Keyword Index**: `export_data.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

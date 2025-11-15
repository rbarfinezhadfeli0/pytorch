# Documentation: `docs/torch/csrc/jit/mobile/import_data.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/import_data.cpp_docs.md`
- **Size**: 12,491 bytes (12.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/import_data.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/import_data.cpp`
- **Size**: 9,466 bytes (9.24 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/mobile/import_data.h>

#include <ATen/Functions.h>
#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/import_export_common.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <torch/custom_class.h>

#include <caffe2/serialize/in_memory_adapter.h>
#include <string>
#include <vector>

namespace torch::jit {
using caffe2::serialize::PyTorchStreamReader;

namespace {

/**
 * Given a ZIP file containing a file named "data.pkl", uses Pickle to
 * deserialize the file and returns the IValue inside it.
 */
class IValueUnpickler final {
 public:
  explicit IValueUnpickler(std::unique_ptr<PyTorchStreamReader> reader);
  c10::IValue deserialize(std::optional<at::Device> device);

 private:
  c10::IValue readArchive(
      const std::string& archive_name,
      std::shared_ptr<mobile::CompilationUnit> mcu,
      std::optional<at::Device> device);

  std::shared_ptr<CompilationUnit> compilation_unit_;
  std::unique_ptr<PyTorchStreamReader> reader_;
};

IValueUnpickler::IValueUnpickler(std::unique_ptr<PyTorchStreamReader> reader)
    : compilation_unit_(std::make_shared<CompilationUnit>()),
      reader_(std::move(reader)) {}

c10::IValue IValueUnpickler::deserialize(std::optional<at::Device> device) {
  auto mcu = std::make_shared<mobile::CompilationUnit>();

  return readArchive("data", mcu, device);
}

c10::IValue IValueUnpickler::readArchive(
    const std::string& archive_name,
    std::shared_ptr<mobile::CompilationUnit> mcu,
    std::optional<at::Device> device) {
  std::stringstream picklename;
  picklename << archive_name << ".pkl";
  at::DataPtr pickle_ptr;
  size_t pickle_size = 0;
  std::tie(pickle_ptr, pickle_size) = reader_->getRecord(picklename.str());

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

  static const c10::QualifiedName torchPrefix = "__torch__";
  auto type_resolver = [&](const c10::QualifiedName& qn) {
    TypePtr type;
    // HACK: first we check whether the name starts with `__torch__` to tell if
    // it's "supposed" to be a class type. This is a reliable check today, but
    // there is no guarantee that this is the case. The real solution is to
    // merge type parsers so we can share class resolution logic.
    if (torchPrefix.isPrefixOf(qn)) {
      if (compilation_unit_->get_class(qn) == nullptr) {
        auto typeptr = ClassType::create(qn, compilation_unit_, true);
        compilation_unit_->register_type(typeptr);
      }
      type = compilation_unit_->get_class(qn);
    } else {
      type = c10::parseType(qn.qualifiedName());
    }
    return c10::StrongTypePtr(compilation_unit_, type);
  };

  auto obj_loader = [&](const at::StrongTypePtr& type, IValue input) {
    auto cls = type.type_->expect<at::ClassType>();
    auto qn = cls->name();
    c10::QualifiedName method_name(qn.value(), "__setstate__");
    auto setstate = mcu->find_function(method_name);
    auto find_custom_class_with_setstate = [&qn]() -> c10::ClassTypePtr {
      auto custom_class_type = torch::jit::getCustomClass(qn->qualifiedName());
      if (custom_class_type && custom_class_type->findMethod("__setstate__")) {
        return custom_class_type;
      }
      return nullptr;
    };
    if (setstate) {
      auto obj = c10::ivalue::Object::create(type, 0);
      Stack stack({obj, input});
      setstate->run(stack);
      return obj;
    } else if (auto custom_class_type = find_custom_class_with_setstate()) {
      auto obj = c10::ivalue::Object::create(
          c10::StrongTypePtr(nullptr, custom_class_type), 1);
      Stack stack({obj, input});
      custom_class_type->getMethod("__setstate__").run(stack);
      return obj;
    } else {
      auto dict = std::move(input).toGenericDict();
      size_t ndict = dict.size();
      auto obj = c10::ivalue::Object::create(type, ndict);
      auto it = dict.begin();
      for (const auto i : c10::irange(ndict)) {
        std::stringstream name;
        name << it->key();
        cls->addOrCheckAttribute(name.str(), it->key().type());
        obj->setSlot(i, it->value());
        ++it;
      }
      return obj;
    }
  };

  auto read_record = [&](const std::string& name) {
    std::stringstream ss;
    ss << archive_name << "/" << name;
    return std::get<0>(reader_->getRecord(ss.str()));
  };

  Unpickler unpickler(
      reader,
      std::move(type_resolver),
      std::move(obj_loader),
      std::move(read_record),
      device,
      false,
      nullptr);
  return unpickler.parse_ivalue();
}

/**
 * Extracts and returns the parameter map serialized as ZIP + Pickle in @p rai.
 */
std::map<std::string, at::Tensor> load_parameters_from_zip(
    std::unique_ptr<ReadAdapterInterface> rai,
    std::optional<c10::Device> device) {
  auto reader = std::make_unique<PyTorchStreamReader>(std::move(rai));
  IValueUnpickler unpickler(std::move(reader));
  auto result = unpickler.deserialize(device).toGenericDict();
  std::map<std::string, at::Tensor> map;
  for (const auto& e : result) {
    auto key = e.key().toStringRef();
    auto value = e.value().toTensor().tensor_data();
    map[key] = value;
  }
  return map;
}

} // namespace

/**
 * Extracts the parameter map stored in @p module. Expects a layout
 * compatible with the one created by #_save_parameters().
 */
std::map<std::string, at::Tensor> mobile_module_to_parameter_map(
    const mobile::Module& module) {
  // Safely look for a slot with the expected name. Note that
  // c10::ivalue::Object::getAttr() is not safe if the attribute isn't present.
  auto obj = module._ivalue();
  const std::vector<IValue>& slots = obj->slots();
  for (const auto i : c10::irange(slots.size())) {
    if (obj->type()->getAttributeName(i) ==
        mobile::internal::kSavedParametersAttributeName) {
      // Found a slot with the right name; make sure it's a
      // Dict<string, Tensor>.
      c10::IValue data = slots[i];
      if (data.isGenericDict()) {
        auto data_dict = data.toGenericDict();

        // The key and value should be DynamicTypes that wrap String and Tensor.
        c10::DynamicType* keyType =
            data_dict.keyType()->castRaw<c10::DynamicType>();
        c10::DynamicType* valueType =
            data_dict.valueType()->castRaw<c10::DynamicType>();
        if (keyType != nullptr &&
            keyType->fallback()->kind() == TypeKind::StringType &&
            valueType != nullptr &&
            valueType->fallback()->kind() == TypeKind::TensorType) {
          // Name and type are good; copy the contents to the output map.
          std::map<std::string, at::Tensor> params;
          for (const auto& e : data_dict) {
            // The source Tensor points into the flatbuffer data associated with
            // the Module. But, this Tensor needs to outlive the Module, since
            // the caller of _load_parameters() won't have a pointer to the
            // Module. So, return a deep copy.
            const auto& source = e.value().toTensor();
            at::Tensor copy = at::empty_like(source); // Must be the same shape.
            copy.copy_(source);

            params[e.key().toStringRef()] = copy;
          }
          return params;
        }
      }
    }
  }

  TORCH_CHECK(
      false,
      "Could not find Dict<string, Tensor> named '",
      mobile::internal::kSavedParametersAttributeName,
      "' in deserialized mobile::Module");
}

static std::map<std::string, at::Tensor> _load_parameters_bytes(
    const std::shared_ptr<char>& data,
    size_t size,
    std::optional<at::Device> device) {
  TORCH_CHECK(size >= kFileFormatHeaderSize, "Unrecognized data format");
  FileFormat format = getFileFormat(data.get());
  // Call the appropriate parser.
  std::map<std::string, at::Tensor> map;
  switch (format) {
    case FileFormat::FlatbufferFileFormat: {
      auto m = parse_flatbuffer_no_object(data, size, device);
      map = mobile_module_to_parameter_map(m);
      break;
    }

    case FileFormat::ZipFileFormat: {
      auto rai = std::make_unique<caffe2::serialize::MemoryReadAdapter>(
          data.get(), size);
      map = load_parameters_from_zip(std::move(rai), device);
      break;
    }

    default:
      TORCH_CHECK(false, "Unrecognized data format");
  }
  return map;
}

std::map<std::string, at::Tensor> _load_parameters(
    std::istream& in,
    std::optional<at::Device> device) {
  auto [data, size] = get_stream_content(in);
  return _load_parameters_bytes(data, size, device);
}

std::map<std::string, at::Tensor> _load_parameters(
    const std::string& filename,
    std::optional<at::Device> device) {
  auto [data, size] = get_file_content(filename.c_str());
  return _load_parameters_bytes(data, size, device);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `IValueUnpickler`, `type`, `resolution`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/mobile/import_data.h`
- `ATen/Functions.h`
- `ATen/core/ivalue.h`
- `c10/util/irange.h`
- `torch/csrc/jit/api/compilation_unit.h`
- `torch/csrc/jit/mobile/file_format.h`
- `torch/csrc/jit/mobile/flatbuffer_loader.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/import_export_common.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/mobile/observer.h`
- `torch/csrc/jit/mobile/type_parser.h`
- `torch/csrc/jit/runtime/instruction.h`
- `torch/csrc/jit/serialization/unpickler.h`
- `torch/custom_class.h`
- `caffe2/serialize/in_memory_adapter.h`
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

- **File Documentation**: `import_data.cpp_docs.md`
- **Keyword Index**: `import_data.cpp_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data

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

- **File Documentation**: `import_data.cpp_docs.md_docs.md`
- **Keyword Index**: `import_data.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

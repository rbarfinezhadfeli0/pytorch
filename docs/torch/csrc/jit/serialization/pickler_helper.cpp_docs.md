# Documentation: `torch/csrc/jit/serialization/pickler_helper.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/pickler_helper.cpp`
- **Size**: 3,907 bytes (3.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/core/jit_type.h>

#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/serialization/pickler_helper.h>

namespace torch::jit {

WriteableTensorData getWriteableTensorData(
    const at::Tensor& tensor,
    bool to_cpu) {
  WriteableTensorData result;
  result.tensor_ = tensor;
  result.size_ = tensor.storage().nbytes();
  // TODO HIP support
  if (tensor.storage().device_type() != DeviceType::CPU && to_cpu) {
    // NB: This new tensor is created to support cuda tensors.
    // Storages can be mutated when converting tensors from cuda to cpu,
    // and we need a cpu tensor to copy data from.
    result.tensor_ =
        at::empty({0}, tensor.options())
            .set_(
                tensor.storage(),
                /* storage_offset = */ 0,
                /* size = */
                {static_cast<int64_t>(
                    tensor.storage().nbytes() / tensor.element_size())},
                /* stride = */ {1})
            .cpu();
    TORCH_CHECK(
        result.tensor_.storage().nbytes() == result.size_,
        "Storage tensor size did not match record size");
  }
  return result;
}

bool checkHasValidSetGetState(const c10::ClassType& cls) {
  // Check that the schemas for __getstate__ and __setstate__ are correct
  auto getstate = cls.findMethod("__getstate__");
  if (getstate == nullptr) {
    return false;
  }
  auto get_schema = getstate->getSchema();

  // Check __getstate__
  //   __getstate__ is expected to be (self) -> T
  TORCH_CHECK(
      get_schema.arguments().size() == 1,
      "'__getstate__' must have 'self' as its only argument, but found ",
      get_schema.arguments().size(),
      " arguments");
  TORCH_CHECK(
      get_schema.returns().size() == 1,
      "'__getstate__' must return 1 value, but found ",
      get_schema.returns().size());

  // Check __setstate__ if the method exists
  //   __setstate__ is expected to be (self, T) -> None
  auto setstate = cls.findMethod("__setstate__");
  if (!setstate) {
    return false;
  }
  auto set_schema = setstate->getSchema();

  TORCH_CHECK(
      set_schema.arguments().size() == 2,
      "'__setstate__' must have 'self' and the state as its "
      "only arguments, but found ",
      set_schema.arguments().size(),
      " arguments");
  TORCH_CHECK(
      set_schema.returns().size() == 1,
      "'__setstate__' must return None, but found ",
      set_schema.returns().size(),
      " return values");
  TORCH_CHECK(
      set_schema.returns().at(0).type()->isSubtypeOf(*NoneType::get()),
      "'__setstate__' must return None, but found value of type",
      set_schema.returns().at(0).type()->annotation_str());

  // Check that the return type of __getstate__ matches the input to
  // __setstate__
  auto get_type = get_schema.returns().at(0).type();
  auto set_type = set_schema.arguments().at(1).type();

  TORCH_CHECK(
      get_type->isSubtypeOf(*set_type),
      "'__getstate__'s return type (",
      get_type->annotation_str(),
      ") does not match '__setstate__'s argument type (",
      set_type->annotation_str(),
      ")");

  return true;
}

std::unordered_set<c10::DeviceType>& GetBackendMetaAllowlist() {
  static std::unordered_set<c10::DeviceType> DeviceTypeAllowlist{
      c10::DeviceType::PrivateUse1};
  return DeviceTypeAllowlist;
}

std::array<
    std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>,
    at::COMPILE_TIME_MAX_DEVICE_TYPES>&
GetBackendMetaSerialization() {
  // The array to save function pointer for BackendMeta serialization.
  // key is the DeviceType, value is std::pair obj.
  // value.first represent get function and value.second represent set function
  static std::array<
      std::optional<std::pair<BackendMetaPtr, BackendMetaPtr>>,
      at::COMPILE_TIME_MAX_DEVICE_TYPES>
      BackendMetaSerialization;
  return BackendMetaSerialization;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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

- `ATen/ATen.h`
- `ATen/core/jit_type.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/serialization/pickler_helper.h`


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

- **File Documentation**: `pickler_helper.cpp_docs.md`
- **Keyword Index**: `pickler_helper.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

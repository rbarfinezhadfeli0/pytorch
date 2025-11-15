# Documentation: `c10/core/StorageImpl.cpp`

## File Metadata

- **Path**: `c10/core/StorageImpl.cpp`
- **Size**: 4,373 bytes (4.27 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/StorageImpl.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {

// The array to save function pointer for custom storageImpl create.
static std::array<StorageImplCreateHelper, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    StorageImplCreate;

// A allowlist of device type, currently available is PrivateUse1
static ska::flat_hash_set<c10::DeviceType>& GetBackendMetaAllowlist() {
  static ska::flat_hash_set<c10::DeviceType> DeviceTypeAllowList{
      DeviceType::PrivateUse1};
  return DeviceTypeAllowList;
}

void throwNullDataPtrError() {
  TORCH_CHECK(
      false,
      "Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor). "
      "If you're using torch.compile/export/fx, it is likely that we are erroneously "
      "tracing into a custom kernel. To fix this, please wrap the custom kernel into "
      "an opaque custom op. Please see the following for details: "
      "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html");
}

// NOTE: [FakeTensor.data_ptr deprecation]
// Today:
// - FakeTensor.data_ptr errors out in torch.compile.
// - FakeTensor.data_ptr raises the following deprecation warning otherwise.
// - the following deprecation warning is only for FakeTensor (for now).
//   In the future we can consider extending to more wrapper Tensor subclasses.
void warnDeprecatedDataPtr() {
  TORCH_WARN_ONCE(
      "Accessing the data pointer of FakeTensor is deprecated and will error in "
      "PyTorch 2.5. This is almost definitely a bug in your code and will "
      "cause undefined behavior with subsystems like torch.compile. "
      "Please wrap calls to tensor.data_ptr() in an opaque custom op; "
      "If all else fails, you can guard accesses to tensor.data_ptr() on "
      "isinstance(tensor, FakeTensor).")
}

[[noreturn]] void StorageImpl::throw_data_ptr_access_error() const {
  if (extra_meta_ && extra_meta_->custom_data_ptr_error_msg_) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    TORCH_CHECK(false, *extra_meta_->custom_data_ptr_error_msg_);
  }
  TORCH_CHECK(false, "Cannot access data pointer of Storage that is invalid.");
}

void SetStorageImplCreate(DeviceType t, StorageImplCreateHelper fptr) {
  // Allowlist verification.
  // Only if the devicetype is in the allowlist,
  // we allow the extension to be registered for storageImpl create.
  const auto& DeviceTypeAllowlist = GetBackendMetaAllowlist();
  TORCH_CHECK(
      DeviceTypeAllowlist.find(t) != DeviceTypeAllowlist.end(),
      "It is only allowed to register the storageImpl create method ",
      "for PrivateUse1. ",
      "If you have related storageImpl requirements, ",
      "please expand the allowlist");
  // Register function pointer.
  int device_type = static_cast<int>(t);
  TORCH_CHECK(
      StorageImplCreate[device_type] == nullptr,
      "The StorageImplCreate function pointer for ",
      t,
      " has been registered.");
  StorageImplCreate[device_type] = fptr;
}

StorageImplCreateHelper GetStorageImplCreate(DeviceType t) {
  int device_type = static_cast<int>(t);
  return StorageImplCreate[device_type];
}

c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable,
    std::optional<at::Device> device_opt) {
  // This will be non-nullptr only when there is a custom StorageImpl
  // constructor for the given device
  c10::StorageImplCreateHelper fptr = nullptr;
  if (device_opt.has_value()) {
    // We only need to check this here as this is the only case where we can
    // have a device that is not CPU (and thus for which the StorageImpl
    // constructor can be overwritten).
    fptr = c10::GetStorageImplCreate(device_opt.value().type());
  }

  if (fptr != nullptr) {
    return fptr(
        use_byte_size,
        std::move(size_bytes),
        std::move(data_ptr),
        allocator,
        resizable);
  }

  // Create a c10::StorageImpl object.
  if (data_ptr != nullptr) {
    return c10::make_intrusive<c10::StorageImpl>(
        use_byte_size,
        std::move(size_bytes),
        std::move(data_ptr),
        allocator,
        resizable);
  }
  return c10::make_intrusive<c10::StorageImpl>(
      use_byte_size, std::move(size_bytes), allocator, resizable);
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/StorageImpl.h`
- `c10/util/flat_hash_map.h`


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

Files in the same folder (`c10/core`):

- [`DispatchKey.cpp_docs.md`](./DispatchKey.cpp_docs.md)
- [`CopyBytes.h_docs.md`](./CopyBytes.h_docs.md)
- [`OptionalRef.h_docs.md`](./OptionalRef.h_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`SafePyObject.cpp_docs.md`](./SafePyObject.cpp_docs.md)
- [`DeviceType.cpp_docs.md`](./DeviceType.cpp_docs.md)
- [`SymBool.cpp_docs.md`](./SymBool.cpp_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `StorageImpl.cpp_docs.md`
- **Keyword Index**: `StorageImpl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

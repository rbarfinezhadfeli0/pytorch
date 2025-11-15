# Documentation: `docs/aten/src/ATen/core/DeprecatedTypeProperties.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/DeprecatedTypeProperties.h_docs.md`
- **Size**: 6,559 bytes (6.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/DeprecatedTypeProperties.h`

## File Metadata

- **Path**: `aten/src/ATen/core/DeprecatedTypeProperties.h`
- **Size**: 3,879 bytes (3.79 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Layout.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/Storage.h>
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
#include <ATen/core/Generator.h>


namespace at {

class Tensor;

// This class specifies a Backend and a ScalarType. Currently, it primarily
// serves as a replacement return value for Tensor::type(). Previously,
// Tensor::type() returned Type&, but we are changing Type to not be
// dtype-specific.
class TORCH_API DeprecatedTypeProperties {
 public:
  DeprecatedTypeProperties(Backend backend, ScalarType scalar_type)
    : backend_(backend), scalar_type_(scalar_type) {}

  Backend backend() const {
    return backend_;
  }

  Layout layout() const {
    return layout_from_backend(backend_);
  }

  bool is_sparse() const {
    return layout_from_backend(backend()) == kSparse;
  }

  bool is_sparse_csr() const {
    return layout_from_backend(backend()) == kSparseCsr;
  }

  c10::DeviceType device_type() const {
    return backendToDeviceType(backend_);
  }

  bool is_cuda() const {
    return backendToDeviceType(backend_) == kCUDA;
  }

  ScalarType scalarType() const {
    return scalar_type_;
  }

  caffe2::TypeMeta typeMeta() const {
    return scalarTypeToTypeMeta(scalar_type_);
  }

  bool operator==(const DeprecatedTypeProperties& other) const {
    return backend_ == other.backend() && scalar_type_ == other.scalarType();
  }

  bool operator!=(const DeprecatedTypeProperties& other) const {
    return !(*this == other);
  }

  std::string toString() const {
    std::string base_str;
    if (backend_ == Backend::Undefined || scalar_type_ == ScalarType::Undefined) {
      base_str = "UndefinedType";
    } else {
      base_str = std::string(at::toString(backend_)) + at::toString(scalar_type_) + "Type";
    }
    return base_str;
  }

  DeprecatedTypeProperties & toBackend(Backend b) const {
    return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
        b, scalar_type_);
  }

  DeprecatedTypeProperties & toScalarType(ScalarType s) const {
    return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
        backend_, s);
  }

  DeprecatedTypeProperties & cpu() const {
    return toBackend(Backend::CPU);
  }

  DeprecatedTypeProperties & cuda() const {
    return toBackend(Backend::CUDA);
  }

  DeprecatedTypeProperties & hip() const {
    return toBackend(Backend::HIP);
  }

  DeprecatedTypeProperties & privateUser1() const {
    return toBackend(Backend::PrivateUse1);
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  TensorOptions options(int16_t device_index = -1) const {
    return TensorOptions().dtype(typeMeta())
                          .device(device_type(), static_cast<c10::DeviceIndex>(device_index))
                          .layout(layout());
  }

  /// Constructs the `TensorOptions` from a type and a Device.  Asserts that
  /// the device type matches the device type of the type.
  TensorOptions options(std::optional<Device> device_opt) const {
    if (!device_opt.has_value()) {
      return options(-1);
    } else {
      Device device = device_opt.value();
      AT_ASSERT(device.type() == device_type());
      return options(device.index());
    }
  }

  operator TensorOptions() const {
    return options();
  }

  int64_t id() const {
    return static_cast<int64_t>(backend()) *
        static_cast<int64_t>(ScalarType::NumOptions) +
        static_cast<int64_t>(scalarType());
  }

  Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const;
  Storage unsafeStorageFromTH(void * th_pointer, bool retain) const;
  Tensor copy(const Tensor & src, bool non_blocking=false, std::optional<Device> to_device={}) const;

 private:
  Backend backend_;
  ScalarType scalar_type_;
};

}  // namespace at

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 32 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Tensor`, `specifies`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Backend.h`
- `c10/core/ScalarType.h`
- `c10/core/Layout.h`
- `c10/core/TensorOptions.h`
- `c10/core/Storage.h`
- `ATen/core/DeprecatedTypePropertiesRegistry.h`
- `ATen/core/Generator.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `DeprecatedTypeProperties.h_docs.md`
- **Keyword Index**: `DeprecatedTypeProperties.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `DeprecatedTypeProperties.h_docs.md_docs.md`
- **Keyword Index**: `DeprecatedTypeProperties.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

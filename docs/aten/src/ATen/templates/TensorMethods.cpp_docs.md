# Documentation: `aten/src/ATen/templates/TensorMethods.cpp`

## File Metadata

- **Path**: `aten/src/ATen/templates/TensorMethods.cpp`
- **Size**: 2,613 bytes (2.55 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/Scalar.h>
#include <ATen/core/TensorBody.h>

#include <string_view>

namespace at {

namespace {

// Verifies the requested type is the same as the Tensor's type.
void check_type(const TensorBase& tensor, ScalarType type, std::string_view type_name) {
  TORCH_CHECK(
      tensor.scalar_type() == type
      || (isQIntType(tensor.scalar_type())
          && toUnderlying(tensor.scalar_type()) == type),
      "expected scalar type ", type_name, " but found ", tensor.scalar_type());
}

} // namespace

#define DEFINE_CAST(T, name)                                         \
   template <>                                                       \
   TORCH_API const T* TensorBase::const_data_ptr() const {           \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->data_ptr_impl<T>();         \
   }                                                                 \
                                                                     \
   template <>                                                       \
   TORCH_API const T* TensorBase::const_data_ptr<const T>() const {  \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->data_ptr_impl<std::remove_const_t<T>>(); \
   }                                                                 \
                                                                     \
   template <>                                                       \
   TORCH_API T* TensorBase::mutable_data_ptr() const {               \
     check_type(*this, ScalarType::name, #name);                     \
     return this->unsafeGetTensorImpl()->mutable_data_ptr_impl<T>(); \
   }                                                                 \
                                                                     \
   template <>                                                       \
   TORCH_API T* TensorBase::data_ptr() const {                       \
     return mutable_data_ptr<T>();                                   \
   }                                                                 \

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
 AT_FORALL_QINT_TYPES(DEFINE_CAST)
 DEFINE_CAST(uint16_t, UInt16)
 DEFINE_CAST(uint32_t, UInt32)
 DEFINE_CAST(uint64_t, UInt64)
 #undef DEFINE_CAST

 #define DEFINE_ITEM(T, name)      \
   template <>                     \
   TORCH_API T Tensor::item() const { \
     return item().to##name();     \
   }

 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ITEM)
 #undef DEFINE_ITEM

 } //namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/templates`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Scalar.h`
- `ATen/core/TensorBody.h`
- `string_view`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/templates`):

- [`NativeFunction.h_docs.md`](./NativeFunction.h_docs.md)
- [`DispatchKeyFunctions.h_docs.md`](./DispatchKeyFunctions.h_docs.md)
- [`aten_interned_strings.h_docs.md`](./aten_interned_strings.h_docs.md)
- [`UfuncCPUKernel.cpp_docs.md`](./UfuncCPUKernel.cpp_docs.md)
- [`DispatchKeyFunction.h_docs.md`](./DispatchKeyFunction.h_docs.md)
- [`LazyIr.h_docs.md`](./LazyIr.h_docs.md)
- [`RegisterDispatchDefinitions.ini_docs.md`](./RegisterDispatchDefinitions.ini_docs.md)
- [`Functions.cpp_docs.md`](./Functions.cpp_docs.md)
- [`RegisterDispatchKey.cpp_docs.md`](./RegisterDispatchKey.cpp_docs.md)
- [`MethodOperators.h_docs.md`](./MethodOperators.h_docs.md)


## Cross-References

- **File Documentation**: `TensorMethods.cpp_docs.md`
- **Keyword Index**: `TensorMethods.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `c10/util/typeid.cpp`

## File Metadata

- **Path**: `c10/util/typeid.cpp`
- **Size**: 3,169 bytes (3.09 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <c10/util/typeid.h>

#include <algorithm>
#include <atomic>

namespace caffe2 {
namespace detail {
C10_EXPORT void _ThrowRuntimeTypeLogicError(const std::string& msg) {
  // In earlier versions it used to be std::abort() but it's a bit hard-core
  // for a library
  TORCH_CHECK(false, msg);
}
} // namespace detail

[[noreturn]] void TypeMeta::error_unsupported_typemeta(caffe2::TypeMeta dtype) {
  TORCH_CHECK(
      false,
      "Unsupported TypeMeta in ATen: ",
      dtype,
      " (please report this error)");
}

std::mutex& TypeMeta::getTypeMetaDatasLock() {
  static std::mutex lock;
  return lock;
}

uint16_t TypeMeta::nextTypeIndex(NumScalarTypes);

// fixed length array of TypeMetaData instances
detail::TypeMetaData* TypeMeta::typeMetaDatas() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static detail::TypeMetaData instances[MaxTypeIndex + 1] = {
#define SCALAR_TYPE_META(T, name)        \
  /* ScalarType::name */                 \
  detail::TypeMetaData(                  \
      sizeof(T),                         \
      detail::_PickNew<T>(),             \
      detail::_PickPlacementNew<T>(),    \
      detail::_PickCopy<T>(),            \
      detail::_PickPlacementDelete<T>(), \
      detail::_PickDelete<T>(),          \
      TypeIdentifier::Get<T>(),          \
      c10::util::get_fully_qualified_type_name<T>()),
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
#undef SCALAR_TYPE_META
      // The remainder of the array is padded with TypeMetaData blanks.
      // The first of these is the entry for ScalarType::Undefined.
      // The rest are consumed by CAFFE_KNOWN_TYPE entries.
  };
  return instances;
}

uint16_t TypeMeta::existingMetaDataIndexForType(TypeIdentifier identifier) {
  auto* metaDatas = typeMetaDatas();
  const auto end = metaDatas + nextTypeIndex;
  // MaxTypeIndex is not very large; linear search should be fine.
  auto it = std::find_if(metaDatas, end, [identifier](const auto& metaData) {
    return metaData.id_ == identifier;
  });
  if (it == end) {
    return MaxTypeIndex;
  }
  return static_cast<uint16_t>(it - metaDatas);
}

CAFFE_DEFINE_KNOWN_TYPE(std::string, std_string)
CAFFE_DEFINE_KNOWN_TYPE(uint16_t, uint16_t)
CAFFE_DEFINE_KNOWN_TYPE(char, char)
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::mutex>, std_unique_ptr_std_mutex)
CAFFE_DEFINE_KNOWN_TYPE(
    std::unique_ptr<std::atomic<bool>>,
    std_unique_ptr_std_atomic_bool)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int32_t>, std_vector_int32_t)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int64_t>, std_vector_int64_t)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<unsigned long>, std_vector_unsigned_long)
CAFFE_DEFINE_KNOWN_TYPE(bool*, bool_ptr)
CAFFE_DEFINE_KNOWN_TYPE(char*, char_ptr)
CAFFE_DEFINE_KNOWN_TYPE(int*, int_ptr)

CAFFE_DEFINE_KNOWN_TYPE(
    detail::_guard_long_unique<long>,
    detail_guard_long_unique_long)
CAFFE_DEFINE_KNOWN_TYPE(
    detail::_guard_long_unique<std::vector<long>>,
    detail_guard_long_unique_std_vector_long)

CAFFE_DEFINE_KNOWN_TYPE(float*, float_ptr)
CAFFE_DEFINE_KNOWN_TYPE(at::Half*, at_Half)

} // namespace caffe2

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `detail`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `c10/util/typeid.h`
- `algorithm`
- `atomic`


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

Files in the same folder (`c10/util`):

- [`CallOnce.h_docs.md`](./CallOnce.h_docs.md)
- [`Unicode.cpp_docs.md`](./Unicode.cpp_docs.md)
- [`logging_is_not_google_glog.h_docs.md`](./logging_is_not_google_glog.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`complex_math.h_docs.md`](./complex_math.h_docs.md)
- [`order_preserving_flat_hash_map.h_docs.md`](./order_preserving_flat_hash_map.h_docs.md)
- [`flags_use_gflags.cpp_docs.md`](./flags_use_gflags.cpp_docs.md)
- [`flags_use_no_gflags.cpp_docs.md`](./flags_use_no_gflags.cpp_docs.md)
- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)


## Cross-References

- **File Documentation**: `typeid.cpp_docs.md`
- **Keyword Index**: `typeid.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

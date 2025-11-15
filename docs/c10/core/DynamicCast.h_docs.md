# Documentation: `c10/core/DynamicCast.h`

## File Metadata

- **Path**: `c10/core/DynamicCast.h`
- **Size**: 4,614 bytes (4.51 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Load.h>
#include <c10/util/TypeCast.h>

namespace c10 {

// Dynamic type casting utils:
// - fetch_and_cast
// - cast_and_store
//
// fetch_and_cast fetch a value with dynamic type specified by a ScalarType
// from a void pointer and cast it to a static type.
//
// cast_and_store casts a static typed value into dynamic type specified
// by a ScalarType, and store it into a void pointer.
//
// NOTE:
//
// Dynamic casting allows us to support type promotion without blowing up
// the combination space: For example, without dynamic cast, in order to
// implement `add_` with type promotion, we would need something like
//
// AT_DISPATCH_ALL_TYPES(output.dtype(),
//    AT_DISPATCH_ALL_TYPES(input1.dtype(),
//       AT_DISPATCH_ALL_TYPES(input2.dtype(),
//           [](arg0_t a, arg1_t b) -> out_t { return a + b; }
//       )
//    )
// )
//
// If we support N dtypes, the above code would generate the a+b kernel for
// all the N * N * N different supported types, the compilation time and
// binary size would become horrible.
//
// Dynamic casting might sounds like a bad idea in terms of performance.
// Especially if you ever do it in a loop, you are going to do a billion tests.
// But in practice it is not as bad as it might look:
//
// - on CPU, this is a branch that always has the same outcome, therefore
//   hopefully the branch predictor could do the job pretty well
// - on GPU, these branches will not diverge, so we could still have the same
//   warp executing the same line of code
// - Most kernels, like `add`, are bandwidth bound, adding a few clock cycles to
//   check an integer does not hurt the performance much because the ALUs would
//   wait for load instructions anyway.
//
// For the discussion and benchmark, refer to:
// - https://github.com/pytorch/pytorch/pull/28343
// - https://github.com/pytorch/pytorch/pull/28344
// - https://github.com/pytorch/pytorch/pull/28345
//

#ifdef C10_HOST_DEVICE
#define ERROR_UNSUPPORTED_CAST CUDA_KERNEL_ASSERT(false);
#else
#define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
#endif

// Fetch a value with dynamic type src_type from ptr, and cast it to static type
// dest_t.
#define FETCH_AND_CAST_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    return c10::convert<dest_t>(c10::load<type>(ptr));

template <typename dest_t>
C10_HOST_DEVICE inline dest_t fetch_and_cast(
    const ScalarType src_type,
    const void* ptr) {
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(FETCH_AND_CAST_CASE)
    FETCH_AND_CAST_CASE(uint16_t, UInt16)
    FETCH_AND_CAST_CASE(uint32_t, UInt32)
    FETCH_AND_CAST_CASE(uint64_t, UInt64)
    default:
      ERROR_UNSUPPORTED_CAST
  }
  C10_DIAGNOSTIC_POP()
  return dest_t(0); // just to avoid compiler warning
}

// Cast a value with static type src_t into dynamic dest_type, and store it to
// ptr.
#define CAST_AND_STORE_CASE(type, scalartype) \
  case ScalarType::scalartype:                \
    *(type*)ptr = c10::convert<type>(value);  \
    return;
template <typename src_t>
C10_HOST_DEVICE inline void cast_and_store(
    const ScalarType dest_type,
    void* ptr,
    src_t value) {
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CAST_AND_STORE_CASE)
    CAST_AND_STORE_CASE(uint16_t, UInt16)
    CAST_AND_STORE_CASE(uint32_t, UInt32)
    CAST_AND_STORE_CASE(uint64_t, UInt64)
    default:;
  }
  C10_DIAGNOSTIC_POP()
  ERROR_UNSUPPORTED_CAST
}

#define DEFINE_UNCASTABLE(T, scalartype_)                     \
  template <>                                                 \
  C10_HOST_DEVICE inline T fetch_and_cast<T>(                 \
      const ScalarType src_type, const void* ptr) {           \
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == src_type);  \
    return c10::load<T>(ptr);                                 \
  }                                                           \
  template <>                                                 \
  C10_HOST_DEVICE inline void cast_and_store<T>(              \
      const ScalarType dest_type, void* ptr, T value) {       \
    CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == dest_type); \
    *(T*)ptr = value;                                         \
  }

AT_FORALL_QINT_TYPES(DEFINE_UNCASTABLE)

#undef FETCH_AND_CAST_CASE
#undef CAST_AND_STORE_CASE
#undef DEFINE_UNCASTABLE
#undef ERROR_UNSUPPORTED_CAST

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

- `c10/core/ScalarType.h`
- `c10/macros/Macros.h`
- `c10/util/Load.h`
- `c10/util/TypeCast.h`


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

- **File Documentation**: `DynamicCast.h_docs.md`
- **Keyword Index**: `DynamicCast.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

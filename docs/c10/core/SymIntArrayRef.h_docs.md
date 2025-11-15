# Documentation: `c10/core/SymIntArrayRef.h`

## File Metadata

- **Path**: `c10/core/SymIntArrayRef.h`
- **Size**: 3,276 bytes (3.20 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cstdint>
#include <optional>

namespace c10 {
using SymIntArrayRef = ArrayRef<SymInt>;

inline at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar) {
  return IntArrayRef(reinterpret_cast<const int64_t*>(ar.data()), ar.size());
}

// TODO: a SymIntArrayRef containing a heap allocated large negative integer
// can actually technically be converted to an IntArrayRef... but not with
// the non-owning API we have here.  We can't reinterpet cast; we have to
// allocate another buffer and write the integers into it.  If you need it,
// we can do it.  But I don't think you need it.

inline std::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar) {
  for (const c10::SymInt& sci : ar) {
    if (sci.is_heap_allocated()) {
      return std::nullopt;
    }
  }

  return {asIntArrayRefUnchecked(ar)};
}

inline at::IntArrayRef asIntArrayRefSlow(
    c10::SymIntArrayRef ar,
    const char* file,
    int64_t line) {
  for (const c10::SymInt& sci : ar) {
    TORCH_CHECK(
        !sci.is_heap_allocated(),
        file,
        ":",
        line,
        ": SymIntArrayRef expected to contain only concrete integers");
  }
  return asIntArrayRefUnchecked(ar);
}

// Even slower than asIntArrayRefSlow, as it forces an allocation for a
// destination int, BUT it is able to force specialization (it never errors)
inline c10::DimVector asIntArrayRefSlowAlloc(
    c10::SymIntArrayRef ar,
    const char* file,
    int64_t line) {
  c10::DimVector res(ar.size(), 0);
  for (const auto i : c10::irange(ar.size())) {
    res[i] = ar[i].guard_int(file, line);
  }
  return res;
}

#define C10_AS_INTARRAYREF_SLOW(a) c10::asIntArrayRefSlow(a, __FILE__, __LINE__)
#define C10_AS_INTARRAYREF_SLOW_ALLOC(a) \
  c10::asIntArrayRefSlowAlloc(a, __FILE__, __LINE__)

// Prefer using a more semantic constructor, like
// fromIntArrayRefKnownNonNegative
inline SymIntArrayRef fromIntArrayRefUnchecked(IntArrayRef array_ref) {
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

inline SymIntArrayRef fromIntArrayRefKnownNonNegative(IntArrayRef array_ref) {
  return fromIntArrayRefUnchecked(array_ref);
}

inline SymIntArrayRef fromIntArrayRefSlow(IntArrayRef array_ref) {
  for (long i : array_ref) {
    TORCH_CHECK(
        SymInt::check_range(i),
        "IntArrayRef contains an int that cannot be represented as a SymInt: ",
        i);
  }
  return SymIntArrayRef(
      reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
}

inline c10::SymBool sym_equals(SymIntArrayRef LHS, SymIntArrayRef RHS) {
  if (LHS.size() != RHS.size()) {
    return c10::SymBool(false);
  }

  c10::SymBool result = sym_eq(LHS.size(), RHS.size());
  for (size_t i = 0; i < RHS.size(); ++i) {
    c10::SymBool equals = sym_eq(LHS[i], RHS[i]);
    std::optional<bool> equals_bool = equals.maybe_as_bool();

    if (equals_bool.has_value() && !*equals_bool) {
      // Early return if element comparison is known to be false
      return equals;
    }
    result = result.sym_and(equals);
  }
  return result;
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

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

- `c10/core/SymInt.h`
- `c10/util/ArrayRef.h`
- `c10/util/DimVector.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`
- `cstdint`
- `optional`


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

- **File Documentation**: `SymIntArrayRef.h_docs.md`
- **Keyword Index**: `SymIntArrayRef.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

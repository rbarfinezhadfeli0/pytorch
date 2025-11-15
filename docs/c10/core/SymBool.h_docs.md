# Documentation: `c10/core/SymBool.h`

## File Metadata

- **Path**: `c10/core/SymBool.h`
- **Size**: 4,638 bytes (4.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c

#pragma once

#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>
#include <optional>
#include <ostream>
#include <utility>

namespace c10 {

class SymInt;

class C10_API SymBool {
 public:
  /*implicit*/ SymBool(bool b) : data_(b) {}
  SymBool(SymNode ptr) : data_(false), ptr_(std::move(ptr)) {
    TORCH_CHECK(ptr_->is_bool());
  }
  SymBool() : data_(false) {}

  SymNodeImpl* toSymNodeImplUnowned() const {
    return ptr_.get();
  }

  SymNodeImpl* release() && {
    return std::move(ptr_).release();
  }

  // Only valid if is_heap_allocated()
  SymNode toSymNodeImpl() const;

  // Guaranteed to return a SymNode, wrapping using base if necessary
  SymNode wrap_node(const SymNode& base) const;

  bool expect_bool() const {
    std::optional<bool> c = maybe_as_bool();
    TORCH_CHECK(c.has_value());
    return *c;
  }

  SymBool sym_and(const SymBool& /*sci*/) const;
  SymBool sym_or(const SymBool& /*sci*/) const;
  SymBool sym_not() const;

  SymBool operator&(const SymBool& other) const {
    return sym_and(other);
  }
  SymBool operator|(const SymBool& other) const {
    return sym_or(other);
  }
  SymBool operator||(const SymBool& other) const {
    return sym_or(other);
  }
  SymBool operator~() const {
    return sym_not();
  }

  // Insert a guard for the bool to be its concrete value, and then return
  // that value.  Note that C++ comparison operations default to returning
  // bool, so it's not so common to have to call this
  bool guard_bool(const char* file, int64_t line) const;
  bool expect_true(const char* file, int64_t line) const;
  bool guard_size_oblivious(const char* file, int64_t line) const;
  bool statically_known_true(const char* file, int64_t line) const;
  bool guard_or_false(const char* file, int64_t line) const;
  bool guard_or_true(const char* file, int64_t line) const;

  bool has_hint() const;

  bool as_bool_unchecked() const {
    return data_;
  }

  std::optional<bool> maybe_as_bool() const {
    if (!is_heap_allocated()) {
      return data_;
    }
    return toSymNodeImplUnowned()->constant_bool();
  }

  // Convert SymBool to SymInt (0 or 1)
  // This is the C++ equivalent of Python's cast_symbool_to_symint_guardless
  SymInt toSymInt() const;

  bool is_heap_allocated() const {
    return ptr_;
  }

 private:
  // TODO: optimize to union
  bool data_;
  SymNode ptr_;
};

C10_API std::ostream& operator<<(std::ostream& os, const SymBool& s);

#define TORCH_SYM_CHECK(cond, ...) \
  TORCH_CHECK((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)
#define TORCH_SYM_INTERNAL_ASSERT(cond, ...) \
  TORCH_INTERNAL_ASSERT((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)
#define TORCH_MAYBE_SYM_CHECK(cond, ...)                                 \
  if constexpr (std::is_same_v<std::decay_t<decltype(cond)>, SymBool>) { \
    TORCH_CHECK((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)     \
  } else {                                                               \
    TORCH_CHECK((cond), __VA_ARGS__)                                     \
  }

inline bool guard_size_oblivious(
    bool b,
    const char* file [[maybe_unused]],
    int64_t line [[maybe_unused]]) {
  return b;
}

inline bool guard_size_oblivious(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
  return b.guard_size_oblivious(file, line);
}

inline bool guard_or_false(
    bool b,
    const char* file [[maybe_unused]],
    int64_t line [[maybe_unused]]) {
  return b;
}

inline bool guard_or_false(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
  return b.guard_or_false(file, line);
}

inline bool statically_known_true(
    bool b,
    const char* file [[maybe_unused]],
    int64_t line [[maybe_unused]]) {
  return b;
}

inline bool statically_known_true(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
  return b.statically_known_true(file, line);
}

inline bool guard_or_true(
    bool b,
    const char* file [[maybe_unused]],
    int64_t line [[maybe_unused]]) {
  return b;
}

inline bool guard_or_true(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
  return b.guard_or_true(file, line);
}

#define TORCH_GUARD_SIZE_OBLIVIOUS(cond) \
  c10::guard_size_oblivious((cond), __FILE__, __LINE__)

#define TORCH_STATICALLY_KNOWN_TRUE(cond) \
  c10::statically_known_true((cond), __FILE__, __LINE__)

#define TORCH_GUARD_OR_FALSE(cond) \
  c10::guard_or_false((cond), __FILE__, __LINE__)

#define TORCH_GUARD_OR_TRUE(cond) c10::guard_or_true((cond), __FILE__, __LINE__)

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 39 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `SymInt`, `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/SymNodeImpl.h`
- `c10/macros/Export.h`
- `c10/util/Exception.h`
- `c10/util/intrusive_ptr.h`
- `cstdint`
- `optional`
- `ostream`
- `utility`


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

- **File Documentation**: `SymBool.h_docs.md`
- **Keyword Index**: `SymBool.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

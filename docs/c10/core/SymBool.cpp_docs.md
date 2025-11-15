# Documentation: `c10/core/SymBool.cpp`

## File Metadata

- **Path**: `c10/core/SymBool.cpp`
- **Size**: 3,907 bytes (3.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>

namespace c10 {

SymNode SymBool::toSymNodeImpl() const {
  TORCH_CHECK(is_heap_allocated());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymBool::wrap_node(const SymNode& base) const {
  if (auto ma = maybe_as_bool()) {
    return base->wrap_bool(*ma);
  } else {
    return toSymNodeImpl();
  }
}

#define DEFINE_BINARY(API, OP, METHOD, RET)                              \
  RET SymBool::API(const SymBool& sci) const {                           \
    if (auto ma = maybe_as_bool()) {                                     \
      if (auto mb = sci.maybe_as_bool()) {                               \
        return RET(OP(*ma, *mb));                                        \
      } else {                                                           \
        auto b = sci.toSymNodeImpl();                                    \
        return RET(b->wrap_bool(*ma)->METHOD(b));                        \
      }                                                                  \
    } else {                                                             \
      if (auto mb = sci.maybe_as_bool()) {                               \
        auto a = toSymNodeImplUnowned();                                 \
        return RET(a->METHOD(a->wrap_bool(*mb)));                        \
      } else {                                                           \
        return RET(toSymNodeImplUnowned()->METHOD(sci.toSymNodeImpl())); \
      }                                                                  \
    }                                                                    \
  }

// clang-format off
DEFINE_BINARY(sym_and, std::logical_and<>(), sym_and, SymBool)
DEFINE_BINARY(sym_or, std::logical_or<>(), sym_or, SymBool)
// clang-format on

SymBool SymBool::sym_not() const {
  if (auto ma = maybe_as_bool()) {
    return SymBool(!*ma);
  }
  return SymBool(toSymNodeImpl()->sym_not());
}

std::ostream& operator<<(std::ostream& os, const SymBool& s) {
  if (auto ma = s.maybe_as_bool()) {
    os << *ma;
  } else {
    os << s.toSymNodeImpl()->str();
  }
  return os;
}

bool SymBool::guard_bool(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_bool(file, line);
}

bool SymBool::guard_size_oblivious(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_size_oblivious(file, line);
}

bool SymBool::guard_or_false(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_or_false(file, line);
}

bool SymBool::statically_known_true(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->statically_known_true(file, line);
}

bool SymBool::guard_or_true(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_or_true(file, line);
}

bool SymBool::expect_true(const char* file, int64_t line) const {
  if (auto ma = maybe_as_bool()) {
    return *ma;
  }
  SymNode a = toSymNodeImpl();
  return a->expect_true(file, line);
}

bool SymBool::has_hint() const {
  if (maybe_as_bool()) {
    return true;
  }
  return toSymNodeImpl()->has_hint();
}

SymInt SymBool::toSymInt() const {
  // If concrete bool, return concrete SymInt
  if (auto ma = maybe_as_bool()) {
    return SymInt(*ma ? 1 : 0);
  }

  // Symbolic case: use sym_ite to convert bool to int (0 or 1)
  auto node = toSymNodeImpl();
  auto one_node = node->wrap_int(1);
  auto zero_node = node->wrap_int(0);
  return SymInt(node->sym_ite(one_node, zero_node));
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

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

- `c10/core/SymBool.h`
- `c10/core/SymInt.h`
- `c10/core/SymNodeImpl.h`


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
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `SymBool.cpp_docs.md`
- **Keyword Index**: `SymBool.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

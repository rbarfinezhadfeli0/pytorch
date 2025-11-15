# Documentation: `c10/core/ConstantSymNodeImpl.h`

## File Metadata

- **Path**: `c10/core/ConstantSymNodeImpl.h`
- **Size**: 2,990 bytes (2.92 KB)
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
#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace c10 {

// Unlike other SymNodeImpl, this cannot be "dispatched" conventionally,
// as it typically needs to defer to another SymNodeImpl
//
// Can either represent a bool, int (don't support float yet) this is useful
// for representing otherwise unrepresentable large negative integer constant.
template <typename T>
class C10_API ConstantSymNodeImpl : public SymNodeImpl {
  static_assert(
      ::std::is_same_v<T, int64_t> || ::std::is_same_v<T, bool>,
      "ConstantSymNodeImpl can only accept int64_t or bool types");

 public:
  ConstantSymNodeImpl(T val) : value_(val) {}

  bool is_int() override {
    return is_int_();
  }
  bool is_bool() override {
    return is_bool_();
  }
  bool is_float() override {
    return false;
  }
  int64_t guard_int(
      const char* file [[maybe_unused]],
      int64_t line [[maybe_unused]]) override {
    TORCH_CHECK(is_int(), "not an int");
    return int_();
  }
  bool guard_bool(
      const char* file [[maybe_unused]],
      int64_t line [[maybe_unused]]) override {
    TORCH_CHECK(is_bool(), "not a bool");
    return bool_();
  }
  double guard_float(
      const char* file [[maybe_unused]],
      int64_t line [[maybe_unused]]) override {
    TORCH_CHECK(false, "not a float");
  }
  int64_t int_() override {
    TORCH_CHECK(is_int(), "not an int");
    return ::std::get<int64_t>(value_);
  }
  bool bool_() override {
    TORCH_CHECK(is_bool(), "not a bool");
    return ::std::get<bool>(value_);
  }
  bool has_hint() override {
    return true;
  }
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  c10::SymNode ge(const c10::SymNode& other) override;
  c10::SymNode le(const c10::SymNode& other) override;
  c10::SymNode lt(const c10::SymNode& other) override;
  c10::SymNode gt(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;
  ::std::string str() override {
    if constexpr (is_int_()) {
      return ::std::to_string(::std::get<int64_t>(value_));
    } else {
      return ::std::get<bool>(value_) ? "true" : "false";
    }
  }
  std::optional<int64_t> constant_int() override {
    if constexpr (is_int_()) {
      return ::std::get<int64_t>(value_);
    } else {
      return std::nullopt;
    }
  }
  std::optional<bool> constant_bool() override {
    if constexpr (is_bool_()) {
      return ::std::get<bool>(value_);
    } else {
      return std::nullopt;
    }
  }
  bool is_constant() override {
    return true;
  }
  bool is_symbolic() override {
    return false;
  }

 private:
  ::std::variant<int64_t, bool> value_;

  static constexpr bool is_int_() {
    return ::std::is_same_v<T, int64_t>;
  }
  static constexpr bool is_bool_() {
    return ::std::is_same_v<T, bool>;
  }
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 28 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_API`


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
- `cstdint`
- `optional`
- `string`
- `variant`


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

- **File Documentation**: `ConstantSymNodeImpl.h_docs.md`
- **Keyword Index**: `ConstantSymNodeImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

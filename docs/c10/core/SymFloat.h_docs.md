# Documentation: `c10/core/SymFloat.h`

## File Metadata

- **Path**: `c10/core/SymFloat.h`
- **Size**: 3,637 bytes (3.55 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/SymBool.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <limits>
#include <ostream>
#include <utility>

namespace c10 {

// NB: this is actually double precision; we're using the Python naming here
class C10_API SymFloat {
 public:
  /*implicit*/ SymFloat(double d) : data_(d) {}
  SymFloat(SymNode ptr)
      : data_(std::numeric_limits<double>::quiet_NaN()), ptr_(std::move(ptr)) {
    TORCH_CHECK(ptr_->is_float());
  }
  SymFloat() : data_(0.0) {}

  SymNodeImpl* toSymNodeImplUnowned() const {
    return ptr_.get();
  }

  SymNodeImpl* release() && {
    return std::move(ptr_).release();
  }

  // Only valid if is_symbolic()
  SymNode toSymNodeImpl() const;

  // Guaranteed to return a SymNode, wrapping using base if necessary
  SymNode wrap_node(const SymNode& base) const;

  double expect_float() const {
    TORCH_CHECK(!is_symbolic());
    return data_;
  }

  SymFloat operator+(const SymFloat& /*sci*/) const;
  SymFloat operator-(const SymFloat& /*sci*/) const;
  SymFloat operator*(const SymFloat& /*sci*/) const;
  SymFloat operator/(const SymFloat& /*sci*/) const;

  SymBool sym_eq(const SymFloat& /*sci*/) const;
  SymBool sym_ne(const SymFloat& /*sci*/) const;
  SymBool sym_lt(const SymFloat& /*sci*/) const;
  SymBool sym_le(const SymFloat& /*sci*/) const;
  SymBool sym_gt(const SymFloat& /*sci*/) const;
  SymBool sym_ge(const SymFloat& /*sci*/) const;

  bool operator==(const SymFloat& o) const {
    return sym_eq(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator!=(const SymFloat& o) const {
    return sym_ne(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<(const SymFloat& o) const {
    return sym_lt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<=(const SymFloat& o) const {
    return sym_le(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>(const SymFloat& o) const {
    return sym_gt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>=(const SymFloat& o) const {
    return sym_ge(o).guard_bool(__FILE__, __LINE__);
  }

  SymFloat min(const SymFloat& sci) const;
  SymFloat max(const SymFloat& sci) const;

  // Need guidance on where to put this code
  SymFloat sqrt() const;

  // Insert a guard for the float to be its concrete value, and then return
  // that value.  This operation always works, even if the float is symbolic,
  // so long as we know what the underlying value is. Don't blindly put this
  // everywhere; you can cause overspecialization of PyTorch programs with
  // this method.
  //
  // It should be called as guard_float(__FILE__, __LINE__).  The file and line
  // number can be used to diagnose overspecialization.
  double guard_float(const char* file, int64_t line) const;

  bool has_hint() const;

  // N.B. It's important to keep this definition in the header
  // as we expect if checks to be folded for mobile builds
  // where `is_symbolic` is always false
  C10_ALWAYS_INLINE bool is_symbolic() const {
    return ptr_;
  }

  // UNSAFELY coerce this SymFloat into a double.  You MUST have
  // established that this is a non-symbolic by some other means,
  // typically by having tested is_symbolic().  You will get garbage
  // from this function if is_symbolic()
  double as_float_unchecked() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!is_symbolic());
    return data_;
  }

 private:
  // TODO: optimize to union
  double data_;
  SymNode ptr_;
};

C10_API std::ostream& operator<<(std::ostream& os, const SymFloat& s);
} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 26 function(s).

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

- `c10/core/SymBool.h`
- `c10/core/SymNodeImpl.h`
- `c10/macros/Export.h`
- `c10/macros/Macros.h`
- `c10/util/Exception.h`
- `c10/util/intrusive_ptr.h`
- `cstdint`
- `limits`
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

- **File Documentation**: `SymFloat.h_docs.md`
- **Keyword Index**: `SymFloat.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

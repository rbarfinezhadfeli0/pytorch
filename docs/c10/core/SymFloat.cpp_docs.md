# Documentation: `c10/core/SymFloat.cpp`

## File Metadata

- **Path**: `c10/core/SymFloat.cpp`
- **Size**: 4,325 bytes (4.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/SymFloat.h>
#include <c10/core/SymNodeImpl.h>
#include <array>
#include <cmath>
#include <utility>

namespace c10 {

SymNode SymFloat::toSymNodeImpl() const {
  TORCH_CHECK(is_symbolic());
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

SymNode SymFloat::wrap_node(const SymNode& base) const {
  if (is_symbolic()) {
    return toSymNodeImpl();
  } else {
    return base->wrap_float(as_float_unchecked());
  }
}

static std::array<SymNode, 2> normalize_symfloats(
    const SymFloat& a_,
    const SymFloat& b_) {
  SymNode a, b;
  if (a_.is_symbolic())
    a = a_.toSymNodeImpl();
  if (b_.is_symbolic())
    b = b_.toSymNodeImpl();

  SymNodeImpl* common = a ? a.get() : b.get();
  if (!a) {
    a = common->wrap_float(a_.as_float_unchecked());
  }
  if (!b) {
    b = common->wrap_float(b_.as_float_unchecked());
  }
  return {std::move(a), std::move(b)};
}

SymFloat SymFloat::operator+(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ + sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->add(res[1]));
}

SymFloat SymFloat::operator-(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ - sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sub(res[1]));
}

SymFloat SymFloat::operator*(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ * sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->mul(res[1]));
}

SymFloat SymFloat::operator/(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return SymFloat(data_ / sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->truediv(res[1]));
}

SymBool SymFloat::sym_eq(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ == sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->eq(res[1]);
}

SymBool SymFloat::sym_ne(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ != sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->ne(res[1]);
}

SymBool SymFloat::sym_lt(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ < sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->lt(res[1]);
}

SymBool SymFloat::sym_le(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ <= sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->le(res[1]);
}

SymBool SymFloat::sym_gt(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ > sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->gt(res[1]);
}

SymBool SymFloat::sym_ge(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return data_ >= sci.data_;
  }
  auto res = normalize_symfloats(*this, sci);
  return res[0]->ge(res[1]);
}

SymFloat SymFloat::min(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::min(data_, sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sym_min(res[1]));
}
SymFloat SymFloat::max(const SymFloat& sci) const {
  if (!is_symbolic() && !sci.is_symbolic()) {
    return std::max(data_, sci.data_);
  }
  auto res = normalize_symfloats(*this, sci);
  return SymFloat(res[0]->sym_max(res[1]));
}

std::ostream& operator<<(std::ostream& os, const SymFloat& s) {
  if (s.is_symbolic()) {
    os << s.toSymNodeImpl()->str();
  } else {
    os << s.as_float_unchecked();
  }
  return os;
}

SymFloat SymFloat::sqrt() const {
  if (!is_symbolic()) {
    return SymFloat(std::sqrt(data_));
  }
  auto other = SymFloat(0.5);
  auto res = normalize_symfloats(*this, other);
  return SymFloat(res[0]->pow(res[1]));
}

double SymFloat::guard_float(const char* file, int64_t line) const {
  if (!is_symbolic()) {
    return data_;
  }
  SymNode a = toSymNodeImpl();
  return a->guard_float(file, line);
}

bool SymFloat::has_hint() const {
  if (!is_symbolic()) {
    return true;
  }
  return toSymNodeImpl()->has_hint();
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

- `c10/core/SymFloat.h`
- `c10/core/SymNodeImpl.h`
- `array`
- `cmath`
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

- **File Documentation**: `SymFloat.cpp_docs.md`
- **Keyword Index**: `SymFloat.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

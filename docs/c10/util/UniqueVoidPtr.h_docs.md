# Documentation: `c10/util/UniqueVoidPtr.h`

## File Metadata

- **Path**: `c10/util/UniqueVoidPtr.h`
- **Size**: 4,581 bytes (4.47 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <cstddef>
#include <memory>
#include <utility>

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

namespace c10 {

using DeleterFnPtr = void (*)(void*);

namespace detail {

// Does not delete anything
C10_API void deleteNothing(void* /*unused*/);

// A detail::UniqueVoidPtr is an owning smart pointer like unique_ptr, but
// with three major differences:
//
//    1) It is specialized to void
//
//    2) It is specialized for a function pointer deleter
//       void(void* ctx); i.e., the deleter doesn't take a
//       reference to the data, just to a context pointer
//       (erased as void*).  In fact, internally, this pointer
//       is implemented as having an owning reference to
//       context, and a non-owning reference to data; this is why
//       you release_context(), not release() (the conventional
//       API for release() wouldn't give you enough information
//       to properly dispose of the object later.)
//
//    3) The deleter is guaranteed to be called when the unique
//       pointer is destructed and the context is non-null; this is different
//       from std::unique_ptr where the deleter is not called if the
//       data pointer is null.
//
// Some of the methods have slightly different types than std::unique_ptr
// to reflect this.
//
class UniqueVoidPtr {
 private:
  // Lifetime tied to ctx_
  void* data_;
  std::unique_ptr<void, DeleterFnPtr> ctx_;

 public:
  UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, &deleteNothing) {}
  explicit UniqueVoidPtr(void* data)
      : data_(data), ctx_(nullptr, &deleteNothing) {}
  UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)
      : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing) {}
  void* operator->() const {
    return data_;
  }
  void clear() {
    ctx_ = nullptr;
    data_ = nullptr;
  }
  void* get() const {
    return data_;
  }

  bool /* success */ unsafe_reset_data_and_ctx(void* new_data_and_ctx) {
    if (C10_UNLIKELY(ctx_.get_deleter() != &deleteNothing)) {
      return false;
    }
    // seems quicker than calling the no-op deleter when we reset
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    ctx_.release();
    ctx_.reset(new_data_and_ctx);
    data_ = new_data_and_ctx;
    return true;
  }

  void* get_context() const {
    return ctx_.get();
  }
  void* release_context() {
    return ctx_.release();
  }
  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return std::move(ctx_);
  }
  [[nodiscard]] bool compare_exchange_deleter(
      DeleterFnPtr expected_deleter,
      DeleterFnPtr new_deleter) {
    if (get_deleter() != expected_deleter)
      return false;
    ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
    return true;
  }

  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    if (get_deleter() != expected_deleter)
      return nullptr;
    return static_cast<T*>(get_context());
  }
  operator bool() const {
    return data_ || ctx_;
  }
  DeleterFnPtr get_deleter() const {
    return ctx_.get_deleter();
  }
};

// Note [How UniqueVoidPtr is implemented]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UniqueVoidPtr solves a common problem for allocators of tensor data, which
// is that the data pointer (e.g., float*) which you are interested in, is not
// the same as the context pointer (e.g., DLManagedTensor) which you need
// to actually deallocate the data.  Under a conventional deleter design, you
// have to store extra context in the deleter itself so that you can actually
// delete the right thing.  Implementing this with standard C++ is somewhat
// error-prone: if you use a std::unique_ptr to manage tensors, the deleter will
// not be called if the data pointer is nullptr, which can cause a leak if the
// context pointer is non-null (and the deleter is responsible for freeing both
// the data pointer and the context pointer).
//
// So, in our reimplementation of unique_ptr, which just store the context
// directly in the unique pointer, and attach the deleter to the context
// pointer itself.  In simple cases, the context pointer is just the pointer
// itself.

inline bool operator==(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return !sp;
}
inline bool operator==(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return !sp;
}
inline bool operator!=(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return sp;
}
inline bool operator!=(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return sp;
}

} // namespace detail
} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `UniqueVoidPtr`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `cstddef`
- `memory`
- `utility`
- `c10/macros/Export.h`
- `c10/macros/Macros.h`


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
- [`typeid.cpp_docs.md`](./typeid.cpp_docs.md)


## Cross-References

- **File Documentation**: `UniqueVoidPtr.h_docs.md`
- **Keyword Index**: `UniqueVoidPtr.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

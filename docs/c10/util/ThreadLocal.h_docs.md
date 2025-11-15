# Documentation: `c10/util/ThreadLocal.h`

## File Metadata

- **Path**: `c10/util/ThreadLocal.h`
- **Size**: 4,020 bytes (3.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Macros.h>

/**
 * Android versions with libgnustl incorrectly handle thread_local C++
 * qualifier with composite types. NDK up to r17 version is affected.
 *
 * (A fix landed on Jun 4 2018:
 * https://android-review.googlesource.com/c/toolchain/gcc/+/683601)
 *
 * In such cases, use c10::ThreadLocal<T> wrapper
 * which is `pthread_*` based with smart pointer semantics.
 *
 * In addition, convenient macro C10_DEFINE_TLS_static is available.
 * To define static TLS variable of type std::string, do the following
 * ```
 *  C10_DEFINE_TLS_static(std::string, str_tls_);
 *  ///////
 *  {
 *    *str_tls_ = "abc";
 *    assert(str_tls_->length(), 3);
 *  }
 * ```
 *
 * (see c10/test/util/ThreadLocal_test.cpp for more examples)
 */
#if !defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(C10_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604
#define C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE
#endif // defined(C10_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604

#endif // !defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
#include <c10/util/Exception.h>
#include <errno.h>
#include <pthread.h>
#include <memory>
namespace c10 {

/**
 * @brief Temporary thread_local C++ qualifier replacement for Android
 * based on `pthread_*`.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal {
 public:
  ThreadLocal() {
    pthread_key_create(
        &key_, [](void* buf) { delete static_cast<Type*>(buf); });
  }

  ~ThreadLocal() {
    if (void* current = pthread_getspecific(key_)) {
      delete static_cast<Type*>(current);
    }

    pthread_key_delete(key_);
  }

  ThreadLocal(const ThreadLocal&) = delete;
  ThreadLocal& operator=(const ThreadLocal&) = delete;

  Type& get() {
    if (void* current = pthread_getspecific(key_)) {
      return *static_cast<Type*>(current);
    }

    std::unique_ptr<Type> ptr = std::make_unique<Type>();
    if (0 == pthread_setspecific(key_, ptr.get())) {
      return *ptr.release();
    }

    int err = errno;
    TORCH_INTERNAL_ASSERT(false, "pthread_setspecific() failed, errno = ", err);
  }

  Type& operator*() {
    return get();
  }

  Type* operator->() {
    return &get();
  }

 private:
  pthread_key_t key_;
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name) static ::c10::ThreadLocal<Type> Name

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \
  static ::c10::ThreadLocal<Type> Name

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \
  ::c10::ThreadLocal<Type> Class::Name

#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

namespace c10 {

/**
 * @brief Default thread_local implementation for non-Android cases.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal {
 public:
  using Accessor = Type* (*)();
  explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}

  ThreadLocal(const ThreadLocal&) = delete;
  ThreadLocal(ThreadLocal&&) noexcept = default;
  ThreadLocal& operator=(const ThreadLocal&) = delete;
  ThreadLocal& operator=(ThreadLocal&&) noexcept = default;
  ~ThreadLocal() = default;

  Type& get() {
    return *accessor_();
  }

  Type& operator*() {
    return get();
  }

  Type* operator->() {
    return &get();
  }

 private:
  Accessor accessor_;
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name)     \
  static ::c10::ThreadLocal<Type> Name([]() { \
    static thread_local Type var;             \
    return &var;                              \
  })

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \
  static ::c10::ThreadLocal<Type> Name

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \
  ::c10::ThreadLocal<Type> Class::Name([]() {          \
    static thread_local Type var;                      \
    return &var;                                       \
  })

#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `ThreadLocal`, `ThreadLocal`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `c10/util/Exception.h`
- `errno.h`
- `pthread.h`
- `memory`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `ThreadLocal.h_docs.md`
- **Keyword Index**: `ThreadLocal.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `c10/util/logging_is_google_glog.h`

## File Metadata

- **Path**: `c10/util/logging_is_google_glog.h`
- **Size**: 2,909 bytes (2.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#ifndef C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_
#define C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_

#include <map>
#include <set>
#include <vector>

#include <iomanip> // because some of the caffe2 code uses e.g. std::setw
// Using google glog. For glog 0.3.2 versions, stl_logging.h needs to be before
// logging.h to actually use stl_logging. Because template magic.
// In addition, we do not do stl logging in .cu files because nvcc does not like
// it. Some mobile platforms do not like stl_logging, so we add an
// overload in that case as well.

#ifdef __CUDACC__
#include <cuda.h>
#endif

#if !defined(__CUDACC__) && !defined(C10_USE_MINIMAL_GLOG)
#include <glog/stl_logging.h>

// Old versions of glog don't declare this using declaration, so help
// them out.  Fortunately, C++ won't complain if you declare the same
// using declaration multiple times.
namespace std {
using ::operator<<;
}

#else // !defined(__CUDACC__) && !defined(C10_USE_MINIMAL_GLOG)

// In the cudacc compiler scenario, we will simply ignore the container
// printout feature. Basically we need to register a fake overload for
// vector/string - here, we just ignore the entries in the logs.

namespace std {
#define INSTANTIATE_FOR_CONTAINER(container)                      \
  template <class... Types>                                       \
  ostream& operator<<(ostream& out, const container<Types...>&) { \
    return out;                                                   \
  }

INSTANTIATE_FOR_CONTAINER(vector)
INSTANTIATE_FOR_CONTAINER(map)
INSTANTIATE_FOR_CONTAINER(set)
#undef INSTANTIATE_FOR_CONTAINER
} // namespace std

#endif

#include <c10/util/logging_common.h>
#include <glog/logging.h>

namespace c10 {

[[noreturn]] void ThrowEnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg,
    const void* caller);

template <typename T>
T& CheckNotNullCommon(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  if (t == nullptr) {
    MessageLogger(file, line, ::google::GLOG_FATAL, fatal).stream()
        << "Check failed: '" << names << "' must be non NULL. ";
  }
  return t;
}

template <typename T>
T* CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T* t,
    bool fatal) {
  return CheckNotNullCommon(file, line, names, t, fatal);
}

template <typename T>
T& CheckNotNull(
    const char* file,
    int line,
    const char* names,
    T& t,
    bool fatal) {
  return CheckNotNullCommon(file, line, names, t, fatal);
}

} // namespace c10

// Log with source location information override (to be used in generic
// warning/error handlers implemented as functions, not macros)
//
// Note, we don't respect GOOGLE_STRIP_LOG here for simplicity
#define LOG_AT_FILE_LINE(n, file, line) \
  ::google::LogMessage(file, line, ::google::GLOG_##n).stream()

#endif // C10_UTIL_LOGGING_IS_GOOGLE_GLOG_H_

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`, `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `map`
- `set`
- `vector`
- `iomanip`
- `cuda.h`
- `glog/stl_logging.h`
- `c10/util/logging_common.h`
- `glog/logging.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `logging_is_google_glog.h_docs.md`
- **Keyword Index**: `logging_is_google_glog.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `c10/util/Deprecated.h`

## File Metadata

- **Path**: `c10/util/Deprecated.h`
- **Size**: 3,579 bytes (3.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

/**
 * This file provides portable macros for marking declarations
 * as deprecated.  You should generally use C10_DEPRECATED,
 * except when marking 'using' declarations as deprecated,
 * in which case you should use C10_DEFINE_DEPRECATED_USING
 * (due to portability concerns).
 */

// Sample usage:
//
//    C10_DEPRECATED void bad_func();
//    struct C10_DEPRECATED BadStruct {
//      ...
//    };

// NB: __cplusplus doesn't work for MSVC, so for now MSVC always uses
// the "__declspec(deprecated)" implementation and not the C++14
// "[[deprecated]]" attribute. We tried enabling "[[deprecated]]" for C++14 on
// MSVC, but ran into issues with some older MSVC versions.
#if (defined(__cplusplus) && __cplusplus >= 201402L)
#define C10_DEPRECATED [[deprecated]]
#define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]
#elif defined(__GNUC__)
#define C10_DEPRECATED __attribute__((deprecated))
// TODO Is there some way to implement this?
#define C10_DEPRECATED_MESSAGE(message) __attribute__((deprecated))

#elif defined(_MSC_VER)
#define C10_DEPRECATED __declspec(deprecated)
#define C10_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
#warning "You need to implement C10_DEPRECATED for this compiler"
#define C10_DEPRECATED
#endif

// Sample usage:
//
//    C10_DEFINE_DEPRECATED_USING(BadType, int)
//
//   which is the portable version of
//
//    using BadType [[deprecated]] = int;

// technically [[deprecated]] syntax is from c++14 standard, but it works in
// many compilers.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated) && !defined(__CUDACC__)
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;
#endif
#endif

#if defined(_MSC_VER)
#if defined(__CUDACC__)
// neither [[deprecated]] nor __declspec(deprecated) work on nvcc on Windows;
// you get the error:
//
//    error: attribute does not apply to any entity
//
// So we just turn the macro off in this case.
#if defined(C10_DEFINE_DEPRECATED_USING)
#undef C10_DEFINE_DEPRECATED_USING
#endif
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = TypeThingy;
#else
// [[deprecated]] does work in windows without nvcc, though msc doesn't support
// `__has_cpp_attribute` when c++14 is supported, otherwise
// __declspec(deprecated) is used as the alternative.
#ifndef C10_DEFINE_DEPRECATED_USING
#if defined(_MSVC_LANG) && _MSVC_LANG >= 201402L
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;
#else
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = __declspec(deprecated) TypeThingy;
#endif
#endif
#endif
#endif

#if !defined(C10_DEFINE_DEPRECATED_USING) && defined(__GNUC__)
// nvcc has a bug where it doesn't understand __attribute__((deprecated))
// declarations even when the host compiler supports it. We'll only use this gcc
// attribute when not cuda, and when using a GCC compiler that doesn't support
// the c++14 syntax we checked for above (available in __GNUC__ >= 5)
#if !defined(__CUDACC__)
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName __attribute__((deprecated)) = TypeThingy;
#else
// using cuda + gcc < 5, neither deprecated syntax is available so turning off.
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = TypeThingy;
#endif
#endif

#if !defined(C10_DEFINE_DEPRECATED_USING)
#warning "You need to implement C10_DEFINE_DEPRECATED_USING for this compiler"
#define C10_DEFINE_DEPRECATED_USING
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `C10_DEPRECATED`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

*No includes detected.*


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

- **File Documentation**: `Deprecated.h_docs.md`
- **Keyword Index**: `Deprecated.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

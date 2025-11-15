# Documentation: `c10/util/AlignOf.h`

## File Metadata

- **Path**: `c10/util/AlignOf.h`
- **Size**: 4,906 bytes (4.79 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
//===--- AlignOf.h - Portable calculation of type alignment -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AlignedCharArray and AlignedCharArrayUnion classes.
//
//===----------------------------------------------------------------------===//

// ATen: modified from llvm::AlignOf
// replaced LLVM_ALIGNAS with alignas

#pragma once

#include <cstddef>

namespace c10 {

/// \struct AlignedCharArray
/// \brief Helper for building an aligned character array type.
///
/// This template is used to explicitly build up a collection of aligned
/// character array types. We have to build these up using a macro and explicit
/// specialization to cope with MSVC (at least till 2015) where only an
/// integer literal can be used to specify an alignment constraint. Once built
/// up here, we can then begin to indirect between these using normal C++
/// template parameters.

// MSVC requires special handling here.
#ifndef _MSC_VER

template <size_t Alignment, size_t Size>
struct AlignedCharArray {
  // NOLINTNEXTLINE(*c-arrays)
  alignas(Alignment) char buffer[Size];
};

#else // _MSC_VER

/// \brief Create a type with an aligned char buffer.
template <size_t Alignment, size_t Size>
struct AlignedCharArray;

// We provide special variations of this template for the most common
// alignments because __declspec(align(...)) doesn't actually work when it is
// a member of a by-value function argument in MSVC, even if the alignment
// request is something reasonably like 8-byte or 16-byte. Note that we can't
// even include the declspec with the union that forces the alignment because
// MSVC warns on the existence of the declspec despite the union member forcing
// proper alignment.

template <size_t Size>
struct AlignedCharArray<1, Size> {
  union {
    char aligned;
    char buffer[Size];
  };
};

template <size_t Size>
struct AlignedCharArray<2, Size> {
  union {
    short aligned;
    char buffer[Size];
  };
};

template <size_t Size>
struct AlignedCharArray<4, Size> {
  union {
    int aligned;
    char buffer[Size];
  };
};

template <size_t Size>
struct AlignedCharArray<8, Size> {
  union {
    double aligned;
    char buffer[Size];
  };
};

// The rest of these are provided with a __declspec(align(...)) and we simply
// can't pass them by-value as function arguments on MSVC.

#define AT_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(x) \
  template <size_t Size>                          \
  struct AlignedCharArray<x, Size> {              \
    __declspec(align(x)) char buffer[Size];       \
  };

AT_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(16)
AT_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(32)
AT_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(64)
AT_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT(128)

#undef AT_ALIGNEDCHARARRAY_TEMPLATE_ALIGNMENT

#endif // _MSC_VER

namespace detail {
template <
    typename T1,
    typename T2 = char,
    typename T3 = char,
    typename T4 = char,
    typename T5 = char,
    typename T6 = char,
    typename T7 = char,
    typename T8 = char,
    typename T9 = char,
    typename T10 = char>
class AlignerImpl {
  T1 t1;
  T2 t2;
  T3 t3;
  T4 t4;
  T5 t5;
  T6 t6;
  T7 t7;
  T8 t8;
  T9 t9;
  T10 t10;

 public:
  AlignerImpl() = delete;
};

template <
    typename T1,
    typename T2 = char,
    typename T3 = char,
    typename T4 = char,
    typename T5 = char,
    typename T6 = char,
    typename T7 = char,
    typename T8 = char,
    typename T9 = char,
    typename T10 = char>
union SizerImpl {
  // NOLINTNEXTLINE(*c-arrays)
  char arr1[sizeof(T1)], arr2[sizeof(T2)], arr3[sizeof(T3)], arr4[sizeof(T4)],
      arr5[sizeof(T5)], arr6[sizeof(T6)], arr7[sizeof(T7)], arr8[sizeof(T8)],
      arr9[sizeof(T9)], arr10[sizeof(T10)];
};
} // end namespace detail

/// \brief This union template exposes a suitably aligned and sized character
/// array member which can hold elements of any of up to ten types.
///
/// These types may be arrays, structs, or any other types. The goal is to
/// expose a char array buffer member which can be used as suitable storage for
/// a placement new of any of these types. Support for more than ten types can
/// be added at the cost of more boilerplate.
template <
    typename T1,
    typename T2 = char,
    typename T3 = char,
    typename T4 = char,
    typename T5 = char,
    typename T6 = char,
    typename T7 = char,
    typename T8 = char,
    typename T9 = char,
    typename T10 = char>
struct AlignedCharArrayUnion
    : AlignedCharArray<
          alignof(detail::AlignerImpl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>),
          sizeof(::c10::detail::
                     SizerImpl<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>)> {};
} // end namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `AlignedCharArray`, `AlignedCharArray`, `AlignedCharArray`, `AlignedCharArray`, `AlignedCharArray`, `AlignedCharArray`, `AlignedCharArray`, `AlignedCharArray`, `AlignerImpl`, `AlignedCharArrayUnion`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `cstddef`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

- **File Documentation**: `AlignOf.h_docs.md`
- **Keyword Index**: `AlignOf.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

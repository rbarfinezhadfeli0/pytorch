# Documentation: `c10/test/util/string_view_test.cpp`

## File Metadata

- **Path**: `c10/test/util/string_view_test.cpp`
- **Size**: 3,629 bytes (3.54 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/string_view.h>

#include <gmock/gmock.h>

// NOLINTBEGIN(modernize*, readability*, bugprone-string-constructor)
using string_view = c10::c10_string_view;

namespace {
namespace testutils {
constexpr bool string_equal(const char* lhs, const char* rhs, size_t size) {
  return (size == 0)   ? true
      : (*lhs != *rhs) ? false
                       : string_equal(lhs + 1, rhs + 1, size - 1);
}
static_assert(string_equal("hi", "hi", 2), "");
static_assert(string_equal("", "", 0), "");
static_assert(string_equal("hi", "hi2", 2), "");
static_assert(string_equal("hi2", "hi", 2), "");
static_assert(!string_equal("hi", "hi2", 3), "");
static_assert(!string_equal("hi2", "hi", 3), "");
static_assert(!string_equal("hi", "ha", 2), "");

template <class Exception, class Functor>
inline void expectThrows(Functor&& functor, const char* expectMessageContains) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Exception& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr(expectMessageContains));
    return;
  }
  ADD_FAILURE() << "Expected to throw exception containing \""
                << expectMessageContains << "\" but didn't throw";
}
} // namespace testutils

using testutils::expectThrows;
using testutils::string_equal;

namespace test_starts_with {
static_assert(string_view("hi").starts_with(string_view("hi")), "");
static_assert(string_view("").starts_with(string_view("")), "");
static_assert(string_view("hi2").starts_with(string_view("")), "");
static_assert(!string_view("").starts_with(string_view("hi")), "");
static_assert(string_view("hi2").starts_with(string_view("hi")), "");
static_assert(!string_view("hi").starts_with(string_view("hi2")), "");
static_assert(!string_view("hi").starts_with(string_view("ha")), "");

static_assert(string_view("hi").starts_with("hi"), "");
static_assert(string_view("").starts_with(""), "");
static_assert(string_view("hi2").starts_with(""), "");
static_assert(!string_view("").starts_with("hi"), "");
static_assert(string_view("hi2").starts_with("hi"), "");
static_assert(!string_view("hi").starts_with("hi2"), "");
static_assert(!string_view("hi").starts_with("ha"), "");

static_assert(!string_view("").starts_with('a'), "");
static_assert(!string_view("").starts_with('\0'), "");
static_assert(!string_view("hello").starts_with('a'), "");
static_assert(string_view("hello").starts_with('h'), "");
} // namespace test_starts_with

namespace test_ends_with {
static_assert(string_view("hi").ends_with(string_view("hi")), "");
static_assert(string_view("").ends_with(string_view("")), "");
static_assert(string_view("hi2").ends_with(string_view("")), "");
static_assert(!string_view("").ends_with(string_view("hi")), "");
static_assert(string_view("hi2").ends_with(string_view("i2")), "");
static_assert(!string_view("i2").ends_with(string_view("hi2")), "");
static_assert(!string_view("hi").ends_with(string_view("ha")), "");

static_assert(string_view("hi").ends_with("hi"), "");
static_assert(string_view("").ends_with(""), "");
static_assert(string_view("hi2").ends_with(""), "");
static_assert(!string_view("").ends_with("hi"), "");
static_assert(string_view("hi2").ends_with("i2"), "");
static_assert(!string_view("i2").ends_with("hi2"), "");
static_assert(!string_view("hi").ends_with("ha"), "");

static_assert(!string_view("").ends_with('a'), "");
static_assert(!string_view("").ends_with('\0'), "");
static_assert(!string_view("hello").ends_with('a'), "");
static_assert(string_view("hello").ends_with('o'), "");
} // namespace test_ends_with

} // namespace
// NOLINTEND(modernize*, readability*, bugprone-string-constructor)

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `test_ends_with`, `test_starts_with`, `testutils`

**Classes/Structs**: `Exception`, `Functor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/string_view.h`
- `gmock/gmock.h`


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

This is a test file. Run it with:

```bash
python c10/test/util/string_view_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/util`):

- [`bfloat16_test.cpp_docs.md`](./bfloat16_test.cpp_docs.md)
- [`complex_test_common.h_docs.md`](./complex_test_common.h_docs.md)
- [`TypeIndex_test.cpp_docs.md`](./TypeIndex_test.cpp_docs.md)
- [`generic_math_test.cpp_docs.md`](./generic_math_test.cpp_docs.md)
- [`Half_test.cpp_docs.md`](./Half_test.cpp_docs.md)
- [`nofatal_test.cpp_docs.md`](./nofatal_test.cpp_docs.md)
- [`small_vector_test.cpp_docs.md`](./small_vector_test.cpp_docs.md)
- [`exception_test.cpp_docs.md`](./exception_test.cpp_docs.md)
- [`Enumerate_test.cpp_docs.md`](./Enumerate_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `string_view_test.cpp_docs.md`
- **Keyword Index**: `string_view_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

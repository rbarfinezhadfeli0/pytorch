# Documentation: `c10/test/util/exception_test.cpp`

## File Metadata

- **Path**: `c10/test/util/exception_test.cpp`
- **Size**: 2,966 bytes (2.90 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <stdexcept>

using c10::Error;

namespace {

template <class Functor>
inline void expectThrowsEq(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Error& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expectedMessage);
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw";
}
} // namespace

TEST(ExceptionTest, TORCH_INTERNAL_ASSERT_DEBUG_ONLY) {
#ifdef NDEBUG
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false));
  // Does nothing - `throw ...` should not be evaluated
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (throw std::runtime_error("I'm throwing..."), true)));
#else
  ASSERT_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false), c10::Error);
  ASSERT_NO_THROW(TORCH_INTERNAL_ASSERT_DEBUG_ONLY(true));
#endif
}

// On these platforms there's no assert
#if !defined(__ANDROID__) && !defined(__APPLE__)
TEST(ExceptionTest, CUDA_KERNEL_ASSERT) {
  // This function always throws even in NDEBUG mode
  ASSERT_DEATH_IF_SUPPORTED({ CUDA_KERNEL_ASSERT(false); }, "Assert");
}
#endif

TEST(WarningTest, JustPrintWarning) {
  TORCH_WARN("I'm a warning");
}

TEST(ExceptionTest, ErrorFormatting) {
  expectThrowsEq(
      []() { TORCH_CHECK(false, "This is invalid"); }, "This is invalid");

  expectThrowsEq(
      []() {
        try {
          TORCH_CHECK(false, "This is invalid");
        } catch (Error& e) {
          TORCH_RETHROW(e, "While checking X");
        }
      },
      "This is invalid (While checking X)");

  expectThrowsEq(
      []() {
        try {
          try {
            TORCH_CHECK(false, "This is invalid");
          } catch (Error& e) {
            TORCH_RETHROW(e, "While checking X");
          }
        } catch (Error& e) {
          TORCH_RETHROW(e, "While checking Y");
        }
      },
      R"msg(This is invalid
  While checking X
  While checking Y)msg");
}

static int assertionArgumentCounter = 0;
static int getAssertionArgument() {
  return ++assertionArgumentCounter;
}

static void failCheck() {
  TORCH_CHECK(false, "message ", getAssertionArgument());
}

static void failInternalAssert() {
  TORCH_INTERNAL_ASSERT(false, "message ", getAssertionArgument());
}

TEST(ExceptionTest, DontCallArgumentFunctionsTwiceOnFailure) {
  assertionArgumentCounter = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failCheck());
  EXPECT_EQ(assertionArgumentCounter, 1) << "TORCH_CHECK called argument twice";

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(failInternalAssert());
  EXPECT_EQ(assertionArgumentCounter, 2)
      << "TORCH_INTERNAL_ASSERT called argument twice";
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `TEST`

**Classes/Structs**: `Functor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `gtest/gtest.h`
- `stdexcept`


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

This is a test file. Run it with:

```bash
python c10/test/util/exception_test.cpp
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
- [`string_view_test.cpp_docs.md`](./string_view_test.cpp_docs.md)
- [`Enumerate_test.cpp_docs.md`](./Enumerate_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `exception_test.cpp_docs.md`
- **Keyword Index**: `exception_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

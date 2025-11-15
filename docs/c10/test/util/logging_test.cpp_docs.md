# Documentation: `c10/test/util/logging_test.cpp`

## File Metadata

- **Path**: `c10/test/util/logging_test.cpp`
- **Size**: 6,228 bytes (6.08 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <algorithm>
#include <optional>

#include <c10/util/ArrayRef.h>
#include <c10/util/Logging.h>
#include <gtest/gtest.h>

namespace c10_test {

using std::set;
using std::string;
using std::vector;

TEST(LoggingTest, TestEnforceTrue) {
  // This should just work.
  CAFFE_ENFORCE(true, "Isn't it?");
}

TEST(LoggingTest, TestEnforceFalse) {
  bool kFalse = false;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
  try {
    CAFFE_ENFORCE(false, "This throws.");
    // This should never be triggered.
    ADD_FAILURE();
    // NOLINTNEXTLINE(*catch*)
  } catch (const ::c10::Error&) {
  }
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
}

TEST(LoggingTest, TestEnforceEquals) {
  int x = 4;
  int y = 5;
  int z = 0;
  try {
    CAFFE_ENFORCE_THAT(std::equal_to<void>(), ==, ++x, ++y, "Message: ", z++);
    // This should never be triggered.
    ADD_FAILURE();
  } catch (const ::c10::Error& err) {
    auto errStr = std::string(err.what());
    EXPECT_NE(errStr.find("5 vs 6"), string::npos);
    EXPECT_NE(errStr.find("Message: 0"), string::npos);
  }

  // arguments are expanded only once
  CAFFE_ENFORCE_THAT(std::equal_to<void>(), ==, ++x, y);
  EXPECT_EQ(x, 6);
  EXPECT_EQ(y, 6);
  EXPECT_EQ(z, 1);
}

namespace {
struct EnforceEqWithCaller {
  void test(const char* x) {
    CAFFE_ENFORCE_EQ_WITH_CALLER(1, 1, "variable: ", x, " is a variable");
  }
};
} // namespace

TEST(LoggingTest, TestEnforceMessageVariables) {
  const char* const x = "hello";
  CAFFE_ENFORCE_EQ(1, 1, "variable: ", x, " is a variable");

  EnforceEqWithCaller e;
  e.test(x);
}

TEST(
    LoggingTest,
    EnforceEqualsObjectWithReferenceToTemporaryWithoutUseOutOfScope) {
  std::vector<int> x = {1, 2, 3, 4};
  // This case is a little tricky. We have a temporary
  // std::initializer_list to which our temporary ArrayRef
  // refers. Temporary lifetime extension by binding a const reference
  // to the ArrayRef doesn't extend the lifetime of the
  // std::initializer_list, just the ArrayRef, so we end up with a
  // dangling ArrayRef. This test forces the implementation to get it
  // right.
  CAFFE_ENFORCE_EQ(x, (at::ArrayRef<int>{1, 2, 3, 4}));
}

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct Noncopyable {
  int x;

  explicit Noncopyable(int a) : x(a) {}

  Noncopyable(const Noncopyable&) = delete;
  Noncopyable(Noncopyable&&) = delete;
  Noncopyable& operator=(const Noncopyable&) = delete;
  Noncopyable& operator=(Noncopyable&&) = delete;

  bool operator==(const Noncopyable& rhs) const {
    return x == rhs.x;
  }
};

std::ostream& operator<<(std::ostream& out, const Noncopyable& nc) {
  out << "Noncopyable(" << nc.x << ")";
  return out;
}
} // namespace

TEST(LoggingTest, DoesntCopyComparedObjects) {
  CAFFE_ENFORCE_EQ(Noncopyable(123), Noncopyable(123));
}

TEST(LoggingTest, EnforceShowcase) {
  // It's not really a test but rather a convenient thing that you can run and
  // see all messages
  int one = 1;
  int two = 2;
  int three = 3;
#define WRAP_AND_PRINT(exp)                    \
  try {                                        \
    exp;                                       \
  } catch (const ::c10::Error&) {              \
    /* ::c10::Error already does LOG(ERROR) */ \
  }
  WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_NE(one * 2, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_GT(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_GE(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_LT(three, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_LE(three, two));

  WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(
      one * two + three, three * two, "It's a pretty complicated expression"));

  WRAP_AND_PRINT(CAFFE_ENFORCE_THAT(
      std::equal_to<void>(), ==, one * two + three, three * two));
}

TEST(LoggingTest, Join) {
  auto s = c10::Join(", ", vector<int>({1, 2, 3}));
  EXPECT_EQ(s, "1, 2, 3");
  s = c10::Join(":", vector<string>());
  EXPECT_EQ(s, "");
  s = c10::Join(", ", set<int>({3, 1, 2}));
  EXPECT_EQ(s, "1, 2, 3");
}

TEST(LoggingTest, TestDanglingElse) {
  if (true)
    TORCH_DCHECK_EQ(1, 1);
  else
    GTEST_FAIL();
}

#if GTEST_HAS_DEATH_TEST
TEST(LoggingDeathTest, TestEnforceUsingFatal) {
  bool kTrue = true;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_DEATH(CAFFE_ENFORCE(false, "This goes fatal."), "");
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
}
#endif

#ifdef FBCODE_CAFFE2
static C10_NOINLINE void f1() {
  CAFFE_THROW("message");
}

static C10_NOINLINE void f2() {
  f1();
}

static C10_NOINLINE void f3() {
  f2();
}
TEST(LoggingTest, ExceptionWhat) {
  std::optional<::c10::Error> error;
  try {
    f3();
  } catch (const ::c10::Error& e) {
    error = e;
  }

  ASSERT_TRUE(error);
  std::string what = error->what();

  EXPECT_TRUE(what.find("c10_test::f1()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f2()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f3()") != std::string::npos) << what;

  // what() should be recomputed.
  error->add_context("NewContext");
  what = error->what();
  EXPECT_TRUE(what.find("c10_test::f1()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f2()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f3()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("NewContext") != std::string::npos) << what;
}
#endif

TEST(LoggingTest, LazyBacktrace) {
  struct CountingLazyString : ::c10::OptimisticLazyValue<std::string> {
    mutable size_t invocations{0};

    std::string compute() const override {
      ++invocations;
      return "A string";
    }
  };

  auto backtrace = std::make_shared<CountingLazyString>();
  ::c10::Error ex("", backtrace);
  // The backtrace is not computed on construction, and then it is not computed
  // more than once.
  EXPECT_EQ(backtrace->invocations, 0);
  const char* w1 = ex.what();
  EXPECT_EQ(backtrace->invocations, 1);
  const char* w2 = ex.what();
  EXPECT_EQ(backtrace->invocations, 1);
  // what() should not be recomputed.
  EXPECT_EQ(w1, w2);

  ex.add_context("");
  ex.what();
  EXPECT_EQ(backtrace->invocations, 1);
}

} // namespace c10_test

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `TEST`, `c10_test`

**Classes/Structs**: `EnforceEqWithCaller`, `Noncopyable`, `CountingLazyString`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `algorithm`
- `optional`
- `c10/util/ArrayRef.h`
- `c10/util/Logging.h`
- `gtest/gtest.h`


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
python c10/test/util/logging_test.cpp
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
- [`string_view_test.cpp_docs.md`](./string_view_test.cpp_docs.md)
- [`Enumerate_test.cpp_docs.md`](./Enumerate_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `logging_test.cpp_docs.md`
- **Keyword Index**: `logging_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

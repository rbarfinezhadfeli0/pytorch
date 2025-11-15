# Documentation: `c10/test/util/ThreadLocal_test.cpp`

## File Metadata

- **Path**: `c10/test/util/ThreadLocal_test.cpp`
- **Size**: 4,543 bytes (4.44 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/ThreadLocal.h>
#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace {

TEST(ThreadLocal, TestNoOpScopeWithOneVar) {
  C10_DEFINE_TLS_static(std::string, str);
}

TEST(ThreadLocalTest, TestNoOpScopeWithTwoVars) {
  C10_DEFINE_TLS_static(std::string, str);
  C10_DEFINE_TLS_static(std::string, str2);
}

TEST(ThreadLocalTest, TestScopeWithOneVar) {
  C10_DEFINE_TLS_static(std::string, str);
  EXPECT_EQ(*str, std::string());
  EXPECT_EQ(*str, "");

  *str = "abc";
  EXPECT_EQ(*str, "abc");
  EXPECT_EQ(str->length(), 3);
  EXPECT_EQ(str.get(), "abc");
}

TEST(ThreadLocalTest, TestScopeWithTwoVars) {
  C10_DEFINE_TLS_static(std::string, str);
  EXPECT_EQ(*str, "");

  C10_DEFINE_TLS_static(std::string, str2);

  *str = "abc";
  EXPECT_EQ(*str, "abc");
  EXPECT_EQ(*str2, "");

  *str2 = *str;
  EXPECT_EQ(*str, "abc");
  EXPECT_EQ(*str2, "abc");

  str->clear();
  EXPECT_EQ(*str, "");
  EXPECT_EQ(*str2, "abc");
}

TEST(ThreadLocalTest, TestInnerScopeWithTwoVars) {
  C10_DEFINE_TLS_static(std::string, str);
  *str = "abc";

  {
    C10_DEFINE_TLS_static(std::string, str2);
    EXPECT_EQ(*str2, "");

    *str2 = *str;
    EXPECT_EQ(*str, "abc");
    EXPECT_EQ(*str2, "abc");

    str->clear();
    EXPECT_EQ(*str2, "abc");
  }

  EXPECT_EQ(*str, "");
}

struct Foo {
  C10_DECLARE_TLS_class_static(Foo, std::string, str_);
};

C10_DEFINE_TLS_class_static(Foo, std::string, str_);

TEST(ThreadLocalTest, TestClassScope) {
  EXPECT_EQ(*Foo::str_, "");

  *Foo::str_ = "abc";
  EXPECT_EQ(*Foo::str_, "abc");
  EXPECT_EQ(Foo::str_->length(), 3);
  EXPECT_EQ(Foo::str_.get(), "abc");
}

C10_DEFINE_TLS_static(std::string, global_);
C10_DEFINE_TLS_static(std::string, global2_);
TEST(ThreadLocalTest, TestTwoGlobalScopeVars) {
  EXPECT_EQ(*global_, "");
  EXPECT_EQ(*global2_, "");

  *global_ = "abc";
  EXPECT_EQ(global_->length(), 3);
  EXPECT_EQ(*global_, "abc");
  EXPECT_EQ(*global2_, "");

  *global2_ = *global_;
  EXPECT_EQ(*global_, "abc");
  EXPECT_EQ(*global2_, "abc");

  global_->clear();
  EXPECT_EQ(*global_, "");
  EXPECT_EQ(*global2_, "abc");
  EXPECT_EQ(global2_.get(), "abc");
}

C10_DEFINE_TLS_static(std::string, global3_);
TEST(ThreadLocalTest, TestGlobalWithLocalScopeVars) {
  *global3_ = "abc";

  C10_DEFINE_TLS_static(std::string, str);

  std::swap(*global3_, *str);
  EXPECT_EQ(*str, "abc");
  EXPECT_EQ(*global3_, "");
}

TEST(ThreadLocalTest, TestThreadWithLocalScopeVar) {
  C10_DEFINE_TLS_static(std::string, str);
  *str = "abc";

  std::atomic_bool b(false);
  std::thread t([&b]() {
    EXPECT_EQ(*str, "");
    *str = "def";
    b = true;
    EXPECT_EQ(*str, "def");
  });
  t.join();

  EXPECT_TRUE(b);
  EXPECT_EQ(*str, "abc");
}

C10_DEFINE_TLS_static(std::string, global4_);
TEST(ThreadLocalTest, TestThreadWithGlobalScopeVar) {
  *global4_ = "abc";

  std::atomic_bool b(false);
  std::thread t([&b]() {
    EXPECT_EQ(*global4_, "");
    *global4_ = "def";
    b = true;
    EXPECT_EQ(*global4_, "def");
  });
  t.join();

  EXPECT_TRUE(b);
  EXPECT_EQ(*global4_, "abc");
}

TEST(ThreadLocalTest, TestObjectsAreReleased) {
  static std::atomic<int> ctors{0};
  static std::atomic<int> dtors{0};
  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
  struct A {
    A() {
      ++ctors;
    }

    ~A() {
      ++dtors;
    }

    A(const A&) = delete;
    A& operator=(const A&) = delete;

    int i{};
  };

  C10_DEFINE_TLS_static(A, a);

  std::atomic_bool b(false);
  std::thread t([&b]() {
    EXPECT_EQ(a->i, 0);
    a->i = 1;
    EXPECT_EQ(a->i, 1);
    b = true;
  });
  t.join();

  EXPECT_TRUE(b);

  EXPECT_EQ(ctors, 1);
  EXPECT_EQ(dtors, 1);
}

TEST(ThreadLocalTest, TestObjectsAreReleasedByNonstaticThreadLocal) {
  static std::atomic<int> ctors(0);
  static std::atomic<int> dtors(0);
  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
  struct A {
    A() {
      ++ctors;
    }

    ~A() {
      ++dtors;
    }

    A(const A&) = delete;
    A& operator=(const A&) = delete;

    int i{};
  };

  std::atomic_bool b(false);
  std::thread t([&b]() {
#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
    ::c10::ThreadLocal<A> a;
#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
    ::c10::ThreadLocal<A> a([]() {
      static thread_local A var;
      return &var;
    });
#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

    EXPECT_EQ(a->i, 0);
    a->i = 1;
    EXPECT_EQ(a->i, 1);
    b = true;
  });
  t.join();

  EXPECT_TRUE(b);

  EXPECT_EQ(ctors, 1);
  EXPECT_EQ(dtors, 1);
}

} // namespace

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `Foo`, `A`, `A`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/ThreadLocal.h`
- `gtest/gtest.h`
- `atomic`
- `thread`


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
python c10/test/util/ThreadLocal_test.cpp
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

- **File Documentation**: `ThreadLocal_test.cpp_docs.md`
- **Keyword Index**: `ThreadLocal_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

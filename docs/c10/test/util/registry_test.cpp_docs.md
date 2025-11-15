# Documentation: `c10/test/util/registry_test.cpp`

## File Metadata

- **Path**: `c10/test/util/registry_test.cpp`
- **Size**: 2,691 bytes (2.63 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <c10/util/Registry.h>

// Note: we use a different namespace to test if the macros defined in
// Registry.h actually works with a different namespace from c10.
namespace c10_test {

class Foo {
 public:
  explicit Foo(int x) {
    // LOG(INFO) << "Foo " << x;
  }
  virtual ~Foo() = default;
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DECLARE_REGISTRY(FooRegistry, Foo, int);
C10_DEFINE_REGISTRY(FooRegistry, Foo, int);
#define REGISTER_FOO(clsname) C10_REGISTER_CLASS(FooRegistry, clsname, clsname)

class Bar : public Foo {
 public:
  explicit Bar(int x) : Foo(x) {
    // LOG(INFO) << "Bar " << x;
  }
};
REGISTER_FOO(Bar);

class AnotherBar : public Foo {
 public:
  explicit AnotherBar(int x) : Foo(x) {
    // LOG(INFO) << "AnotherBar " << x;
  }
};
REGISTER_FOO(AnotherBar);

TEST(RegistryTest, CanRunCreator) {
  std::unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1));
  EXPECT_TRUE(bar != nullptr) << "Cannot create bar.";
  std::unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
  EXPECT_TRUE(another_bar != nullptr);
}

TEST(RegistryTest, ReturnNullOnNonExistingCreator) {
  EXPECT_EQ(FooRegistry()->Create("Non-existing bar", 1), nullptr);
}

// C10_REGISTER_CLASS_WITH_PRIORITY defines static variable
static void RegisterFooDefault() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_DEFAULT, Foo);
}

static void RegisterFooDefaultAgain() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_DEFAULT, Foo);
}

static void RegisterFooBarFallback() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_FALLBACK, Bar);
}

static void RegisterFooBarPreferred() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_PREFERRED, Bar);
}

TEST(RegistryTest, RegistryPriorities) {
  FooRegistry()->SetTerminate(false);
  RegisterFooDefault();

  // throws because Foo is already registered with default priority
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(RegisterFooDefaultAgain(), std::runtime_error);

#ifdef __GXX_RTTI
  // not going to register Bar because Foo is registered with Default priority
  RegisterFooBarFallback();
  std::unique_ptr<Foo> bar1(FooRegistry()->Create("FooWithPriority", 1));
  EXPECT_EQ(dynamic_cast<Bar*>(bar1.get()), nullptr);

  // will register Bar because of higher priority
  RegisterFooBarPreferred();
  std::unique_ptr<Foo> bar2(FooRegistry()->Create("FooWithPriority", 1));
  EXPECT_NE(dynamic_cast<Bar*>(bar2.get()), nullptr);
#endif
}

} // namespace c10_test

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10_test`, `to`, `from`

**Classes/Structs**: `Foo`, `Bar`, `AnotherBar`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `iostream`
- `memory`
- `c10/util/Registry.h`


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
python c10/test/util/registry_test.cpp
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

- **File Documentation**: `registry_test.cpp_docs.md`
- **Keyword Index**: `registry_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

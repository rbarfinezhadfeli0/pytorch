# Documentation: `c10/test/util/TypeIndex_test.cpp`

## File Metadata

- **Path**: `c10/test/util/TypeIndex_test.cpp`
- **Size**: 6,585 bytes (6.43 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeIndex.h>
#include <gtest/gtest.h>

using c10::util::get_fully_qualified_type_name;
using c10::util::get_type_index;
using std::string_view;

// NOLINTBEGIN(modernize-unary-static-assert)
namespace {

static_assert(get_type_index<int>() == get_type_index<int>(), "");
static_assert(get_type_index<float>() == get_type_index<float>(), "");
static_assert(get_type_index<int>() != get_type_index<float>(), "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int(double, double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() != get_type_index<int(double)>(),
    "");
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int (*)(double, double)>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() ==
        get_type_index<std::function<int(double, double)>>(),
    "");
static_assert(
    get_type_index<std::function<int(double, double)>>() !=
        get_type_index<std::function<int(double)>>(),
    "");

static_assert(get_type_index<int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() == get_type_index<int&&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int&>(), "");
static_assert(get_type_index<int>() == get_type_index<const int>(), "");
static_assert(get_type_index<const int>() == get_type_index<int&>(), "");
static_assert(get_type_index<int>() != get_type_index<int*>(), "");
static_assert(get_type_index<int*>() != get_type_index<int**>(), "");
static_assert(
    get_type_index<int(double&, double)>() !=
        get_type_index<int(double, double)>(),
    "");

struct Dummy final {};
struct Functor final {
  int64_t operator()(uint32_t, Dummy&&, const Dummy&) const;
};
static_assert(
    get_type_index<int64_t(uint32_t, Dummy&&, const Dummy&)>() ==
        get_type_index<
            c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");

namespace test_top_level_name {

static_assert(
    string_view::npos != get_fully_qualified_type_name<Dummy>().find("Dummy"),
    "");

TEST(TypeIndex, TopLevelName) {
  EXPECT_NE(
      string_view::npos, get_fully_qualified_type_name<Dummy>().find("Dummy"));
}
} // namespace test_top_level_name

namespace test_nested_name {
struct Dummy final {};

static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy"),
    "");

TEST(TypeIndex, NestedName) {
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy"));
}
} // namespace test_nested_name

namespace test_type_template_parameter {
template <class T>
struct Outer final {};
struct Inner final {};

static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Outer"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Inner"),
    "");

TEST(TypeIndex, TypeTemplateParameter) {
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Outer<Inner>>().find(
          "test_type_template_parameter::Outer"));
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Outer<Inner>>().find(
          "test_type_template_parameter::Inner"));
}
} // namespace test_type_template_parameter

namespace test_nontype_template_parameter {
template <size_t N>
struct Class final {};

static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Class<38474355>>().find("38474355"),
    "");

TEST(TypeIndex, NonTypeTemplateParameter) {
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Class<38474355>>().find("38474355"));
}
} // namespace test_nontype_template_parameter

namespace test_type_computations_are_resolved {
template <class T>
struct Type final {
  using type = const T*;
};

static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("int"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find('*'),
    "");

// but with remove_pointer applied, there is no '*' in the type name anymore
static_assert(
    string_view::npos ==
        get_fully_qualified_type_name<
            std::remove_pointer_t<typename Type<int>::type>>()
            .find('*'),
    "");

TEST(TypeIndex, TypeComputationsAreResolved) {
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<typename Type<int>::type>().find("int"));
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<typename Type<int>::type>().find('*'));
  // but with remove_pointer applied, there is no '*' in the type name anymore
  EXPECT_EQ(
      string_view::npos,
      get_fully_qualified_type_name<
          std::remove_pointer_t<typename Type<int>::type>>()
          .find('*'));
}

struct Functor final {
  std::string operator()(int64_t a, const Type<int>& b) const;
};

static_assert(
    // NOLINTNEXTLINE(misc-redundant-expression)
    get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>() ==
        get_fully_qualified_type_name<
            typename c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");

TEST(TypeIndex, FunctionTypeComputationsAreResolved) {
  EXPECT_EQ(
      get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>(),
      get_fully_qualified_type_name<
          typename c10::guts::infer_function_traits_t<Functor>::func_type>());
}
} // namespace test_type_computations_are_resolved

namespace test_function_arguments_and_returns {
class Dummy final {};

static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy(int)>().find(
            "test_function_arguments_and_returns::Dummy"),
    "");
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<void(Dummy)>().find(
            "test_function_arguments_and_returns::Dummy"),
    "");

TEST(TypeIndex, FunctionArgumentsAndReturns) {
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Dummy(int)>().find(
          "test_function_arguments_and_returns::Dummy"));
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<void(Dummy)>().find(
          "test_function_arguments_and_returns::Dummy"));
}
} // namespace test_function_arguments_and_returns
} // namespace
// NOLINTEND(modernize-unary-static-assert)

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `test_type_computations_are_resolved`, `test_type_template_parameter`, `test_nontype_template_parameter`, `test_function_arguments_and_returns`, `test_top_level_name`, `test_nested_name`

**Classes/Structs**: `Dummy`, `Functor`, `Dummy`, `T`, `Outer`, `Inner`, `Class`, `T`, `Type`, `Functor`, `Dummy`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Metaprogramming.h`
- `c10/util/TypeIndex.h`
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
python c10/test/util/TypeIndex_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/util`):

- [`bfloat16_test.cpp_docs.md`](./bfloat16_test.cpp_docs.md)
- [`complex_test_common.h_docs.md`](./complex_test_common.h_docs.md)
- [`generic_math_test.cpp_docs.md`](./generic_math_test.cpp_docs.md)
- [`Half_test.cpp_docs.md`](./Half_test.cpp_docs.md)
- [`nofatal_test.cpp_docs.md`](./nofatal_test.cpp_docs.md)
- [`small_vector_test.cpp_docs.md`](./small_vector_test.cpp_docs.md)
- [`exception_test.cpp_docs.md`](./exception_test.cpp_docs.md)
- [`string_view_test.cpp_docs.md`](./string_view_test.cpp_docs.md)
- [`Enumerate_test.cpp_docs.md`](./Enumerate_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `TypeIndex_test.cpp_docs.md`
- **Keyword Index**: `TypeIndex_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

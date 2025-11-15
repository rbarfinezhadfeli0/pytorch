# Documentation: `docs/test/cpp/aoti_abi_check/test_metaprogramming.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/aoti_abi_check/test_metaprogramming.cpp_docs.md`
- **Size**: 12,323 bytes (12.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/aoti_abi_check/test_metaprogramming.cpp`

## File Metadata

- **Path**: `test/cpp/aoti_abi_check/test_metaprogramming.cpp`
- **Size**: 9,593 bytes (9.37 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <torch/headeronly/util/Metaprogramming.h>
#include <cstdlib>

using namespace torch::headeronly::guts;

// NOLINTBEGIN(modernize*, cppcoreguidelines-special-member-functions)
namespace {

namespace test_function_traits {
static_assert(
    std::is_same<
        void,
        typename function_traits<void(int, float)>::return_type>::value,
    "");
static_assert(
    std::is_same<int, typename function_traits<int(int, float)>::return_type>::
        value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename function_traits<void(int, float)>::parameter_types>::value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename function_traits<int(int, float)>::parameter_types>::value,
    "");

static_assert(
    std::is_same<
        bool,
        typename make_function_traits_t<bool, typelist::typelist<int, float>>::
            return_type>::value,
    "");
static_assert(
    std::is_same<
        void,
        typename make_function_traits_t<void, typelist::typelist<int, float>>::
            return_type>::value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename make_function_traits_t<bool, typelist::typelist<int, float>>::
            parameter_types>::value,
    "");
static_assert(
    std::is_same<
        typelist::typelist<int, float>,
        typename make_function_traits_t<void, typelist::typelist<int, float>>::
            parameter_types>::value,
    "");
static_assert(
    std::is_same<
        bool(int, float),
        typename make_function_traits_t<bool, typelist::typelist<int, float>>::
            func_type>::value,
    "");
static_assert(
    std::is_same<
        void(int, float),
        typename make_function_traits_t<void, typelist::typelist<int, float>>::
            func_type>::value,
    "");

struct Functor final {
  std::string operator()(int64_t a, float b) const;
};
static_assert(
    std::is_same<
        std::string(int64_t, float),
        typename infer_function_traits_t<Functor>::func_type>::value,
    "");
} // namespace test_function_traits

struct MovableOnly {
  constexpr MovableOnly(int val_) : val(val_) { /* no default constructor */ }
  MovableOnly(const MovableOnly&) = delete;
  MovableOnly(MovableOnly&&) = default;
  MovableOnly& operator=(const MovableOnly&) = delete;
  MovableOnly& operator=(MovableOnly&&) = default;

  friend bool operator==(const MovableOnly& lhs, const MovableOnly& rhs) {
    return lhs.val == rhs.val;
  }

 private:
  int val;
};

template <class T>
using is_my_movable_only_class =
    std::is_same<MovableOnly, std::remove_cv_t<std::remove_reference_t<T>>>;

struct CopyCounting {
  int move_count{0};
  int copy_count{0};

  CopyCounting() {}
  CopyCounting(const CopyCounting& rhs)
      : move_count(rhs.move_count), copy_count(rhs.copy_count + 1) {}
  CopyCounting(CopyCounting&& rhs) noexcept
      : move_count(rhs.move_count + 1), copy_count(rhs.copy_count) {}
  CopyCounting& operator=(const CopyCounting& rhs) {
    move_count = rhs.move_count;
    copy_count = rhs.copy_count + 1;
    return *this;
  }
  CopyCounting& operator=(CopyCounting&& rhs) noexcept {
    move_count = rhs.move_count + 1;
    copy_count = rhs.copy_count;
    return *this;
  }
};

template <class T>
using is_my_copy_counting_class =
    std::is_same<CopyCounting, std::remove_cv_t<std::remove_reference_t<T>>>;

namespace test_tuple_elements {
// note: not testing empty selection, as some compilers will raise
// "parameter set but not used" in tuple_elements(). a good example
// of the friction that comes with using these tools

TEST(MetaprogrammingTest, TupleElements_subsetSelection) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_elements(x, std::index_sequence<0, 2>());
  auto z = std::make_tuple(0, 2.0);
  EXPECT_EQ(y, z);
}

TEST(MetaprogrammingTest, TupleElements_reorderSelection) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_elements(x, std::index_sequence<0, 2, 1>());
  auto z = std::make_tuple(0, 2.0, "HEY");
  EXPECT_EQ(y, z);
}
} // namespace test_tuple_elements

namespace test_tuple_take {
// note: not testing empty prefix, see note on empty selection above.

TEST(MetaprogrammingTest, TupleTake_nonemptyPrefix) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), 2>(x);
  auto z = std::make_tuple(0, "HEY");
  EXPECT_EQ(y, z);
}

TEST(MetaprogrammingTest, TupleTake_fullPrefix) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), 3>(x);
  EXPECT_EQ(x, y);
}

TEST(MetaprogrammingTest, TupleTake_negative) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_take<decltype(x), -2>(x);
  auto z = std::make_tuple("HEY", 2.0);
  EXPECT_EQ(y, z);
}
} // namespace test_tuple_take

namespace test_tuple_slice {
TEST(MetaprogrammingTest, TupleSlice_middle) {
  auto x = std::make_tuple(0, "HEY", 2.0, false);
  auto y = tuple_slice<decltype(x), 1, 2>(x);
  auto z = std::make_tuple("HEY", 2.0);
  EXPECT_EQ(y, z);
}

TEST(MetaprogrammingTest, TupleSlice_full) {
  auto x = std::make_tuple(0, "HEY", 2.0);
  auto y = tuple_slice<decltype(x), 0, 3>(x);
  EXPECT_EQ(x, y);
}
} // namespace test_tuple_slice

namespace test_tuple_map {
TEST(MetaprogrammingTest, TupleMap_simple) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](int32_t a) -> int16_t { return static_cast<int16_t>(a + 1); });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapperTakesDifferentButConvertibleType) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](int64_t a) -> int16_t { return static_cast<int16_t>(a + 1); });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapperTakesConstRef) {
  auto result = tuple_map(
      std::tuple<int32_t, int32_t, int32_t>(3, 4, 5),
      [](const int32_t& a) -> int16_t { return static_cast<int16_t>(a + 1); });
  static_assert(
      std::is_same<std::tuple<int16_t, int16_t, int16_t>, decltype(result)>::
          value,
      "");
  EXPECT_EQ(4, std::get<0>(result));
  EXPECT_EQ(5, std::get<1>(result));
  EXPECT_EQ(6, std::get<2>(result));
}

TEST(MetaprogrammingTest, TupleMap_mapsToDifferentTypes) {
  struct Mapper {
    std::string operator()(int32_t a) const {
      return std::to_string(a);
    }
    int32_t operator()(const std::string& a) const {
      return atoi(a.c_str());
    }
  };
  auto result = tuple_map(std::tuple<int32_t, std::string>(3, "4"), Mapper());
  static_assert(
      std::is_same<std::tuple<std::string, int32_t>, decltype(result)>::value,
      "");
  EXPECT_EQ("3", std::get<0>(result));
  EXPECT_EQ(4, std::get<1>(result));
}

TEST(MetaprogrammingTest, TupleMap_differentiatesLRValueReferences) {
  struct Mapper {
    // NOLINTNEXTLINE(*move*)
    std::string operator()(std::string&& a) const {
      return "moved";
    }
    std::string operator()(const std::string& a) const {
      return "copied";
    }
  };
  std::string str1, str2;
  auto result = tuple_map(
      std::tuple<const std::string&, std::string&&>(str1, std::move(str2)),
      Mapper());
  static_assert(
      std::is_same<std::tuple<std::string, std::string>, decltype(result)>::
          value,
      "");
  EXPECT_EQ("copied", std::get<0>(result));
  EXPECT_EQ("moved", std::get<1>(result));
}

TEST(MetaprogrammingTest, TupleMap_canWorkWithMovableOnlyType) {
  auto result = tuple_map(
      std::tuple<MovableOnly>(MovableOnly(7)), [](MovableOnly a) { return a; });
  static_assert(
      std::is_same<std::tuple<MovableOnly>, decltype(result)>::value, "");
  EXPECT_EQ(MovableOnly(7), std::get<0>(result));
}

TEST(MetaprogrammingTest, TupleMap_doesntUnecessarilyCopyValues) {
  auto result = tuple_map(
      std::tuple<CopyCounting>(CopyCounting()),
      [](CopyCounting a) { return a; });
  static_assert(
      std::is_same<std::tuple<CopyCounting>, decltype(result)>::value, "");
  EXPECT_EQ(4, std::get<0>(result).move_count);
  EXPECT_EQ(0, std::get<0>(result).copy_count);
}

TEST(MetaprogrammingTest, TupleMap_doesntUnecessarilyMoveValues) {
  CopyCounting a;
  auto result = tuple_map(
      std::tuple<CopyCounting&&>(std::move(a)),
      [](CopyCounting&& a) -> CopyCounting&& { return std::move(a); });
  static_assert(
      std::is_same<std::tuple<CopyCounting&&>, decltype(result)>::value, "");
  EXPECT_EQ(&a, &std::get<0>(result));
  EXPECT_EQ(0, std::get<0>(result).move_count);
  EXPECT_EQ(0, std::get<0>(result).copy_count);
}

TEST(MetaprogrammingTest, TupleMap_canBeUsedWithAutoLambdas) {
  struct A final {
    int32_t func() {
      return 5;
    }
  };
  struct B final {
    std::string func() {
      return "5";
    }
  };
  auto result =
      tuple_map(std::make_tuple(A(), B()), [](auto a) { return a.func(); });
  static_assert(
      std::is_same<std::tuple<int32_t, std::string>, decltype(result)>::value,
      "");
  EXPECT_EQ(5, std::get<0>(result));
  EXPECT_EQ("5", std::get<1>(result));
}
} // namespace test_tuple_map

} // namespace
// NOLINTEND(modernize*, cppcoreguidelines-special-member-functions)

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `test_tuple_elements`, `test_tuple_map`, `test_tuple_take`, `test_function_traits`, `test_tuple_slice`

**Classes/Structs**: `Functor`, `MovableOnly`, `T`, `CopyCounting`, `T`, `Mapper`, `Mapper`, `A`, `B`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/aoti_abi_check`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/headeronly/util/Metaprogramming.h`
- `cstdlib`


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

This is a test file. Run it with:

```bash
python test/cpp/aoti_abi_check/test_metaprogramming.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/aoti_abi_check`):

- [`test_headeronlyarrayref.cpp_docs.md`](./test_headeronlyarrayref.cpp_docs.md)
- [`test_cast.cpp_docs.md`](./test_cast.cpp_docs.md)
- [`test_scalartype.cpp_docs.md`](./test_scalartype.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_typetraits.cpp_docs.md`](./test_typetraits.cpp_docs.md)
- [`test_dtype.cpp_docs.md`](./test_dtype.cpp_docs.md)
- [`test_math.cpp_docs.md`](./test_math.cpp_docs.md)
- [`test_dispatch.cpp_docs.md`](./test_dispatch.cpp_docs.md)
- [`test_exception.cpp_docs.md`](./test_exception.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_metaprogramming.cpp_docs.md`
- **Keyword Index**: `test_metaprogramming.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/aoti_abi_check`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/aoti_abi_check`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/aoti_abi_check/test_metaprogramming.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/aoti_abi_check`):

- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`test_math.cpp_docs.md_docs.md`](./test_math.cpp_docs.md_docs.md)
- [`test_dtype.cpp_kw.md_docs.md`](./test_dtype.cpp_kw.md_docs.md)
- [`test_typetraits.cpp_docs.md_docs.md`](./test_typetraits.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_exception.cpp_docs.md_docs.md`](./test_exception.cpp_docs.md_docs.md)
- [`test_headeronlyarrayref.cpp_kw.md_docs.md`](./test_headeronlyarrayref.cpp_kw.md_docs.md)
- [`test_rand.cpp_kw.md_docs.md`](./test_rand.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_metaprogramming.cpp_docs.md_docs.md`
- **Keyword Index**: `test_metaprogramming.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

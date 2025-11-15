# Documentation: Enumerate_test.cpp

## File Metadata
- **Path**: `c10/test/util/Enumerate_test.cpp`
- **Size**: 6314 bytes
- **Lines**: 269
- **Extension**: .cpp
- **Type**: Regular file

## Original Source

```cpp
/*
 * Ported from folly/container/test/EnumerateTest.cpp
 */

#include <c10/util/Enumerate.h>
#include <gtest/gtest.h>
#include <array>

namespace {

template <class T>
struct IsConstReference {
  constexpr static bool value = false;
};
template <class T>
struct IsConstReference<const T&> {
  constexpr static bool value = true;
};

constexpr int basicSum(const std::array<int, 3>& test) {
  int sum = 0;
  for (auto it : c10::enumerate(test)) {
    sum += *it;
  }
  return sum;
}

constexpr int cpp17StructuredBindingSum(const std::array<int, 3>& test) {
  int sum = 0;
  for (auto&& [_, integer] : c10::enumerate(test)) {
    sum += integer;
  }
  return sum;
}

} // namespace

TEST(Enumerate, Basic) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (auto it : c10::enumerate(v)) {
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());

    /* Test mutability. */
    std::string newValue = "x";
    *it = newValue;
    EXPECT_EQ(newValue, v[i]);

    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicRRef) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (auto&& it : c10::enumerate(v)) {
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());

    /* Test mutability. */
    std::string newValue = "x";
    *it = newValue;
    EXPECT_EQ(newValue, v[i]);

    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicConst) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (const auto it : c10::enumerate(v)) {
    static_assert(IsConstReference<decltype(*it)>::value, "Const enumeration");
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicConstRef) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (const auto& it : c10::enumerate(v)) {
    static_assert(IsConstReference<decltype(*it)>::value, "Const enumeration");
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicConstRRef) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (const auto&& it : c10::enumerate(v)) {
    static_assert(IsConstReference<decltype(*it)>::value, "Const enumeration");
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicVecBool) {
  std::vector<bool> v = {true, false, false, true};
  size_t i = 0;
  for (auto it : c10::enumerate(v)) {
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicVecBoolRRef) {
  std::vector<bool> v = {true, false, false, true};
  size_t i = 0;
  for (auto it : c10::enumerate(v)) {
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, Temporary) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (auto&& it : c10::enumerate(decltype(v)(v))) { // Copy v.
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, BasicConstArg) {
  const std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (auto&& it : c10::enumerate(v)) {
    static_assert(
        IsConstReference<decltype(*it)>::value, "Enumerating a const vector");
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, TemporaryConstEnumerate) {
  std::vector<std::string> v = {"abc", "a", "ab"};
  size_t i = 0;
  for (const auto&& it : c10::enumerate(decltype(v)(v))) { // Copy v.
    static_assert(IsConstReference<decltype(*it)>::value, "Const enumeration");
    EXPECT_EQ(it.index, i);
    EXPECT_EQ(*it, v[i]);
    EXPECT_EQ(it->size(), v[i].size());
    ++i;
  }

  EXPECT_EQ(i, v.size());
}

TEST(Enumerate, EmptyRange) {
  std::vector<std::string> v;
  for (auto&& it : c10::enumerate(v)) {
    (void)it; // Silence warnings.
    ADD_FAILURE();
  }
}

class CStringRange {
  const char* cstr;

 public:
  struct Sentinel {};

  explicit CStringRange(const char* cstr_) : cstr(cstr_) {}

  const char* begin() const {
    return cstr;
  }
  Sentinel end() const {
    return Sentinel{};
  }
};

static bool operator==(const char* c, CStringRange::Sentinel) {
  return *c == 0;
}

TEST(Enumerate, Cpp17Support) {
  std::array<char, 5> test = {"test"};
  for (const auto&& it : c10::enumerate(CStringRange{test.data()})) {
    ASSERT_LT(it.index, test.size());
    EXPECT_EQ(*it, test[it.index]);
  }
}

TEST(Enumerate, Cpp17StructuredBindingConstRef) {
  std::vector<std::string> test = {"abc", "a", "ab"};
  for (const auto& [index, str] : c10::enumerate(test)) {
    ASSERT_LT(index, test.size());
    EXPECT_EQ(str, test[index]);
  }
}

TEST(Enumerate, Cpp17StructuredBindingConstRRef) {
  std::vector<std::string> test = {"abc", "a", "ab"};
  for (const auto&& [index, str] : c10::enumerate(test)) {
    ASSERT_LT(index, test.size());
    EXPECT_EQ(str, test[index]);
  }
}

TEST(Enumerate, Cpp17StructuredBindingConstVector) {
  const std::vector<std::string> test = {"abc", "a", "ab"};
  for (auto&& [index, str] : c10::enumerate(test)) {
    static_assert(
        IsConstReference<decltype(str)>::value, "Enumerating const vector");
    ASSERT_LT(index, test.size());
    EXPECT_EQ(str, test[index]);
  }
}

TEST(Enumerate, Cpp17StructuredBindingModify) {
  std::vector<int> test = {1, 2, 3, 4, 5};
  for (auto&& [index, integer] : c10::enumerate(test)) {
    integer = 0;
  }

  for (const auto& integer : test) {
    EXPECT_EQ(integer, 0);
  }
}

TEST(Enumerate, BasicConstexpr) {
  constexpr std::array<int, 3> test = {1, 2, 3};
  static_assert(basicSum(test) == 6, "Basic enumerating is not constexpr");
  EXPECT_EQ(basicSum(test), 6);
}

TEST(Enumerate, Cpp17StructuredBindingConstexpr) {
  constexpr std::array<int, 3> test = {1, 2, 3};
  static_assert(
      cpp17StructuredBindingSum(test) == 6,
      "C++17 structured binding enumerating is not constexpr");
  EXPECT_EQ(cpp17StructuredBindingSum(test), 6);
}

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): T, T, CStringRange

### Structures
This file defines 3 struct(s): IsConstReference, IsConstReference, Sentinel


## Key Components

The file contains 684 words across 269 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 6314 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

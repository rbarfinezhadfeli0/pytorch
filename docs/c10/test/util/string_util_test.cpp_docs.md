# Documentation: `c10/test/util/string_util_test.cpp`

## File Metadata

- **Path**: `c10/test/util/string_util_test.cpp`
- **Size**: 4,905 bytes (4.79 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/StringUtil.h>

#include <gtest/gtest.h>

namespace {

namespace test_str_narrow_single {
TEST(StringUtilTest, testStrNarrowSingle) {
  std::string s = "narrow test string";
  EXPECT_EQ(s, c10::str(s));

  const char* c_str = s.c_str();
  EXPECT_EQ(s, c10::str(c_str));

  char c = 'a';
  EXPECT_EQ(std::string(1, c), c10::str(c));
}
} // namespace test_str_narrow_single

namespace test_str_wide_single {
TEST(StringUtilTest, testStrWideSingle) {
  std::wstring s = L"wide test string";
  std::string narrow = "wide test string";
  EXPECT_EQ(narrow, c10::str(s));

  const wchar_t* c_str = s.c_str();
  EXPECT_EQ(narrow, c10::str(c_str));

  wchar_t c = L'a';
  std::string narrowC = "a";
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_single

namespace test_str_wide_single_multibyte {
TEST(StringUtilTest, testStrWideSingleMultibyte) {
  std::wstring s = L"\u00EC blah";
  std::string narrow = "\xC3\xAC blah";
  EXPECT_EQ(narrow, c10::str(s));

  const wchar_t* c_str = s.c_str();
  EXPECT_EQ(narrow, c10::str(c_str));

  wchar_t c = L'\u00EC';
  std::string narrowC = "\xC3\xAC";
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_single_multibyte

namespace test_str_wide_empty {
TEST(StringUtilTest, testStrWideEmpty) {
  std::wstring s;
  std::string narrow;
  EXPECT_EQ(narrow, c10::str(s));

  const wchar_t* c_str = s.c_str();
  EXPECT_EQ(narrow, c10::str(c_str));

  wchar_t c = L'\0';
  std::string narrowC(1, '\0');
  EXPECT_EQ(narrowC, c10::str(c));
}
} // namespace test_str_wide_empty

namespace test_str_multi {
TEST(StringUtilTest, testStrMulti) {
  std::string result = c10::str(
      "c_str ",
      'c',
      std::string(" std::string "),
      42,
      L" wide c_str ",
      L'w',
      std::wstring(L" std::wstring "));
  std::string expected = "c_str c std::string 42 wide c_str w std::wstring ";
  EXPECT_EQ(expected, result);
}
} // namespace test_str_multi

namespace test_try_to {
TEST(tryToTest, Int64T) {
  const std::vector<std::pair<const char*, int64_t>> valid_examples = {
      {"123", 123},
      {"+456", 456},
      {"-123", -123},
      {"0x123", 291},
      {"00123", 83},
      {"000", 0},
  };
  for (const auto& [str, num] : valid_examples) {
    EXPECT_EQ(c10::tryToNumber<int64_t>(str), num);
    EXPECT_EQ(c10::tryToNumber<int64_t>(std::string{str}), num);
  }

  const std::vector<const char*> invalid_examples = {
      "123abc",
      "123.45",
      "",
      "12345678901234567890", // overflow
  };
  for (const auto str : invalid_examples) {
    EXPECT_FALSE(c10::tryToNumber<int64_t>(str).has_value());
    EXPECT_FALSE(c10::tryToNumber<int64_t>(std::string{str}).has_value());
  }
  EXPECT_FALSE(c10::tryToNumber<int64_t>(nullptr).has_value());
}

TEST(tryToTest, Double) {
  const std::vector<std::pair<const char*, double>> valid_examples = {
      {"123.45", 123.45},
      {"-123.45", -123.45},
      {"123", 123.},
      {".5", 0.5},
      {"-.02", -0.02},
      {"5e-2", 5e-2},
      {"1e+3", 1e3},
      {"0x123.45", 291.26953125},
  };
  for (const auto& [str, num] : valid_examples) {
    EXPECT_EQ(c10::tryToNumber<double>(str), num);
    EXPECT_EQ(c10::tryToNumber<double>(std::string{str}), num);
  }

  const std::vector<const char*> invalid_examples = {
      "123abc",
      "",
      "1e309", // overflow
  };
  for (const auto str : invalid_examples) {
    EXPECT_FALSE(c10::tryToNumber<double>(str).has_value());
    EXPECT_FALSE(c10::tryToNumber<double>(std::string{str}).has_value());
  }
  EXPECT_FALSE(c10::tryToNumber<double>(nullptr).has_value());
}
} // namespace test_try_to

namespace test_split {
TEST(SplitTest, NormalCase) {
  std::string str = "torch.ops.aten.linear";
  auto result = c10::split(str, '.');
  ASSERT_EQ(4, result.size());
  EXPECT_EQ("torch", result[0]);
  EXPECT_EQ("ops", result[1]);
  EXPECT_EQ("aten", result[2]);
  EXPECT_EQ("linear", result[3]);
}
TEST(SplitTest, EmptyString) {
  auto result = c10::split("", '.');
  EXPECT_TRUE(result.empty());
}
TEST(SplitTest, NoDelimiter) {
  std::string str = "single";
  auto result = c10::split(str, '.');
  ASSERT_EQ(1, result.size());
  EXPECT_EQ("single", result[0]);
}
TEST(SplitTest, ConsecutiveDelimiters) {
  std::string str = "atom1..atom2";
  auto result = c10::split(str, '.');
  ASSERT_EQ(3, result.size());
  EXPECT_EQ("atom1", result[0]);
  EXPECT_EQ("", result[1]);
  EXPECT_EQ("atom2", result[2]);
}
} // namespace test_split

namespace test_str_enum {
TEST(StringUtilTest, testStrEnum) {
  enum class Foo { Bar = 1, Baz = 2 };
  EXPECT_EQ(c10::str(Foo::Baz, Foo::Bar), "21");

  enum UnscopedEnum { Bar2 = 1, Baz2 = 2 };
  EXPECT_EQ(c10::str(Baz2, Bar2), "21");

  static_assert(c10::detail::Streamable<int>::value);
  static_assert(!c10::detail::Streamable<Foo>::value);

  static_assert(c10::detail::Streamable<UnscopedEnum>::value);
}
} // namespace test_str_enum
} // namespace

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `test_str_wide_single_multibyte`, `test_str_enum`, `test_str_wide_empty`, `test_str_wide_single`, `test_str_multi`, `test_try_to`, `test_split`, `test_str_narrow_single`

**Classes/Structs**: `Foo`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/StringUtil.h`
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
python c10/test/util/string_util_test.cpp
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

- **File Documentation**: `string_util_test.cpp_docs.md`
- **Keyword Index**: `string_util_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

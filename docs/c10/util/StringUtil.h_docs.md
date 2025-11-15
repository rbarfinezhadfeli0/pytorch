# Documentation: `c10/util/StringUtil.h`

## File Metadata

- **Path**: `c10/util/StringUtil.h`
- **Size**: 6,803 bytes (6.64 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#ifndef C10_UTIL_STRINGUTIL_H_
#define C10_UTIL_STRINGUTIL_H_

#include <c10/macros/Macros.h>
#include <c10/util/string_utils.h>

#include <cstddef>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wshorten-64-to-32")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wshorten-64-to-32")
#endif

namespace c10 {

namespace detail {

// Obtains the base name from a full path.
C10_API std::string StripBasename(const std::string& full_path);

C10_API std::string ExcludeFileExtension(const std::string& full_path);

struct CompileTimeEmptyString {
  operator const std::string&() const {
    static const std::string empty_string_literal;
    return empty_string_literal;
  }
  operator const char*() const {
    return "";
  }
};

template <typename T>
struct CanonicalizeStrTypes {
  using type = const T&;
};

template <size_t N>
// NOLINTNEXTLINE(*c-arrays*)
struct CanonicalizeStrTypes<char[N]> {
  using type = const char*;
};

inline std::ostream& _str(std::ostream& ss) {
  return ss;
}

template <class T, class = std::ostream&>
struct Streamable : std::false_type {};

template <class T>
struct Streamable<T, decltype(std::declval<std::ostream&>() << T{})>
    : std::true_type {};

template <typename T>
inline std::ostream& _str(std::ostream& ss, const T& t) {
  if constexpr (std::is_enum_v<T> && !Streamable<T>::value) {
    // NOLINTNEXTLINE(modernize-type-traits)
    return _str(ss, static_cast<typename std::underlying_type<T>::type>(t));
  } else {
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    ss << t;
    return ss;
  }
}

template <typename T>
inline std::ostream& _str(std::ostream& ss, const std::optional<T>& t) {
  if (t.has_value()) {
    return _str(ss, t.value());
  }
  ss << "std::nullopt";
  return ss;
}
// Overloads of _str for wide types; forces narrowing.
C10_API std::ostream& _str(std::ostream& ss, const wchar_t* wCStr);
C10_API std::ostream& _str(std::ostream& ss, const wchar_t& wChar);
C10_API std::ostream& _str(std::ostream& ss, const std::wstring& wString);

template <>
inline std::ostream& _str<CompileTimeEmptyString>(
    std::ostream& ss,
    const CompileTimeEmptyString& /*unused*/) {
  return ss;
}

template <typename T, typename... Args>
inline std::ostream& _str(std::ostream& ss, const T& t, const Args&... args) {
  return _str(_str(ss, t), args...);
}

template <typename... Args>
struct _str_wrapper final {
  static std::string call(const Args&... args) {
    std::ostringstream ss;
    _str(ss, args...);
    return ss.str();
  }
};

// Specializations for already-a-string types.
template <>
struct _str_wrapper<std::string> final {
  // return by reference to avoid the binary size of a string copy
  static const std::string& call(const std::string& str) {
    return str;
  }
};

template <>
struct _str_wrapper<const char*> final {
  static const char* call(const char* str) {
    return str;
  }
};

// For c10::str() with an empty argument list (which is common in our assert
// macros), we don't want to pay the binary size for constructing and
// destructing a stringstream or even constructing a string.
template <>
struct _str_wrapper<> final {
  static CompileTimeEmptyString call() {
    return CompileTimeEmptyString();
  }
};

} // namespace detail

// Convert a list of string-like arguments into a single string.
template <typename... Args>
inline auto str(const Args&... args) {
  return detail::_str_wrapper<
      typename detail::CanonicalizeStrTypes<Args>::type...>::call(args...);
}

template <class Container>
inline std::string Join(const std::string& delimiter, const Container& v) {
  std::stringstream s;
  int cnt = static_cast<int64_t>(v.size()) - 1;
  for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
    s << (*i) << (cnt ? delimiter : "");
  }
  return std::move(s).str();
}

// Replace all occurrences of "from" substring to "to" string.
// Returns number of replacements
size_t C10_API
ReplaceAll(std::string& s, std::string_view from, std::string_view to);

/// Represents a location in source code (for debugging).
struct C10_API SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

// unix isprint but insensitive to locale
inline bool isPrint(char s) {
  return s > 0x1f && s < 0x7f;
}

inline void printQuotedString(std::ostream& stmt, const std::string_view str) {
  stmt << "\"";
  for (auto s : str) {
    switch (s) {
      case '\\':
        stmt << "\\\\";
        break;
      case '\'':
        stmt << "\\'";
        break;
      case '\"':
        stmt << "\\\"";
        break;
      case '\a':
        stmt << "\\a";
        break;
      case '\b':
        stmt << "\\b";
        break;
      case '\f':
        stmt << "\\f";
        break;
      case '\n':
        stmt << "\\n";
        break;
      case '\r':
        stmt << "\\r";
        break;
      case '\t':
        stmt << "\\t";
        break;
      case '\v':
        stmt << "\\v";
        break;
      default:
        if (isPrint(s)) {
          stmt << s;
        } else {
          // C++ io has stateful formatting settings. Messing with
          // them is probably worse than doing this manually.
          // NOLINTNEXTLINE(*c-arrays*)
          char buf[4] = "000";
          // NOLINTNEXTLINE(*narrowing-conversions)
          buf[2] += s % 8;
          s /= 8;
          // NOLINTNEXTLINE(*narrowing-conversions)
          buf[1] += s % 8;
          s /= 8;
          // NOLINTNEXTLINE(*narrowing-conversions)
          buf[0] += s;
          stmt << "\\" << buf;
        }
        break;
    }
  }
  stmt << "\"";
}

template <typename T>
std::optional<T> tryToNumber(const char* symbol) = delete;
template <typename T>
std::optional<T> tryToNumber(const std::string& symbol) = delete;

/*
 * Convert a string to a 64 bit integer. Trailing whitespaces are not supported.
 * Similarly, integer string with trailing characters like "123abc" will be
 * rejected.
 */
template <>
C10_API std::optional<int64_t> tryToNumber<int64_t>(const char* symbol);
template <>
C10_API std::optional<int64_t> tryToNumber<int64_t>(const std::string& symbol);

/*
 * Convert a string to a double. Trailing whitespaces are not supported.
 * Similarly, integer string with trailing characters like "123abc" will
 * be rejected.
 */
template <>
C10_API std::optional<double> tryToNumber<double>(const char* symbol);
template <>
C10_API std::optional<double> tryToNumber<double>(const std::string& symbol);

C10_API std::vector<std::string_view> split(
    std::string_view target,
    char delimiter);
} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

#endif // C10_UTIL_STRINGUTIL_H_

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `CompileTimeEmptyString`, `CanonicalizeStrTypes`, `CanonicalizeStrTypes`, `T`, `Streamable`, `T`, `Streamable`, `_str_wrapper`, `_str_wrapper`, `_str_wrapper`, `_str_wrapper`, `Container`, `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `c10/util/string_utils.h`
- `cstddef`
- `optional`
- `ostream`
- `sstream`
- `string`
- `string_view`
- `type_traits`
- `vector`


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

- **File Documentation**: `StringUtil.h_docs.md`
- **Keyword Index**: `StringUtil.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

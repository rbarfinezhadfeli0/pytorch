# Documentation: `docs/torch/headeronly/util/Metaprogramming.h_docs.md`

## File Metadata

- **Path**: `docs/torch/headeronly/util/Metaprogramming.h_docs.md`
- **Size**: 10,224 bytes (9.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/headeronly/util/Metaprogramming.h`

## File Metadata

- **Path**: `torch/headeronly/util/Metaprogramming.h`
- **Size**: 7,444 bytes (7.27 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/TypeList.h>
#include <type_traits>

namespace c10::guts {

/**
 * Access information about result type or arguments from a function type.
 * Example:
 * using A = function_traits<int (float, double)>::return_type // A == int
 * using A = function_traits<int (float, double)>::parameter_types::tuple_type
 * // A == tuple<float, double>
 */
template <class Func>
struct function_traits {
  static_assert(
      !std::is_same_v<Func, Func>,
      "In function_traits<Func>, Func must be a plain function type.");
};
template <class Result, class... Args>
struct function_traits<Result(Args...)> {
  using func_type = Result(Args...);
  using return_type = Result;
  using parameter_types = typelist::typelist<Args...>;
  static constexpr auto number_of_parameters = sizeof...(Args);
};

/**
 * infer_function_traits: creates a `function_traits` type for a simple
 * function (pointer) or functor (lambda/struct). Currently does not support
 * class methods.
 */

template <typename Functor>
struct infer_function_traits {
  using type = function_traits<
      c10::guts::detail::strip_class_t<decltype(&Functor::operator())>>;
};

template <typename Result, typename... Args>
struct infer_function_traits<Result (*)(Args...)> {
  using type = function_traits<Result(Args...)>;
};

template <typename Result, typename... Args>
struct infer_function_traits<Result(Args...)> {
  using type = function_traits<Result(Args...)>;
};

template <typename T>
using infer_function_traits_t = typename infer_function_traits<T>::type;

/**
 * make_function_traits: creates a `function_traits` type given a Return type
 * and a typelist of Argument types
 *
 * Example:
 * bool f(int, int);
 *
 * infer_function_traits_t<f> == make_function_traits_t<bool,
 * typelist::typelist<int, int>>
 */
template <typename Result, typename ArgList>
struct make_function_traits {
  static_assert(
      false_t<ArgList>::value,
      "In guts::make_function_traits<Result, TypeList>, the ArgList argument must be typelist<...>.");
};

template <typename Result, typename... Args>
struct make_function_traits<Result, typelist::typelist<Args...>> {
  using type = function_traits<Result(Args...)>;
};

template <typename Result, typename ArgList>
using make_function_traits_t =
    typename make_function_traits<Result, ArgList>::type;

/**
 * make_offset_index_sequence<Start, N>
 * Like make_index_sequence<N>, but starting from Start instead of 0.
 *
 * Example:
 *  make_offset_index_sequence<10, 3> == std::index_sequence<10, 11, 12>
 */
template <size_t Start, size_t N, size_t... Is>
struct make_offset_index_sequence_impl
    : make_offset_index_sequence_impl<Start, N - 1, Start + N - 1, Is...> {
  static_assert(
      static_cast<int>(Start) >= 0,
      "make_offset_index_sequence: Start < 0");
  static_assert(static_cast<int>(N) >= 0, "make_offset_index_sequence: N < 0");
};

template <size_t Start, size_t... Is>
struct make_offset_index_sequence_impl<Start, 0, Is...> {
  typedef std::index_sequence<Is...> type;
};

template <size_t Start, size_t N>
using make_offset_index_sequence =
    typename make_offset_index_sequence_impl<Start, N>::type;

/**
 * Use tuple_elements to extract a position-indexed subset of elements
 * from the argument tuple into a result tuple.
 *
 * Example:
 *  std::tuple<int, const char*, double> t = std::make_tuple(0, "HEY", 2.0);
 *  std::tuple<int, double> result = tuple_elements(t, std::index_sequence<0,
 * 2>());
 */
template <class Tuple, size_t... Is>
constexpr auto tuple_elements(Tuple t, std::index_sequence<Is...> /*unused*/) {
  return std::tuple<std::tuple_element_t<Is, Tuple>...>(std::get<Is>(t)...);
}

/**
 * Use tuple_take to extract the first or last n elements from the argument
 * tuple into a result tuple.
 *
 * Example:
 *  std::tuple<int, const char*, double> t = std::make_tuple(0, "HEY", 2.0);
 *  std::tuple<int, const char*> first_two = tuple_take<decltype(t), 2>(t);
 *  std::tuple<const char*, double> last_two = tuple_take<decltype(t), -2>(t);
 */
template <class Tuple, int N, class Enable = void>
struct TupleTake {};

template <class Tuple, int N>
struct TupleTake<Tuple, N, std::enable_if_t<N >= 0, void>> {
  static auto call(Tuple t) {
    constexpr size_t size = std::tuple_size<Tuple>();
    static_assert(N <= size, "tuple_take: N > size");
    return tuple_elements(t, std::make_index_sequence<N>{});
  }
};

template <class Tuple, int N>
    struct TupleTake < Tuple,
    N, std::enable_if_t<N<0, void>> {
  static auto call(Tuple t) {
    constexpr size_t size = std::tuple_size<Tuple>();
    static_assert(-N <= size, "tuple_take: -N > size");
    return tuple_elements(t, make_offset_index_sequence<size + N, -N>{});
  }
};

template <class Tuple, int N>
auto tuple_take(Tuple t) {
  return TupleTake<Tuple, N>::call(t);
}

/**
 * Use tuple_slice to extract a contiguous subtuple from the argument.
 *
 * Example:
 *  std::tuple<int, const char*, double, bool> t = std::make_tuple(0,
 * "HEY", 2.0, false); std::tuple<int, const char*> middle_two =
 * tuple_slice<decltype(t), 1, 2>(t);
 */
template <class Tuple, size_t Start, size_t N>
constexpr auto tuple_slice(Tuple t) {
  constexpr size_t size = std::tuple_size<Tuple>();
  static_assert(Start + N <= size, "tuple_slice: Start + N > size");
  return tuple_elements(t, make_offset_index_sequence<Start, N>{});
}

/**
 * Use tuple_map to run a mapping function over a tuple to get a new tuple.
 *
 * Example 1:
 *   auto result = tuple_map(std::tuple<int32_t, int32_t, int32_t>(3, 4, 5), []
 * (int32_t a) -> int16_t {return a+1;});
 *   // result == std::tuple<int16_t, int16_t, int16_t>(4, 5, 6)
 *
 * Example 2:
 *   struct Mapper {
 *     std::string operator()(int32_t a) const {
 *       return std::to_string(a);
 *     }
 *     int64_t operator()(const std::string& a) const {
 *        return atoi(a.c_str());
 *     }
 *   };
 *   auto result = tuple_map(std::tuple<int32_t, std::string>(3, "4"),
 * Mapper());
 *   // result == std::tuple<std::string, int64_t>("3", 4)
 *
 * Example 3:
 *   struct A final {
 *    int32_t func() {
 *      return 5;
 *    }
 *  };
 *  struct B final {
 *    std::string func() {
 *      return "5";
 *    }
 *  };
 *  auto result = tuple_map(std::make_tuple(A(), B()), [] (auto a) { return
 * a.func(); });
 *  // result == std::tuple<int32_t, std::string>(5, "5");
 */
namespace detail {
template <class Mapper, class... Args, size_t... Indices>
auto tuple_map(
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    std::tuple<Args...>&& tuple,
    const Mapper& mapper,
    std::index_sequence<Indices...> /*unused*/) {
  return std::tuple<decltype(mapper(std::forward<Args>(std::get<Indices>(
      tuple))))...>(mapper(std::forward<Args>(std::get<Indices>(tuple)))...);
}
} // namespace detail

template <class Mapper, class... Args>
auto tuple_map(std::tuple<Args...>&& tuple, const Mapper& mapper) {
  return detail::tuple_map(
      std::move(tuple), mapper, std::index_sequence_for<Args...>());
}

} // namespace c10::guts

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, guts)

using c10::guts::function_traits;
using c10::guts::infer_function_traits_t;
using c10::guts::make_function_traits_t;
using c10::guts::tuple_elements;
using c10::guts::tuple_map;
using c10::guts::tuple_slice;
using c10::guts::tuple_take;

HIDDEN_NAMESPACE_END(torch, headeronly, guts)

```



## High-Level Overview


This C++ file contains approximately 12 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `Func`, `function_traits`, `Result`, `function_traits`, `methods`, `infer_function_traits`, `infer_function_traits`, `infer_function_traits`, `make_function_traits`, `make_function_traits`, `make_offset_index_sequence_impl`, `make_offset_index_sequence_impl`, `Tuple`, `Tuple`, `Enable`, `TupleTake`, `Tuple`, `TupleTake`, `Tuple`, `TupleTake`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/headeronly/util`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/headeronly/macros/Macros.h`
- `torch/headeronly/util/TypeList.h`
- `type_traits`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/headeronly/util`):

- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`HeaderOnlyArrayRef.h_docs.md`](./HeaderOnlyArrayRef.h_docs.md)
- [`Float8_e5m2.h_docs.md`](./Float8_e5m2.h_docs.md)
- [`floating_point_utils.h_docs.md`](./floating_point_utils.h_docs.md)
- [`shim_utils.h_docs.md`](./shim_utils.h_docs.md)
- [`TypeList.h_docs.md`](./TypeList.h_docs.md)
- [`Float4_e2m1fn_x2.h_docs.md`](./Float4_e2m1fn_x2.h_docs.md)
- [`Float8_e8m0fnu.h_docs.md`](./Float8_e8m0fnu.h_docs.md)
- [`BFloat16.h_docs.md`](./BFloat16.h_docs.md)
- [`Float8_fnuz_cvt.h_docs.md`](./Float8_fnuz_cvt.h_docs.md)


## Cross-References

- **File Documentation**: `Metaprogramming.h_docs.md`
- **Keyword Index**: `Metaprogramming.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/headeronly/util`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/headeronly/util`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/headeronly/util`):

- [`quint8.h_docs.md_docs.md`](./quint8.h_docs.md_docs.md)
- [`TypeTraits.h_kw.md_docs.md`](./TypeTraits.h_kw.md_docs.md)
- [`Half.h_kw.md_docs.md`](./Half.h_kw.md_docs.md)
- [`TypeSafeSignMath.h_kw.md_docs.md`](./TypeSafeSignMath.h_kw.md_docs.md)
- [`qint32.h_docs.md_docs.md`](./qint32.h_docs.md_docs.md)
- [`Float8_e4m3fnuz.h_kw.md_docs.md`](./Float8_e4m3fnuz.h_kw.md_docs.md)
- [`Exception.h_docs.md_docs.md`](./Exception.h_docs.md_docs.md)
- [`quint8.h_kw.md_docs.md`](./quint8.h_kw.md_docs.md)
- [`quint2x4.h_docs.md_docs.md`](./quint2x4.h_docs.md_docs.md)
- [`quint4x2.h_docs.md_docs.md`](./quint4x2.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Metaprogramming.h_docs.md_docs.md`
- **Keyword Index**: `Metaprogramming.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

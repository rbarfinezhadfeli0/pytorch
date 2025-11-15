# Documentation: `docs/torch/headeronly/util/TypeTraits.h_docs.md`

## File Metadata

- **Path**: `docs/torch/headeronly/util/TypeTraits.h_docs.md`
- **Size**: 8,322 bytes (8.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/headeronly/util/TypeTraits.h`

## File Metadata

- **Path**: `torch/headeronly/util/TypeTraits.h`
- **Size**: 5,698 bytes (5.56 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <functional>
#include <type_traits>

namespace c10::guts {

/**
 * is_equality_comparable<T> is true_type iff the equality operator is defined
 * for T.
 */
template <class T, class Enable = void>
struct is_equality_comparable : std::false_type {};
template <class T>
struct is_equality_comparable<
    T,
    std::void_t<decltype(std::declval<T&>() == std::declval<T&>())>>
    : std::true_type {};
template <class T>
using is_equality_comparable_t = typename is_equality_comparable<T>::type;

/**
 * is_hashable<T> is true_type iff std::hash is defined for T
 */
template <class T, class Enable = void>
struct is_hashable : std::false_type {};
template <class T>
struct is_hashable<T, std::void_t<decltype(std::hash<T>()(std::declval<T&>()))>>
    : std::true_type {};
template <class T>
using is_hashable_t = typename is_hashable<T>::type;

/**
 * is_function_type<T> is true_type iff T is a plain function type (i.e.
 * "Result(Args...)")
 */
template <class T>
struct is_function_type : std::false_type {};
template <class Result, class... Args>
struct is_function_type<Result(Args...)> : std::true_type {};
template <class T>
using is_function_type_t = typename is_function_type<T>::type;

/**
 * is_instantiation_of<T, I> is true_type iff I is a template instantiation of T
 * (e.g. vector<int> is an instantiation of vector) Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template <template <class...> class Template, class T>
using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;

namespace detail {
/**
 * strip_class: helper to remove the class type from pointers to `operator()`.
 */

template <typename T>
struct strip_class {};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...)> {
  using type = Result(Args...);
};
template <typename Class, typename Result, typename... Args>
struct strip_class<Result (Class::*)(Args...) const> {
  using type = Result(Args...);
};
template <typename T>
using strip_class_t = typename strip_class<T>::type;
} // namespace detail

/**
 * Evaluates to true_type, iff the given class is a Functor
 * (i.e. has a call operator with some set of arguments)
 */

template <class Functor, class Enable = void>
struct is_functor : std::false_type {};
template <class Functor>
struct is_functor<
    Functor,
    std::enable_if_t<is_function_type<
        detail::strip_class_t<decltype(&Functor::operator())>>::value>>
    : std::true_type {};

/**
 * lambda_is_stateless<T> is true iff the lambda type T is stateless
 * (i.e. does not have a closure).
 * Example:
 *  auto stateless_lambda = [] (int a) {return a;};
 *  lambda_is_stateless<decltype(stateless_lambda)> // true
 *  auto stateful_lambda = [&] (int a) {return a;};
 *  lambda_is_stateless<decltype(stateful_lambda)> // false
 */
namespace detail {
template <class LambdaType, class FuncType>
struct is_stateless_lambda__ final {
  static_assert(
      !std::is_same_v<LambdaType, LambdaType>,
      "Base case shouldn't be hit");
};
// implementation idea: According to the C++ standard, stateless lambdas are
// convertible to function pointers
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...) const>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};
template <class LambdaType, class C, class Result, class... Args>
struct is_stateless_lambda__<LambdaType, Result (C::*)(Args...)>
    : std::is_convertible<LambdaType, Result (*)(Args...)> {};

// case where LambdaType is not even a functor
template <class LambdaType, class Enable = void>
struct is_stateless_lambda_ final : std::false_type {};
// case where LambdaType is a functor
template <class LambdaType>
struct is_stateless_lambda_<
    LambdaType,
    std::enable_if_t<is_functor<LambdaType>::value>>
    : is_stateless_lambda__<LambdaType, decltype(&LambdaType::operator())> {};
} // namespace detail
template <class T>
using is_stateless_lambda = detail::is_stateless_lambda_<std::decay_t<T>>;

/**
 * is_type_condition<C> is true_type iff C<...> is a type trait representing a
 * condition (i.e. has a constexpr static bool ::value member) Example:
 *   is_type_condition<std::is_reference>  // true
 */
template <template <class> class C, class Enable = void>
struct is_type_condition : std::false_type {};
template <template <class> class C>
struct is_type_condition<
    C,
    std::enable_if_t<
        std::is_same_v<bool, std::remove_cv_t<decltype(C<int>::value)>>>>
    : std::true_type {};

/**
 * is_fundamental<T> is true_type iff the lambda type T is a fundamental type
 * (that is, arithmetic type, void, or nullptr_t). Example: is_fundamental<int>
 * // true We define it here to resolve a MSVC bug. See
 * https://github.com/pytorch/pytorch/issues/30932 for details.
 */
template <class T>
struct is_fundamental : std::is_fundamental<T> {};
} // namespace c10::guts

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, guts)

using c10::guts::is_equality_comparable;
using c10::guts::is_function_type;
using c10::guts::is_hashable;
using c10::guts::is_instantiation_of;
using c10::guts::is_stateless_lambda;
using c10::guts::is_type_condition;

HIDDEN_NAMESPACE_END(torch, headeronly, guts)

```



## High-Level Overview


This C++ file contains approximately 37 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `T`, `Enable`, `is_equality_comparable`, `T`, `is_equality_comparable`, `T`, `T`, `Enable`, `is_hashable`, `T`, `is_hashable`, `T`, `T`, `is_function_type`, `Result`, `is_function_type`, `T`, `Template`, `T`, `is_instantiation_of`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/headeronly/util`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/headeronly/macros/Macros.h`
- `functional`
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

- **File Documentation**: `TypeTraits.h_docs.md`
- **Keyword Index**: `TypeTraits.h_kw.md`
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

- **File Documentation**: `TypeTraits.h_docs.md_docs.md`
- **Keyword Index**: `TypeTraits.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

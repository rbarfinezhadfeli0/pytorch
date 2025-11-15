# Documentation: `torch/custom_class_detail.h`

## File Metadata

- **Path**: `torch/custom_class_detail.h`
- **Size**: 7,832 bytes (7.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h>
#include <ATen/core/function.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeTraits.h>
#include <c10/util/irange.h>

#include <functional>

namespace torch {

namespace detail {
/**
 * In the Facebook internal build (using BUCK), this macro is enabled by
 * passing in -c pt.enable_record_kernel_dtype=1 when building the tracer
 * binary.
 */
#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
TORCH_API void record_custom_class(std::string name);

/**
 * Record an instance of a custom class being loaded
 * grab portion of string after final '.' from qualified name
 * as this seemingly aligns with how users name their custom classes
 * example: __torch__.torch.classes.xnnpack.Conv2dOpContext
 */
#define RECORD_CUSTOM_CLASS(NAME) \
  auto name = std::string(NAME);  \
  detail::record_custom_class(name.substr(name.find_last_of(".") + 1));
#else
#define RECORD_CUSTOM_CLASS(NAME)
#endif
} // namespace detail

/// This struct is used to represent default values for arguments
/// when registering methods for custom classes.
///     static auto register_foo = torch::class_<Foo>("myclasses", "Foo")
///       .def("myMethod", &Foo::myMethod, {torch::arg("name") = name});
struct arg {
  // Static method for representing a default value of None. This is meant to
  // be used like so:
  //     torch::arg("name") = torch::arg::none
  // and is identical to:
  //     torch::arg("name") = IValue()
  static c10::IValue none() {
    return c10::IValue();
  }

  // Explicit constructor.
  explicit arg(std::string name)
      : name_(std::move(name)), value_(std::nullopt) {}
  // Assignment operator. This enables the pybind-like syntax of
  // torch::arg("name") = value.
  arg& operator=(const c10::IValue& rhs) {
    value_ = rhs;
    return *this;
  }

  // The name of the argument. This is copied to the schema; argument
  // names cannot be extracted from the C++ declaration.
  std::string name_;
  // IValue's default constructor makes it None, which is not distinguishable
  // from an actual, user-provided default value that is None. This boolean
  // helps distinguish between the two cases.
  std::optional<c10::IValue> value_;
};

namespace detail {

// Argument type utilities
template <class R, class...>
struct types {
  using type = types;
};

template <typename Method>
struct WrapMethod;

template <typename R, typename CurrClass, typename... Args>
struct WrapMethod<R (CurrClass::*)(Args...)> {
  WrapMethod(R (CurrClass::*m)(Args...)) : m(std::move(m)) {}

  R operator()(c10::intrusive_ptr<CurrClass> cur, Args... args) {
    return std::invoke(m, *cur, args...);
  }

  R (CurrClass::*m)(Args...);
};

template <typename R, typename CurrClass, typename... Args>
struct WrapMethod<R (CurrClass::*)(Args...) const> {
  WrapMethod(R (CurrClass::*m)(Args...) const) : m(std::move(m)) {}

  R operator()(c10::intrusive_ptr<CurrClass> cur, Args... args) {
    return std::invoke(m, *cur, args...);
  }

  R (CurrClass::*m)(Args...) const;
};

// Adapter for different callable types
template <
    typename CurClass,
    typename Func,
    std::enable_if_t<
        std::is_member_function_pointer_v<std::decay_t<Func>>,
        bool> = false>
WrapMethod<Func> wrap_func(Func f) {
  return WrapMethod<Func>(std::move(f));
}

template <
    typename CurClass,
    typename Func,
    std::enable_if_t<
        !std::is_member_function_pointer_v<std::decay_t<Func>>,
        bool> = false>
Func wrap_func(Func f) {
  return f;
}

template <
    class Functor,
    bool AllowDeprecatedTypes,
    size_t... ivalue_arg_indices>
typename c10::guts::infer_function_traits_t<Functor>::return_type
call_torchbind_method_from_stack(
    Functor& functor,
    jit::Stack& stack,
    std::index_sequence<ivalue_arg_indices...> /*unused*/) {
  (void)stack; // when sizeof...(ivalue_arg_indices) == 0, this argument would
                 // be unused and we have to silence the compiler warning.

  constexpr size_t num_ivalue_args = sizeof...(ivalue_arg_indices);

  using IValueArgTypes =
      typename c10::guts::infer_function_traits_t<Functor>::parameter_types;
  // TODO We shouldn't use c10::impl stuff directly here. We should use the
  // KernelFunction API instead.
  return functor(c10::impl::ivalue_to_arg<
                   typename c10::impl::decay_if_not_tensor<
                       c10::guts::typelist::
                           element_t<ivalue_arg_indices, IValueArgTypes>>::type,
                   AllowDeprecatedTypes>::
                       call(torch::jit::peek(
                           stack, ivalue_arg_indices, num_ivalue_args))...);
}

template <class Functor, bool AllowDeprecatedTypes>
typename c10::guts::infer_function_traits_t<Functor>::return_type
call_torchbind_method_from_stack(Functor& functor, jit::Stack& stack) {
  constexpr size_t num_ivalue_args =
      c10::guts::infer_function_traits_t<Functor>::number_of_parameters;
  return call_torchbind_method_from_stack<Functor, AllowDeprecatedTypes>(
      functor, stack, std::make_index_sequence<num_ivalue_args>());
}

template <class RetType, class Func>
struct BoxedProxy;

template <class RetType, class Func>
struct BoxedProxy {
  void operator()(jit::Stack& stack, Func& func) {
    auto retval = call_torchbind_method_from_stack<Func, false>(func, stack);
    constexpr size_t num_ivalue_args =
        c10::guts::infer_function_traits_t<Func>::number_of_parameters;
    torch::jit::drop(stack, num_ivalue_args);
    stack.emplace_back(c10::ivalue::from(std::move(retval)));
  }
};

template <class Func>
struct BoxedProxy<void, Func> {
  void operator()(jit::Stack& stack, Func& func) {
    call_torchbind_method_from_stack<Func, false>(func, stack);
    constexpr size_t num_ivalue_args =
        c10::guts::infer_function_traits_t<Func>::number_of_parameters;
    torch::jit::drop(stack, num_ivalue_args);
    stack.emplace_back();
  }
};

inline bool validIdent(size_t i, char n) {
  return isalpha(n) || n == '_' || (i > 0 && isdigit(n));
}

inline void checkValidIdent(const std::string& str, const char* type) {
  for (const auto i : c10::irange(str.size())) {
    TORCH_CHECK(
        validIdent(i, str[i]),
        type,
        " must be a valid Python/C++ identifier."
        " Character '",
        str[i],
        "' at index ",
        i,
        " is illegal.");
  }
}

class TORCH_API class_base {
 protected:
  explicit class_base(
      const std::string& namespaceName,
      const std::string& className,
      std::string doc_string,
      const std::type_info& intrusivePtrClassTypeid,
      const std::type_info& taggedCapsuleClass);

  static c10::FunctionSchema withNewArguments(
      const c10::FunctionSchema& schema,
      std::initializer_list<arg> default_args);
  std::string qualClassName;
  at::ClassTypePtr classTypePtr;
};

} // namespace detail

TORCH_API void registerCustomClass(at::ClassTypePtr class_type);
TORCH_API void registerCustomClassMethod(std::unique_ptr<jit::Function> method);

// Given a qualified name (e.g. __torch__.torch.classes.Foo), return
// the ClassType pointer to the Type that describes that custom class,
// or nullptr if no class by that name was found.
TORCH_API at::ClassTypePtr getCustomClass(const std::string& name);

// Given an IValue, return true if the object contained in that IValue
// is a custom C++ class, otherwise return false.
// NOLINTNEXTLINE(readability-redundant-declaration)
TORCH_API bool isCustomClass(const c10::IValue& v);

// This API is for testing purposes ONLY. It should not be used in
// any load-bearing code.
TORCH_API std::vector<c10::FunctionSchema> customClassSchemasForBCCheck();

namespace jit {
using ::torch::registerCustomClass;
using ::torch::registerCustomClassMethod;
} // namespace jit

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 11 class(es)/struct(s) and 24 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`, `detail`

**Classes/Structs**: `being`, `is`, `arg`, `R`, `types`, `WrapMethod`, `WrapMethod`, `WrapMethod`, `Functor`, `Functor`, `RetType`, `Func`, `BoxedProxy`, `RetType`, `Func`, `BoxedProxy`, `Func`, `BoxedProxy`, `TORCH_API`, `by`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h`
- `ATen/core/function.h`
- `c10/util/Metaprogramming.h`
- `c10/util/TypeTraits.h`
- `c10/util/irange.h`
- `functional`


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

Files in the same folder (`torch`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_tensor_docs.py_docs.md`](./_tensor_docs.py_docs.md)
- [`_classes.py_docs.md`](./_classes.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`_meta_registrations.py_docs.md`](./_meta_registrations.py_docs.md)
- [`_appdirs.py_docs.md`](./_appdirs.py_docs.md)
- [`_tensor.py_docs.md`](./_tensor.py_docs.md)
- [`_streambase.py_docs.md`](./_streambase.py_docs.md)
- [`_lowrank.py_docs.md`](./_lowrank.py_docs.md)
- [`_size_docs.py_docs.md`](./_size_docs.py_docs.md)


## Cross-References

- **File Documentation**: `custom_class_detail.h_docs.md`
- **Keyword Index**: `custom_class_detail.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

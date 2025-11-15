# Documentation: `aten/src/ATen/core/boxing/KernelFunction.h`

## File Metadata

- **Path**: `aten/src/ATen/core/boxing/KernelFunction.h`
- **Size**: 10,290 bytes (10.05 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/TypeList.h>
#include <c10/util/intrusive_ptr.h>
#include <atomic>
#include <memory>
#include <type_traits>

namespace c10 {

using Stack = torch::jit::Stack; // TODO Instead of this, move torch::jit::Stack
                                 // to the c10 namespace.

class OperatorHandle;
struct OperatorKernel;
class KernelFunction;

class KernelToken;
class SafeKernelFunction;

template <typename T>
using has_symint = std::disjunction<
    std::is_same<c10::SymInt, T>,
    std::is_same<c10::SymIntArrayRef, T>,
    std::is_same<at::OptionalSymIntArrayRef, T>,
    std::is_same<std::optional<c10::SymInt>, T>>;

template <typename T>
struct remove_symint {
  using type = T;
};

template <>
struct remove_symint<c10::SymInt> {
  using type = int64_t;
};

template <>
struct remove_symint<at::OptionalSymIntArrayRef> {
  using type = OptionalIntArrayRef;
};

template <>
struct remove_symint<c10::SymIntArrayRef> {
  using type = c10::IntArrayRef;
};

template <>
struct remove_symint<std::optional<c10::SymInt>> {
  using type = std::optional<int64_t>;
};

template <bool symint, typename T>
struct maybe_keep_symint final {};

template <typename T>
struct maybe_keep_symint<true, T> {
  using type = T;
};

template <typename T>
struct maybe_keep_symint<false, T> {
  using type = typename remove_symint<T>::type;
};

template <typename T>
using fn_has_symint = typename guts::typelist::true_for_any_type<
    has_symint,
    typename guts::infer_function_traits<T>::type::parameter_types>;

template <typename T>
struct fn_remove_symint;

template <typename Ret, typename... Args>
struct fn_remove_symint<Ret(Args...)> {
  using type = Ret(typename remove_symint<Args>::type...);
};

/**
 * KernelFunction is similar to std::function but stores a kernel function.
 * You can create a KernelFunction from a boxed or unboxed
 * function/functor/lambda and call it in a boxed or unboxed way. If the way it
 * was created doesn't match the way it was called, it will do boxing or
 * unboxing as necessary.
 */
class TORCH_API KernelFunction final {
 public:
  using InternalBoxedKernelFunction = BoxedKernel::InternalBoxedKernelFunction;
  using BoxedKernelFunction = BoxedKernel::BoxedKernelFunction;
  using BoxedKernelFunction_withDispatchKeys =
      BoxedKernel::BoxedKernelFunction_withDispatchKeys;

  KernelFunction();
  ~KernelFunction();

  KernelFunction(const KernelFunction& other);
  KernelFunction& operator=(const KernelFunction& other);

  KernelFunction(KernelFunction&&) noexcept = default;

  // Fast path for dispatch to allow not touching the boxed kernel in
  // the common case where unboxed is available.
  bool isValidUnboxed() const;
  bool isValidSymUnboxed() const;
  bool isValid() const;
  bool isFallthrough() const;

  /**
   * Call the function in a boxed way.
   * If the kernel function was created with an unboxed function,
   * this will call an unboxing wrapper which then calls into that
   * unboxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.callBoxed(stack);
   *
   * Or, with an unboxed implementation:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.callBoxed(stack);
   */
  void callBoxed(
      const OperatorHandle& opHandle,
      DispatchKeySet dispatchKeySet,
      Stack* stack) const;

  /**
   * Call the function in an unboxed way.
   * If the kernel function was created with a boxed function,
   * this will box all inputs and then call into that boxed function.
   *
   * Note that this doesn't work for all types yet.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   *
   * Or, with a boxed implementation:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func = KernelFunction::makeFromBoxedFunction(&boxed_func);
   * > Tensor result = func.call<Tensor, Tensor, bool>(tensor1, true);
   */
  template <class Return, class... Args>
  Return call(
      const OperatorHandle& opHandle,
      DispatchKeySet dispatchKeySet,
      Args... args) const;

  /**
   * Create a KernelFunction from a BoxedKernel.
   */
  static KernelFunction makeFromBoxedKernel(BoxedKernel boxed_fn);

  /**
   * Create a KernelFunction from a boxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > KernelFunction func =
   * KernelFunction::makeFromBoxedFunction<&boxed_func>();
   */
  template <BoxedKernelFunction* func>
  static KernelFunction makeFromBoxedFunction();

  /**
   * TODO: This will only be useful if we write a backend fallback that plumbs
   * dispatch keys (currently there are none) See Note [Plumbing Keys Through
   * The Dispatcher] for details.
   */
  template <BoxedKernelFunction_withDispatchKeys* func>
  static KernelFunction makeFromBoxedFunction();

  /**
   * Create a KernelFunction from an unboxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     Tensor operator()(Tensor a, Tensor b) {...}
   * > };
   * > KernelFunction func =
   * KernelFunction::makeFromUnboxedFunctor<MyFunctor>(std::make_unique<MyFunctor>());
   */
  template <bool AllowLegacyTypes = false, class KernelFunctor>
  static KernelFunction makeFromUnboxedFunctor(
      std::unique_ptr<OperatorKernel> kernelFunctor);

  /**
   * Create a KernelFunction from a boxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     void operator()(const OperatorHandle&, DispatchKeySet, Stack*) {...}
   * > };
   * > KernelFunction func =
   * KernelFunction::makeFromBoxedFunctor(std::make_unique<MyFunctor>());
   */
  template <class KernelFunctor>
  static KernelFunction makeFromBoxedFunctor(
      std::unique_ptr<KernelFunctor> kernelFunctor);

  /**
   * Create a KernelFunction from an unboxed function.
   * This is usually better than KernelFunction::makeFromUnboxedRuntimeFunction
   * because knowing the function pointer as a template argument (i.e. at
   * compile time) allows the compiler to inline the function into its
   * unboxing wrapper and yields better performance when calling the function.
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func =
   * KernelFunction::makeFromUnboxedFunction<decltype(unboxed_func),
   * &unboxed_func>();
   */
  template <class FuncPtr, bool AllowLegacyTypes = false>
  static KernelFunction makeFromUnboxedFunction(FuncPtr /*func_ptr*/);

  /**
   * Create a KernelFunction from an unboxed function.
   * KernelFunction::makeFromUnboxedFunction is usually a better choice than
   * this if you know the function pointer at compile time, see doc comment
   * there for an explanation.
   *
   * Example:
   *
   * > Tensor unboxed_func(Tensor a, Tensor b) {...}
   * > KernelFunction func =
   * KernelFunction::makeFromUnboxedRuntimeFunction(&unboxed_func);
   */
  template <bool AllowLegacyTypes = false, class FuncType>
  static KernelFunction makeFromUnboxedRuntimeFunction(FuncType* func);

  static KernelFunction makeFallthrough();
  static KernelFunction makeAmbiguousAutogradOther();
  static KernelFunction makeNamedNotSupported();

  /**
   * Create a KernelFunction from an unboxed lambda.
   *
   * Example:
   *
   * > KernelFunction func = KernelFunction::makeFromUnboxedLambda(
   * >      [] (Tensor a, bool b) -> Tensor {...});
   */
  template <bool AllowLegacyTypes = false, class Lambda>
  static std::enable_if_t<
      guts::is_stateless_lambda<std::decay_t<Lambda>>::value,
      KernelFunction>
  makeFromUnboxedLambda(Lambda&& lambda);
  template <bool AllowLegacyTypes = false, class Lambda>
  static std::enable_if_t<
      !guts::is_stateless_lambda<std::decay_t<Lambda>>::value,
      KernelFunction>
  makeFromUnboxedLambda(Lambda&& lambda);

  std::string dumpState() const;
  // For testing internal invariants only
  bool _equalsBoxedAndUnboxed(const KernelFunction& /*other*/) const;

  // Register a token to be invalidated when this KernelFunction is destroyed
  void registerToken(std::weak_ptr<KernelToken> token) const;

 private:
  explicit KernelFunction(
      std::unique_ptr<OperatorKernel> functor,
      InternalBoxedKernelFunction* boxed_kernel_func,
      void* unboxed_kernel_func,
      void* sym_unboxed_kernel_func);
  explicit KernelFunction(
      BoxedKernel boxed_fn,
      void* unboxed_kernel_func,
      void* sym_unboxed_kernel_func);

  BoxedKernel boxed_kernel_func_;
  void* unboxed_kernel_func_;
  void* sym_unboxed_kernel_func_;
  // List of tokens that need to be invalidated when this KernelFunction is
  // destroyed (lazy allocation to save memory when empty)
  mutable std::unique_ptr<std::vector<std::weak_ptr<KernelToken>>> tokens_;
};

// Token held by SafeKernelFunction that gets invalidated when KernelFunction is
// destroyed
class KernelToken {
 public:
  bool isValid() const;
  void invalidate();

 private:
  std::atomic<bool> invalid_{false};
};

class SafeKernelFunction {
 public:
  SafeKernelFunction(
      const KernelFunction* kernel,
      std::string debug,
      std::shared_ptr<OperatorHandle> opHandle);

  // Safe callBoxed - checks token validity first
  void callBoxed(
      const OperatorHandle& opHandle,
      DispatchKeySet dispatchKeySet,
      Stack* stack) const;

  // Get debug information
  const std::string& debug() const {
    return debug_;
  }

  // Get the OpHandle that lives on this SafeKernelFunction
  const OperatorHandle& opHandle() const {
    return *opHandle_;
  }

 private:
  KernelFunction kernel_;
  std::shared_ptr<KernelToken> token_;
  std::string debug_;
  std::shared_ptr<OperatorHandle> opHandle_;
};

} // namespace c10

#include <ATen/core/boxing/KernelFunction_impl.h>

```



## High-Level Overview


This C++ file contains approximately 16 class(es)/struct(s) and 33 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `OperatorHandle`, `OperatorKernel`, `KernelFunction`, `KernelToken`, `SafeKernelFunction`, `remove_symint`, `remove_symint`, `remove_symint`, `remove_symint`, `remove_symint`, `maybe_keep_symint`, `maybe_keep_symint`, `maybe_keep_symint`, `fn_remove_symint`, `fn_remove_symint`, `TORCH_API`, `Return`, `MyFunctor`, `KernelFunctor`, `MyFunctor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core/boxing`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ATen_fwd.h`
- `ATen/core/boxing/BoxedKernel.h`
- `ATen/core/stack.h`
- `c10/core/DispatchKeySet.h`
- `c10/util/TypeList.h`
- `c10/util/intrusive_ptr.h`
- `atomic`
- `memory`
- `type_traits`
- `ATen/core/boxing/KernelFunction_impl.h`


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

Files in the same folder (`aten/src/ATen/core/boxing`):

- [`BoxedKernel_impl.h_docs.md`](./BoxedKernel_impl.h_docs.md)
- [`KernelFunction_impl.h_docs.md`](./KernelFunction_impl.h_docs.md)
- [`KernelFunction_test.cpp_docs.md`](./KernelFunction_test.cpp_docs.md)
- [`KernelFunction.cpp_docs.md`](./KernelFunction.cpp_docs.md)
- [`OperatorKernel.h_docs.md`](./OperatorKernel.h_docs.md)
- [`BoxedKernel.h_docs.md`](./BoxedKernel.h_docs.md)


## Cross-References

- **File Documentation**: `KernelFunction.h_docs.md`
- **Keyword Index**: `KernelFunction.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

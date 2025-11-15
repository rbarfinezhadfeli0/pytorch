# Documentation: `docs/aten/src/ATen/core/boxing/BoxedKernel.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/boxing/BoxedKernel.h_docs.md`
- **Size**: 10,841 bytes (10.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/boxing/BoxedKernel.h`

## File Metadata

- **Path**: `aten/src/ATen/core/boxing/BoxedKernel.h`
- **Size**: 8,301 bytes (8.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/boxing/OperatorKernel.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {

struct IValue;
using Stack = std::vector<IValue>;

class OperatorHandle;
class KernelFunction;

// This kernel implements the behavior of falling through to the next available
// registered dispatch key.  The implementation of this function is FAST; it is
// no overhead to fallthrough to the next key.  See cpp file for some more
// implementation notes; notably, this does NOT actually go through the
// boxing/unboxing codepath.
TORCH_API void fallthrough_kernel(
    OperatorKernel* /*unused*/,
    const OperatorHandle& /*unused*/,
    DispatchKeySet /*unused*/,
    Stack* /*unused*/);

// Note [Ambiguity in AutogradOther kernel]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This error-reporting kernel is registered to the AutogradOther entry in the
// dispatch table when there is both a CompositeImplicitAutograd kernel and a
// backend kernel for ANY backend that maps to AutogradOther.  To see why
// this is necessary in the AutogradOther case, it's helpful to first see
// why everything works out fine for a backend that has a reserved Autograd
// entry (see rule 2.2 in [Note] DispatchTable computation):
//
//    CPU   AutogradCPU
//    reg?  registers with...
//    -------------------------------------------------
//    y     Autograd registration takes precedence
//          over CompositeImplicitAutograd.
//          This is good, because the CPU specific backend
//          implementation is more specialized and typically better;
//          if we used the composite, we would bypass it.
//          (NB: the Autograd key is guaranteed to exist because
//          the autograd codegen requires it!)
//
//    n     CompositeImplicitAutograd takes precedence.
//          This is also good, because the Autograd
//          registration (if it exists) would try to redispatch
//          to the (non-existent) CPU implementation; by
//          using the composite, we ensure the operator
//          actually works.
//
// As you can see, when we have a specific Autograd key (AutogradCPU), we can
// decide whether or not to use the CompositeImplicitAutograd kernel or the
// Autograd kernel based on whether or not the backend kernel exists.
//
// However, for AutogradOther (which is the catchall autograd kernel for
// everything that doesn't have a specific Autograd key), we can't do this
// trick because there isn't any unique backend to peek at to disambiguate;
// if there are some backends that have implementations they prefer Autograd,
// but unimplemented backends would prefer CompositeImplicitAutograd.  Rather
// than arbitrarily pick one or the other, we just register a kernel that raises
// an error and let the user decide how to proceed.
TORCH_API void ambiguous_autogradother_kernel(
    OperatorKernel* /*unused*/,
    const OperatorHandle& /*op*/,
    DispatchKeySet /*unused*/,
    Stack* /*unused*/);

// Note [named_not_supported_kernel]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This kernel implements reporting an error message saying that named tensor is
// not supported.  This kernel doesn't rely on the Stack, and so it is special
// cased in the dispatcher to be triggered before we attempt boxing (so we can
// give a good error message in cases when boxing is not supported).  When
// boxing is universally supported this can be removed.
[[noreturn]] TORCH_API void named_not_supported_kernel(
    OperatorKernel* /*unused*/,
    const OperatorHandle& /*op*/,
    DispatchKeySet /*unused*/,
    Stack* /*unused*/);

/**
 * BoxedKernel is similar to a std::function storing a boxed kernel.
 */
class TORCH_API BoxedKernel final {
 public:
  // This is how boxed kernels are actually stored
  //
  // Note [Plumbing Keys Through The Dispatcher]
  // Benchmarks have shown that it is expensive for the dispatcher to read from
  // thread-local storage (TLS) upon every dispatch call into order to compute
  // which kernel to dispatch to.
  //
  // To mitigate this, we've updated the calling convention inside the
  // dispatcher to expect every kernel that it stores to have a first argument
  // of type DispatchKeySet.
  //
  // What are the invariants of the DispatchKeySet when it gets passed to a
  // kernel?
  // - All keys to the left of the current dispatch key have been masked out.
  //   (e.g. a Tracing kernel that takes in the DispatchKeySet will expect the
  //   highest bit to be DispatchKey::Tracer)
  // - All other keys that dispatcher normally would have computed through TLS +
  // global state + op arguments
  //   are still in the set.
  //
  // Kernels can then opt into using this keyset to save the dispatcher from
  // doing repeated work during redispatches: recalculating the highest-priority
  // dispatch key, which involves reading from TLS. Instead, the kernels that
  // opt in will calculate an updated DispatchKeySet directly from the old one,
  // and pass the updated set directly into the dispatcher upon redispatching.
  //
  // This is an opt-in mechanism: Kernels can automatically opt in by setting
  // the first argument in their signature to be of type DispatchKeySet. See the
  // kernels in VariableTypeEverything.cpp and TraceTypeEverything.cpp for
  // examples.
  //
  // The mechanism for optionally passing that DispatchKeySet into the kernel
  // lives in make_boxed_from_unboxed_functor.h. See Note [Plumbing Keys Through
  // The Dispatcher 2] for details.
  using InternalBoxedKernelFunction =
      void(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*);
  // This is the public API for how boxed kernels are defined
  using BoxedKernelFunction = void(const OperatorHandle&, Stack*);
  using BoxedKernelFunction_withDispatchKeys =
      void(const OperatorHandle&, DispatchKeySet, Stack*);

  BoxedKernel();

  // Fast path for dispatch to allow not touching the boxed kernel in
  // the common case where unboxed is available.
  bool isValid() const;
  bool isFallthrough() const;

  /**
   * Call the function with boxed arguments.
   */
  void callBoxed(
      const OperatorHandle& opHandle,
      DispatchKeySet dispatchKeySet,
      Stack* stack) const;

  /**
   * Create a KernelFunction from a boxed function.
   *
   * Example:
   *
   * > void boxed_func(OperatorKernel*, Stack* stack) {...}
   * > BoxedFunction func = BoxedKernel::makeFromFunction<&boxed_func>();
   */
  template <BoxedKernelFunction* func>
  static BoxedKernel makeFromFunction();

  /**
   * TODO: This will only be useful if we write a backend fallback that plumbs
   * dispatch keys (currently there are none) See Note [Plumbing Keys Through
   * The Dispatcher] for details.
   */
  template <BoxedKernelFunction_withDispatchKeys* func>
  static BoxedKernel makeFromFunction();

  /**
   * Create a KernelFunction from a boxed functor.
   *
   * Example:
   *
   * > class MyFunctor final : public c10::OperatorKernel {
   * >   public:
   * >     void operator()(const OperatorHandle&, DispatchKeySet, Stack*) {...}
   * > };
   * > BoxedKernel func =
   * BoxedKernel::makeFromFunctor(std::make_unique<MyFunctor>());
   */
  template <class KernelFunctor>
  static BoxedKernel makeFromFunctor(
      std::unique_ptr<KernelFunctor> kernelFunctor);

  static BoxedKernel makeFallthrough();
  static BoxedKernel makeAmbiguousAutogradOther();
  static BoxedKernel makeNamedNotSupported();

 private:
  friend class KernelFunction;

  template <BoxedKernelFunction* func>
  static void make_boxed_function(
      OperatorKernel* /*unused*/,
      const OperatorHandle& opHandle,
      DispatchKeySet /*unused*/,
      Stack* stack);

  template <BoxedKernelFunction_withDispatchKeys* func>
  static void make_boxed_function(
      OperatorKernel* /*unused*/,
      const OperatorHandle& opHandle,
      DispatchKeySet /*ks*/,
      Stack* stack);

  explicit BoxedKernel(
      std::unique_ptr<OperatorKernel> functor,
      InternalBoxedKernelFunction* boxed_kernel_func);

  OperatorKernel* getFunctor() const;
  InternalBoxedKernelFunction* getFnPtr() const;

  c10::intrusive_ptr<OperatorKernel> functor_;
  InternalBoxedKernelFunction* boxed_kernel_func_;
};

} // namespace c10

#include <ATen/core/boxing/BoxedKernel_impl.h>

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 23 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `IValue`, `OperatorHandle`, `KernelFunction`, `TORCH_API`, `MyFunctor`, `KernelFunctor`, `KernelFunction`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core/boxing`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/boxing/OperatorKernel.h`
- `c10/core/DispatchKeySet.h`
- `c10/util/intrusive_ptr.h`
- `ATen/core/boxing/BoxedKernel_impl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/core/boxing`):

- [`BoxedKernel_impl.h_docs.md`](./BoxedKernel_impl.h_docs.md)
- [`KernelFunction_impl.h_docs.md`](./KernelFunction_impl.h_docs.md)
- [`KernelFunction_test.cpp_docs.md`](./KernelFunction_test.cpp_docs.md)
- [`KernelFunction.cpp_docs.md`](./KernelFunction.cpp_docs.md)
- [`OperatorKernel.h_docs.md`](./OperatorKernel.h_docs.md)
- [`KernelFunction.h_docs.md`](./KernelFunction.h_docs.md)


## Cross-References

- **File Documentation**: `BoxedKernel.h_docs.md`
- **Keyword Index**: `BoxedKernel.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core/boxing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core/boxing`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/aten/src/ATen/core/boxing`):

- [`KernelFunction.cpp_kw.md_docs.md`](./KernelFunction.cpp_kw.md_docs.md)
- [`BoxedKernel.h_kw.md_docs.md`](./BoxedKernel.h_kw.md_docs.md)
- [`KernelFunction.h_docs.md_docs.md`](./KernelFunction.h_docs.md_docs.md)
- [`KernelFunction_impl.h_docs.md_docs.md`](./KernelFunction_impl.h_docs.md_docs.md)
- [`KernelFunction_test.cpp_docs.md_docs.md`](./KernelFunction_test.cpp_docs.md_docs.md)
- [`OperatorKernel.h_kw.md_docs.md`](./OperatorKernel.h_kw.md_docs.md)
- [`KernelFunction_impl.h_kw.md_docs.md`](./KernelFunction_impl.h_kw.md_docs.md)
- [`KernelFunction_test.cpp_kw.md_docs.md`](./KernelFunction_test.cpp_kw.md_docs.md)
- [`BoxedKernel_impl.h_docs.md_docs.md`](./BoxedKernel_impl.h_docs.md_docs.md)
- [`KernelFunction.h_kw.md_docs.md`](./KernelFunction.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `BoxedKernel.h_docs.md_docs.md`
- **Keyword Index**: `BoxedKernel.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

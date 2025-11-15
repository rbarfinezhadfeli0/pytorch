# Documentation: `docs/aten/src/ATen/core/boxing/BoxedKernel_impl.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/boxing/BoxedKernel_impl.h_docs.md`
- **Size**: 5,474 bytes (5.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/boxing/BoxedKernel_impl.h`

## File Metadata

- **Path**: `aten/src/ATen/core/boxing/BoxedKernel_impl.h`
- **Size**: 3,276 bytes (3.20 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

namespace c10 {

inline BoxedKernel::BoxedKernel() : boxed_kernel_func_(nullptr) {}

inline BoxedKernel::BoxedKernel(
    std::unique_ptr<OperatorKernel> functor,
    InternalBoxedKernelFunction* boxed_kernel_func)
    : functor_(std::move(functor)), boxed_kernel_func_(boxed_kernel_func) {}

template <BoxedKernel::BoxedKernelFunction* func>
inline void BoxedKernel::make_boxed_function(
    OperatorKernel* /*unused*/,
    const OperatorHandle& opHandle,
    DispatchKeySet /*unused*/,
    Stack* stack) {
  // Note that we're dropping the DispatchKeySet argument.
  // See Note [Plumbing Keys Through The Dispatcher 2] for details.
  func(opHandle, stack);
}

template <BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline void BoxedKernel::make_boxed_function(
    OperatorKernel* /*unused*/,
    const OperatorHandle& opHandle,
    DispatchKeySet ks,
    Stack* stack) {
  // See Note [Plumbing Keys Through The Dispatcher 2] for details.
  func(opHandle, ks, stack);
}

inline bool BoxedKernel::isValid() const {
  return boxed_kernel_func_ != nullptr;
}

inline bool BoxedKernel::isFallthrough() const {
  return boxed_kernel_func_ == &fallthrough_kernel;
}

inline void BoxedKernel::callBoxed(
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Stack* stack) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      boxed_kernel_func_ != nullptr,
      "Tried to call BoxedKernel::callBoxed() on an uninitialized BoxedKernel.");
  (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
}

template <BoxedKernel::BoxedKernelFunction* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &make_boxed_function<func>);
}

template <BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &make_boxed_function<func>);
}

inline BoxedKernel BoxedKernel::makeFallthrough() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &fallthrough_kernel);
}

inline BoxedKernel BoxedKernel::makeAmbiguousAutogradOther() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &ambiguous_autogradother_kernel);
}

inline BoxedKernel BoxedKernel::makeNamedNotSupported() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &named_not_supported_kernel);
}

template <class KernelFunctor>
inline BoxedKernel BoxedKernel::makeFromFunctor(
    std::unique_ptr<KernelFunctor> kernelFunctor) {
  static_assert(
      std::is_base_of_v<OperatorKernel, KernelFunctor>,
      "Tried to call BoxedKernel::makeFromFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
  return BoxedKernel(
      std::move(kernelFunctor),
      [](OperatorKernel* kernel,
         const OperatorHandle& op,
         DispatchKeySet ks,
         Stack* stack) {
        (*static_cast<KernelFunctor*>(kernel))(op, ks, stack);
      });
}

inline OperatorKernel* BoxedKernel::getFunctor() const {
  return functor_.get();
}
inline BoxedKernel::InternalBoxedKernelFunction* BoxedKernel::getFnPtr() const {
  return boxed_kernel_func_;
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `KernelFunctor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core/boxing`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*No includes detected.*


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

Files in the same folder (`aten/src/ATen/core/boxing`):

- [`KernelFunction_impl.h_docs.md`](./KernelFunction_impl.h_docs.md)
- [`KernelFunction_test.cpp_docs.md`](./KernelFunction_test.cpp_docs.md)
- [`KernelFunction.cpp_docs.md`](./KernelFunction.cpp_docs.md)
- [`OperatorKernel.h_docs.md`](./OperatorKernel.h_docs.md)
- [`BoxedKernel.h_docs.md`](./BoxedKernel.h_docs.md)
- [`KernelFunction.h_docs.md`](./KernelFunction.h_docs.md)


## Cross-References

- **File Documentation**: `BoxedKernel_impl.h_docs.md`
- **Keyword Index**: `BoxedKernel_impl.h_kw.md`
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
- [`KernelFunction.h_kw.md_docs.md`](./KernelFunction.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `BoxedKernel_impl.h_docs.md_docs.md`
- **Keyword Index**: `BoxedKernel_impl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

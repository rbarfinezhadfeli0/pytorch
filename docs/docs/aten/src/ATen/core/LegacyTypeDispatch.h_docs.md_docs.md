# Documentation: `docs/aten/src/ATen/core/LegacyTypeDispatch.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/LegacyTypeDispatch.h_docs.md`
- **Size**: 7,370 bytes (7.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/LegacyTypeDispatch.h`

## File Metadata

- **Path**: `aten/src/ATen/core/LegacyTypeDispatch.h`
- **Size**: 4,857 bytes (4.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

// The legacy mechanism for dispatching operators in ATen is a Type
// object, which is essentially a giant virtual dispatch table
// for every operation we support dynamically dispatching over.
//
// This has been deprecated in favor of ATenDispatch, and in the future,
// c10 dispatcher.
// TODO: Clean up what remains here

#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at {

// A RAII, thread local (!) guard that will disable dispatch to variable
// handler.
//
// NOTE [ Treating Variables as non-Variables in type dispatch ]
//
// What exactly does AutoDispatchBelowAutograd do?  The short answer is, it causes
// dispatches on ATen functions to go to the non-variable implementation,
// bypassing autograd handling (and also profiling and tracing).
//
// To understand why this guard exists, it's helpful to understand the history
// behind how Variable was implemented.  Previously, Variables were implemented
// as a wrapper on Tensors; so the act of processing a Variable involved
// unwrapping the underlying Tensor, and then calling the underlying base
// operation on /that/ operation
//
// However, after the Variable/Tensor merge, there is no concept of unwrapping
// a tensor anymore.  If you just call the operation on the same variable
// again inside your VariableType handler, you'll dispatch back to
// VariableType, which is not what we want.
//
// The solution to the above problem is to add `at::AutoDispatchBelowAutograd`, which
// when enabled will cause `legacyTensorType()` and `getType()` to always return
// non-Variable type, even if the tensor being called on is a variable.

/* Note [AutoDispatchBelowAutograd]
 * AutoDispatchBelowAutograd is **INTERNAL ONLY** that it should be used
 * for kernel implementations and customized C++ kernels.
 * If you are looking for a guard to run workload in inference mode, please use
 * c10::InferenceMode RAII which is user facing API.
 * In the past AutoDispatchBelowAutograd(or its old version AutoNonVariableTypeMode)
 * was used in the user code for inference-only workload, this was under risk of
 * producing wrong results silently in some edge cases. For example:
 * ```
 *  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
 *  torch::Tensor out = s * s;
 *  {
 *    at::AutoDispatchBelowAutograd guard;
 *    s.add_(1);  // Skips version bump on `s`.
 *  }
 *  // WRONG GRADIENT! s.grad() are now computed using `s` value after the
 *  // inplace update.
 *  out.backward(torch::ones_like(out));
 * ```
 * Users should use `c10::InferenceMode` here so that it'll properly throw an
 * error saying "one of the variables needed for gradient computation has be modified."
 */
struct TORCH_API AutoDispatchBelowAutograd {
  AutoDispatchBelowAutograd() :
    autograd_guard_(c10::autograd_dispatch_keyset) {
  }

  // disable all autograd dispatch keys
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

// TODO: AutoNonVariableTypeMode should be removed in release 1.10.
struct TORCH_API AutoNonVariableTypeMode {
  AutoNonVariableTypeMode(bool enabled = true) :
    autograd_guard_(c10::autograd_dispatch_keyset) {
    TORCH_WARN_ONCE("AutoNonVariableTypeMode is deprecated and will be removed in 1.10 release. "
        "For kernel implementations please use AutoDispatchBelowADInplaceOrView instead, "
        "If you are looking for a user facing API to enable running your inference-only "
        "workload, please use c10::InferenceMode. Using AutoDispatchBelowADInplaceOrView in user code "
        "is under risk of producing silent wrong result in some edge cases. "
        "See Note [AutoDispatchBelowAutograd] for more details.");
    TORCH_INTERNAL_ASSERT(enabled);
  }

  // disable all autograd dispatch keys
  c10::impl::ExcludeDispatchKeyGuard autograd_guard_;
};

struct TORCH_API AutoDispatchSkipFunctionalize {
  AutoDispatchSkipFunctionalize() :
    dispatch_key_guard_(c10::DispatchKeySet(c10::DispatchKey::Functionalize)) {
  }
  c10::impl::ExcludeDispatchKeyGuard dispatch_key_guard_;
};

/* Note [AutoDispatchBelowADInplaceOrView]
 * AutoDispatchBelowADInplaceOrView is equivalent to AutoNonVariableTypeMode
 * before we split inplace & view ops out of VariableType kernel.
 * Note this guard is used in VariableType kernels for functional ops
 * as well as ADInplaceOrView kernels for inplace/view ops to enforce the
 * Invariant:
 *   Once you are in VariableType/ADInplaceOrView kernel for an op,
 *   you never go back to a kernel on same dispatch key until
 *   you finish the current op.
 */
struct TORCH_API AutoDispatchBelowADInplaceOrView {
  AutoDispatchBelowADInplaceOrView() :
    dispatch_key_guard_(c10::autograd_dispatch_keyset_with_ADInplaceOrView) {
  }
  // disable Autograd & ADInplaceOrView dispatch keys
  c10::impl::ExcludeDispatchKeyGuard dispatch_key_guard_;
};
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/LocalDispatchKeySet.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `LegacyTypeDispatch.h_docs.md`
- **Keyword Index**: `LegacyTypeDispatch.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `LegacyTypeDispatch.h_docs.md_docs.md`
- **Keyword Index**: `LegacyTypeDispatch.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/TracerMode.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/TracerMode.h_docs.md`
- **Size**: 8,116 bytes (7.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/TracerMode.h`

## File Metadata

- **Path**: `aten/src/ATen/TracerMode.h`
- **Size**: 5,509 bytes (5.38 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>

// NOTE [Tracing Mode Switches]
//
// Historically, tracing function was controlled by two switches:
//
// - `AutoDispatchBelowADInplaceOrView` guard
//
//    Tracing function used to be script-generated inside `VariableType_*.cpp`
//    kernels, sharing the same `Autograd` dispatch key with autograd function.
//    Therefore, before tracing function was moved out of VariableType,
//    `AutoDispatchBelowADInplaceOrView` guard can also disable tracing as a
//    side effect of disabling `Autograd` dispatching.
//
// - `setTracingState()` API in `torch/csrc/jit/frontend/tracer.h`
//
//    It stores tracing data in a `TracingState` object in TLS. If the
//    `TracingState` object in TLS is `null`, then tracing is paused.
//
//    The `TracingState` object is created in `tracer::trace()` - the main
//    entrance of tracing function. It's temporarily set to `null` inside
//    generated VariableType (now TraceType) to bypass tracing for intermediate
//    ops (ops being called by other ops). After the intermediate op call
//    finishes it's set back to the original `TracingState` object.
//
//    The `TracingState` object in TLS can also be read/written via its Python
//    binding in `python_tracer.cpp`, and `get/setTracingState()` C++ APIs,
//    which are also exposed as `TORCH_API`.
//
// Two new switches were introduced since tracing function was moved out of
// VariableType:
//
// - `tracer::impl::set_dispatch_enabled()` API
//
//    Unlike the special `Autograd` dispatch key which is included in dispatch
//    key set by default, `Tracer` dispatch key is off by default. The
//    dispatching switch can be toggled via this new API.
//
// - `tracer::impl::NoTracerDispatchMode` guard
//
//    It's used to cover the old semantics of `AutoDispatchBelowADInplaceOrView`
//    after tracing was moved out of VariableType.
//
// Before tracing function was moved out of VariableType, tracing was enabled
// when the following conditions are satisfied:
//
//    1) `TracingState` object in TLS != null;
//       - Either inside the execution scope of `tracer::trace()`, or
//       - Eagerly called `setTracingState()` with non-null object.
//    2) Not inside `AutoDispatchBelowADInplaceOrView` scope;
//
// After:
//
//    1) `TracingState` object in TLS != null;
//    2) Has called `tracer::impl::set_dispatch_enabled(true)`;
//    3) Not inside `tracer::impl::NonDispatchGuard` scope;
//
// [TODOs]
//
// - `setTracingState()` v.s. `tracer::impl::set_dispatch_enabled()`
//
//   Currently `set_dispatch_enabled()` is set/unset inside `setTracingState()`
//   to keep the semantics exactly the same as before - it's confusing to keep
//   both switches, though. We should consider simplifying/limiting the exposed
//   `setTracingState()` Python/C++ APIs (and other APIs calling it) so that
//   these two can be unified.
//
// - `AutoDispatchBelowADInplaceOrView` v.s.
// `tracer::impl::NoTracerDispatchMode`
//
//   We don't need to always set both guards together to keep semantics
//   unchanged. For the follow use cases of `AutoDispatchBelowADInplaceOrView`
//   we don't need set the new tracer guard:
//
//   * Script-generated VariableType kernels. The guard is not necessary as
//     tracing is already disabled explicitly by `setTracingState(null)` in
//     generated TraceType kernels - we could keep it as is or use the new guard
//     instead.
//
//   * Custom ops. Will be handled by fallback kernel for `Tracer`.
//
//   * Functions that are not likely to be called in tracing context (no python
//     binding / not an operator), e.g.: all mobile forward() wrappers, test
//     binaries, and etc.
//
//   * Where new threads are spawned, e.g.: ATen/native/ConvolutionMM2d.cpp.
//     It's not necessary as tracing is off by default.
//
//   For the rest of cases we might need have both:
//
//   * Functions that might be reachable from eager mode python (especially
//     factory methods), e.g.:
//     `internal_new_from_data()` in `torch/csrc/utils/tensor_new.cpp`.
//     Without the new guard it will add `aten::empty` to the traced graph.
//
//   * Some manually maintained functions, e.g.:
//     `torch/csrc/autograd/VariableTypeManual.cpp`.
//     Set the new guard if it's not obvious whether `setTracingState(null)`
//     has been called before it reaches the `AutoDispatchBelowADInplaceOrView`
//     guard.
//
//   We might need tweak the usage of the new guard to optimize/fix things.
//   It should only affect the correctness of tracing function, because the
//   guard is essentially no-op when the master `setTracingState()` switch is
//   off.

// TODO: move this from `at::` to `jit::torch::` after
// `aten/src/ATen/cpp_custom_type_hack.h` is removed.

namespace at::tracer::impl {

inline bool is_dispatch_enabled() {
  return c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Tracer) &&
      !c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Tracer);
}

inline void set_dispatch_enabled(bool enabled) {
  TORCH_INTERNAL_ASSERT(
      !c10::impl::tls_is_dispatch_key_excluded(at::DispatchKey::Tracer),
      "Cannot enable tracing within the scope of NoTracerDispatchMode!");
  c10::impl::tls_set_dispatch_key_included(at::DispatchKey::Tracer, enabled);
}

struct NoTracerDispatchMode {
  c10::impl::ExcludeDispatchKeyGuard guard_{at::DispatchKey::Tracer};
};

} // namespace at::tracer::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `NoTracerDispatchMode`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/LocalDispatchKeySet.h`
- `c10/macros/Export.h`
- `c10/macros/Macros.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `TracerMode.h_docs.md`
- **Keyword Index**: `TracerMode.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `TracerMode.h_docs.md_docs.md`
- **Keyword Index**: `TracerMode.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

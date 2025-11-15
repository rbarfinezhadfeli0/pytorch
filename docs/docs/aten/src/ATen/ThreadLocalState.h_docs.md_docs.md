# Documentation: `docs/aten/src/ATen/ThreadLocalState.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/ThreadLocalState.h_docs.md`
- **Size**: 7,285 bytes (7.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/ThreadLocalState.h`

## File Metadata

- **Path**: `aten/src/ATen/ThreadLocalState.h`
- **Size**: 4,354 bytes (4.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <ATen/FuncTorchTLS.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/ThreadLocalPythonObjects.h>
#include <ATen/record_function.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace at {

// Thread local state contains values that are preserved across
// thread boundaries (e.g. at::launch/JIT fork, autograd).
// Note at::parallel_for doesn't preserve TLS across thread boundaries.
class TORCH_API ThreadLocalState {
 public:
  // Saves the thread local variables' values and
  // returns them as a ThreadLocalState
  ThreadLocalState();

  // set_grad_mode - force the value of the grad mode TLS in
  //  the current state object. This is used for example in the
  //  autograd engine.
  void set_grad_mode(bool enabled);

  // set_multithreading_enabled - force the value of the multithreadinmaximum
  // threads TLS in
  //  the current state object. This is used for example in the
  //  autograd engine.
  void set_multithreading_enabled(bool enabled);

  // Sets thread local variables in the current thread,
  // according to the thread boundary specified
  static void setThreadLocalState(const ThreadLocalState& state);

 private:
  c10::impl::LocalDispatchKeySet dispatch_key_;

  // ThreadLocalDebugInfo does not change after being created
  // with DebugInfoGuard
  std::shared_ptr<c10::ThreadLocalDebugInfo> debug_info_;

  // RecordFunction TLS
  RecordFunctionTLS rf_tls_;

  // TLS for out-of-tree functorch
  // See NOTE [functorch TLS in pytorch/pytorch] for why this needs to be a
  // pointer (spoiler alert: it's due to the indirection)
  // This needs to be a shared_ptr instead of a unique_ptr because
  // ThreadLocalState is copy-able and does indeed get copied. Maybe we can
  // consider adding an explicit copy constructor for ThreadLocalState in the
  // future but I didn't want to add one just for this.
  std::shared_ptr<const functorch::FuncTorchTLSBase> functorch_tls_;

  // TLS for AutogradModes
  AutogradState autograd_tls_;

  // TLS for enable_torch_dispatch_mode
  c10::impl::TorchDispatchModeTLS torch_dispatch_mode_state_;

  // TLS for enable_python_dispatcher
  c10::impl::PyInterpreter* python_dispatcher_state_;

  // TLS for __torch_function__ (mode and disable_torch_function)
  at::impl::PythonTorchFunctionTLS python_torch_function_state_;

  // TLS for saved tensors default hooks
  at::impl::SavedTensorDefaultHooksTLS saved_tensors_default_hooks_state_;

  bool functionalization_reapply_views_state_;

  bool dtensor_allow_implicit_replication_;

  // TLS for arbitrary python objects that is registered via hooks
  at::impl::ThreadLocalPythonObjects saved_objects_;

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && \
    !defined(BUILD_LITE_INTERPRETER)
  // TLS for autocast dtypes
  std::array<at::ScalarType, at::COMPILE_TIME_MAX_DEVICE_TYPES>
      autocast_dtypes_{};
#endif

  friend class ThreadLocalStateGuard;
};

// Guard to set and reset the thread local state
class TORCH_API ThreadLocalStateGuard {
 public:
  explicit ThreadLocalStateGuard(const ThreadLocalState& state)
      : prev_state_(ThreadLocalState()) {
    // set the given state across the thread boundary
    ThreadLocalState::setThreadLocalState(state);
  }
  ThreadLocalStateGuard(ThreadLocalStateGuard&& other) = delete;
  ThreadLocalStateGuard(const ThreadLocalStateGuard&) = delete;
  ThreadLocalStateGuard& operator=(const ThreadLocalStateGuard&) = delete;
  ThreadLocalStateGuard& operator=(ThreadLocalStateGuard&&) = delete;

  ~ThreadLocalStateGuard() {
    // restore previously set variables
    ThreadLocalState::setThreadLocalState(prev_state_);
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ThreadLocalState prev_state_;
};

template <typename T>
auto wrapPropagateTLSState(T callback) {
  return [tls_state = ThreadLocalState(),
          callback = std::move(callback)](auto&&... args) {
    ThreadLocalStateGuard g(tls_state);
    // Propagate value returned by callback().
    return callback(std::forward<decltype(args)>(args)...);
  };
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_API`, `ThreadLocalStateGuard`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/InferenceMode.h`
- `c10/core/impl/LocalDispatchKeySet.h`
- `c10/util/Exception.h`
- `c10/util/ThreadLocalDebugInfo.h`
- `ATen/FuncTorchTLS.h`
- `ATen/PythonTorchFunctionTLS.h`
- `ATen/SavedTensorHooks.h`
- `ATen/ThreadLocalPythonObjects.h`
- `ATen/record_function.h`
- `c10/core/impl/PythonDispatcherTLS.h`
- `c10/core/impl/TorchDispatchModeTLS.h`


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

- **File Documentation**: `ThreadLocalState.h_docs.md`
- **Keyword Index**: `ThreadLocalState.h_kw.md`
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

- **File Documentation**: `ThreadLocalState.h_docs.md_docs.md`
- **Keyword Index**: `ThreadLocalState.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

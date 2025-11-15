# Documentation: `aten/src/ATen/ThreadLocalState.cpp`

## File Metadata

- **Path**: `aten/src/ATen/ThreadLocalState.cpp`
- **Size**: 3,177 bytes (3.10 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ThreadLocalState.h>

#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#endif

#include <ATen/record_function.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/DTensorState.h>

namespace at {

ThreadLocalState::ThreadLocalState()
    : dispatch_key_(c10::impl::tls_local_dispatch_key_set()),
      debug_info_(c10::ThreadLocalDebugInfo::current()),
      rf_tls_(at::get_record_function_tls_()), functorch_tls_(functorch::getCopyOfFuncTorchTLS()),
      autograd_tls_(c10::AutogradState::get_tls_state()),
      torch_dispatch_mode_state_(c10::impl::TorchDispatchModeTLS::get_state()), python_dispatcher_state_(c10::impl::PythonDispatcherTLS::get_state()),
      python_torch_function_state_(at::impl::PythonTorchFunctionTLS::get_state()),
      saved_tensors_default_hooks_state_(at::SavedTensorDefaultHooks::get_tls_state()), functionalization_reapply_views_state_(at::functionalization::impl::getFunctionalizationReapplyViewsTLS()),
      dtensor_allow_implicit_replication_(at::get_dtensor_allow_implicit_replication()),
      saved_objects_(at::impl::ThreadLocalPythonObjects::get_state()) {
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
  for(size_t i=0; i<autocast_dtypes_.size(); i++) {
     autocast_dtypes_[i] = at::autocast::get_autocast_dtype(static_cast<at::DeviceType>(i));
  }
#endif
}

void ThreadLocalState::set_grad_mode(bool enabled) {
  autograd_tls_.set_grad_mode(enabled);
}

void ThreadLocalState::set_multithreading_enabled(bool enabled) {
  autograd_tls_.set_multithreading_enabled(enabled);
}

/* static */
void ThreadLocalState::setThreadLocalState(
    const ThreadLocalState& state) {
  // Note that setting the InferenceMode TLS in this function is ONLY ok because we always
  // restore the dispatch key set TLS at the same time.
  c10::AutogradState::set_tls_state(state.autograd_tls_);

  c10::impl::TorchDispatchModeTLS::set_state(state.torch_dispatch_mode_state_);

  at::impl::PythonTorchFunctionTLS::set_state(state.python_torch_function_state_);

  at::set_record_function_tls_(state.rf_tls_);

  at::SavedTensorDefaultHooks::set_tls_state(state.saved_tensors_default_hooks_state_);

  c10::impl::PythonDispatcherTLS::set_state(state.python_dispatcher_state_);

  at::set_dtensor_allow_implicit_replication(state.dtensor_allow_implicit_replication_);

  c10::ThreadLocalDebugInfo::_forceCurrentDebugInfo(state.debug_info_);

  c10::impl::_force_tls_local_dispatch_key_set(state.dispatch_key_);

  functorch::setFuncTorchTLS(state.functorch_tls_);

  at::functionalization::impl::setFunctionalizationReapplyViewsTLS(state.functionalization_reapply_views_state_);

  at::impl::ThreadLocalPythonObjects::set_state(state.saved_objects_);
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE) && !defined(BUILD_LITE_INTERPRETER)
  for(size_t i=0; i<state.autocast_dtypes_.size(); i++) {
     at::autocast::set_autocast_dtype(static_cast<at::DeviceType>(i), state.autocast_dtypes_[i]);
  }
#endif
}

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ThreadLocalState.h`
- `ATen/autocast_mode.h`
- `ATen/core/grad_mode.h`
- `ATen/record_function.h`
- `ATen/SavedTensorHooks.h`
- `ATen/FunctionalTensorWrapper.h`
- `ATen/DTensorState.h`


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

- **File Documentation**: `ThreadLocalState.cpp_docs.md`
- **Keyword Index**: `ThreadLocalState.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

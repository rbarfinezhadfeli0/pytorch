# Documentation: `docs/aten/src/ATen/core/PythonFallbackKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/PythonFallbackKernel.cpp_docs.md`
- **Size**: 11,206 bytes (10.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/PythonFallbackKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/core/PythonFallbackKernel.cpp`
- **Size**: 8,494 bytes (8.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <ATen/core/PythonFallbackKernel.h>
#include <c10/core/SafePyObject.h>
#include <ATen/record_function.h>

namespace {

// This TLS is used to track the state of the dispatcher to be able to restore
// it when calling back into python.
// It has the following invariant:
//  - It must be empty while python code is executed.
//  - It should only be set once even for multiple dispatcher calls that do not come
//    back to python.
// To achieve this, we ensure that the tls is empty by default and emptied again both when
// we call into user torch_dispatch or returning back to python after this call.

thread_local std::optional<c10::impl::LocalDispatchKeySet> tls_on_entry;

c10::impl::LocalDispatchKeySet safe_get_tls_on_entry() {
  TORCH_CHECK(tls_on_entry.has_value(), "Accessing torch dispatch state outside of '__torch_dispatch__' "
              "is not allowed.");
  return tls_on_entry.value();
}

// All the keys below the Python key
constexpr c10::DispatchKeySet after_Python_keyset = c10::DispatchKeySet(c10::DispatchKeySet::FULL) ^
  (c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python) |
   c10::DispatchKeySet(c10::DispatchKey::Python));


// This guard assumes that tls_on_entry has a value.
struct StashTLSOnEntryGuard {
public:
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  StashTLSOnEntryGuard(): saved_(tls_on_entry.value()) {
    tls_on_entry = std::nullopt;
  }
  StashTLSOnEntryGuard(const StashTLSOnEntryGuard&) = delete;
  StashTLSOnEntryGuard(StashTLSOnEntryGuard&&) = delete;
  StashTLSOnEntryGuard& operator=(const StashTLSOnEntryGuard&) = delete;
  StashTLSOnEntryGuard& operator=(StashTLSOnEntryGuard&&) = delete;

  ~StashTLSOnEntryGuard() {
    TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
    tls_on_entry = saved_;
  }

private:
  c10::impl::LocalDispatchKeySet saved_;
};

void pythonFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
  // c10::impl::ForceDispatchKeyGuard dispatcher_guard(tls_on_entry.value());
  // StashTLSOnEntryGuard stash_guard;
  c10::impl::ExcludeDispatchKeyGuard exclude_guard(after_Python_keyset);

  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();

  // If Torch Dispatch Mode is active, use its PyInterpreter for dispatch
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  if (mode_stack_len > 0) {
    RECORD_FUNCTION("PythonDispatchMode", torch::jit::last(*stack, num_arguments));
    const auto& cur_torch_dispatch_mode_state = c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    cur_torch_dispatch_mode_state->pyinterpreter()->dispatch(op, stack);
    return;
  }

  RECORD_FUNCTION("PythonSubclass", torch::jit::last(*stack, num_arguments));

  // Otherwise, find a PyInterpreter on a Tensor

  // It is safe to dispatch on the very first Tensor with a pyobj_interpreter
  // without checking the interpreters of any of the arguments, because when
  // we actually run dispatch(), we will take out PyObjects in the context
  // of that interpreter, and this will ensure that everyone is on the same
  // interpreter.
  bool tensors_with_python_key_present = false;
  c10::impl::PyInterpreter* interpreter = nullptr;
  for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
    if (ivalue.isTensor()) {
      auto* t = ivalue.unsafeToTensorImpl();
      if (t->key_set().has(c10::DispatchKey::Python)) {
        tensors_with_python_key_present = true;
      }

      if (!interpreter) {
        auto* t_interpreter = t->pyobj_slot()->pyobj_interpreter();
        if (t_interpreter) {
          interpreter = t_interpreter;
        }
      }
    } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
      // NB: use toListRef as it doesn't induce refcount bumps (toTensorListRef
      // is not a thing)
      for (const auto& nv : ivalue.toListRef()) {
        if (nv.isNone()) {
          continue;
        }

        auto* t = nv.unsafeToTensorImpl();
        if (t->key_set().has(c10::DispatchKey::Python)) {
          tensors_with_python_key_present = true;
        }

        if (!interpreter) {
          auto* t_interpreter = t->pyobj_slot()->pyobj_interpreter();
          if (t_interpreter) {
            interpreter = t_interpreter;
          }
        }
      }
    }
  }

  if (interpreter) {
    if (tensors_with_python_key_present) {
      (*interpreter)->dispatch(op, stack);
    } else {
      // At this point, there are no modes in the stack and no tensors with the python key.
      // so disable the python key before redispatching.
      // See https://github.com/pytorch/pytorch/issues/136565
      c10::DispatchKeySet keyset = dispatch_keys.remove(c10::DispatchKey::Python);

      // Remove Python key from the included set as well (modes add it there).
      c10::impl::LocalDispatchKeySet local_keyset = c10::impl::tls_local_dispatch_key_set();
      c10::impl::ForceDispatchKeyGuard no_python_guard(
        local_keyset.included_.remove(c10::DispatchKey::Python),
        local_keyset.excluded_
      );

      op.redispatchBoxed(keyset, stack);
    }
    return;
  }

  TORCH_INTERNAL_ASSERT(0, "Hit Python dispatch key but no arguments had PyInterpreter (no tensor args?)");
}

void pythonDispatcherFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  auto* state = c10::impl::PythonDispatcherTLS::get_state();
  TORCH_INTERNAL_ASSERT(state, "Hit PythonDispatcher dispatch key but PythonDispatcherTLS was not set");
  (*state)->python_dispatcher(op, dispatch_keys.remove(c10::DispatchKey::PythonDispatcher), stack);
}

void pythonTLSSnapshotFallback(const c10::OperatorHandle &op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // It is ok for the tls to be already set here.
  // It means that there are multiple calls into the dispatcher not originating from python code.
  // The guard below will properly ignore such calls.
  at::impl::MaybeSetTLSOnEntryGuard guard;

  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PythonTLSSnapshot), stack);
}

// The PreDispatch key gets a no-op fallback that just redispatches.
// The main way this key is used is that we can register a mode to it from python (e.g. TorchProxyDispatchMode, for pre_dispatch tracing)
// Can't this be a fallthrough kernel, instead of a fallback that just no-ops and redispatches?
// Unfortunately, no: we need a real kernel that is not a fallthrough, in order for the PythonDispatcher to interpose on it.
// Alternatively, we could have hardcoded this kernel (in C++) to directly call in TorchProxyDispatchMode.
// Doing that in C++ is a pain though, so it's done in python using the PythonDispatcher for convenience.
void preDispatchFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PreDispatch), stack);
}

} // anonymous namespace


namespace at::impl {

RestorePythonTLSSnapshot::RestorePythonTLSSnapshot() : saved_(safe_get_tls_on_entry()), guard_(safe_get_tls_on_entry()) {
  tls_on_entry = std::nullopt;
}

RestorePythonTLSSnapshot::~RestorePythonTLSSnapshot() {
  TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
  tls_on_entry = saved_;
}

MaybeSetTLSOnEntryGuard::MaybeSetTLSOnEntryGuard() {
  if (tls_on_entry.has_value()) {
    value_set_ = false;
  } else {
    value_set_ = true;
    tls_on_entry = c10::impl::tls_local_dispatch_key_set();
  }
}
MaybeSetTLSOnEntryGuard::~MaybeSetTLSOnEntryGuard() {
  if (value_set_) {
    TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
    tls_on_entry = std::nullopt;
  }
}


} // namespace at::impl

TORCH_LIBRARY_IMPL(_, Python, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}

TORCH_LIBRARY_IMPL(_, PythonDispatcher, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonDispatcherFallback>());
}

TORCH_LIBRARY_IMPL(_, PythonTLSSnapshot, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonTLSSnapshotFallback>());
}

TORCH_LIBRARY_IMPL(_, PreDispatch, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&preDispatchFallback>());
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `at`

**Classes/Structs**: `StashTLSOnEntryGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/TorchDispatchModeTLS.h`
- `c10/core/impl/PythonDispatcherTLS.h`
- `ATen/core/PythonFallbackKernel.h`
- `c10/core/SafePyObject.h`
- `ATen/record_function.h`


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

- **File Documentation**: `PythonFallbackKernel.cpp_docs.md`
- **Keyword Index**: `PythonFallbackKernel.cpp_kw.md`
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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `PythonFallbackKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `PythonFallbackKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

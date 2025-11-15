# Documentation: `c10/core/impl/TorchDispatchModeTLS.cpp`

## File Metadata

- **Path**: `c10/core/impl/TorchDispatchModeTLS.cpp`
- **Size**: 6,801 bytes (6.64 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/irange.h>

#include <utility>

namespace c10::impl {

thread_local static TorchDispatchModeTLS torchDispatchModeState;

bool TorchDispatchModeTLS::any_modes_set(bool skip_infra_modes) {
  if (!torchDispatchModeState.stack_.empty())
    return true;
  if (!skip_infra_modes) {
    for (const auto i : c10::irange(
             static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
      if (torchDispatchModeState.infra_modes_[i] != std::nullopt) {
        return true;
      }
    }
  }
  return false;
}

void TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
    std::shared_ptr<PyObject_TorchDispatchMode> mode) {
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  torchDispatchModeState.stack_.push_back(std::move(mode));
}

const std::shared_ptr<PyObject_TorchDispatchMode> TorchDispatchModeTLS::
    pop_stack() {
  std::shared_ptr<PyObject_TorchDispatchMode> out;
  if (!torchDispatchModeState.stack_.empty()) {
    out = torchDispatchModeState.stack_.back();
    torchDispatchModeState.stack_.pop_back();
  } else {
    for (int64_t i =
             static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS) - 1;
         i >= 0;
         --i) {
      if (torchDispatchModeState.infra_modes_[i].has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        out = std::move(torchDispatchModeState.infra_modes_[i].value());
        torchDispatchModeState.infra_modes_[i] = std::nullopt;
        break;
      }
    }
  }
  TORCH_CHECK(out, "trying to pop from empty mode stack");
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}
const std::
    tuple<std::shared_ptr<PyObject_TorchDispatchMode>, TorchDispatchModeKey>
    TorchDispatchModeTLS::pop_highest_infra_mode() {
  for (int64_t i = static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS) - 1;
       i >= 0;
       --i) {
    if (torchDispatchModeState.infra_modes_[i].has_value()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto out_mode = torchDispatchModeState.infra_modes_[i].value();
      torchDispatchModeState.infra_modes_[i] = std::nullopt;
      if (!any_modes_set()) {
        c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
        c10::impl::tls_set_dispatch_key_included(
            DispatchKey::PythonTLSSnapshot, false);
      }
      return std::make_tuple(
          std::move(out_mode), static_cast<TorchDispatchModeKey>(i));
    }
  }
  TORCH_CHECK(
      false, "Called pop_highest_infra_mode, but no infra modes were active.")
}

const std::shared_ptr<PyObject_TorchDispatchMode>& TorchDispatchModeTLS::
    get_stack_at(int64_t idx) {
  TORCH_CHECK(idx < stack_len(), "Tried to get stack at idx that's too big");
  // Our "logical" stack includes both:
  // - any user modes (the entire torchDispatchModeState.stack_)
  // - any infra modes (members of torchDispatchModeState.infra_modes_ that are
  // not None)

  // idx == 0 means the "bottom" of the stack, which starts with any infra
  // modes (iterating from lowest-priority to highest-priority).
  auto curr_idx = idx;
  for (const auto i :
       c10::irange(static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
    if (torchDispatchModeState.infra_modes_[i].has_value()) {
      if (curr_idx == 0) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        return torchDispatchModeState.infra_modes_[i].value();
      }
      curr_idx -= 1;
    }
  }
  // At this point, we're guaranteed that curr_idx < stack_.size()
  return torchDispatchModeState.stack_[curr_idx];
}

int64_t TorchDispatchModeTLS::stack_len() {
  auto stack_len = static_cast<int64_t>(torchDispatchModeState.stack_.size());
  int64_t infra_modes_len = 0;
  for (const auto i :
       c10::irange(static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
    if (torchDispatchModeState.infra_modes_[i] != std::nullopt) {
      infra_modes_len += 1;
    }
  }
  return stack_len + infra_modes_len;
}

const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>
TorchDispatchModeTLS::get_mode(TorchDispatchModeKey mode_key) {
  return torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
}

void TorchDispatchModeTLS::set_mode(
    const std::shared_ptr<PyObject_TorchDispatchMode>& mode,
    TorchDispatchModeKey mode_key) {
  TORCH_CHECK(
      torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] ==
          std::nullopt,
      "trying to set the current ",
      to_string(mode_key),
      ", but one already exists");

  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }

  torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] = mode;
}

const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>
TorchDispatchModeTLS::unset_mode(TorchDispatchModeKey mode_key) {
  auto out = torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
  torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] =
      std::nullopt;
  if (out.has_value() && !any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}

const TorchDispatchModeTLS& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::set_state(TorchDispatchModeTLS state) {
  torchDispatchModeState = std::move(state);
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  } else {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
}

// UTIL

bool dispatch_mode_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      TorchDispatchModeTLS::any_modes_set();
}

std::string to_string(TorchDispatchModeKey mode_key) {
  switch (mode_key) {
    case TorchDispatchModeKey::PROXY:
      return "ProxyTorchDispatchMode";
    case TorchDispatchModeKey::FAKE:
      return "FakeTensorMode";
    default:
      return "UNKNOWN_MODE";
  }
}

} // namespace c10::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/DispatchKey.h`
- `c10/core/impl/LocalDispatchKeySet.h`
- `c10/core/impl/TorchDispatchModeTLS.h`
- `c10/util/irange.h`
- `utility`


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

Files in the same folder (`c10/core/impl`):

- [`LocalDispatchKeySet.h_docs.md`](./LocalDispatchKeySet.h_docs.md)
- [`PythonDispatcherTLS.h_docs.md`](./PythonDispatcherTLS.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)
- [`alloc_cpu.h_docs.md`](./alloc_cpu.h_docs.md)
- [`PythonDispatcherTLS.cpp_docs.md`](./PythonDispatcherTLS.cpp_docs.md)
- [`InlineEvent.h_docs.md`](./InlineEvent.h_docs.md)
- [`PyInterpreterHooks.h_docs.md`](./PyInterpreterHooks.h_docs.md)
- [`LocalDispatchKeySet.cpp_docs.md`](./LocalDispatchKeySet.cpp_docs.md)
- [`DeviceGuardImplInterface.cpp_docs.md`](./DeviceGuardImplInterface.cpp_docs.md)


## Cross-References

- **File Documentation**: `TorchDispatchModeTLS.cpp_docs.md`
- **Keyword Index**: `TorchDispatchModeTLS.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

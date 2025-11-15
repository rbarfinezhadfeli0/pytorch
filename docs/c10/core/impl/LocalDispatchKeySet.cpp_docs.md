# Documentation: `c10/core/impl/LocalDispatchKeySet.cpp`

## File Metadata

- **Path**: `c10/core/impl/LocalDispatchKeySet.cpp`
- **Size**: 4,153 bytes (4.06 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace c10::impl {

// NB: POD, must be zero initialized!
// Note [TLS Initialization]
// We wanted raw_local_dispatch_key_set to be initialized with non-zero state
// e.g. BackendSelect and ADInplaceOrView in included set.  But certain Windows
// compiler (e.g the one used in ARVR tests) only allow TLS to be
// zero-initialized. To preserve the invariant that raw TLS storage of the
// default state is zero, we obtain the actual include keyset by XORing
// raw_local_dispatch_key_set.included_ with c10::default_included_set.  This
// logic is encapsulated in struct PODLocalDispatchKeySet.
thread_local PODLocalDispatchKeySet raw_local_dispatch_key_set;

#if defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)
LocalDispatchKeySet tls_local_dispatch_key_set() {
  return raw_local_dispatch_key_set;
}
#endif // defined(_MSC_VER) || defined(C10_ANDROID) || defined(C10_IPHONE)

void _force_tls_local_dispatch_key_set(LocalDispatchKeySet key_set) {
  raw_local_dispatch_key_set.set_included(key_set.included_);
  raw_local_dispatch_key_set.set_excluded(key_set.excluded_);
}

// An RAII guard could snapshot and restore the entire state (entire
// DispatchKeySet) as opposed to only snapshotting and restoring the state of
// its assigned DispatchKeySet. I'm not sure which is better.  If only the RAII
// API is used, the two choices are not distinguishable.
//
// However, if the guard chooses to snapshot and restore the entire
// DispatchKeySet, the interaction with the non-RAII API changes.  Consider this
// sequence of events:
// - An RAII guard is declared for a particular DispatchKeySet, but snapshots
// the entire
//   current DispatchKeySet.
// - A call to the non-RAII API changes the state for DispatchKeys outside the
// assigned
//   set.
// - The RAII guard goes out of scope, restoring the entire DispatchKeySet it
// snapshotted
//   (which restores the state for its own assigned DispatchKey and wipes out
//   the state for the other DispatchKeys set by the non-RAII API).

// RAII API

IncludeDispatchKeyGuard::IncludeDispatchKeyGuard(DispatchKeySet include)
    : tls_(&raw_local_dispatch_key_set), include_(include - tls_->included()) {
  if (!include_.empty()) {
    tls_->set_included(tls_->included() | include_);
  }
}

IncludeDispatchKeyGuard::~IncludeDispatchKeyGuard() {
  if (!include_.empty()) {
    tls_->set_included(tls_->included() - include_);
  }
}

ExcludeDispatchKeyGuard::ExcludeDispatchKeyGuard(DispatchKeySet exclude)
    : tls_(&raw_local_dispatch_key_set), exclude_(exclude - tls_->excluded()) {
  if (!exclude_.empty()) {
    tls_->set_excluded(tls_->excluded() | exclude_);
  }
}

ExcludeDispatchKeyGuard::~ExcludeDispatchKeyGuard() {
  if (!exclude_.empty()) {
    tls_->set_excluded(tls_->excluded() - exclude_);
  }
}

// Non-RAII API
// Please prefer using the RAII API. See declarations in LocalDispatchKeySet.h
// for details.

bool tls_is_dispatch_key_excluded(DispatchKey x) {
  return raw_local_dispatch_key_set.excluded().has(x);
}

void tls_set_dispatch_key_excluded(DispatchKey x, bool desired_state) {
  auto* tls = &raw_local_dispatch_key_set;
  bool current_state = tls->excluded().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls->set_excluded(tls->excluded().add(x));
    } else {
      tls->set_excluded(tls->excluded().remove(x));
    }
  }
}

bool tls_is_dispatch_key_included(DispatchKey x) {
  return raw_local_dispatch_key_set.included().has(x);
}

void tls_set_dispatch_key_included(DispatchKey x, bool desired_state) {
  auto* tls = &raw_local_dispatch_key_set;
  bool current_state = tls->included().has(x);
  if (desired_state != current_state) {
    if (desired_state) {
      tls->set_included(tls->included().add(x));
    } else {
      tls->set_included(tls->included().remove(x));
    }
  }
}

bool tls_is_dispatch_keyset_excluded(DispatchKeySet ks) {
  return raw_local_dispatch_key_set.excluded().isSupersetOf(ks);
}

bool tls_is_dispatch_keyset_included(DispatchKeySet ks) {
  return raw_local_dispatch_key_set.included().isSupersetOf(ks);
}
} // namespace c10::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `PODLocalDispatchKeySet`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



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

Files in the same folder (`c10/core/impl`):

- [`LocalDispatchKeySet.h_docs.md`](./LocalDispatchKeySet.h_docs.md)
- [`TorchDispatchModeTLS.cpp_docs.md`](./TorchDispatchModeTLS.cpp_docs.md)
- [`PythonDispatcherTLS.h_docs.md`](./PythonDispatcherTLS.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)
- [`alloc_cpu.h_docs.md`](./alloc_cpu.h_docs.md)
- [`PythonDispatcherTLS.cpp_docs.md`](./PythonDispatcherTLS.cpp_docs.md)
- [`InlineEvent.h_docs.md`](./InlineEvent.h_docs.md)
- [`PyInterpreterHooks.h_docs.md`](./PyInterpreterHooks.h_docs.md)
- [`DeviceGuardImplInterface.cpp_docs.md`](./DeviceGuardImplInterface.cpp_docs.md)


## Cross-References

- **File Documentation**: `LocalDispatchKeySet.cpp_docs.md`
- **Keyword Index**: `LocalDispatchKeySet.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

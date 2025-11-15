# Documentation: `docs/aten/src/ATen/SavedTensorHooks.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/SavedTensorHooks.cpp_docs.md`
- **Size**: 5,165 bytes (5.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/SavedTensorHooks.cpp`

## File Metadata

- **Path**: `aten/src/ATen/SavedTensorHooks.cpp`
- **Size**: 2,620 bytes (2.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/SavedTensorHooks.h>
#include <c10/util/Exception.h>
#include <stack>
#include <utility>
#include <c10/core/SafePyObject.h>

namespace at {

namespace {
  thread_local impl::SavedTensorDefaultHooksTLS tls;

  // This flag is set to true the first time default hooks are registered
  // and left at true for the rest of the execution.
  // It's an optimization so that users who never use default hooks don't need to
  // read the thread_local variables pack_hook_ and unpack_hook_.
  bool is_initialized(false);
}

static void assertSavedTensorHooksNotDisabled() {
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  TORCH_CHECK(SavedTensorDefaultHooks::is_enabled(), tls.disabled_error_message.value());
}

bool SavedTensorDefaultHooks::is_enabled() {
  // See NOTE: [disabled_error_message invariant]
  return !tls.disabled_error_message.has_value();
}

void SavedTensorDefaultHooks::disable(const std::string& message, const bool fail_if_non_empty) {
  tls.disabled_error_message = message;
  if (fail_if_non_empty && !tls.stack.empty()) {
    assertSavedTensorHooksNotDisabled();
  }
}

void SavedTensorDefaultHooks::enable() {
  tls.disabled_error_message = std::nullopt;
}

/* static */ bool SavedTensorDefaultHooks::set_tracing(bool is_tracing) {
  bool prior  = tls.is_tracing;
  tls.is_tracing = is_tracing;
  return prior;
}

const std::optional<std::string>& SavedTensorDefaultHooks::get_disabled_error_message() {
  return tls.disabled_error_message;
}

const impl::SavedTensorDefaultHooksTLS& SavedTensorDefaultHooks::get_tls_state() {
  return tls;
}

void SavedTensorDefaultHooks::set_tls_state(const impl::SavedTensorDefaultHooksTLS& state) {
  tls = state;
}

void SavedTensorDefaultHooks::lazy_initialize() {
  is_initialized = true;
}

void SavedTensorDefaultHooks::push_hooks(SafePyObject pack_hook, SafePyObject unpack_hook) {
  TORCH_INTERNAL_ASSERT(is_initialized);
  assertSavedTensorHooksNotDisabled();
  tls.stack.emplace(std::move(pack_hook), std::move(unpack_hook));
}

std::pair<SafePyObject, SafePyObject> SavedTensorDefaultHooks::pop_hooks() {
  TORCH_INTERNAL_ASSERT(is_initialized && !tls.stack.empty());
  std::pair<SafePyObject, SafePyObject> hooks = std::move(tls.stack.top());
  tls.stack.pop();
  return hooks;
}

std::optional<std::pair<SafePyObject, SafePyObject>> SavedTensorDefaultHooks::get_hooks(bool ignore_is_tracing) {
  // For tls.is_tracing, see NOTE: [Deferring tensor pack/unpack hooks until runtime]
  if (!is_initialized || tls.stack.empty() || (!ignore_is_tracing && tls.is_tracing)) {
    return std::nullopt;
  }
  return tls.stack.top();
}

}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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

- `ATen/SavedTensorHooks.h`
- `c10/util/Exception.h`
- `stack`
- `utility`
- `c10/core/SafePyObject.h`


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

- **File Documentation**: `SavedTensorHooks.cpp_docs.md`
- **Keyword Index**: `SavedTensorHooks.cpp_kw.md`
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

- **File Documentation**: `SavedTensorHooks.cpp_docs.md_docs.md`
- **Keyword Index**: `SavedTensorHooks.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

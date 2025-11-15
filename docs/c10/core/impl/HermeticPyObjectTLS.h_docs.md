# Documentation: `c10/core/impl/HermeticPyObjectTLS.h`

## File Metadata

- **Path**: `c10/core/impl/HermeticPyObjectTLS.h`
- **Size**: 2,546 bytes (2.49 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Export.h>
#include <atomic>

namespace c10::impl {

// This TLS controls whether or not we permanently associate PyObject
// with Tensor the first time it is allocated.  When hermetic PyObject
// TLS is enabled (state is true), we DO NOT save PyObjects to Tensor,
// meaning you get a distinct PyObject whenever you execute the code in
// question.
struct C10_API HermeticPyObjectTLS {
  static void set_state(bool state);
  static bool get_state() {
    // Hypothetical fastpath if torchdeploy/multipy // codespell:ignore multipy
    // isn't used. Per
    // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
    // this qualifies relaxed access because it is a single-location data
    // structure (only the boolean here).
    //
    // Forgetting about data races for a moment, is there a logical race?
    //
    //  - Boolean only ever transitions from false to true.  So the
    //    critical situation is when one interpreter is already running
    //    when a second interpreter switches haveState from false to true.
    //
    //  - The first interpreter is indifferent whether or not it sees
    //    hasState true/false; obviously false works (this is what the
    //    interpreter was previously using; more directly, the interpreter
    //    calls into itself as the handler, so being hermetic is not
    //    required), and true simply means serviced python operator calls will
    //    be hermetic; in these cases it is expected to be functionally
    //    equivalent.
    //
    //  - The second interpreter MUST see hasState true (as its requests will
    //    be forwarded to the first interpreter), but it is assumed that there
    //    is a synchronization between the interpreter initialization, and
    //    when we actually perform operations, so it is guaranteed to see
    //    hasState true.
    //
    // QED.
    //
    // This fastpath is currently disabled so that we can more easily test that
    // hermetic mode works correctly even on stock build of PyTorch.
    if (false && !haveState_.load(std::memory_order_relaxed))
      return false;
    return get_tls_state();
  }
  // Call this from the multipy/torchdeploy // codespell:ignore multipy
  // top level
  static void init_state();

 private:
  // This only flipped once from false to true during
  // torchdeploy/multipy initialization, // codespell:ignore multipy
  // and never again.
  static std::atomic<bool> haveState_;
  static bool get_tls_state();
};

} // namespace c10::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Export.h`
- `atomic`


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

Files in the same folder (`c10/core/impl`):

- [`LocalDispatchKeySet.h_docs.md`](./LocalDispatchKeySet.h_docs.md)
- [`TorchDispatchModeTLS.cpp_docs.md`](./TorchDispatchModeTLS.cpp_docs.md)
- [`PythonDispatcherTLS.h_docs.md`](./PythonDispatcherTLS.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)
- [`alloc_cpu.h_docs.md`](./alloc_cpu.h_docs.md)
- [`PythonDispatcherTLS.cpp_docs.md`](./PythonDispatcherTLS.cpp_docs.md)
- [`InlineEvent.h_docs.md`](./InlineEvent.h_docs.md)
- [`PyInterpreterHooks.h_docs.md`](./PyInterpreterHooks.h_docs.md)
- [`LocalDispatchKeySet.cpp_docs.md`](./LocalDispatchKeySet.cpp_docs.md)
- [`DeviceGuardImplInterface.cpp_docs.md`](./DeviceGuardImplInterface.cpp_docs.md)


## Cross-References

- **File Documentation**: `HermeticPyObjectTLS.h_docs.md`
- **Keyword Index**: `HermeticPyObjectTLS.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

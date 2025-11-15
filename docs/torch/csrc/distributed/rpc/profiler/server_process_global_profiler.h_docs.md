# Documentation: `torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h`
- **Size**: 4,399 bytes (4.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <shared_mutex>
#include <utility>

#include <torch/csrc/autograd/profiler.h>

namespace torch::distributed::rpc::profiler::processglobal {

using namespace torch::autograd::profiler;

// Process global profiler state.
//
// This class holds information about a profiling range, from "enable" to
// "disable".
// An instance of this ``State`` will be
// pushed into a global stack, so nested profiling range is supported.
//
// It has 2 members.
// One is ``autograd::profiler::ProfilerConfig``. It's set by user and
// will be copied to thread-local profiler state of RPC threads.
// The other is a container that aggregates recorded
// ``autograd::profiler::Event``s from all thread-local profilers on RPC
// threads.
class State {
 public:
  explicit State(ProfilerConfig config) : config_(std::move(config)) {}
  ~State() = default;

  const ProfilerConfig& config() const {
    return config_;
  }

  void pushResult(thread_event_lists result) {
    std::unique_lock<std::mutex> lock(resultsMutex_);

    // NB: When a thread wants to push an entry into the this container,
    // main control logic might have exited the process-global profile range.
    results_.emplace_back(std::move(result));
  }

  std::vector<thread_event_lists> results();

 private:
  // Each result comes from a profile range. In each profile range, there is a
  // "__profiler_start" marker event that all following events calculate time
  // relative to it, so it's required to call
  // parse_cpu_trace(result) for results of all profile range.
  std::mutex resultsMutex_;
  std::vector<thread_event_lists> results_;
  const ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
};

class StateStackEntry;

#if defined(__MACH__)
// Compiler error: 'shared_timed_mutex' is unavailable: introduced in
// macOS 10.12
using mutexType = std::mutex;
// Compiler error: 'shared_lock' is unavailable: introduced in
// macOS 10.12
using rLockType = std::unique_lock<std::mutex>;
using wLockType = std::unique_lock<std::mutex>;
#else
using mutexType = std::shared_timed_mutex;
using rLockType = std::shared_lock<std::shared_timed_mutex>;
using wLockType = std::unique_lock<std::shared_timed_mutex>;
#endif

// This is the global stack of ``State``s.
TORCH_API extern std::shared_ptr<StateStackEntry> currentStateStackEntryPtr;
TORCH_API extern mutexType currentStateStackEntryMutex;

// This class is used to implement a stack of ``State``s.
// It has 2 members.
// One is `prevPtr`, a shared_ptr pointing to previous element in the
// stack.
// The other is ``statePtr``, a shared_ptr pointing to ``State``.
class StateStackEntry {
 public:
  StateStackEntry(
      std::shared_ptr<StateStackEntry> prevPtr,
      std::shared_ptr<State> statePtr)
      : prevPtr_(std::move(prevPtr)), statePtr_(std::move(statePtr)) {}

  static void pushRange(std::shared_ptr<State> profilerProcessGlobalStatePtr);
  static std::shared_ptr<State> popRange();

  static std::shared_ptr<StateStackEntry> current() {
    rLockType rlock(currentStateStackEntryMutex);

    return currentStateStackEntryPtr;
  }

  std::shared_ptr<StateStackEntry> prevPtr() const {
    return prevPtr_;
  }

  std::shared_ptr<State> statePtr() const {
    return statePtr_;
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::shared_ptr<StateStackEntry> prevPtr_{nullptr};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::shared_ptr<State> statePtr_{nullptr};
};

// Push the result to ``State``s of current profile range and recursively outer
// profile ranges.
TORCH_API void pushResultRecursive(
    std::shared_ptr<StateStackEntry> stateStackEntryPtr,
    const thread_event_lists& result);

// User-facing API.
//
// Enter a server-side process-global profiling range.
// Profiling range can be neste, so it's ok to call this API for multiple
// times. This enables all RPC threads running server-side request callbacks.
TORCH_API void enableServer(const ProfilerConfig& new_config);
//
// Exit a server-side process-global profiling range.
// Profiling range can be neste, so it's possible that profiler is still on
// after calling this API.
// This enables all RPC threads running server-side request callbacks.
TORCH_API std::vector<thread_event_lists> disableServer();

} // namespace torch::distributed::rpc::profiler::processglobal

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `holds`, `State`, `StateStackEntry`, `is`, `StateStackEntry`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc/profiler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `shared_mutex`
- `utility`
- `torch/csrc/autograd/profiler.h`


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

Files in the same folder (`torch/csrc/distributed/rpc/profiler`):

- [`remote_profiler_manager.cpp_docs.md`](./remote_profiler_manager.cpp_docs.md)
- [`remote_profiler_manager.h_docs.md`](./remote_profiler_manager.h_docs.md)
- [`server_process_global_profiler.cpp_docs.md`](./server_process_global_profiler.cpp_docs.md)


## Cross-References

- **File Documentation**: `server_process_global_profiler.h_docs.md`
- **Keyword Index**: `server_process_global_profiler.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

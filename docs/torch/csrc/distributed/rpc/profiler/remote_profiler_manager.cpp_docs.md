# Documentation: `torch/csrc/distributed/rpc/profiler/remote_profiler_manager.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/profiler/remote_profiler_manager.cpp`
- **Size**: 3,068 bytes (3.00 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch::distributed::rpc {
const std::string REMOTE_PROFILING_KEY_PREFIX = "#remote_op: ";
constexpr int kAutoIncrementBits = 48;
/*static */ thread_local std::optional<std::string>
    RemoteProfilerManager::currentThreadLocalKey_ = std::nullopt;
/*static */ RemoteProfilerManager& RemoteProfilerManager::getInstance() {
  static RemoteProfilerManager* handler = new RemoteProfilerManager();
  return *handler;
}

void RemoteProfilerManager::setCurrentKey(std::string key) {
  // We should not allow overriding the current key, it needs to be committed
  // with writeKey() explicitly first.
  if (RemoteProfilerManager::currentThreadLocalKey_) {
    TORCH_CHECK(
        false,
        "Cannot call RemoteProfilerManager::setCurrentKey when current key is already set.");
  }
  currentThreadLocalKey_ = std::move(key);
}

bool RemoteProfilerManager::isCurrentKeySet() const {
  return currentThreadLocalKey_.has_value();
}

void RemoteProfilerManager::unsetCurrentKey() {
  currentThreadLocalKey_ = std::nullopt;
}

void RemoteProfilerManager::eraseKey(const ProfilingId& globallyUniqueId) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  profiledRpcKeys_.erase(it);
}

std::string RemoteProfilerManager::retrieveRPCProfilingKey(
    const ProfilingId& globallyUniqueId) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = profiledRpcKeys_.find(globallyUniqueId);
  TORCH_INTERNAL_ASSERT(it != profiledRpcKeys_.end());
  return it->second;
}

ProfilingId RemoteProfilerManager::getNextProfilerId() {
  auto localId = getNextLocalId();
  auto localWorkerId = RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_;
  auto globallyUniqueId =
      torch::distributed::rpc::ProfilingId(localWorkerId, localId);
  return globallyUniqueId;
}

local_id_t RemoteProfilerManager::getNextLocalId() {
  std::lock_guard<std::mutex> guard(mutex_);
  return currentLocalId_++;
}

std::string& RemoteProfilerManager::getCurrentProfilingKey() {
  TORCH_CHECK(
      RemoteProfilerManager::currentThreadLocalKey_,
      "Must set currentThreadLocalKey_ before calling getCurrentProfilingKey");
  return *currentThreadLocalKey_;
}

void RemoteProfilerManager::saveRPCKey(
    ProfilingId globallyUniqueId,
    const std::string& rpcProfilingKey) {
  std::lock_guard<std::mutex> guard(mutex_);
  profiledRpcKeys_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(globallyUniqueId),
      std::forward_as_tuple(rpcProfilingKey));
}

RemoteProfilerManager::RemoteProfilerManager() {
  auto workerId =
      static_cast<int64_t>(RpcAgent::getCurrentRpcAgent()->getWorkerInfo().id_);
  currentLocalId_ = workerId << kAutoIncrementBits;
}
} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc/profiler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h`
- `torch/csrc/distributed/rpc/rpc_agent.h`
- `torch/csrc/distributed/rpc/rpc_command_base.h`
- `torch/csrc/jit/serialization/pickle.h`
- `torch/csrc/utils/byte_order.h`


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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/rpc/profiler`):

- [`server_process_global_profiler.h_docs.md`](./server_process_global_profiler.h_docs.md)
- [`remote_profiler_manager.h_docs.md`](./remote_profiler_manager.h_docs.md)
- [`server_process_global_profiler.cpp_docs.md`](./server_process_global_profiler.cpp_docs.md)


## Cross-References

- **File Documentation**: `remote_profiler_manager.cpp_docs.md`
- **Keyword Index**: `remote_profiler_manager.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/csrc/jit/runtime/logging.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/logging.h_docs.md`
- **Size**: 5,342 bytes (5.22 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/logging.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/logging.h`
- **Size**: 2,614 bytes (2.55 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/csrc/Export.h>

namespace torch::jit::logging {

class LoggerBase {
 public:
  TORCH_API virtual void addStatValue(
      const std::string& stat_name,
      int64_t val) = 0;
  virtual ~LoggerBase() = default;
};

TORCH_API LoggerBase* getLogger();
TORCH_API LoggerBase* setLogger(LoggerBase* logger);

// No-op logger. This is the default and is meant to incur almost no runtime
// overhead.

class NoopLogger : public LoggerBase {
 public:
  void addStatValue(
      const std::string& stat_name [[maybe_unused]],
      int64_t val [[maybe_unused]]) override {}
  ~NoopLogger() override = default;
};

// Trivial locking logger. Pass in an instance of this to setLogger() to use it.
// This keeps track of the sum of all statistics.
//
// NOTE: this is not written in a scalable way and should probably only be used
// in the single-threaded case or for testing.
class TORCH_API LockingLogger : public LoggerBase {
 public:
  void addStatValue(const std::string& stat_name, int64_t val) override;
  virtual int64_t getCounterValue(const std::string& name) const;
  enum class AggregationType { SUM = 0, AVG = 1 };
  void setAggregationType(const std::string& stat_name, AggregationType type);
  ~LockingLogger() override = default;

 private:
  mutable std::mutex m;
  struct RawCounter {
    RawCounter() = default;
    int64_t sum{0};
    size_t count{0};
  };
  std::unordered_map<std::string, RawCounter> raw_counters;
  std::unordered_map<std::string, AggregationType> agg_types;
};

// Make this struct so the timer internals are opaque to the user.
struct JITTimePoint {
  std::chrono::time_point<std::chrono::high_resolution_clock> point;
};

TORCH_API JITTimePoint timePoint();
TORCH_API void recordDurationSince(
    const std::string& name,
    const JITTimePoint& tp);

namespace runtime_counters {
constexpr const char* GRAPH_EXECUTORS_CONSTRUCTED =
    "pytorch_runtime.graph_executors_constructed";
constexpr const char* GRAPH_EXECUTOR_INVOCATIONS =
    "pytorch_runtime.graph_executor_invocations";
constexpr const char* EXECUTION_PLAN_CACHE_HIT =
    "pytorch_runtime.execution_plan_cache_hit";
constexpr const char* EXECUTION_PLAN_CACHE_MISS =
    "pytorch_runtime.execution_plan_cache_miss";

inline std::vector<const char*> allRuntimeCounters() {
  return {
      GRAPH_EXECUTORS_CONSTRUCTED,
      GRAPH_EXECUTOR_INVOCATIONS,
      EXECUTION_PLAN_CACHE_HIT,
      EXECUTION_PLAN_CACHE_MISS};
}

} // namespace runtime_counters

} // namespace torch::jit::logging

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `runtime_counters`

**Classes/Structs**: `LoggerBase`, `NoopLogger`, `TORCH_API`, `AggregationType`, `RawCounter`, `so`, `JITTimePoint`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `chrono`
- `mutex`
- `string`
- `unordered_map`
- `vector`
- `torch/csrc/Export.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)


## Cross-References

- **File Documentation**: `logging.h_docs.md`
- **Keyword Index**: `logging.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `logging.h_docs.md_docs.md`
- **Keyword Index**: `logging.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

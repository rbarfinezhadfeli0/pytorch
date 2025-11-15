# Documentation: `docs/torch/csrc/profiler/kineto_shim.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/profiler/kineto_shim.h_docs.md`
- **Size**: 6,578 bytes (6.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/profiler/kineto_shim.h`

## File Metadata

- **Path**: `torch/csrc/profiler/kineto_shim.h`
- **Size**: 3,915 bytes (3.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <memory>
#include <string>

// Skip Kineto dependency on mobile unless explicitly asked for.
// When is it explicitly asked for?
//   KinetoEdgeCPUProfiler uses KinetoProfiler for cpu
//   event profiling. This has a dependency on cpu only libkineto
#if defined(USE_KINETO) && defined(C10_MOBILE) && \
    !defined(EDGE_PROFILER_USE_KINETO)
#undef USE_KINETO
#endif

#include <ActivityType.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/api.h>

#ifdef USE_KINETO
// Forward declarations so we don't have to include `libkineto.h` in a header.
namespace libkineto {
class GenericTraceActivity;
struct CpuTraceBuffer;
class ActivityTraceInterface;
} // namespace libkineto
#endif

namespace torch {
namespace profiler {

#ifdef USE_KINETO
constexpr bool kKinetoAvailable{true};
#else
constexpr bool kKinetoAvailable{false};
#endif

namespace impl::kineto {

// ----------------------------------------------------------------------------
// -- Interface (Does not require Kineto) -------------------------------------
// ----------------------------------------------------------------------------
struct DeviceAndResource {
  int32_t device;
  int32_t resource;
};
const DeviceAndResource kineto_ids();

#ifdef USE_KINETO
using trace_t = libkineto::CpuTraceBuffer;
using interface_trace_t = libkineto::ActivityTraceInterface;
using activity_t = libkineto::GenericTraceActivity;
#else
struct DummyTraceBuffer {};
struct DummyTraceInterface {};

using trace_t = DummyTraceBuffer;
using interface_trace_t = DummyTraceBuffer;
struct activity_t;
#endif // USE_KINETO

void addMetadata(
    activity_t* activity,
    const std::string& key,
    const std::string& value);

// Wraps: libkineto::CpuTraceBuffer
struct TraceWrapper {
  TraceWrapper(const int64_t start_time, const std::string& name);

  // The caller is expected to hold a mutex when calling `addCPUActivity`.
  activity_t* addCPUActivity(
      const std::string& name,
      const libkineto::ActivityType type,
      const DeviceAndResource device_and_resource,
      const uint64_t correlation_id,
      const int64_t start_time,
      const int64_t end_time);

  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;

  std::unique_ptr<trace_t>& get() {
    return cpu_trace_;
  }

 private:
  std::unique_ptr<trace_t> cpu_trace_;
};

// Wraps libkineto::ActivityTraceInterface
struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t>&& trace);
  ActivityTraceWrapper() = default;
  explicit operator bool() const;
  void save(const std::string& path);

  const std::unique_ptr<interface_trace_t>& get() {
    return trace_;
  }

 private:
  std::unique_ptr<interface_trace_t> trace_;
#ifdef USE_KINETO
  bool saved_ = false; // Kineto's save is destructive
#endif
};

using ActivitySet = std::set<torch::autograd::profiler::ActivityType>;
void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config,
    const std::string& trace_id = "");

void toggleCollectionDynamic(const bool enable);
void startTrace();
ActivityTraceWrapper stopTrace();
void pushCorrelationId(uint64_t correlation_id);
void pushUserCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void popUserCorrelationId();
void recordThreadInfo();
bool collectivesProfilerExists();

void logInvariantViolation(
    const std::string& assertion,
    const std::string& error,
    const std::string& profile_id,
    const std::string& group_profile_id);

} // namespace impl::kineto

} // namespace profiler

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type);

TORCH_API void addMetadataJson(
    const std::string& key,
    const std::string& value);

TORCH_API void profilerStep();

} // namespace autograd::profiler

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `profiler`, `impl`, `libkineto`, `autograd`

**Classes/Structs**: `GenericTraceActivity`, `CpuTraceBuffer`, `ActivityTraceInterface`, `DeviceAndResource`, `DummyTraceBuffer`, `DummyTraceInterface`, `activity_t`, `TraceWrapper`, `ActivityTraceWrapper`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `memory`
- `string`
- `ActivityType.h`
- `torch/csrc/Export.h`
- `torch/csrc/profiler/api.h`


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

Files in the same folder (`torch/csrc/profiler`):

- [`perf-inl.h_docs.md`](./perf-inl.h_docs.md)
- [`perf.cpp_docs.md`](./perf.cpp_docs.md)
- [`kineto_client_interface.cpp_docs.md`](./kineto_client_interface.cpp_docs.md)
- [`combined_traceback.h_docs.md`](./combined_traceback.h_docs.md)
- [`collection.h_docs.md`](./collection.h_docs.md)
- [`kineto_shim.cpp_docs.md`](./kineto_shim.cpp_docs.md)
- [`combined_traceback.cpp_docs.md`](./combined_traceback.cpp_docs.md)
- [`kineto_client_interface.h_docs.md`](./kineto_client_interface.h_docs.md)
- [`perf.h_docs.md`](./perf.h_docs.md)


## Cross-References

- **File Documentation**: `kineto_shim.h_docs.md`
- **Keyword Index**: `kineto_shim.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/profiler`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/profiler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/profiler`):

- [`containers.h_docs.md_docs.md`](./containers.h_docs.md_docs.md)
- [`perf-inl.h_docs.md_docs.md`](./perf-inl.h_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`combined_traceback.cpp_docs.md_docs.md`](./combined_traceback.cpp_docs.md_docs.md)
- [`collection.cpp_kw.md_docs.md`](./collection.cpp_kw.md_docs.md)
- [`collection.h_docs.md_docs.md`](./collection.h_docs.md_docs.md)
- [`kineto_client_interface.h_docs.md_docs.md`](./kineto_client_interface.h_docs.md_docs.md)
- [`combined_traceback.cpp_kw.md_docs.md`](./combined_traceback.cpp_kw.md_docs.md)
- [`kineto_client_interface.cpp_docs.md_docs.md`](./kineto_client_interface.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `kineto_shim.h_docs.md_docs.md`
- **Keyword Index**: `kineto_shim.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

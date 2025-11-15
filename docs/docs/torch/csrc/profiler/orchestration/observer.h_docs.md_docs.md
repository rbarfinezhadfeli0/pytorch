# Documentation: `docs/torch/csrc/profiler/orchestration/observer.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/profiler/orchestration/observer.h_docs.md`
- **Size**: 9,707 bytes (9.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/profiler/orchestration/observer.h`

## File Metadata

- **Path**: `torch/csrc/profiler/orchestration/observer.h`
- **Size**: 7,450 bytes (7.28 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/Export.h>

#include <utility>

namespace torch::profiler::impl {

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
enum class C10_API_ENUM ActivityType {
  CPU = 0,
  XPU, // XPU kernels, runtime
  CUDA, // CUDA kernels, runtime
  HPU, // HPU kernels, runtime
  MTIA, // MTIA kernels, runtime
  PrivateUse1, // PrivateUse1 kernels, runtime
  NUM_KINETO_ACTIVITIES, // must be the last one
};

inline std::string actToString(ActivityType t) {
  const std::array<
      std::string,
      static_cast<size_t>(ActivityType::NUM_KINETO_ACTIVITIES)>
      ActivityTypeNames = {"CPU", "XPU", "CUDA", "MTIA", "PrivateUse1"};
  return ActivityTypeNames[static_cast<int>(t)];
}

enum class C10_API_ENUM ProfilerState {
  Disabled = 0,
  CPU, // CPU-only profiling
  CUDA, // CPU + CUDA events
  NVTX, // only emit NVTX markers
  ITT, // only emit ITT markers
  PRIVATEUSE1, // only emit PRIVATEUSE1 markers
  KINETO, // use libkineto
  KINETO_GPU_FALLBACK, // use CUDA events when CUPTI is not available
  KINETO_PRIVATEUSE1_FALLBACK, // use PrivateUse1 events
  KINETO_ONDEMAND, // run the profiler in on-demand mode
  NUM_PROFILER_STATES, // must be the last one
};

enum class C10_API_ENUM ActiveProfilerType {
  NONE = 0,
  LEGACY,
  KINETO,
  NVTX,
  ITT,
  PRIVATEUSE1
};

struct TORCH_API ExperimentalConfig {
  ExperimentalConfig(
      std::vector<std::string> profiler_metrics = {},
      bool profiler_measure_per_kernel = false,
      bool verbose = false,
      std::vector<std::string> performance_events = {},
      bool enable_cuda_sync_events = false,
      bool adjust_profiler_step = false,
      bool disable_external_correlation = false,
      bool profile_all_threads = false,
      bool capture_overload_names = false,
      bool record_python_gc_info = false,
      bool expose_kineto_event_metadata = false,
      std::string custom_profiler_config = "",
      bool adjust_timestamps = false);
  explicit operator bool() const;

  std::vector<std::string> profiler_metrics;
  bool profiler_measure_per_kernel;
  bool verbose;
  /*
   * List of performance events to be profiled.
   * An empty list will disable performance event based profiling altogether.
   */
  std::vector<std::string> performance_events;
  /*
   * For CUDA profiling mode, enable adding CUDA synchronization events
   * that expose CUDA device, stream and event synchronization activities.
   * This feature is new and currently disabled by default.
   */
  bool enable_cuda_sync_events;
  /*
   * Controls whether or not timestamp adjustment for ProfilerStep and parent
   * Python events occurs after profiling. This occurs at an O(n) cost and
   * affects only the start of profiler step events.
   */
  bool adjust_profiler_step;
  /*
   * Controls whether or not external correlation is disabled. This is used to
   * lower the amount of events received by CUPTI as correlation events are
   * paired with runtime/gpu events for each kind of correlation
   */
  bool disable_external_correlation;

  /* controls whether profiler records cpu events on threads
   * that are not spawned from the main thread on which the
   * profiler was enabled, similar to on_demand mode */
  bool profile_all_threads;

  /* controls whether overload names are queried from an ATen
   * function schema and stored in the profile  */
  bool capture_overload_names;

  /*
   * Controls whether or not python gc info is recorded. This is used to
   * determine if gc collect is slowing down your profile.
   */
  bool record_python_gc_info;

  /* controls whether KinetoEvent metadata is exposed to FunctionEvent
   * in the PyTorch Profiler as a JSON string */
  bool expose_kineto_event_metadata;

  /*
   * A custom_profiler_config option is introduced to allow custom backends
   * to apply custom configurations as needed.
   */
  std::string custom_profiler_config;

  /*
   * Controls whether or not timestamp adjustment occurs after profiling.
   * The purpose of this is to adjust Vulkan event timelines to align with those
   * of their parent CPU events.
   * This sometimes requires increasing CPU event durations (to fully contain
   * their child events) and delaying CPU event start times (to
   * prevent overlaps), so this should not be used unless Vulkan events are
   * being profiled and it is ok to use this modified timestamp/duration
   * information instead of the original information.
   */
  bool adjust_timestamps;
};

struct TORCH_API ProfilerConfig {
  explicit ProfilerConfig(
      ProfilerState state,
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false,
      ExperimentalConfig experimental_config = ExperimentalConfig(),
      std::string trace_id = "");

  bool disabled() const;
  bool global() const;
  bool pushGlobalCallbacks() const;

  ProfilerState state;
  ExperimentalConfig experimental_config;
  bool report_input_shapes;
  bool profile_memory;
  bool with_stack;
  bool with_flops;
  bool with_modules;
  std::string trace_id;

  // For serialization
  at::IValue toIValue() const;
  static ProfilerConfig fromIValue(const at::IValue& profilerConfigIValue);
};

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
struct TORCH_API ProfilerStateBase : public c10::MemoryReportingInfoBase {
  explicit ProfilerStateBase(ProfilerConfig config);
  ProfilerStateBase(const ProfilerStateBase&) = delete;
  ProfilerStateBase(ProfilerStateBase&&) = delete;
  ProfilerStateBase& operator=(const ProfilerStateBase&) = delete;
  ProfilerStateBase& operator=(ProfilerStateBase&&) = delete;
  ~ProfilerStateBase() override;

  static ProfilerStateBase* get(bool global);
  static ProfilerStateBase* get() {
    auto* out = get(/*global=*/true);
    return out ? out : get(/*global=*/false);
  }

  static void push(std::shared_ptr<ProfilerStateBase>&& state);

  static std::shared_ptr<ProfilerStateBase> pop(bool global);
  static std::shared_ptr<ProfilerStateBase> pop() {
    auto out = pop(/*global=*/true);
    return out ? std::move(out) : pop(/*global=*/false);
  }

  const ProfilerConfig& config() const {
    return config_;
  }

  void setCallbackHandle(at::CallbackHandle handle);
  void removeCallback();

  bool memoryProfilingEnabled() const override {
    return config_.profile_memory;
  }

  virtual ActiveProfilerType profilerType() = 0;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::mutex state_mutex_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ProfilerConfig config_ = ProfilerConfig(ProfilerState::Disabled);
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::CallbackHandle handle_ = 0;
};

// Note: The following are only for the active *thread local* profiler.
TORCH_API bool profilerEnabled();
TORCH_API ActiveProfilerType profilerType();
TORCH_API ProfilerConfig getProfilerConfig();

} // namespace torch::profiler::impl

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `C10_API_ENUM`, `C10_API_ENUM`, `C10_API_ENUM`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/orchestration`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/record_function.h`
- `torch/csrc/Export.h`
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

Files in the same folder (`torch/csrc/profiler/orchestration`):

- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`vulkan.h_docs.md`](./vulkan.h_docs.md)
- [`observer.cpp_docs.md`](./observer.cpp_docs.md)
- [`vulkan.cpp_docs.md`](./vulkan.cpp_docs.md)
- [`python_tracer.cpp_docs.md`](./python_tracer.cpp_docs.md)


## Cross-References

- **File Documentation**: `observer.h_docs.md`
- **Keyword Index**: `observer.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/profiler/orchestration`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/profiler/orchestration`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/profiler/orchestration`):

- [`observer.h_kw.md_docs.md`](./observer.h_kw.md_docs.md)
- [`vulkan.cpp_docs.md_docs.md`](./vulkan.cpp_docs.md_docs.md)
- [`vulkan.h_kw.md_docs.md`](./vulkan.h_kw.md_docs.md)
- [`vulkan.cpp_kw.md_docs.md`](./vulkan.cpp_kw.md_docs.md)
- [`python_tracer.cpp_kw.md_docs.md`](./python_tracer.cpp_kw.md_docs.md)
- [`python_tracer.cpp_docs.md_docs.md`](./python_tracer.cpp_docs.md_docs.md)
- [`vulkan.h_docs.md_docs.md`](./vulkan.h_docs.md_docs.md)
- [`python_tracer.h_kw.md_docs.md`](./python_tracer.h_kw.md_docs.md)
- [`observer.cpp_docs.md_docs.md`](./observer.cpp_docs.md_docs.md)
- [`observer.cpp_kw.md_docs.md`](./observer.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `observer.h_docs.md_docs.md`
- **Keyword Index**: `observer.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

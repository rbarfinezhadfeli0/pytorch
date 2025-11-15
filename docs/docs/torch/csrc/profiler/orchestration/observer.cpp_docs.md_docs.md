# Documentation: `docs/torch/csrc/profiler/orchestration/observer.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/profiler/orchestration/observer.cpp_docs.md`
- **Size**: 9,217 bytes (9.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/profiler/orchestration/observer.cpp`

## File Metadata

- **Path**: `torch/csrc/profiler/orchestration/observer.cpp`
- **Size**: 6,991 bytes (6.83 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/profiler/orchestration/observer.h>

#include <torch/csrc/profiler/util.h>

#include <utility>

namespace torch::profiler::impl {

using GlobalManager = GlobalStateManager<ProfilerStateBase>;

// ----------------------------------------------------------------------------
// -- Profiler Config ---------------------------------------------------------
// ----------------------------------------------------------------------------
ExperimentalConfig::ExperimentalConfig(
    std::vector<std::string> profiler_metrics,
    bool profiler_measure_per_kernel,
    bool verbose,
    std::vector<std::string> performance_events,
    bool enable_cuda_sync_events,
    bool adjust_profiler_step,
    bool disable_external_correlation,
    bool profile_all_threads,
    bool capture_overload_names,
    bool record_python_gc_info,
    bool expose_kineto_event_metadata,
    std::string custom_profiler_config,
    bool adjust_timestamps)
    : profiler_metrics{std::move(profiler_metrics)},
      profiler_measure_per_kernel{profiler_measure_per_kernel},
      verbose{verbose},
      performance_events(std::move(performance_events)),
      enable_cuda_sync_events{enable_cuda_sync_events},
      adjust_profiler_step{adjust_profiler_step},
      disable_external_correlation{disable_external_correlation},
      profile_all_threads{profile_all_threads},
      capture_overload_names{capture_overload_names},
      record_python_gc_info{record_python_gc_info},
      expose_kineto_event_metadata{expose_kineto_event_metadata},
      custom_profiler_config(std::move(custom_profiler_config)),
      adjust_timestamps{adjust_timestamps} {}

/*explicit*/ ExperimentalConfig::operator bool() const {
  return !profiler_metrics.empty();
}

ProfilerConfig::ProfilerConfig(
    ProfilerState state,
    bool report_input_shapes,
    bool profile_memory,
    bool with_stack,
    bool with_flops,
    bool with_modules,
    ExperimentalConfig experimental_config,
    std::string trace_id)
    : state{state},
      experimental_config{std::move(experimental_config)},
      report_input_shapes{report_input_shapes},
      profile_memory{profile_memory},
      with_stack{with_stack},
      with_flops{with_flops},
      with_modules{with_modules},
      trace_id{std::move(trace_id)} {}

bool ProfilerConfig::disabled() const {
  return state == torch::profiler::impl::ProfilerState::Disabled;
}

bool ProfilerConfig::global() const {
  return state == torch::profiler::impl::ProfilerState::KINETO_ONDEMAND;
}

bool ProfilerConfig::pushGlobalCallbacks() const {
  return global() || experimental_config.profile_all_threads;
}

namespace {
enum ProfilerIValueIdx {
  STATE = 0,
  REPORT_INPUT_SHAPES,
  PROFILE_MEMORY,
  NUM_PROFILER_CFG_IVALUE_IDX // must be last in list
};
} // namespace

at::IValue ProfilerConfig::toIValue() const {
  c10::impl::GenericList eventIValueList(at::AnyType::get());
  eventIValueList.reserve(NUM_PROFILER_CFG_IVALUE_IDX);
  eventIValueList.emplace_back(static_cast<int64_t>(state));
  eventIValueList.emplace_back(report_input_shapes);
  eventIValueList.emplace_back(profile_memory);
  return eventIValueList;
}

ProfilerConfig ProfilerConfig::fromIValue(
    const at::IValue& profilerConfigIValue) {
  TORCH_INTERNAL_ASSERT(
      profilerConfigIValue.isList(),
      "Expected IValue to contain type c10::impl::GenericList");
  auto ivalues = profilerConfigIValue.toList();
  TORCH_INTERNAL_ASSERT(
      ivalues.size() == NUM_PROFILER_CFG_IVALUE_IDX,
      c10::str(
          "Expected exactly ",
          NUM_PROFILER_CFG_IVALUE_IDX,
          " ivalues to resconstruct ProfilerConfig."));
  return ProfilerConfig(
      static_cast<ProfilerState>(ivalues.get(ProfilerIValueIdx::STATE).toInt()),
      ivalues.get(ProfilerIValueIdx::REPORT_INPUT_SHAPES).toBool(),
      ivalues.get(ProfilerIValueIdx::PROFILE_MEMORY).toBool());
}

// ----------------------------------------------------------------------------
// -- Profiler base class -----------------------------------------------------
// ----------------------------------------------------------------------------
/*explicit*/ ProfilerStateBase::ProfilerStateBase(ProfilerConfig config)
    : c10::MemoryReportingInfoBase(), config_(std::move(config)) {}

ProfilerStateBase::~ProfilerStateBase() {
  if (handle_) {
    auto handle = handle_;
    removeCallback();
    SOFT_ASSERT(false, "Leaked callback handle: ", handle);
  }
}

/*static*/ ProfilerStateBase* ProfilerStateBase::get(bool global) {
  auto* out = global
      ? GlobalManager::get()
      : static_cast<ProfilerStateBase*>(
            c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !out || out->config().pushGlobalCallbacks() == global);
  return out;
}

/*static*/ void ProfilerStateBase::push(
    std::shared_ptr<ProfilerStateBase>&& state) {
  TORCH_INTERNAL_ASSERT(state != nullptr);
  if (state->config().pushGlobalCallbacks()) {
    GlobalManager::push(std::move(state));
  } else {
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, state);
  }
}

namespace {
std::shared_ptr<ProfilerStateBase> popTLS() {
  // If there is no active thread local profiler then we simply return null.
  // However if there is an active profiler but it is not the top
  // `DebugInfoBase`then `c10::ThreadLocalDebugInfo::_pop` will throw.
  // TODO(robieta): make `noexcept` version.
  return c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PROFILER_STATE)
      ? std::static_pointer_cast<ProfilerStateBase>(
            c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE))
      : nullptr;
}
} // namespace

/*static*/ std::shared_ptr<ProfilerStateBase> ProfilerStateBase::pop(
    bool global) {
  auto out = global ? GlobalManager::pop() : popTLS();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!out || out->config().global() == global);
  return out;
}

void ProfilerStateBase::setCallbackHandle(at::CallbackHandle handle) {
  if (handle_) {
    at::removeCallback(handle_);
    SOFT_ASSERT(
        false,
        "ProfilerStateBase already has a registered callback. "
        "Removing to avoid leaked callback.");
  }

  handle_ = handle;
}

void ProfilerStateBase::removeCallback() {
  if (handle_) {
    at::removeCallback(handle_);
    handle_ = 0;
  }
}

bool profilerEnabled() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  return state_ptr && !state_ptr->config().disabled();
}

TORCH_API ActiveProfilerType profilerType() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  return state_ptr == nullptr ? ActiveProfilerType::NONE
                              : state_ptr->profilerType();
}

torch::profiler::impl::ProfilerConfig getProfilerConfig() {
  auto* state_ptr = ProfilerStateBase::get(/*global=*/false);
  TORCH_CHECK(
      state_ptr,
      "Tried to access profiler config, but profiler is not enabled!");
  return state_ptr->config();
}

} // namespace torch::profiler::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`

**Classes/Structs**: `ProfilerConfig`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/orchestration`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/profiler/orchestration/observer.h`
- `torch/csrc/profiler/util.h`
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

- [`observer.h_docs.md`](./observer.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`vulkan.h_docs.md`](./vulkan.h_docs.md)
- [`vulkan.cpp_docs.md`](./vulkan.cpp_docs.md)
- [`python_tracer.cpp_docs.md`](./python_tracer.cpp_docs.md)


## Cross-References

- **File Documentation**: `observer.cpp_docs.md`
- **Keyword Index**: `observer.cpp_kw.md`
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
- [`observer.cpp_kw.md_docs.md`](./observer.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `observer.cpp_docs.md_docs.md`
- **Keyword Index**: `observer.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/csrc/jit/mobile/profiler_edge.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/profiler_edge.cpp_docs.md`
- **Size**: 7,362 bytes (7.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/profiler_edge.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/profiler_edge.cpp`
- **Size**: 4,757 bytes (4.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/mobile/profiler_edge.h>
#include <string>
#include <vector>

namespace torch::jit::mobile {

thread_local KinetoEdgeCPUProfiler* tls_edge_profiler{nullptr};

KinetoEdgeCPUProfiler::KinetoEdgeCPUProfiler(
    const torch::jit::mobile::Module& m,
    const std::string& fname,
    const bool report_input_shapes,
    const bool profile_memory,
    const bool with_stack,
    const bool with_flops,
    const bool with_modules,
    std::vector<std::string> events,
    const bool adjust_vulkan_timestamps)
    : m_(m), trace_file_name_(fname) {
  torch::profiler::impl::ExperimentalConfig experimental_config;
  // Enable hardware counters
  if (!events.empty()) {
    experimental_config.performance_events = std::move(events);
  }

  // Adjust vulkan timestamps from query pool to align with cpu event times
  experimental_config.adjust_timestamps = adjust_vulkan_timestamps;

  torch::profiler::impl::ProfilerConfig config(
      torch::profiler::impl::ProfilerState::KINETO,
      report_input_shapes,
      profile_memory,
      with_stack,
      with_flops,
      with_modules,
      experimental_config);
  torch::autograd::profiler::prepareProfiler(
      config, {torch::autograd::profiler::ActivityType::CPU});
  if (with_modules || with_stack) {
    auto post_processing = [this, with_stack, with_modules](
                               int64_t debug_handle,
                               std::vector<std::string>& jit_stack,
                               std::vector<std::string>& jit_modules) {
      std::string no_debug_info("Model was not saved with debug information");
      if (with_modules) {
        // Since KinetoEvents's module hierarchy takes vector of strings
        // we just construct a temporary vector using one string element
        jit_modules = std::vector<std::string>(
            {this->m_.hasDebugHandles()
                 ? this->m_.getModuleHierarchy(debug_handle)
                 : no_debug_info});
      } else if (with_stack) {
        // Since KinetoEvents's stack trace takes vector of strings we
        // just construct a temporary vector using one string element
        jit_stack = std::vector<std::string>(
            {this->m_.hasDebugHandles() ? this->m_.getCallStack(debug_handle)
                                        : no_debug_info});
      }
    };
    torch::autograd::profiler::enableProfilerWithEventPostProcess(
        config,
        {torch::autograd::profiler::ActivityType::CPU},
        post_processing,
        {at::RecordScope::LITE_INTERPRETER});
  } else {
    torch::autograd::profiler::enableProfiler(
        config,
        {torch::autograd::profiler::ActivityType::CPU},
        {at::RecordScope::LITE_INTERPRETER});
  }
  trace_file_name_ = fname;
  TORCH_CHECK(
      tls_edge_profiler == nullptr, "Edge profiler is already profiling.")
  tls_edge_profiler = this;
}

void KinetoEdgeCPUProfiler::recordBackendMemoryEvent(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    c10::Device device) {
  c10::reportMemoryUsageToProfiler(
      ptr, alloc_size, total_allocated, total_reserved, device);
}

void KinetoEdgeCPUProfiler::recordBackendEvent(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const std::string& event_name,
    const std::string& backend_name) {
  torch::autograd::profiler::reportBackendEventToActiveKinetoProfiler(
      start_time_us,
      end_time_us,
      debug_handle,
      at::RecordScope::LITE_INTERPRETER,
      event_name,
      backend_name);
}

const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
KinetoEdgeCPUProfiler::disableProfiler() {
  TORCH_CHECK(
      !profiler_result_,
      "KinetoEdgeCPUProfiler already disabled. "
      "To get list of events use getProfilerResults()");
  profiler_result_ = torch::autograd::profiler::disableProfiler();
  return profiler_result_;
}

const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
KinetoEdgeCPUProfiler::getProfilerResult() {
  TORCH_CHECK(
      profiler_result_,
      "KinetoEdgeCPUProfiler has not been disabled. "
      "use disableProfiler() API first, which returns the ProfilerResult.");
  return profiler_result_;
}

KinetoEdgeCPUProfiler::~KinetoEdgeCPUProfiler() {
  if (!trace_file_name_.empty()) {
    if (profiler_result_) {
      profiler_result_->save(trace_file_name_);
    } else {
      torch::autograd::profiler::disableProfiler()->save(trace_file_name_);
    }
  }
  tls_edge_profiler = nullptr;
}

KinetoEdgeCPUProfiler* getCurrentEdgeProfiler() {
  return tls_edge_profiler;
}

} // namespace torch::jit::mobile

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `a`, `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Allocator.h`
- `c10/util/Exception.h`
- `c10/util/overloaded.h`
- `torch/csrc/jit/mobile/profiler_edge.h`
- `string`
- `vector`


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

Files in the same folder (`torch/csrc/jit/mobile`):

- [`register_ops_common_utils.cpp_docs.md`](./register_ops_common_utils.cpp_docs.md)
- [`import.h_docs.md`](./import.h_docs.md)
- [`prim_ops_registery.h_docs.md`](./prim_ops_registery.h_docs.md)
- [`profiler_edge.h_docs.md`](./profiler_edge.h_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`file_format.h_docs.md`](./file_format.h_docs.md)
- [`module.h_docs.md`](./module.h_docs.md)
- [`observer.h_docs.md`](./observer.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`flatbuffer_loader.cpp_docs.md`](./flatbuffer_loader.cpp_docs.md)


## Cross-References

- **File Documentation**: `profiler_edge.cpp_docs.md`
- **Keyword Index**: `profiler_edge.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/mobile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/jit/mobile`):

- [`code.h_docs.md_docs.md`](./code.h_docs.md_docs.md)
- [`register_ops_common_utils.cpp_docs.md_docs.md`](./register_ops_common_utils.cpp_docs.md_docs.md)
- [`observer.h_kw.md_docs.md`](./observer.h_kw.md_docs.md)
- [`prim_ops_registery.cpp_kw.md_docs.md`](./prim_ops_registery.cpp_kw.md_docs.md)
- [`quantization.h_docs.md_docs.md`](./quantization.h_docs.md_docs.md)
- [`debug_info.cpp_kw.md_docs.md`](./debug_info.cpp_kw.md_docs.md)
- [`interpreter.cpp_kw.md_docs.md`](./interpreter.cpp_kw.md_docs.md)
- [`debug_info.h_docs.md_docs.md`](./debug_info.h_docs.md_docs.md)
- [`interpreter.cpp_docs.md_docs.md`](./interpreter.cpp_docs.md_docs.md)
- [`promoted_prim_ops.cpp_docs.md_docs.md`](./promoted_prim_ops.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `profiler_edge.cpp_docs.md_docs.md`
- **Keyword Index**: `profiler_edge.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/csrc/profiler/kineto_client_interface.cpp`

## File Metadata

- **Path**: `torch/csrc/profiler/kineto_client_interface.cpp`
- **Size**: 3,004 bytes (2.93 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_KINETO
#include <ATen/Context.h>
#include <libkineto.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/mtia/profiler/MTIAMemoryProfiler.h>
#include <torch/csrc/profiler/kineto_client_interface.h>
#include <chrono>
#include <thread>

// Ondemand tracing is not supported on Apple or edge platform
#if defined(__APPLE__) || defined(EDGE_PROFILER_USE_KINETO)
#define ENABLE_GLOBAL_OBSERVER (0)
#else
#define ENABLE_GLOBAL_OBSERVER (1)
#endif

namespace torch {

namespace profiler::impl {

namespace {

using namespace torch::autograd::profiler;

class LibKinetoClient : public libkineto::ClientInterface {
 public:
  void init() override {
    ::torch::mtia::initMemoryProfiler();
  }

  void prepare(
      bool report_input_shapes = false,
      bool profile_memory = false,
      bool with_stack = false,
      bool with_flops = false,
      bool with_modules = false) override {
    reportInputShapes_ = report_input_shapes;
    profileMemory_ = profile_memory;
    withStack_ = with_stack;
    withFlops_ = with_flops;
    withModules_ = with_modules;
  }

  void start() override {
    ProfilerConfig cfg{
        ProfilerState::KINETO_ONDEMAND,
        /*report_input_shapes=*/reportInputShapes_,
        /*profile_memory=*/profileMemory_,
        /*with_stack=*/withStack_,
        /*with_flops=*/withFlops_,
        /*with_modules=*/withModules_};
    std::set<ActivityType> activities{ActivityType::CPU};
    std::unordered_set<at::RecordScope> scopes;
    scopes.insert(at::RecordScope::FUNCTION);
    scopes.insert(at::RecordScope::USER_SCOPE);
    scopes.insert(at::RecordScope::BACKWARD_FUNCTION);
    enableProfiler(cfg, activities, scopes);
  }

  void stop() override {
    (void)disableProfiler();
  }

  void start_memory_profile() override {
    LOG(INFO) << "Starting on-demand memory profile";
    startMemoryProfile();
  }

  void stop_memory_profile() override {
    LOG(INFO) << "Stopping on-demand memory profile";
    stopMemoryProfile();
  }

  void export_memory_profile(const std::string& path) override {
    exportMemoryProfile(path);
  }

 private:
  // Temporarily disable shape collection until
  // we re-roll out the feature for on-demand cases
  bool reportInputShapes_{false};
  bool profileMemory_{false};
  bool withStack_{false};
  bool withFlops_{false};
  bool withModules_{false};
};

} // namespace

} // namespace profiler::impl

void global_kineto_init() {
#if ENABLE_GLOBAL_OBSERVER
  if (c10::utils::get_env("KINETO_USE_DAEMON").has_value()) {
    libkineto_init(
        /*cpuOnly=*/!(at::hasCUDA() || at::hasXPU() || at::hasMTIA()),
        /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }
#endif
}

#if ENABLE_GLOBAL_OBSERVER
namespace {

struct RegisterLibKinetoClient {
  RegisterLibKinetoClient() {
    static profiler::impl::LibKinetoClient client;
    libkineto::api().registerClient(&client);
  }
} register_libkineto_client;

} // namespace
#endif

} // namespace torch
#endif // USE_KINETO

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `profiler`

**Classes/Structs**: `LibKinetoClient`, `RegisterLibKinetoClient`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Context.h`
- `libkineto.h`
- `torch/csrc/autograd/profiler_kineto.h`
- `torch/csrc/mtia/profiler/MTIAMemoryProfiler.h`
- `torch/csrc/profiler/kineto_client_interface.h`
- `chrono`
- `thread`


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
- [`combined_traceback.h_docs.md`](./combined_traceback.h_docs.md)
- [`kineto_shim.h_docs.md`](./kineto_shim.h_docs.md)
- [`collection.h_docs.md`](./collection.h_docs.md)
- [`kineto_shim.cpp_docs.md`](./kineto_shim.cpp_docs.md)
- [`combined_traceback.cpp_docs.md`](./combined_traceback.cpp_docs.md)
- [`kineto_client_interface.h_docs.md`](./kineto_client_interface.h_docs.md)
- [`perf.h_docs.md`](./perf.h_docs.md)


## Cross-References

- **File Documentation**: `kineto_client_interface.cpp_docs.md`
- **Keyword Index**: `kineto_client_interface.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

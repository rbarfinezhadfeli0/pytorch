# Documentation: `docs/torch/csrc/profiler/standalone/itt_observer.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/profiler/standalone/itt_observer.cpp_docs.md`
- **Size**: 4,979 bytes (4.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/profiler/standalone/itt_observer.cpp`

## File Metadata

- **Path**: `torch/csrc/profiler/standalone/itt_observer.cpp`
- **Size**: 2,434 bytes (2.38 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/profiler/standalone/itt_observer.h>

#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

namespace torch::profiler::impl {

struct ITTThreadLocalState : ProfilerStateBase {
  explicit ITTThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // Only `report_input_shapes` makes sense in this context.
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~ITTThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::ITT;
  }

  void reportMemoryUsage(
      void* /*ptr*/,
      int64_t /*alloc_size*/,
      size_t /*total_allocated*/,
      size_t /*total_reserved*/,
      c10::Device /*device*/) override {}

  static ITTThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr || tls->profilerType() == ActiveProfilerType::ITT);
    return static_cast<ITTThreadLocalState*>(tls);
  }
};

template <bool report_input_shapes>
static std::unique_ptr<at::ObserverContext> enterITT(
    const at::RecordFunction& fn) {
  if (ITTThreadLocalState::getTLS() != nullptr) {
    torch::profiler::impl::ittStubs()->rangePush(fn.name());
  }
  return nullptr;
}

void pushITTCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      torch::profiler::impl::ittStubs()->enabled(),
      "Can't use ITT profiler - PyTorch was compiled without ITT");

  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<ITTThreadLocalState>(config));

  auto state_ptr = ITTThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &enterITT</*report_input_shapes=*/true>
              : &enterITT</*report_input_shapes=*/false>,
          [](const at::RecordFunction&, at::ObserverContext*) {
            torch::profiler::impl::ittStubs()->rangePop();
          })
          .needsInputs(config.report_input_shapes)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace torch::profiler::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ITTThreadLocalState`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/standalone`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/profiler/standalone/itt_observer.h`
- `torch/csrc/profiler/stubs/base.h`
- `torch/csrc/profiler/util.h`


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

Files in the same folder (`torch/csrc/profiler/standalone`):

- [`privateuse1_observer.h_docs.md`](./privateuse1_observer.h_docs.md)
- [`execution_trace_observer.cpp_docs.md`](./execution_trace_observer.cpp_docs.md)
- [`itt_observer.h_docs.md`](./itt_observer.h_docs.md)
- [`nvtx_observer.cpp_docs.md`](./nvtx_observer.cpp_docs.md)
- [`nvtx_observer.h_docs.md`](./nvtx_observer.h_docs.md)
- [`execution_trace_observer.h_docs.md`](./execution_trace_observer.h_docs.md)
- [`privateuse1_observer.cpp_docs.md`](./privateuse1_observer.cpp_docs.md)


## Cross-References

- **File Documentation**: `itt_observer.cpp_docs.md`
- **Keyword Index**: `itt_observer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/profiler/standalone`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/profiler/standalone`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/profiler/standalone`):

- [`privateuse1_observer.h_kw.md_docs.md`](./privateuse1_observer.h_kw.md_docs.md)
- [`itt_observer.h_docs.md_docs.md`](./itt_observer.h_docs.md_docs.md)
- [`execution_trace_observer.cpp_kw.md_docs.md`](./execution_trace_observer.cpp_kw.md_docs.md)
- [`privateuse1_observer.h_docs.md_docs.md`](./privateuse1_observer.h_docs.md_docs.md)
- [`nvtx_observer.h_kw.md_docs.md`](./nvtx_observer.h_kw.md_docs.md)
- [`itt_observer.cpp_kw.md_docs.md`](./itt_observer.cpp_kw.md_docs.md)
- [`privateuse1_observer.cpp_docs.md_docs.md`](./privateuse1_observer.cpp_docs.md_docs.md)
- [`execution_trace_observer.h_kw.md_docs.md`](./execution_trace_observer.h_kw.md_docs.md)
- [`nvtx_observer.cpp_docs.md_docs.md`](./nvtx_observer.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `itt_observer.cpp_docs.md_docs.md`
- **Keyword Index**: `itt_observer.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

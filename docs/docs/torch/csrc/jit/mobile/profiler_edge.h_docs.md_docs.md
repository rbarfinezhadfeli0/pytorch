# Documentation: `docs/torch/csrc/jit/mobile/profiler_edge.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/profiler_edge.h_docs.md`
- **Size**: 6,968 bytes (6.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/profiler_edge.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/profiler_edge.h`
- **Size**: 4,488 bytes (4.38 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch::jit::mobile {

// If we dont have kineto available then edge profiler does not
// work since it relies on Kineto
#ifdef USE_KINETO
class TORCH_API KinetoEdgeCPUProfiler {
 public:
  // This profiler only profiles KINETO events
  // No GPU_FALLBACK or NVTX
  /*
   * @param m is the instance of mobile Module which is being profiled.
   *        Note that this implies that KinetoEdgeCPUProfiler can be used
   *        to profile specific Module (see usage below), unliked ProfilerKineto
   *        which can profile pytorch runtime in arbitrary scope.
   * @param fname is the name of the file to which chrome trace is written.
   * @param report_input_shapes: whether to record shapes of op's inputs.
   * @param with_stack: whether to record model's python stacktrace for the op.
   * @param with_flops: whether to report flops corresponding to the op.
   * @param with_modules: whether to report original python module
   *        hierarchy to which the op belongs.
   * @param events
   * @param adjust_vulkan_timestamps: whether to adjust vulkan timestamps from
   *        query pool to align with cpu event times
   *
   * Usage pattern for this profiler must be as follows:
   *
   * {
   *   KinetoEdgeCPUProfiler(m, filename, args);
   *   m.forward(...);
   * }
   *
   * The reason being that KinetoEdgeCPUProfiler has a dependency on Module
   * and thus it must not outlive it.
   *
   * Thus, when KinetoEdgeCPUProfiler is used as RAII to do profiling
   * within certain scope. In that scope, the captured reference to
   * Module will outlive KinetoEdgeCPUProfiler. This is guaranteed because
   * KinetoEdgeCPUProfiler must be constructed later than Module, on stack.
   *
   * An example of the anti-pattern and wrong usage is:
   *
   * std::shared_ptr<KinetoMobileCPUProfiler> profiler(m, filename, args);
   * m.forward(...);
   *
   * Since KinetoEdgeCPUProfiler object would then be constructed on heap
   * with its lifetime managed manually or via smart pointers.
   */
  KinetoEdgeCPUProfiler(
      const torch::jit::mobile::Module& m,
      const std::string& fname,
      const bool report_input_shapes = false,
      const bool profile_memory = false,
      const bool with_stack = false,
      const bool with_flops = false,
      const bool with_modules = false,
      std::vector<std::string> events = {},
      const bool adjust_vulkan_timestamps = false);

  const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
  disableProfiler();
  const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
  getProfilerResult();
  void recordBackendEvent(
      const int64_t start_time_us,
      const int64_t end_time_us,
      const int64_t debug_handle,
      const std::string& event_name,
      const std::string& backend_name);
  void recordBackendMemoryEvent(
      void* ptr,
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device);

  ~KinetoEdgeCPUProfiler();

 private:
  /*
   * We store a reference to Module to make such dependency explicit, since
   * a Module reference is already stored in a functor.
   */
  const mobile::Module& m_;
  std::string trace_file_name_;
  std::unique_ptr<torch::autograd::profiler::ProfilerResult> profiler_result_;
};

TORCH_API KinetoEdgeCPUProfiler* getCurrentEdgeProfiler();

#define RECORD_BACKEND_EVENT_TO_EDGE_PROFILER(                               \
    start_time_us, end_time_us, debug_handle, event_name, backend_name)      \
  if (mobile::getCurrentEdgeProfiler()) {                                    \
    mobile::getCurrentEdgeProfiler()->recordBackendEvent(                    \
        start_time_us, end_time_us, debug_handle, event_name, backend_name); \
  }

#define RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER(              \
    ptr, alloc_size, total_allocated, total_reserved, device)      \
  if (mobile::getCurrentEdgeProfiler()) {                          \
    mobile::getCurrentEdgeProfiler()->recordBackendMemoryEvent(    \
        ptr, alloc_size, total_allocated, total_reserved, device); \
  }
#else

#define RECORD_BACKEND_EVENT_TO_EDGE_PROFILER( \
    start_time_us, end_time_us, debug_handle, event_name, backend_name)

#define RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER( \
    ptr, alloc_size, total_allocated, total_reserved, device)
#endif
} // namespace torch::jit::mobile

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/profiler_kineto.h`
- `torch/csrc/jit/mobile/module.h`


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
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`file_format.h_docs.md`](./file_format.h_docs.md)
- [`module.h_docs.md`](./module.h_docs.md)
- [`observer.h_docs.md`](./observer.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`flatbuffer_loader.cpp_docs.md`](./flatbuffer_loader.cpp_docs.md)


## Cross-References

- **File Documentation**: `profiler_edge.h_docs.md`
- **Keyword Index**: `profiler_edge.h_kw.md`
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

- **File Documentation**: `profiler_edge.h_docs.md_docs.md`
- **Keyword Index**: `profiler_edge.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

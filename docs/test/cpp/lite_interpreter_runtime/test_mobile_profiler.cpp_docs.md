# Documentation: `test/cpp/lite_interpreter_runtime/test_mobile_profiler.cpp`

## File Metadata

- **Path**: `test/cpp/lite_interpreter_runtime/test_mobile_profiler.cpp`
- **Size**: 6,905 bytes (6.74 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <ATen/Functions.h>
#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/profiler_edge.h>
#include <fstream>

#include <unordered_set>

#include <torch/csrc/profiler/events.h>

#include "test/cpp/lite_interpreter_runtime/resources.h"

#ifdef EDGE_PROFILER_USE_KINETO
namespace torch {
namespace jit {
namespace mobile {

namespace {
bool checkMetaData(
    const std::string& op_name,
    const std::string& metadata_name,
    const std::string& metadata_val,
    std::ifstream& trace_file) {
  std::string line;
  while (std::getline(trace_file, line)) {
    if (line.find(op_name) != std::string::npos) {
      while (std::getline(trace_file, line)) {
        if (line.find(metadata_name) != std::string::npos) {
          if (line.find(metadata_val) != std::string::npos ||
              !metadata_val.size()) {
            /* if found the right metadata_val OR if expected
             * metadata value is an empty string then ignore the metadata_val */
            return true;
          }
        }
      }
    }
  }
  return false;
}
} // namespace

TEST(MobileProfiler, ModuleHierarchy) {
  auto testModelFile = torch::testing::getResourcePath(
      "test/cpp/lite_interpreter_runtime/to_be_profiled_module.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace.trace");

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        false, // profile memory
        true, // record callstack
        false, // record flops
        true, // record module hierarchy
        {}, // events
        false); // adjust_vulkan_timestamps
    bc.forward(inputs);
  } // End of profiler
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);
  const std::string metadata_name("Module Hierarchy");
  ASSERT_TRUE(checkMetaData(
      "aten::sub",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.aten::sub",
      trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  ASSERT_TRUE(checkMetaData(
      "aten::mul",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method.aten::mul",
      trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  ASSERT_TRUE(checkMetaData(
      "aten::add",
      metadata_name,
      "top(C)::<unknown>.A0(A)::forward.SELF(A)::forward_impl_.aten::add",
      trace_file));
  ASSERT_TRUE(checkMetaData(
      "aten::add",
      metadata_name,
      "top(C)::<unknown>.SELF(C)::call_b.B0(B)::forward.aten::add",
      trace_file));
  ASSERT_TRUE(checkMetaData(
      "aten::add", metadata_name, "top(C)::<unknown>.aten::add", trace_file));
}

TEST(MobileProfiler, Backend) {
  auto testModelFile = torch::testing::getResourcePath(
      "test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace_backend.trace");

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        false, // profile memory
        true, // record callstack
        false, // record flops
        true); // record module hierarchy
    bc.forward(inputs);
  } // End of profiler
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);
  std::string metadata_name("Module Hierarchy");
  ASSERT_TRUE(checkMetaData(
      "aten::add", metadata_name, "top(m)::<unknown>.aten::add", trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  metadata_name = "Backend";
  ASSERT_TRUE(
      checkMetaData("aten::add", metadata_name, "test_backend", trace_file));
}

TEST(MobileProfiler, BackendMemoryEvents) {
  auto testModelFile = torch::testing::getResourcePath(
      "test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace_backend_memory.trace");

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    mobile::KinetoEdgeCPUProfiler profiler(
        bc,
        trace_file_name,
        false, // record input_shapes
        true, // profile memory
        true, // record callstack
        false, // record flops
        true); // record module hierarchy
    bc.forward(inputs);
  }
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());
  trace_file.seekg(0, std::ios_base::beg);
  std::string metadata_name("Bytes");
  ASSERT_TRUE(checkMetaData("[memory]", metadata_name, "16384", trace_file));
  trace_file.seekg(0, std::ios_base::beg);
  metadata_name = "Total Reserved";
  ASSERT_TRUE(checkMetaData("[memory]", metadata_name, "49152", trace_file));
}

TEST(MobileProfiler, ProfilerEvent) {
  auto testModelFile = torch::testing::getResourcePath(
      "test/cpp/lite_interpreter_runtime/test_backend_for_profiling.ptl");

  std::vector<IValue> inputs;
  inputs.emplace_back(at::rand({64, 64}));
  inputs.emplace_back(at::rand({64, 64}));
  std::string trace_file_name("/tmp/test_trace_profiler_event.trace");

  std::vector<std::string> events(
      torch::profiler::ProfilerPerfEvents.begin(),
      torch::profiler::ProfilerPerfEvents.end());

  mobile::Module bc = _load_for_mobile(testModelFile.string());
  {
    // Bail if something goes wrong here
    try {
      KinetoEdgeCPUProfiler profiler(
          bc,
          trace_file_name,
          false, // record input_shapes
          false, // profile memory
          true, // record callstack
          false, // record flops
          true, // record module hierarchy
          events); // performance events
      bc.forward(inputs);
    } catch (...) {
      return;
    }
  } // End of profiler
  std::ifstream trace_file(trace_file_name);
  std::string line;
  ASSERT_TRUE(trace_file.is_open());

  for (auto& event : events) {
    trace_file.seekg(0, std::ios_base::beg);
    /*
     * Just checking if the event entry exists in the chrometrace.
     * Checking the value in a hardware independent matter is tricky.
     */
    ASSERT_TRUE(checkMetaData("aten::__getitem__", event, "", trace_file));
  }
}

} // namespace mobile
} // namespace jit
} // namespace torch
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `mobile`, `torch`, `TEST`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/lite_interpreter_runtime`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Functions.h`
- `gtest/gtest.h`
- `test/cpp/jit/test_utils.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/module.h`
- `torch/csrc/jit/mobile/profiler_edge.h`
- `fstream`
- `unordered_set`
- `torch/csrc/profiler/events.h`
- `test/cpp/lite_interpreter_runtime/resources.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/lite_interpreter_runtime/test_mobile_profiler.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/lite_interpreter_runtime`):

- [`resources.h_docs.md`](./resources.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`main.cpp_docs.md`](./main.cpp_docs.md)
- [`test_lite_interpreter_runtime.cpp_docs.md`](./test_lite_interpreter_runtime.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_mobile_profiler.cpp_docs.md`
- **Keyword Index**: `test_mobile_profiler.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

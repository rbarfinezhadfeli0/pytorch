# Documentation: `torch/csrc/distributed/c10d/FlightRecorderCuda.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/FlightRecorderCuda.cpp`
- **Size**: 4,030 bytes (3.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_C10D_NCCL
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <mutex>
#include <vector>

#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d {
control_plane::RegisterHandler dumpHandler{
    "dump_nccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true},
          {includeStackTracesStr, true},
          {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setContent(
          dump_nccl_trace(
              processedParams[includeCollectivesStr],
              processedParams[includeStackTracesStr],
              processedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

control_plane::RegisterHandler jsonDumpHandler{
    "dump_nccl_trace_json",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setStatus(200);
      res.setContent(
          dump_nccl_trace_json(
              processedParams[includeCollectivesStr],
              processedParams[onlyActiveStr]),
          "application/json");
    }};

/* Helper used by work::getDuration() and nccl flight recorder */
template <>
float getDurationFromEvent<at::cuda::CUDAEvent>(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent) {
  TORCH_CHECK(
      ncclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return ncclStartEvent.elapsed_time(ncclEndEvent);
}

template struct FlightRecorder<at::cuda::CUDAEvent>;
} // namespace c10d
#endif // USE_C10D_NCCL

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `FlightRecorder`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/CUDAEvent.h`
- `cuda_runtime.h`
- `nlohmann/json.hpp`
- `fstream`
- `mutex`
- `vector`
- `torch/csrc/distributed/c10d/FlightRecorderDetail.hpp`
- `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp`
- `torch/csrc/distributed/c10d/control_plane/Handlers.hpp`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`Store.hpp_docs.md`](./Store.hpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `FlightRecorderCuda.cpp_docs.md`
- **Keyword Index**: `FlightRecorderCuda.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

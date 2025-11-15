# Documentation: `torch/csrc/distributed/c10d/FlightRecorder.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/FlightRecorder.cpp`
- **Size**: 4,139 bytes (4.04 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/FileSystem.h>
#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <fstream>

namespace c10d {

void DebugInfoWriter::write(const std::string& trace) {
  std::string filename = filename_;
  if (enable_dynamic_filename_) {
    LOG(INFO) << "Writing Flight Recorder debug info to a dynamic file name";
    filename = c10::str(getCvarString({"TORCH_FR_DUMP_TEMP_FILE"}, ""));
  } else {
    LOG(INFO) << "Writing Flight Recorder debug info to a static file name";
  }
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing Flight Recorder debug info: "
               << filename;
    return;
  }

  if (!file.write(trace.data(), static_cast<std::streamsize>(trace.size()))) {
    const auto bad = file.bad();
    LOG(ERROR) << "Error writing Flight Recorder debug info to file: "
               << filename << " bad bit: " << bad;
    return;
  }

  // Flush the buffer to ensure data is written to the file
  file.flush();
  if (file.bad()) {
    LOG(ERROR) << "Error flushing Flight Recorder debug info: " << filename;
    return;
  }

  LOG(INFO) << "Finished writing Flight Recorder debug info to " << filename;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
    // Attempt to write to running user's HOME directory cache folder - if it
    // exists.
    auto homeDir = getCvarString({"HOME"}, "/tmp");
    auto cacheDirPath = c10::filesystem::path(homeDir + "/.cache/torch");
    // Create the .cache directory if it doesn't exist
    c10::filesystem::create_directories(cacheDirPath);
    auto defaultLocation = cacheDirPath / "comm_lib_trace_rank_";

    // For internal bc compatibility, we keep the old the ENV check.
    std::string fileNamePrefix = getCvarString(
        {"TORCH_FR_DUMP_TEMP_FILE", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"},
        defaultLocation.string().c_str());
    bool useDynamicFileName =
        getCvarBool({"TORCH_FR_DUMP_DYNAMIC_FILE_NAME"}, false);
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(
        new DebugInfoWriter(fileNamePrefix, rank, useDynamicFileName));
    DebugInfoWriter::registerWriter(std::move(writerPtr));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  if (hasWriterRegistered_.load()) {
    TORCH_WARN_ONCE(
        "DebugInfoWriter has already been registered, and since we need the writer to stay "
        "outside ProcessGroup, user needs to ensure that this extra registration is indeed needed. "
        "And we will only use the last registered writer.");
  }
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

template <>
float getDurationFromEvent<c10::Event>(
    c10::Event& startEvent,
    c10::Event& endEvent) {
  TORCH_CHECK(false, "getDuration not supported by c10::Event.");
}

// For any third party library that uses the flight recorder, if one wants to
// use an Event type other than c10::Event, one also needs to registers here to
// avoid linking errors.
template struct FlightRecorder<c10::Event>;

std::string dump_fr_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  return FlightRecorder<c10::Event>::get()->dump(
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>{},
      includeCollectives,
      includeStackTraces,
      onlyActive);
}

std::string dump_fr_trace_json(bool includeCollectives, bool onlyActive) {
  return FlightRecorder<c10::Event>::get()->dump_json(
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>{},
      includeCollectives,
      onlyActive);
}
} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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

- `c10/util/FileSystem.h`
- `torch/csrc/distributed/c10d/FlightRecorderDetail.hpp`
- `fstream`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `FlightRecorder.cpp_docs.md`
- **Keyword Index**: `FlightRecorder.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

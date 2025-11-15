# Documentation: `docs/torch/csrc/distributed/c10d/reducer_timer.hpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/reducer_timer.hpp_docs.md`
- **Size**: 4,761 bytes (4.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/reducer_timer.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/reducer_timer.hpp`
- **Size**: 2,384 bytes (2.33 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once
#include <c10/util/ApproximateClock.h>
#include <torch/csrc/autograd/profiler.h>

namespace c10d {
constexpr int kUnsetTime = -1;

inline int64_t current_time_in_nanos() {
  return c10::getTime();
}

class TORCH_API Timer {
 private:
  // The timestamp of forward call start time in each iteration.
  int64_t forward_start_time = kUnsetTime;
  // The timestamp of backward computation start and end time in each
  // iteration.
  int64_t backward_compute_start_time = kUnsetTime;
  int64_t backward_compute_end_time = kUnsetTime;
  // The timestamp of first communication call start time in each iteration.
  int64_t backward_comm_start_time = kUnsetTime;
  // The timestamp of last communication call end time in each iteration.
  int64_t backward_comm_end_time = kUnsetTime;

 public:
  enum class Event : uint8_t {
    kForwardStart,
    kBackwardComputeStart,
    kBackwardComputeEnd,
    kBackwardCommStart,
    kBackwardCommEnd,
  };

  // Record the current event, i.e., mark it as having occurred now. Default
  // CPU implementation.
  virtual void record(Event event) {
    getTimeRef(event) = current_time_in_nanos();
  }

  // Return the difference between when two events occurred, in nanoseconds.
  // Or nullopt if one of them hasn't been recorded.
  virtual std::optional<int64_t> measureDifference(Event start, Event end) = 0;

  virtual ~Timer() = default;

  // Return host-side timestamp, or nullopt if it has not yet been recorded.
  std::optional<int64_t> getTimestamp(Event event) {
    auto time = getTimeRef(event);
    if (time == kUnsetTime) {
      return std::nullopt;
    } else {
      return time;
    }
  }

  // Return host-side time member variable corresponding to the given event.
  int64_t& getTimeRef(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start_time;
      case Event::kBackwardComputeStart:
        return backward_compute_start_time;
      case Event::kBackwardComputeEnd:
        return backward_compute_end_time;
      case Event::kBackwardCommStart:
        return backward_comm_start_time;
      case Event::kBackwardCommEnd:
        return backward_comm_end_time;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }
};

TORCH_DECLARE_TYPED_REGISTRY(
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device);
} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `TORCH_API`, `Event`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/ApproximateClock.h`
- `torch/csrc/autograd/profiler.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

- **File Documentation**: `reducer_timer.hpp_docs.md`
- **Keyword Index**: `reducer_timer.hpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/distributed/c10d`):

- [`ProcessGroupWrapper.cpp_docs.md_docs.md`](./ProcessGroupWrapper.cpp_docs.md_docs.md)
- [`c10d.h_kw.md_docs.md`](./c10d.h_kw.md_docs.md)
- [`TCPStoreLibUvBackend.cpp_kw.md_docs.md`](./TCPStoreLibUvBackend.cpp_kw.md_docs.md)
- [`ProcessGroupGlooCuda.cpp_docs.md_docs.md`](./ProcessGroupGlooCuda.cpp_docs.md_docs.md)
- [`NanCheck.cu_docs.md_docs.md`](./NanCheck.cu_docs.md_docs.md)
- [`python_callback_work.hpp_kw.md_docs.md`](./python_callback_work.hpp_kw.md_docs.md)
- [`sequence_num.hpp_kw.md_docs.md`](./sequence_num.hpp_kw.md_docs.md)
- [`Functional.hpp_kw.md_docs.md`](./Functional.hpp_kw.md_docs.md)
- [`TCPStoreBackend.cpp_kw.md_docs.md`](./TCPStoreBackend.cpp_kw.md_docs.md)
- [`ProcessGroupUCC.cpp_kw.md_docs.md`](./ProcessGroupUCC.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `reducer_timer.hpp_docs.md_docs.md`
- **Keyword Index**: `reducer_timer.hpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

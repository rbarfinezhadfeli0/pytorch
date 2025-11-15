# Documentation: `docs/torch/csrc/distributed/c10d/reducer_cuda.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/reducer_cuda.cpp_docs.md`
- **Size**: 5,549 bytes (5.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/reducer_cuda.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/reducer_cuda.cpp`
- **Size**: 3,069 bytes (3.00 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/reducer_timer.hpp>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>

namespace c10d {
namespace {

const int kMilliSecondToNanosSecond = 1000000;

class CudaTimer : public Timer {
 private:
  c10::Device device;

  at::cuda::CUDAEvent forward_start = at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_compute_start =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_compute_end =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_comm_start =
      at::cuda::CUDAEvent(cudaEventDefault);
  at::cuda::CUDAEvent backward_comm_end = at::cuda::CUDAEvent(cudaEventDefault);

  at::cuda::CUDAEvent& getEvent(Event event) {
    switch (event) {
      case Event::kForwardStart:
        return forward_start;
      case Event::kBackwardComputeStart:
        return backward_compute_start;
      case Event::kBackwardComputeEnd:
        return backward_compute_end;
      case Event::kBackwardCommStart:
        return backward_comm_start;
      case Event::kBackwardCommEnd:
        return backward_comm_end;
      default:
        TORCH_INTERNAL_ASSERT(false);
    }
  }

 public:
  explicit CudaTimer(c10::Device dev) : device(dev) {}

  void record(Event event) override {
    // Parent class sets the host-side time
    Timer::record(event);
    c10::DeviceGuard g(device);
    getEvent(event).record();
  }

  std::optional<int64_t> measureDifference(Event start, Event end) override {
    c10::DeviceGuard g(device);
    at::cuda::CUDAEvent& start_event = getEvent(start);
    at::cuda::CUDAEvent& end_event = getEvent(end);
    // It is possible users did not call backward or run codes in
    // no-sync mode, in this case, some cudaEvents like "backward_compute_end"
    // or "backward_comm_start" or "backward_comm_end" will not be recorded.
    // cudaEvent is created when it is first time to be recorded.
    // If it is never recorded/created, skip synchronize and calculation.
    // Otherwise it will throw cuda errors.
    if (!start_event.isCreated() || !end_event.isCreated()) {
      return std::nullopt;
    }
    // set_runtime_stats_and_log is called at the beginning of forward call,
    // when it is cheap to synchronize the cuda events of previous iteration,
    // as mostly all cuda operations are finished in previous iteration.
    start_event.synchronize();
    end_event.synchronize();
    float milliseconds = start_event.elapsed_time(end_event);
    // If gpu_end is not recorded in this iteration,
    // milliseconds will have invalid value.
    // For some cases like DDP runs on non-sync mode,
    // gpu_end can not be recorded in this iteration and thus can not
    // calculate the valid avg_time.
    // In this case, skip calculating the avg_time and return.
    if (milliseconds < 0) {
      return std::nullopt;
    }
    return static_cast<int64_t>(milliseconds * kMilliSecondToNanosSecond);
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCUDA, CudaTimer)

} // namespace
} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `CudaTimer`, `sets`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/reducer_timer.hpp`
- `ATen/cuda/CUDAEvent.h`
- `c10/core/DeviceGuard.h`


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

- **File Documentation**: `reducer_cuda.cpp_docs.md`
- **Keyword Index**: `reducer_cuda.cpp_kw.md`
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

- **File Documentation**: `reducer_cuda.cpp_docs.md_docs.md`
- **Keyword Index**: `reducer_cuda.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

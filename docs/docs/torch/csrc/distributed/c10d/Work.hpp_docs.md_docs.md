# Documentation: `docs/torch/csrc/distributed/c10d/Work.hpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/Work.hpp_docs.md`
- **Size**: 8,282 bytes (8.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/Work.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/Work.hpp`
- **Size**: 5,850 bytes (5.71 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#include <ATen/ATen.h>
#include <chrono>
#include <mutex>
#include <vector>

constexpr auto kNoTimeout = std::chrono::milliseconds(0);

namespace c10d {

constexpr const char* const kSeqNumStoreKey = "SEQ_NUM_STORE_KEY";

enum class OpType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_COALESCED = 2,
  REDUCE = 3,
  ALLGATHER = 4,
  _ALLGATHER_BASE = 5,
  ALLGATHER_COALESCED = 6,
  GATHER = 7,
  SCATTER = 8,
  REDUCE_SCATTER = 9,
  ALLTOALL_BASE = 10,
  ALLTOALL = 11,
  SEND = 12,
  RECV = 13,
  RECVANYSOURCE = 14,
  BARRIER = 15,
  _REDUCE_SCATTER_BASE = 16,
  COALESCED = 17,
  _ALLREDUCE_SPARSE = 18,
  UNKNOWN = 100,
};

// TODO: support different types of failures/errors
enum class WorkResult : std::uint8_t {
  SUCCESS = 0,
  TIMEOUT = 1,
  COMM_ERROR = 2,
  UNKNOWN = 100,
};

// Converts OpType to human readable string.
TORCH_API std::string opTypeToString(OpType opType);

// Whether or not an OP is an p2p op (SEND, RECV, RECVANYSOURCE)
TORCH_API bool isP2POp(OpType opType, bool batchP2P = false);

// Please do not use Work API, it is going away, to be
// replaced by ivalue::Future.
// Python binding for this class might change, please do not assume
// this will be bound using pybind.
class TORCH_API Work : public torch::CustomClassHolder {
 public:
  Work(
      int rank = -1,
      OpType opType = OpType::UNKNOWN,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  ~Work() override;

  // Checks if request has completed. Non-blocking operation.
  virtual bool isCompleted();

  // Returns if the work completed successfully.
  // If false, the exception function can be called to get details.
  virtual bool isSuccess() const;

  // Returns exception if isSuccess() returned false.
  virtual std::exception_ptr exception() const;

  // Returns source rank if this objects represents a recv-from-any.
  virtual int sourceRank() const;

  // Returns result tensors, if applicable.
  // If work is not supposed to have result, we return empty list.
  virtual std::vector<at::Tensor> result();

  // Ensures that operations on the output tensors that are invoked
  // after this function returns are correctly sequenced after the
  // asynchronous completion of this work.
  //
  // For CUDA tensors, it inserts stream synchronization such that
  // the streams of the caller wait for completion of the
  // asynchronous operations on the destination tensors.
  //
  // For CPU tensors, it is currently a nop.
  //
  // This function should only be used if the caller polls for
  // completion through the `isCompleted` function, it has returned
  // true, and the `isSuccess` function also has returned true.
  //
  virtual void synchronize();

  // Waits until request completes. Blocking operation.
  // Throws if the work completed with an exception.
  // Returns false if the work is aborted.
  // Otherwise, it always returns true, indicating the work is completed.
  //
  // Functionally equivalent to:
  //
  //   while (!isCompleted()) { /* nop */ }
  //   auto success = isSuccess();
  //   if (!success) { std::rethrow_exception(exception()); }
  //   return success;
  //
  virtual bool wait(std::chrono::milliseconds timeout = kNoTimeout);

  // Blocks the current stream until the work is completed.
  // This is equivalent to synchronize for CUDA tensors but works for both CPU
  // tensors and CUDA tensors by using a spinlock CUDA kernel.
  // This will immediately return.
  // If no stream is active it will throw an error.
  virtual void blockCurrentStream();

  virtual void abort();

  // Returns a Future object that will be associated with the completion of
  // work. Only NCCL backend is currently supported.
  virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture();

  // Get a Future object that would be marked as either success or failure
  // This API can be used by the user to track the completion of the work
  // and handle the exception if any.
  virtual c10::intrusive_ptr<c10::ivalue::Future> getFutureResult();

  virtual float getDuration() const;

  virtual uint64_t getSequencenumber() const;

  OpType retrieveOpType() const;

  static c10::intrusive_ptr<Work> create_from_future(
      const c10::intrusive_ptr<c10::ivalue::Future>& /*future*/);

 protected:
  // Completes the work object and optionally sets the exception in a
  // thread-safe manner. Notifies all waiting condition variables as well.
  void finish(std::exception_ptr exception = nullptr);

  // Similar to finish, but throws an exception if one is already set or
  // provided by the user.
  void finishAndThrow(std::exception_ptr exception);

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exception_;

  // Current rank of the node.
  const int rank_;

  // Operation type that this work object refers to.
  OpType opType_;

  // When profiling, the callback to record end of operation event. This
  // callback needs to be called when collective operation is complete.
  std::function<void()> recordFunctionEndCallback_;
};

struct TORCH_API WorkInfo {
  WorkInfo(
      const OpType& opType,
      const uint64_t seq,
      const std::chrono::time_point<std::chrono::steady_clock>& timeStarted,
      const std::chrono::time_point<std::chrono::steady_clock>& timeFinished,
      const std::chrono::duration<float>& activeDuration)
      : opType(opType),
        seq(seq),
        timeStarted(timeStarted),
        timeFinished(timeFinished),
        activeDuration(activeDuration) {}

  OpType opType;
  uint64_t seq;
  std::chrono::time_point<std::chrono::steady_clock> timeStarted;
  std::chrono::time_point<std::chrono::steady_clock> timeFinished;
  std::chrono::duration<float> activeDuration;
};

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `OpType`, `WorkResult`, `might`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `chrono`
- `mutex`
- `vector`


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

- **File Documentation**: `Work.hpp_docs.md`
- **Keyword Index**: `Work.hpp_kw.md`
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

- **File Documentation**: `Work.hpp_docs.md_docs.md`
- **Keyword Index**: `Work.hpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

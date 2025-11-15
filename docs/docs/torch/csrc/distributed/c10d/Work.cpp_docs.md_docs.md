# Documentation: `docs/torch/csrc/distributed/c10d/Work.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/Work.cpp_docs.md`
- **Size**: 8,285 bytes (8.09 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/Work.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/Work.cpp`
- **Size**: 5,810 bytes (5.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ThreadLocalState.h>
#include <distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/cuda/StreamBlock.hpp>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <utility>

namespace c10d {

Work::Work(
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : rank_(rank), opType_(opType) {
  if (profilingTitle != nullptr) {
    auto recordingFunction =
        std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
      // Work events follow a future like pattern and can potentially be marked
      // as complete by different threads, so explicitly set as async event.
      recordingFunction->_setAsync();
      // Passing input tensor to recordFunction allows for shape information in
      // profiling output.
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }
      recordingFunction->before(
          profilingTitle,
          c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
      std::function<void()> end_handler = [recordingFunction]() {
        recordingFunction->end();
      };
      recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

OpType Work::retrieveOpType() const {
  return opType_;
}

Work::~Work() = default;

bool Work::isCompleted() {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_;
}

bool Work::isSuccess() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !exception_;
}

std::exception_ptr Work::exception() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return exception_;
}

int Work::sourceRank() const {
  TORCH_CHECK(
      false,
      "sourceRank() may only be called on work objects "
      "that correspond to a recv or recv-from-any call.");
}

std::vector<at::Tensor> Work::result() {
  TORCH_CHECK(false, "result() not implemented.");
}

void Work::synchronize() {
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<Work>::unsafe_reclaim_from_nonowning(this));
  }
}

bool Work::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (timeout == kNoTimeout) {
    // This waits without a timeout.
    cv_.wait(lock, [&] { return completed_; });
  } else {
    // Waits for the user-provided timeout.
    cv_.wait_for(lock, timeout, [&] { return completed_; });
    if (!completed_) {
      // Throw exception if the wait operation timed out and the work was not
      // completed.
      TORCH_CHECK(false, "Operation timed out!");
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  synchronize();
  // Always return true, because abort API is not implemented.
  return true;
}

void Work::blockCurrentStream() {
  // block cuda stream indefinitely until work is completed.
  std::shared_ptr<c10d::cuda::StreamBlock> handle =
      c10d::cuda::block_stream(std::chrono::milliseconds(0));

  getFuture()->addCallback(
      [handle](c10::ivalue::Future& future) { handle->abort(); });
}

void Work::abort() {
  TORCH_CHECK(false, "Work::abort not implemented.");
}

c10::intrusive_ptr<c10::ivalue::Future> Work::getFuture(){
    TORCH_CHECK(false, "Work::getFuture not implemented.")}

c10::intrusive_ptr<c10::ivalue::Future> Work::getFutureResult() {
  TORCH_CHECK(false, "Work::getFutureResult not implemented.")
}

void Work::finish(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = std::move(exception);
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  lock.unlock();
  cv_.notify_all();
}

void Work::finishAndThrow(std::exception_ptr exception) {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_ = true;
  exception_ = std::move(exception);
  if (recordFunctionEndCallback_) {
    recordFunctionEndCallback_();
    recordFunctionEndCallback_ = nullptr;
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

float Work::getDuration() const {
  TORCH_CHECK(false, "This Backend doesn't support getDuration.");
}

uint64_t Work::getSequencenumber() const {
  TORCH_CHECK(false, "This Backend doesn't support getSequencenumber.");
}

class FutureWrappingWork : public Work {
 public:
  FutureWrappingWork(c10::intrusive_ptr<c10::ivalue::Future> fut)
      : _fut(std::move(fut)) {}

  ~FutureWrappingWork() override = default;

  bool isCompleted() override {
    return _fut->completed();
  }

  bool isSuccess() const override {
    return _fut->hasValue();
  }

  std::exception_ptr exception() const override {
    return _fut->exception_ptr();
  }

  int sourceRank() const override {
    TORCH_CHECK(false, "FutureWrappingWork::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    return _fut->value().toPyObjectHolder()->extractTensors();
  }

  bool wait(std::chrono::milliseconds timeout) override {
    // FIXME
    TORCH_CHECK(
        timeout == kNoTimeout,
        "FutureWrappingWork::wait() with finite timeout not implemented");
    _fut->wait();
    return true;
  }

  void abort() override {
    TORCH_CHECK(false, "FutureWrappingWork::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    return _fut;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> _fut;
};

c10::intrusive_ptr<Work> Work::create_from_future(
    const c10::intrusive_ptr<c10::ivalue::Future>& future) {
  return c10::make_intrusive<FutureWrappingWork>(future);
}

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `FutureWrappingWork`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ThreadLocalState.h`
- `distributed/c10d/ProcessGroup.hpp`
- `torch/csrc/distributed/c10d/cuda/StreamBlock.hpp`
- `torch/csrc/distributed/c10d/Work.hpp`
- `utility`


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
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `Work.cpp_docs.md`
- **Keyword Index**: `Work.cpp_kw.md`
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

- **File Documentation**: `Work.cpp_docs.md_docs.md`
- **Keyword Index**: `Work.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

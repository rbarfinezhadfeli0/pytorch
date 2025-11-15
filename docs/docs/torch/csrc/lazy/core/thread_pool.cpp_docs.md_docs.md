# Documentation: `docs/torch/csrc/lazy/core/thread_pool.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/core/thread_pool.cpp_docs.md`
- **Size**: 6,885 bytes (6.72 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/core/thread_pool.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/core/thread_pool.cpp`
- **Size**: 4,280 bytes (4.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/core/thread_pool.h>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

namespace torch::lazy {
namespace {

class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads) {
    threads_.reserve(num_threads);
    for ([[maybe_unused]] const auto i : c10::irange(num_threads)) {
      threads_.emplace_back([this]() {
        c10::setThreadName("pt_thread_pool");
        Worker();
      });
    }
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      exiting_ = true;
      cv_.notify_all();
    }
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void Schedule(std::function<void()> closure) {
    // If we have more work scheduled than waiting worker threads, just schedule
    // it on a separate thread. This prevents tricky thread-pool-size-deadlocks
    // caused by an undersized thread pool and closures that end up doing sync
    // waits on the pool threads.
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (work_.size() < waiting_) {
        work_.emplace_back(std::move(closure));
        lock.unlock();
        cv_.notify_one();
        return;
      }
    }
    ScheduleOnThread(std::move(closure));
  }

 private:
  void Worker() {
    while (true) {
      std::function<void()> closure = GetWork();
      if (closure == nullptr) {
        break;
      }
      try {
        closure();
      } catch (const std::exception& ex) {
        TORCH_LAZY_COUNTER("ThreadPoolException", 1);
        LOG(ERROR) << "Exception from running thread pool closure: "
                   << ex.what();
      }
    }
  }

  void ScheduleOnThread(std::function<void()> closure) {
    std::thread thread(std::move(closure));
    thread.detach();
  }

  std::function<void()> GetWork() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++waiting_;
    cv_.wait(lock, [this] { return exiting_ || !work_.empty(); });
    --waiting_;
    if (work_.empty()) {
      return nullptr;
    }
    std::function<void()> closure(std::move(work_.front()));
    work_.pop_front();
    return closure;
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool exiting_ = false;
  std::deque<std::function<void()>> work_;
  size_t waiting_ = 0;
};

ThreadPool* GetIoThreadPool() {
  static ThreadPool* pool =
      new ThreadPool(FLAGS_torch_lazy_io_thread_pool_size);
  return pool;
}

} // namespace

class Completion::Data {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return completed_; });
    if (exptr_ != nullptr) {
      std::rethrow_exception(exptr_);
    }
  }

  static std::function<void()> GetCompleter(
      const std::shared_ptr<Data>& data,
      std::function<void()> closure) {
    auto closure_wrapper = [closure = std::move(closure), data]() {
      std::exception_ptr exptr;
      try {
        closure();
      } catch (...) {
        exptr = std::current_exception();
      }
      data->Complete(exptr);
    };
    return closure_wrapper;
  }

 private:
  void Complete(std::exception_ptr exptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    exptr_ = std::move(exptr);
    completed_ = true;
    cv_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool completed_ = false;
  std::exception_ptr exptr_;
};

Completion::Completion(std::shared_ptr<Data> data) : data_(std::move(data)) {}

void Completion::Wait() {
  data_->Wait();
}

void ScheduleIoClosure(std::function<void()> closure) {
  GetIoThreadPool()->Schedule(std::move(closure));
}

Completion ScheduleIoClosureWithCompletion(std::function<void()> closure) {
  auto data = std::make_shared<Completion::Data>();
  GetIoThreadPool()->Schedule(
      Completion::Data::GetCompleter(data, std::move(closure)));
  return Completion(std::move(data));
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `class`

**Classes/Structs**: `ThreadPool`, `Completion`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/core/thread_pool.h`
- `c10/util/Logging.h`
- `c10/util/irange.h`
- `c10/util/thread_name.h`
- `torch/csrc/lazy/core/config.h`
- `torch/csrc/lazy/core/metrics.h`
- `condition_variable`
- `deque`
- `exception`
- `mutex`
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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `thread_pool.cpp_docs.md`
- **Keyword Index**: `thread_pool.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/lazy/core`):

- [`helpers.cpp_docs.md_docs.md`](./helpers.cpp_docs.md_docs.md)
- [`tensor_util.h_kw.md_docs.md`](./tensor_util.h_kw.md_docs.md)
- [`permutation_util.h_kw.md_docs.md`](./permutation_util.h_kw.md_docs.md)
- [`ir_util.cpp_kw.md_docs.md`](./ir_util.cpp_kw.md_docs.md)
- [`shape_inference.h_kw.md_docs.md`](./shape_inference.h_kw.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`shape_inference.cpp_kw.md_docs.md`](./shape_inference.cpp_kw.md_docs.md)
- [`hash.h_kw.md_docs.md`](./hash.h_kw.md_docs.md)
- [`multi_wait.cpp_kw.md_docs.md`](./multi_wait.cpp_kw.md_docs.md)
- [`lazy_graph_executor.cpp_docs.md_docs.md`](./lazy_graph_executor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `thread_pool.cpp_docs.md_docs.md`
- **Keyword Index**: `thread_pool.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `c10/core/thread_pool.h`

## File Metadata

- **Path**: `c10/core/thread_pool.h`
- **Size**: 2,997 bytes (2.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <c10/macros/Export.h>
#include <c10/util/Registry.h>
#include <c10/util/numa.h>
#include <c10/util/thread_name.h>

namespace c10 {

class C10_API TaskThreadPoolBase {
 public:
  virtual void run(std::function<void()> func) = 0;

  virtual size_t size() const = 0;

  /**
   * The number of available (i.e. idle) threads in this thread pool.
   */
  virtual size_t numAvailable() const = 0;

  /**
   * Check if the current thread is from the thread pool.
   */
  virtual bool inThreadPool() const = 0;

  virtual ~TaskThreadPoolBase() noexcept = default;

  static size_t defaultNumThreads();
};

class C10_API ThreadPool : public c10::TaskThreadPoolBase {
 protected:
  struct task_element_t {
    bool run_with_id;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::function<void()> no_id;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(std::function<void()> f)
        : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
    explicit task_element_t(std::function<void(std::size_t)> f)
        : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
  };

  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
  std::atomic_bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;
  int numa_node_id_;

 public:
  ThreadPool() = delete;

  explicit ThreadPool(
      int pool_size,
      int numa_node_id = -1,
      const std::function<void()>& init_thread = nullptr);

  ~ThreadPool() override;

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(std::function<void()> func) override;

  template <typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void waitWorkComplete();

 private:
  // @brief Entry point for pool threads.
  void main_loop(std::size_t index);
};

class C10_API TaskThreadPool : public c10::ThreadPool {
 public:
  explicit TaskThreadPool(int pool_size, int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
          setThreadName("CaffeTaskThread");
          NUMABind(numa_node_id);
        }) {}
};

C10_DECLARE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPoolBase,
    int,
    int,
    bool);

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `C10_API`, `C10_API`, `task_element_t`, `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `atomic`
- `condition_variable`
- `cstddef`
- `functional`
- `mutex`
- `queue`
- `thread`
- `utility`
- `vector`
- `c10/macros/Export.h`
- `c10/util/Registry.h`
- `c10/util/numa.h`
- `c10/util/thread_name.h`


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

Files in the same folder (`c10/core`):

- [`DispatchKey.cpp_docs.md`](./DispatchKey.cpp_docs.md)
- [`CopyBytes.h_docs.md`](./CopyBytes.h_docs.md)
- [`OptionalRef.h_docs.md`](./OptionalRef.h_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`SafePyObject.cpp_docs.md`](./SafePyObject.cpp_docs.md)
- [`DeviceType.cpp_docs.md`](./DeviceType.cpp_docs.md)
- [`SymBool.cpp_docs.md`](./SymBool.cpp_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `thread_pool.h_docs.md`
- **Keyword Index**: `thread_pool.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

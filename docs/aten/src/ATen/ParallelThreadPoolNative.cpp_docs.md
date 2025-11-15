# Documentation: `aten/src/ATen/ParallelThreadPoolNative.cpp`

## File Metadata

- **Path**: `aten/src/ATen/ParallelThreadPoolNative.cpp`
- **Size**: 2,564 bytes (2.50 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Config.h>
#if AT_PARALLEL_OPENMP || AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>
#include <ATen/ThreadLocalState.h>

#include <atomic>

namespace at {

namespace {
const int NOT_SET = -1;
const int CONSUMED = -2;

// Number of inter-op threads set by the user;
// NOT_SET -> positive value -> CONSUMED
// (CONSUMED - thread pool is initialized)
// or
// NOT_SET -> CONSUMED
std::atomic<int> num_interop_threads{NOT_SET};

// thread pool global instance is hidden,
// users should use at::launch and get/set_num_interop_threads interface
TaskThreadPoolBase& get_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ num_interop_threads.exchange(CONSUMED),
          /* create_new */ true);
  return *pool;
}

// Factory function for ThreadPoolRegistry
std::shared_ptr<TaskThreadPoolBase> create_c10_threadpool(
    int device_id,
    int pool_size,
    bool create_new) {
  // For now, the only accepted device id is 0
  TORCH_CHECK(device_id == 0);
  // Create new thread pool
  TORCH_CHECK(create_new);
  return std::make_shared<PTThreadPool>(pool_size);
}

} // namespace

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, create_c10_threadpool)

void set_num_interop_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");

  int no_value = NOT_SET;
  TORCH_CHECK(num_interop_threads.compare_exchange_strong(no_value, nthreads),
      "Error: cannot set number of interop threads after parallel work "
      "has started or set_num_interop_threads called");
}

size_t get_num_interop_threads() {
  at::internal::lazy_init_num_threads();
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    // return default value
    return TaskThreadPoolBase::defaultNumThreads();
  } else {
    return get_pool().size();
  }
}

namespace internal {
void launch_no_thread_state(std::function<void()> fn) {
#if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
  intraop_launch(std::move(fn));
#else
  get_pool().run(std::move(fn));
#endif
}
} // namespace internal

void launch(std::function<void()> func) {
  // NOLINTNEXTLINE(modernize-avoid-bind)
  internal::launch_no_thread_state(std::bind([](
    const std::function<void()>& f, const ThreadLocalState& thread_locals) {
      ThreadLocalStateGuard guard(thread_locals);
      f();
    },
    std::move(func),
    ThreadLocalState()
  ));
}

} // namespace at
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `C10_REGISTER_CREATOR`, `internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/Parallel.h`
- `ATen/PTThreadPool.h`
- `ATen/ThreadLocalState.h`
- `atomic`


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

Files in the same folder (`aten/src/ATen`):

- [`TensorGeometry.cpp_docs.md`](./TensorGeometry.cpp_docs.md)
- [`ROCmFABackend.h_docs.md`](./ROCmFABackend.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`ParallelCommon.cpp_docs.md`](./ParallelCommon.cpp_docs.md)
- [`ZeroTensorFallback.cpp_docs.md`](./ZeroTensorFallback.cpp_docs.md)
- [`CachedTensorUtils.h_docs.md`](./CachedTensorUtils.h_docs.md)
- [`LegacyBatchedFallback.cpp_docs.md`](./LegacyBatchedFallback.cpp_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`ExpandUtils.h_docs.md`](./ExpandUtils.h_docs.md)
- [`TensorIteratorInternal.h_docs.md`](./TensorIteratorInternal.h_docs.md)


## Cross-References

- **File Documentation**: `ParallelThreadPoolNative.cpp_docs.md`
- **Keyword Index**: `ParallelThreadPoolNative.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

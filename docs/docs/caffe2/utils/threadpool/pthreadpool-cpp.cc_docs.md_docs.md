# Documentation: `docs/caffe2/utils/threadpool/pthreadpool-cpp.cc_docs.md`

## File Metadata

- **Path**: `docs/caffe2/utils/threadpool/pthreadpool-cpp.cc_docs.md`
- **Size**: 6,260 bytes (6.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `caffe2/utils/threadpool/pthreadpool-cpp.cc`

## File Metadata

- **Path**: `caffe2/utils/threadpool/pthreadpool-cpp.cc`
- **Size**: 3,702 bytes (3.62 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <caffe2/utils/threadpool/thread_pool_guard.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <c10/util/Exception.h>

namespace {
// After fork, the child process inherits the data-structures of the parent
// process' thread-pool, but since those threads don't exist, the thread-pool
// is corrupt. It's leaked in order to prevent segfaults.
// Ref: https://github.com/pytorch/pytorch/issues/54752#issuecomment-810315302
bool leak_corrupted_threadpool = false;

void child_atfork() {
  leak_corrupted_threadpool = true;
}

} // namespace

namespace caffe2 {

PThreadPool::PThreadPool(const size_t thread_count)
    : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy) {}

size_t PThreadPool::get_thread_count() const {
  std::lock_guard<std::mutex> lock{mutex_};

  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");
  return pthreadpool_get_threads_count(threadpool_.get());
}

void PThreadPool::set_thread_count(const size_t thread_count) {
  // No need to do anything if the count is same
  if (thread_count == get_thread_count()) {
    return;
  }

  std::lock_guard<std::mutex> lock{mutex_};

  // As it stands, pthreadpool is an entirely data parallel framework with no
  // support for task parallelism.  Hence, all functions are blocking, and no
  // user-provided tasks can be in flight when the control is returned to the
  // user of the API, which means re-initializing the library, without the
  // need to wait on any pending tasks, is all one needs to do to re-adjust
  // the thread count.
  threadpool_.reset(pthreadpool_create(thread_count));
}

void PThreadPool::run(
    const std::function<void(size_t)>& fn,
    const size_t range) {
  // Run on same thread if _NoPThreadPoolGuard guard is enabled
  if (caffe2::_NoPThreadPoolGuard::is_enabled()) {
    for (size_t i = 0; i < range; ++i) {
      fn(i);
    }
    return;
  }

  std::lock_guard<std::mutex> lock{mutex_};

  TORCH_INTERNAL_ASSERT(!caffe2::_NoPThreadPoolGuard::is_enabled(), "Inside a threadpool guard!");
  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");

  struct Context final {
    const std::function<void(size_t)>& fn;
  } context{
      fn,
  };

  pthreadpool_parallelize_1d(
      threadpool_.get(),
      // Note: pthreadpool_parallelize_1d() is a blocking function.  The
      // function pointer to this lambda passed on to
      // pthreadpool_parallelize_1d() cannot go out of scope until
      // pthreadpool_parallelize_1d() returns.
      [](void* const context, const size_t item) {
        reinterpret_cast<Context*>(context)->fn(item);
      },
      &context,
      range,
      0u);
}

PThreadPool* pthreadpool(size_t thread_count) {
  static auto threadpool =
    std::make_unique<PThreadPool>(thread_count);
#if !(defined(WIN32))
  static std::once_flag flag;
  std::call_once(flag, []() {
    pthread_atfork(nullptr, nullptr, child_atfork);
  });
#endif
  if (C10_UNLIKELY(leak_corrupted_threadpool)) {
    leak_corrupted_threadpool = false;
    if (auto leaked = threadpool.release()) {
      auto num_threads = leaked->get_thread_count();
      // NOLINTNEXTLINE(modernize-make-unique)
      threadpool.reset(new PThreadPool(num_threads));
    }
  }
  return threadpool.get();
}

PThreadPool* pthreadpool() {
  return pthreadpool(getDefaultNumThreads());
}

pthreadpool_t pthreadpool_() {
  if (caffe2::_NoPThreadPoolGuard::is_enabled()) {
    return nullptr;
  }
  PThreadPool* const threadpool = pthreadpool();
  TORCH_INTERNAL_ASSERT(
      threadpool, "Failed to acquire an instance of PThreadPool!");
  return threadpool->threadpool_.get();
}

} // namespace caffe2

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `namespace`

**Classes/Structs**: `Context`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `caffe2/utils/threadpool`, which is part of the **Caffe2** deep learning framework.



## Dependencies

### Import Dependencies

This file includes:

- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `caffe2/utils/threadpool/thread_pool_guard.h`
- `caffe2/utils/threadpool/ThreadPool.h`
- `c10/util/Exception.h`


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

Files in the same folder (`caffe2/utils/threadpool`):

- [`thread_pool_guard.cpp_docs.md`](./thread_pool_guard.cpp_docs.md)
- [`thread_pool_guard.h_docs.md`](./thread_pool_guard.h_docs.md)
- [`pthreadpool.h_docs.md`](./pthreadpool.h_docs.md)
- [`pthreadpool_impl.cc_docs.md`](./pthreadpool_impl.cc_docs.md)
- [`ThreadPool.cc_docs.md`](./ThreadPool.cc_docs.md)
- [`WorkersPool.h_docs.md`](./WorkersPool.h_docs.md)
- [`pthreadpool-cpp.h_docs.md`](./pthreadpool-cpp.h_docs.md)
- [`pthreadpool.cc_docs.md`](./pthreadpool.cc_docs.md)
- [`ThreadPool.h_docs.md`](./ThreadPool.h_docs.md)


## Cross-References

- **File Documentation**: `pthreadpool-cpp.cc_docs.md`
- **Keyword Index**: `pthreadpool-cpp.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/caffe2/utils/threadpool`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/caffe2/utils/threadpool`, which is part of the **Caffe2** deep learning framework.



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

Files in the same folder (`docs/caffe2/utils/threadpool`):

- [`ThreadPoolCommon.h_docs.md_docs.md`](./ThreadPoolCommon.h_docs.md_docs.md)
- [`pthreadpool-cpp.h_kw.md_docs.md`](./pthreadpool-cpp.h_kw.md_docs.md)
- [`pthreadpool_impl.cc_docs.md_docs.md`](./pthreadpool_impl.cc_docs.md_docs.md)
- [`pthreadpool_impl.cc_kw.md_docs.md`](./pthreadpool_impl.cc_kw.md_docs.md)
- [`thread_pool_guard.cpp_docs.md_docs.md`](./thread_pool_guard.cpp_docs.md_docs.md)
- [`ThreadPool.h_docs.md_docs.md`](./ThreadPool.h_docs.md_docs.md)
- [`pthreadpool-cpp.cc_kw.md_docs.md`](./pthreadpool-cpp.cc_kw.md_docs.md)
- [`ThreadPool.h_kw.md_docs.md`](./ThreadPool.h_kw.md_docs.md)
- [`ThreadPoolCommon.h_kw.md_docs.md`](./ThreadPoolCommon.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `pthreadpool-cpp.cc_docs.md_docs.md`
- **Keyword Index**: `pthreadpool-cpp.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `aten/src/ATen/ParallelOpenMP.cpp`

## File Metadata

- **Path**: `aten/src/ATen/ParallelOpenMP.cpp`
- **Size**: 2,922 bytes (2.85 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/Config.h>
#include <ATen/core/jit_type.h>
#if AT_PARALLEL_OPENMP
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>

#include <atomic>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/IDeepRegistration.h>
#endif

#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

namespace at {

namespace {
// Number of threads set by the user
std::atomic<int> num_threads{-1};
thread_local int this_thread_id{0};

} // namespace

void init_num_threads() {
  auto nthreads = num_threads.load();
  if (nthreads > 0) {
    set_num_threads(nthreads);
  } else {
#if defined(_OPENMP) && AT_MKL_ENABLED() && !AT_MKL_SEQUENTIAL()
    // If we are using MKL an OpenMP make sure the number of threads match.
    // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
    // size of the OpenMP thread pool, resulting in worse performance (and memory
    // leaks in GCC 5.4)
    omp_set_num_threads(mkl_get_max_threads());
#elif defined(_OPENMP)
    omp_set_num_threads(intraop_default_num_threads());
#endif
  }
}

void set_num_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
  num_threads.store(nthreads);
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
#if AT_MKL_ENABLED()
  mkl_set_num_threads_local(nthreads);

  // because PyTorch uses OpenMP outside of MKL invocations
  // as well, we want this flag to be false, so that
  // threads aren't destroyed and recreated across every
  // MKL / non-MKL boundary of OpenMP usage
  // See https://github.com/pytorch/pytorch/issues/13757
  mkl_set_dynamic(false);
#endif
#ifdef USE_PTHREADPOOL
  // because PyTorch uses caffe2::pthreadpool() in QNNPACK
  caffe2::PThreadPool* const pool = caffe2::pthreadpool(nthreads);
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
#endif
#if AT_MKLDNN_ENABLED()
  at::native::mkldnn::clear_computation_cache();
#endif
}

// Explicitly calling omp_get_max_threads() as the size of the parallel
// region might be different in the new thread;
// Use init_num_threads() during thread initialization to ensure
// consistent size of parallel region in different threads
int get_num_threads() {
#ifdef _OPENMP
  at::internal::lazy_init_num_threads();
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int get_thread_num() {
  return this_thread_id;
}

namespace internal {
void set_thread_num(int id) {
  this_thread_id = id;
}
}

bool in_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel();
#else
  return false;
#endif
}

void intraop_launch(const std::function<void()>& func) {
  // execute inline in openmp case
  func();
}

c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    const std::function<void()>& func) {
  func();
  auto future = c10::make_intrusive<c10::ivalue::Future>(NoneType::get());
  future->markCompleted();
  return future;
}

} // namespace at
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `void`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/core/jit_type.h`
- `ATen/Parallel.h`
- `ATen/ParallelFuture.h`
- `atomic`
- `mkl.h`
- `ATen/native/mkldnn/IDeepRegistration.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `ParallelOpenMP.cpp_docs.md`
- **Keyword Index**: `ParallelOpenMP.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/aten/src/ATen/Parallel.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/Parallel.h_docs.md`
- **Size**: 7,406 bytes (7.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/Parallel.h`

## File Metadata

- **Path**: `aten/src/ATen/Parallel.h`
- **Size**: 4,785 bytes (4.67 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/Config.h>
#include <c10/macros/Macros.h>
#include <functional>
#include <string>

namespace at {

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Called during new thread initialization
TORCH_API void init_num_threads();

// Sets the number of threads to be used in parallel region
TORCH_API void set_num_threads(int /*nthreads*/);

// Returns the maximum number of threads that may be used in a parallel region
TORCH_API int get_num_threads();

// Returns the current thread number (starting from 0)
// in the current parallel region, or 0 in the sequential region
TORCH_API int get_thread_num();

// Checks whether the code runs in parallel region
TORCH_API bool in_parallel_region();

namespace internal {

// Initialise num_threads lazily at first parallel call
inline void lazy_init_num_threads() {
  thread_local bool init = false;
  if (C10_UNLIKELY(!init)) {
    at::init_num_threads();
    init = true;
  }
}

TORCH_API void set_thread_num(int /*id*/);

class TORCH_API ThreadIdGuard {
 public:
  ThreadIdGuard(int new_id) : old_id_(at::get_thread_num()) {
    set_thread_num(new_id);
  }

  ~ThreadIdGuard() {
    set_thread_num(old_id_);
  }

 private:
  int old_id_;
};

} // namespace internal

/*
parallel_for

begin: index at which to start applying user function

end: index at which to stop applying user function

grain_size: number of elements per chunk. impacts the degree of parallelization

f: user function applied in parallel to the chunks, signature:
  void f(int64_t begin, int64_t end)

Warning: parallel_for does NOT copy thread local
states from the current thread to the worker threads.
This means for example that Tensor operations CANNOT be used in the
body of your function, only data pointers.
*/
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f);

/*
parallel_reduce

begin: index at which to start applying reduction

end: index at which to stop applying reduction

grain_size: number of elements per chunk. impacts number of elements in
intermediate results tensor and degree of parallelization.

ident: identity for binary combination function sf. sf(ident, x) needs to return
x.

f: function for reduction over a chunk. f needs to be of signature scalar_t
f(int64_t partial_begin, int64_t partial_end, scalar_t identify)

sf: function to combine two partial results. sf needs to be of signature
scalar_t sf(scalar_t x, scalar_t y)

For example, you might have a tensor of 10000 entries and want to sum together
all the elements. Parallel_reduce with a grain_size of 2500 will then allocate
an intermediate result tensor with 4 elements. Then it will execute the function
"f" you provide and pass the beginning and end index of these chunks, so
0-2499, 2500-4999, etc. and the combination identity. It will then write out
the result from each of these chunks into the intermediate result tensor. After
that it'll reduce the partial results from each chunk into a single number using
the combination function sf and the identity ident. For a total summation this
would be "+" and 0 respectively. This is similar to tbb's approach [1], where
you need to provide a function to accumulate a subrange, a function to combine
two partial results and an identity.

Warning: parallel_reduce does NOT copy thread local
states from the current thread to the worker threads.
This means for example that Tensor operations CANNOT be used in the
body of your function, only data pointers.

[1] https://software.intel.com/en-us/node/506154
*/
template <class scalar_t, class F, class SF>
inline scalar_t parallel_reduce(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const scalar_t ident,
    const F& f,
    const SF& sf);

// Returns a detailed string describing parallelization settings
TORCH_API std::string get_parallel_info();

// Sets number of threads used for inter-op parallelism
TORCH_API void set_num_interop_threads(int /*nthreads*/);

// Returns the number of threads used for inter-op parallelism
TORCH_API size_t get_num_interop_threads();

// Launches inter-op parallel task
TORCH_API void launch(std::function<void()> func);
namespace internal {
void launch_no_thread_state(std::function<void()> fn);
} // namespace internal

// Launches intra-op parallel task
TORCH_API void intraop_launch(const std::function<void()>& func);

// Returns number of intra-op threads used by default
TORCH_API int intraop_default_num_threads();

} // namespace at

#if AT_PARALLEL_OPENMP
#include <ATen/ParallelOpenMP.h> // IWYU pragma: keep
#elif AT_PARALLEL_NATIVE
#include <ATen/ParallelNative.h> // IWYU pragma: keep
#endif

#include <ATen/Parallel-inl.h> // IWYU pragma: keep

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`

**Classes/Structs**: `TORCH_API`, `F`, `scalar_t`, `F`, `SF`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `c10/macros/Macros.h`
- `functional`
- `string`
- `ATen/ParallelOpenMP.h`
- `ATen/ParallelNative.h`
- `ATen/Parallel-inl.h`


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

- **File Documentation**: `Parallel.h_docs.md`
- **Keyword Index**: `Parallel.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen`):

- [`Dispatch.cpp_docs.md_docs.md`](./Dispatch.cpp_docs.md_docs.md)
- [`Context.cpp_docs.md_docs.md`](./Context.cpp_docs.md_docs.md)
- [`ThreadLocalState.cpp_docs.md_docs.md`](./ThreadLocalState.cpp_docs.md_docs.md)
- [`DeviceAccelerator.cpp_kw.md_docs.md`](./DeviceAccelerator.cpp_kw.md_docs.md)
- [`FunctionalInverses.cpp_kw.md_docs.md`](./FunctionalInverses.cpp_kw.md_docs.md)
- [`SequenceNumber.h_kw.md_docs.md`](./SequenceNumber.h_kw.md_docs.md)
- [`ThreadLocalPythonObjects.h_docs.md_docs.md`](./ThreadLocalPythonObjects.h_docs.md_docs.md)
- [`TensorNames.h_docs.md_docs.md`](./TensorNames.h_docs.md_docs.md)
- [`LegacyBatchedTensorImpl.h_docs.md_docs.md`](./LegacyBatchedTensorImpl.h_docs.md_docs.md)
- [`TensorOperators.h_docs.md_docs.md`](./TensorOperators.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Parallel.h_docs.md_docs.md`
- **Keyword Index**: `Parallel.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

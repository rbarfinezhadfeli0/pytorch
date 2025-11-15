# Documentation: `docs/aten/src/ATen/test/test_parallel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/test_parallel.cpp_docs.md`
- **Size**: 5,785 bytes (5.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/test_parallel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/test_parallel.cpp`
- **Size**: 3,028 bytes (2.96 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>

#include <iostream>
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <string.h>
#include <sstream>
#if AT_MKL_ENABLED()
#include <mkl.h>
#include <thread>
#endif

struct NumThreadsGuard {
  int old_num_threads_;
  NumThreadsGuard(int nthreads) {
    old_num_threads_ = at::get_num_threads();
    at::set_num_threads(nthreads);
  }

  ~NumThreadsGuard() {
    at::set_num_threads(old_num_threads_);
  }
};

using namespace at;

TEST(TestParallel, TestParallel) {
  manual_seed(123);
  NumThreadsGuard guard(1);

  Tensor a = rand({1, 3});
  a[0][0] = 1;
  a[0][1] = 0;
  a[0][2] = 0;
  Tensor as = rand({3});
  as[0] = 1;
  as[1] = 0;
  as[2] = 0;
  ASSERT_TRUE(a.sum(0).equal(as));
}

TEST(TestParallel, NestedParallel) {
  Tensor a = ones({1024, 1024});
  auto expected = a.sum();
  // check that calling sum() from within a parallel block computes the same result
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    if (begin == 0) {
      ASSERT_TRUE(a.sum().equal(expected));
    }
  });
}

#ifdef TH_BLAS_MKL
TEST(TestParallel, LocalMKLThreadNumber) {
  auto master_thread_num = mkl_get_max_threads();
  auto f = [](int nthreads){
    set_num_threads(nthreads);
  };
  std::thread t(f, 1);
  t.join();
  ASSERT_EQ(master_thread_num, mkl_get_max_threads());
}
#endif

TEST(TestParallel, NestedParallelThreadId) {
  // check that thread id within a nested parallel block is accurate
  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      // Nested parallel regions execute on a single thread
      ASSERT_EQ(begin, 0);
      ASSERT_EQ(end, 10);

      // Thread id reflects inner parallel region
      ASSERT_EQ(at::get_thread_num(), 0);
    });
  });

  at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
    auto num_threads =
      at::parallel_reduce(0, 10, 1, 0, [&](int64_t begin, int64_t end, int ident) {
        // Thread id + 1 should always be 1
        return at::get_thread_num() + 1;
      }, std::plus<>{});
    ASSERT_EQ(num_threads, 1);
  });
}

TEST(TestParallel, Exceptions) {
  // parallel case
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(
    at::parallel_for(0, 10, 1, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception");
    }),
    std::runtime_error);

  // non-parallel case
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(
    at::parallel_for(0, 1, 1000, [&](int64_t begin, int64_t end) {
      throw std::runtime_error("exception");
    }),
    std::runtime_error);
}

TEST(TestParallel, IntraOpLaunchFuture) {
  int v1 = 0;
  int v2 = 0;

  auto fut1 = at::intraop_launch_future([&v1](){
    v1 = 1;
  });

  auto fut2 = at::intraop_launch_future([&v2](){
    v2 = 2;
  });

  fut1->wait();
  fut2->wait();

  ASSERT_TRUE(v1 == 1 && v2 == 2);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `NumThreadsGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/DLConvertor.h`
- `ATen/Parallel.h`
- `ATen/ParallelFuture.h`
- `iostream`
- `string.h`
- `sstream`
- `mkl.h`
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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/test_parallel.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `test_parallel.cpp_docs.md`
- **Keyword Index**: `test_parallel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/test/test_parallel.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_parallel.cpp_docs.md_docs.md`
- **Keyword Index**: `test_parallel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

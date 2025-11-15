# Documentation: `docs/aten/src/ATen/test/memory_overlapping_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/memory_overlapping_test.cpp_docs.md`
- **Size**: 4,916 bytes (4.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/memory_overlapping_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/memory_overlapping_test.cpp`
- **Size**: 2,430 bytes (2.37 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

std::vector<std::vector<int64_t>> sizes = {{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {3, 1, 2}, {3, 2, 1}, {2, 3, 1}};

TEST(MemoryOverlapTest, TensorExpanded) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::ones({1}).expand(size);
    EXPECT_FALSE(t.is_contiguous());
    EXPECT_FALSE(t.is_non_overlapping_and_dense());
  }
}

TEST(MemoryOverlapTest, ScalarExpanded) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::tensor(1).expand(size);
    EXPECT_FALSE(t.is_contiguous());
    EXPECT_FALSE(t.is_non_overlapping_and_dense());
  }
}

TEST(MemoryOverlapTest, NonContiguousTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::rand(size).transpose(1, 2).transpose(0, 2);
    if (!t.is_contiguous()) {
      EXPECT_TRUE(t.is_non_overlapping_and_dense());
    }
  }
}

TEST(MemoryOverlapTest, NonContiguousExpandedTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::rand(size).transpose(1, 2).transpose(0, 2);
    if (!t.is_contiguous()) {
      for (auto size_to_add : {1, 2, 3, 4}) {
        auto transpose_size = t.sizes().vec();
        std::vector<int64_t> expanded_size(transpose_size);
        expanded_size.insert(expanded_size.begin(), size_to_add);
        auto expanded = t.expand(expanded_size);
        EXPECT_FALSE(t.is_contiguous());
        if (size_to_add == 1) {
          EXPECT_TRUE(expanded.is_non_overlapping_and_dense());
        } else {
          EXPECT_FALSE(expanded.is_non_overlapping_and_dense());
        }
      }
    }
  }
}

TEST(MemoryOverlapTest, ContiguousTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::rand(size);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_TRUE(t.is_non_overlapping_and_dense());
  }
}

TEST(MemoryOverlapTest, ContiguousExpandedTensor) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::rand(size);
    for (auto size_to_add : {1, 2, 3, 4}) {
      std::vector<int64_t> expanded_size(size);
      expanded_size.insert(expanded_size.begin(), size_to_add);
      auto expanded = t.expand(expanded_size);
      EXPECT_TRUE(t.is_contiguous());
      EXPECT_TRUE(t.is_non_overlapping_and_dense());
    }
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python aten/src/ATen/test/memory_overlapping_test.cpp
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
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `memory_overlapping_test.cpp_docs.md`
- **Keyword Index**: `memory_overlapping_test.cpp_kw.md`
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
python docs/aten/src/ATen/test/memory_overlapping_test.cpp_docs.md
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

- **File Documentation**: `memory_overlapping_test.cpp_docs.md_docs.md`
- **Keyword Index**: `memory_overlapping_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

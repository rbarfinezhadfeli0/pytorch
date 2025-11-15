# Documentation: `docs/aten/src/ATen/test/memory_format_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/memory_format_test.cpp_docs.md`
- **Size**: 10,034 bytes (9.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/memory_format_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/memory_format_test.cpp`
- **Size**: 7,492 bytes (7.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

std::vector<std::vector<int64_t>> sizes = {{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 1, 4, 1}, {4, 1, 1, 4}, {1, 4, 1, 4}, {1, 4, 4, 1}};

TEST(MemoryFormatTest, SetMemoryFormat) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto size : sizes) {
    Tensor t = at::rand(size);
    for (auto memory_format : {at::MemoryFormat::ChannelsLast, at::MemoryFormat::Contiguous}) {
      t.resize_(size, memory_format);
      EXPECT_TRUE(t.suggest_memory_format() == memory_format);
    }
  }

  Tensor t = at::rand({4, 1, 1, 1});
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::Contiguous);
  t.resize_({4, 1, 1, 1}, at::MemoryFormat::ChannelsLast);
  // TODO: Should be able to handle this after accumulated permutation is implemented;
  // Ambiguous case where we fallback to Contiguous;
  // This should be `EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);`
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::Contiguous);
}

TEST(MemoryFormatTest, TransposeMemoryFormat) {
  Tensor t = at::rand({2, 3, 4, 5});
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::Contiguous);
  t.transpose_(1, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);
  t = at::rand({2, 3, 4, 5});
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({2, 3, 4, 5});
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);

  // corner cases:
  t = at::rand({1, 4, 1, 4});
  t.transpose_(1, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 1, 4});
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 1, 4});
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 1, 4});
  t.transpose_(2, 3);
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);

  t = at::rand({1, 4, 4, 1});
  t.transpose_(1, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 4, 1});
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 4, 1});
  t.transpose_(2, 3);
  EXPECT_TRUE(t.suggest_memory_format() != at::MemoryFormat::ChannelsLast);
  t = at::rand({1, 4, 4, 1});
  t.transpose_(2, 3);
  t.transpose_(1, 2);
  EXPECT_TRUE(t.suggest_memory_format() == at::MemoryFormat::ChannelsLast);
}

inline void sliceStepTwo(Tensor& t, int dim, at::MemoryFormat format) {
  t = t.slice(dim, 0, 3, 2);
  EXPECT_TRUE(t.suggest_memory_format() == format);
  t = t.slice(dim, 0, 3, 2);
  EXPECT_TRUE(t.suggest_memory_format() == format);
}

TEST(MemoryFormatTest, SliceStepTwoMemoryFormat) {
  Tensor t = at::rand({4, 4, 4, 4});
  sliceStepTwo(t, 1, MemoryFormat::Contiguous);
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);

  t = at::rand({4, 4, 4, 4});
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);
  sliceStepTwo(t, 1, MemoryFormat::Contiguous);

  t = at::rand({4, 4, 4, 4});
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 1, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

  t = at::rand({4, 4, 4, 4});
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 1, MemoryFormat::ChannelsLast);

  t = at::rand({4, 4, 1, 1});
  sliceStepTwo(t, 1, MemoryFormat::Contiguous);
  t = at::rand({4, 4, 1, 1});
  t.resize_({4, 4, 1, 1}, at::MemoryFormat::ChannelsLast);
  t = t.slice(1, 0, 3, 2);
  EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);
  t = t.slice(1, 0, 3, 2);
  // TODO: Should be able to handle this after accumulated permutation is implemented;
  // won't be able to tell how we ended up here
  // [4, 1, 1, 4]@[4, 4, 4, 1] slice twice at dim3
  // [4, 4, 1, 1]@[4, 1, 4, 4] slice twice at dim1
  // EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::ChannelsLast);
  EXPECT_TRUE(t.suggest_memory_format() == MemoryFormat::Contiguous);

  t = at::rand({4, 1, 4, 4});
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);
  t = at::rand({4, 1, 4, 4});
  t.resize_({4, 1, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

  t = at::rand({4, 1, 1, 4});
  sliceStepTwo(t, 3, MemoryFormat::Contiguous);
  t = at::rand({4, 1, 1, 4});
  t.resize_({4, 1, 1, 4}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 3, MemoryFormat::ChannelsLast);

  t = at::rand({4, 1, 4, 1});
  sliceStepTwo(t, 2, MemoryFormat::Contiguous);
  t = at::rand({4, 1, 4, 1});
  t.resize_({4, 1, 4, 1}, at::MemoryFormat::ChannelsLast);
  sliceStepTwo(t, 2, MemoryFormat::ChannelsLast);
}

inline void sliceFirst(Tensor& t, int dim, at::MemoryFormat format) {
  t = t.slice(dim, 0, 1, 1);
  EXPECT_TRUE(t.suggest_memory_format() == format);
}

TEST(MemoryFormatTest, SliceFirstMemoryFormat) {
  Tensor t = at::rand({4, 4, 4, 4});
  sliceFirst(t, 1, MemoryFormat::Contiguous);
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  sliceFirst(t, 3, MemoryFormat::Contiguous);

  t = at::rand({4, 4, 4, 4});
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  sliceFirst(t, 3, MemoryFormat::Contiguous);
  sliceFirst(t, 1, MemoryFormat::Contiguous);

  t = at::rand({4, 4, 4, 4});
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceFirst(t, 1, MemoryFormat::ChannelsLast);
  sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);

  t = at::rand({4, 4, 4, 4});
  t.resize_({4, 4, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);
  sliceFirst(t, 1, MemoryFormat::ChannelsLast);

  t = at::rand({4, 4, 1, 1});
  sliceFirst(t, 1, MemoryFormat::Contiguous);
  t = at::rand({4, 4, 1, 1});
  t.resize_({4, 4, 1, 1}, at::MemoryFormat::ChannelsLast);
  sliceFirst(t, 1, MemoryFormat::ChannelsLast);

  t = at::rand({4, 1, 4, 4});
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  sliceFirst(t, 3, MemoryFormat::Contiguous);
  t = at::rand({4, 1, 4, 4});
  t.resize_({4, 1, 4, 4}, at::MemoryFormat::ChannelsLast);
  sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);

  t = at::rand({4, 1, 1, 4});
  sliceFirst(t, 3, MemoryFormat::Contiguous);
  t = at::rand({4, 1, 1, 4});
  t.resize_({4, 1, 1, 4}, at::MemoryFormat::ChannelsLast);
  sliceFirst(t, 3, MemoryFormat::ChannelsLast);

  t = at::rand({4, 1, 4, 1});
  sliceFirst(t, 2, MemoryFormat::Contiguous);
  t = at::rand({4, 1, 4, 1});
  t.resize_({4, 1, 4, 1}, at::MemoryFormat::ChannelsLast);
  // TODO: Should be able to handle this after accumulated permutation is implemented;
  // [4, 1, 4, 1]@[4, 1, 1, 1] after slice becomes [4, 1, 1, 1]@[4, 1, 1, 1]
  // sliceFirst(t, 2, MemoryFormat::ChannelsLast);
  sliceFirst(t, 2, MemoryFormat::Contiguous);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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
python aten/src/ATen/test/memory_format_test.cpp
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

- **File Documentation**: `memory_format_test.cpp_docs.md`
- **Keyword Index**: `memory_format_test.cpp_kw.md`
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
python docs/aten/src/ATen/test/memory_format_test.cpp_docs.md
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

- **File Documentation**: `memory_format_test.cpp_docs.md_docs.md`
- **Keyword Index**: `memory_format_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

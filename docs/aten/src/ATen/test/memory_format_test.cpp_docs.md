# Documentation: memory_format_test.cpp

## File Metadata
- **Path**: `aten/src/ATen/test/memory_format_test.cpp`
- **Size**: 7492 bytes
- **Lines**: 193
- **Extension**: .cpp
- **Type**: Regular file

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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough


## Key Components

The file contains 750 words across 193 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 7492 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

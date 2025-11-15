# Documentation: `docs/test/cpp/api/nested_int.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/nested_int.cpp_docs.md`
- **Size**: 5,580 bytes (5.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/nested_int.cpp`

## File Metadata

- **Path**: `test/cpp/api/nested_int.cpp`
- **Size**: 3,321 bytes (3.24 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/core/NestedIntSymNodeImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

TEST(NestedIntTest, Comparisons) {
  auto a = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 1)));
  auto b = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 1)));
  auto c = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(2, 1)));
  auto d = c10::SymInt(3);

  ASSERT_TRUE(a == a);
  ASSERT_TRUE(a == b);
  ASSERT_FALSE(a != a);
  ASSERT_FALSE(a != b);
  ASSERT_FALSE(a == c);
  ASSERT_TRUE(a != c);

  ASSERT_FALSE(a == d);
  ASSERT_TRUE(a != d);
  ASSERT_FALSE(d == a);
  ASSERT_TRUE(d != a);

  // ge
  ASSERT_TRUE(a >= a);
  ASSERT_TRUE(a >= b);
  ASSERT_TRUE(b >= a);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a >= c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c >= a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c >= 3), c10::Error);
  ASSERT_TRUE(c >= 2);
  ASSERT_TRUE(c >= 1);
  ASSERT_FALSE(1 >= c);

  // lt
  ASSERT_FALSE(a < a);
  ASSERT_FALSE(a < b);
  ASSERT_FALSE(b < a);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a < c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c < a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(3 < a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(2 < a), c10::Error);
  ASSERT_TRUE(1 < a);

  // le
  ASSERT_TRUE(a <= a);
  ASSERT_TRUE(b <= a);
  ASSERT_TRUE(a <= b);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a <= c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c <= a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(3 <= c), c10::Error);
  ASSERT_TRUE(2 <= c);
  ASSERT_TRUE(1 <= c);
  ASSERT_FALSE(c <= 1);

  // gt
  ASSERT_FALSE(a > a);
  ASSERT_FALSE(b > a);
  ASSERT_FALSE(a > b);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > c), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(c > a), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > 3), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW((void)(a > 2), c10::Error);
  ASSERT_TRUE(a > 1);
}

TEST(NestedIntTest, WithFactor) {
  auto a = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 5)));
  auto b = c10::SymInt(
      c10::SymNode(c10::make_intrusive<c10::NestedIntSymNodeImpl>(1, 10)));
  // eq
  ASSERT_FALSE(a == b);
  ASSERT_FALSE(a >= b);
  ASSERT_TRUE(b >= a);
  ASSERT_TRUE(a <= b);
  ASSERT_FALSE(b <= a);
  // ne
  ASSERT_TRUE(a != b);
  // mul
  ASSERT_TRUE(a * 2 == b);
  ASSERT_TRUE(a * 3 >= b);
  ASSERT_TRUE(a * 2 == 2 * a);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/core/NestedIntSymNodeImpl.h`
- `c10/core/SymInt.h`
- `c10/core/SymNodeImpl.h`
- `torch/torch.h`
- `test/cpp/api/support.h`


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
python test/cpp/api/nested_int.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

- [`fft.cpp_docs.md`](./fft.cpp_docs.md)
- [`tensor_options.cpp_docs.md`](./tensor_options.cpp_docs.md)
- [`any.cpp_docs.md`](./any.cpp_docs.md)
- [`torch_include.cpp_docs.md`](./torch_include.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`jit.cpp_docs.md`](./jit.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)


## Cross-References

- **File Documentation**: `nested_int.cpp_docs.md`
- **Keyword Index**: `nested_int.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/cpp/api/nested_int.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/api`):

- [`init_baseline.py_kw.md_docs.md`](./init_baseline.py_kw.md_docs.md)
- [`support.cpp_kw.md_docs.md`](./support.cpp_kw.md_docs.md)
- [`memory.cpp_docs.md_docs.md`](./memory.cpp_docs.md_docs.md)
- [`parallel_benchmark.cpp_docs.md_docs.md`](./parallel_benchmark.cpp_docs.md_docs.md)
- [`dataloader.cpp_docs.md_docs.md`](./dataloader.cpp_docs.md_docs.md)
- [`moduledict.cpp_kw.md_docs.md`](./moduledict.cpp_kw.md_docs.md)
- [`support.h_kw.md_docs.md`](./support.h_kw.md_docs.md)
- [`ordered_dict.cpp_docs.md_docs.md`](./ordered_dict.cpp_docs.md_docs.md)
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `nested_int.cpp_docs.md_docs.md`
- **Keyword Index**: `nested_int.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

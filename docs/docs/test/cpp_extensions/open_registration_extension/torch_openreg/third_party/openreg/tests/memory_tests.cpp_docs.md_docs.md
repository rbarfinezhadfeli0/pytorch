# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests/memory_tests.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests/memory_tests.cpp_docs.md`
- **Size**: 5,461 bytes (5.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests/memory_tests.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests/memory_tests.cpp`
- **Size**: 3,125 bytes (3.05 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <include/openreg.h>

namespace {

class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

TEST_F(MemoryTest, AllocateAndFreeDevice) {
  void* ptr = nullptr;
  EXPECT_EQ(orMalloc(&ptr, 4096), orSuccess);
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(orFree(ptr), orSuccess);
}

TEST_F(MemoryTest, AllocateAndFreeHost) {
  void* ptr = nullptr;
  EXPECT_EQ(orMallocHost(&ptr, 8192), orSuccess);
  EXPECT_NE(ptr, nullptr);

  EXPECT_EQ(orFreeHost(ptr), orSuccess);
}

TEST_F(MemoryTest, AllocateNullptr) {
  EXPECT_EQ(orMalloc(nullptr, 4096), orErrorUnknown);
  EXPECT_EQ(orMallocHost(nullptr, 4096), orErrorUnknown);
}

TEST_F(MemoryTest, AllocateZeroSize) {
  void* ptr = nullptr;
  EXPECT_EQ(orMalloc(&ptr, 0), orErrorUnknown);
  EXPECT_EQ(orMallocHost(&ptr, 0), orErrorUnknown);
}

TEST_F(MemoryTest, MemcpyHostToDevice) {
  char host_src[] = "data";
  char host_dst[5] = {};

  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 5), orSuccess);

  EXPECT_EQ(orMemcpy(dev_ptr, host_src, 5, orMemcpyHostToDevice), orSuccess);
  EXPECT_EQ(orMemcpy(host_dst, dev_ptr, 5, orMemcpyDeviceToHost), orSuccess);

  EXPECT_STREQ(host_dst, host_src);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, MemcpyDeviceToDevice) {
  const char host_src[5] = "data";
  char host_dst[5] = {};
  void *dev_dst1 = nullptr, *dev_dst2 = nullptr;

  EXPECT_EQ(orMalloc(&dev_dst1, 5), orSuccess);
  EXPECT_EQ(orMalloc(&dev_dst2, 5), orSuccess);

  EXPECT_EQ(orMemcpy(dev_dst1, host_src, 5, orMemcpyHostToDevice), orSuccess);
  EXPECT_EQ(orMemcpy(dev_dst2, dev_dst1, 5, orMemcpyDeviceToDevice), orSuccess);
  EXPECT_EQ(orMemcpy(host_dst, dev_dst2, 5, orMemcpyDeviceToHost), orSuccess);

  EXPECT_STREQ(host_dst, host_src);

  EXPECT_EQ(orFree(dev_dst1), orSuccess);
  EXPECT_EQ(orFree(dev_dst2), orSuccess);
}

TEST_F(MemoryTest, MemcpyInvalidKind) {
  char host_ptr[5] = "data";
  void* dev_ptr = nullptr;

  EXPECT_EQ(orMalloc(&dev_ptr, 5), orSuccess);

  EXPECT_EQ(
      orMemcpy(nullptr, host_ptr, 4, orMemcpyHostToDevice), orErrorUnknown);
  EXPECT_EQ(
      orMemcpy(dev_ptr, nullptr, 4, orMemcpyHostToDevice), orErrorUnknown);
  EXPECT_EQ(
      orMemcpy(dev_ptr, host_ptr, 0, orMemcpyHostToDevice), orErrorUnknown);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, PointerAttributes) {
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 32), orSuccess);

  orPointerAttributes attr{};
  EXPECT_EQ(orPointerGetAttributes(&attr, dev_ptr), orSuccess);
  EXPECT_EQ(attr.type, orMemoryType::orMemoryTypeDevice);
  EXPECT_EQ(attr.pointer, dev_ptr);

  char host_ptr[16];
  EXPECT_EQ(orPointerGetAttributes(&attr, host_ptr), orSuccess);
  EXPECT_EQ(attr.type, orMemoryType::orMemoryTypeUnmanaged);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

TEST_F(MemoryTest, ProtectUnprotectDevice) {
  void* dev_ptr = nullptr;
  EXPECT_EQ(orMalloc(&dev_ptr, 64), orSuccess);

  EXPECT_EQ(orMemoryUnprotect(dev_ptr), orSuccess);
  EXPECT_EQ(orMemoryProtect(dev_ptr), orSuccess);

  EXPECT_EQ(orFree(dev_ptr), orSuccess);
}

} // namespace

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `MemoryTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `include/openreg.h`


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
python test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests/memory_tests.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests`):

- [`device_tests.cpp_docs.md`](./device_tests.cpp_docs.md)
- [`stream_tests.cpp_docs.md`](./stream_tests.cpp_docs.md)
- [`event_tests.cpp_docs.md`](./event_tests.cpp_docs.md)


## Cross-References

- **File Documentation**: `memory_tests.cpp_docs.md`
- **Keyword Index**: `memory_tests.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests/memory_tests.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg/third_party/openreg/tests`):

- [`event_tests.cpp_kw.md_docs.md`](./event_tests.cpp_kw.md_docs.md)
- [`memory_tests.cpp_kw.md_docs.md`](./memory_tests.cpp_kw.md_docs.md)
- [`stream_tests.cpp_kw.md_docs.md`](./stream_tests.cpp_kw.md_docs.md)
- [`event_tests.cpp_docs.md_docs.md`](./event_tests.cpp_docs.md_docs.md)
- [`stream_tests.cpp_docs.md_docs.md`](./stream_tests.cpp_docs.md_docs.md)
- [`device_tests.cpp_docs.md_docs.md`](./device_tests.cpp_docs.md_docs.md)
- [`device_tests.cpp_kw.md_docs.md`](./device_tests.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `memory_tests.cpp_docs.md_docs.md`
- **Keyword Index**: `memory_tests.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

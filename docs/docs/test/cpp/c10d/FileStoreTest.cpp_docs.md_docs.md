# Documentation: `docs/test/cpp/c10d/FileStoreTest.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/c10d/FileStoreTest.cpp_docs.md`
- **Size**: 7,002 bytes (6.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/c10d/FileStoreTest.cpp`

## File Metadata

- **Path**: `test/cpp/c10d/FileStoreTest.cpp`
- **Size**: 4,328 bytes (4.23 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp

#include <c10/util/irange.h>
#include "StoreTestCommon.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif

#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

#ifdef _WIN32
std::string tmppath() {
  return c10d::test::autoGenerateTmpFilePath();
}
#else
std::string tmppath() {
  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }

  // Create template
  std::vector<char> tmp(256);
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
  tmp.resize(len);

  // Create temporary file
  auto fd = mkstemp(&tmp[0]);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());
  }
  close(fd);
  return std::string(tmp.data(), tmp.size());
}
#endif

void testGetSet(const std::string& path, const std::string& prefix = "") {
  // Basic Set/Get on File Store
  {
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, 2);
    c10d::PrefixStore store(prefix, fileStore);
    c10d::test::set(store, "key0", "value0");
    c10d::test::set(store, "key1", "value1");
    c10d::test::set(store, "key2", "value2");
    c10d::test::check(store, "key0", "value0");
    c10d::test::check(store, "key1", "value1");
    c10d::test::check(store, "key2", "value2");
    auto numKeys = fileStore->getNumKeys();
    EXPECT_EQ(numKeys, 4);

    // Check compareSet, does not check return value
    c10d::test::compareSet(store, "key0", "wrongExpectedValue", "newValue");
    c10d::test::check(store, "key0", "value0");
    c10d::test::compareSet(store, "key0", "value0", "newValue");
    c10d::test::check(store, "key0", "newValue");

    // Check deleteKey
    c10d::test::deleteKey(store, "key1");
    numKeys = fileStore->getNumKeys();
    EXPECT_EQ(numKeys, 3);
    c10d::test::check(store, "key0", "newValue");
    c10d::test::check(store, "key2", "value2");

    c10d::test::set(store, "-key0", "value-");
    c10d::test::check(store, "key0", "newValue");
    c10d::test::check(store, "-key0", "value-");
    numKeys = fileStore->getNumKeys();
    EXPECT_EQ(numKeys, 4);
    c10d::test::deleteKey(store, "-key0");
    numKeys = fileStore->getNumKeys();
    EXPECT_EQ(numKeys, 3);
    c10d::test::check(store, "key0", "newValue");
    c10d::test::check(store, "key2", "value2");
  }

  // Perform get on new instance
  {
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, 2);
    c10d::PrefixStore store(prefix, fileStore);
    c10d::test::check(store, "key0", "newValue");
    auto numKeys = fileStore->getNumKeys();
    // There will be 4 keys since we still use the same underlying file as the
    // other store above.
    EXPECT_EQ(numKeys, 4);
  }
}

void stressTestStore(std::string path, std::string prefix = "") {
  // Hammer on FileStore::add
  const auto numThreads = 4;
  const auto numIterations = 100;

  std::vector<std::thread> threads;
  c10d::test::Semaphore sem1, sem2;

  for ([[maybe_unused]] const auto i : c10::irange(numThreads)) {
    threads.emplace_back([&] {
      auto fileStore =
          c10::make_intrusive<c10d::FileStore>(path, numThreads + 1);
      c10d::PrefixStore store(prefix, fileStore);
      sem1.post();
      sem2.wait();
      for ([[maybe_unused]] const auto j : c10::irange(numIterations)) {
        store.add("counter", 1);
      }
    });
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);
  for (auto& thread : threads) {
    thread.join();
  }

  // Check that the counter has the expected value
  {
    auto fileStore = c10::make_intrusive<c10d::FileStore>(path, numThreads + 1);
    c10d::PrefixStore store(prefix, fileStore);
    std::string expected = std::to_string(numThreads * numIterations);
    c10d::test::check(store, "counter", expected);
  }
}

class FileStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    path_ = tmppath();
  }

  void TearDown() override {
    unlink(path_.c_str());
  }

  std::string path_;
};

TEST_F(FileStoreTest, testGetAndSet) {
  testGetSet(path_);
}

TEST_F(FileStoreTest, testGetAndSetWithPrefix) {
  testGetSet(path_, "testPrefix");
}

TEST_F(FileStoreTest, testStressStore) {
  stressTestStore(path_);
}

TEST_F(FileStoreTest, testStressStoreWithPrefix) {
  stressTestStore(path_, "testPrefix");
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `FileStoreTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/c10d`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `StoreTestCommon.hpp`
- `unistd.h`
- `iostream`
- `thread`
- `gtest/gtest.h`
- `torch/csrc/distributed/c10d/FileStore.hpp`
- `torch/csrc/distributed/c10d/PrefixStore.hpp`


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
python test/cpp/c10d/FileStoreTest.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/c10d`):

- [`TCPStoreTest.cpp_docs.md`](./TCPStoreTest.cpp_docs.md)
- [`ProcessGroupUCCTest.cpp_docs.md`](./ProcessGroupUCCTest.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`ProcessGroupNCCLErrorsTest.cpp_docs.md`](./ProcessGroupNCCLErrorsTest.cpp_docs.md)
- [`ProcessGroupMPITest.cpp_docs.md`](./ProcessGroupMPITest.cpp_docs.md)
- [`ProcessGroupNCCLTest.cpp_docs.md`](./ProcessGroupNCCLTest.cpp_docs.md)
- [`HashStoreTest.cpp_docs.md`](./HashStoreTest.cpp_docs.md)
- [`TestUtils.hpp_docs.md`](./TestUtils.hpp_docs.md)
- [`StoreTestCommon.hpp_docs.md`](./StoreTestCommon.hpp_docs.md)


## Cross-References

- **File Documentation**: `FileStoreTest.cpp_docs.md`
- **Keyword Index**: `FileStoreTest.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/c10d`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



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
python docs/test/cpp/c10d/FileStoreTest.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/c10d`):

- [`FileStoreTest.cpp_kw.md_docs.md`](./FileStoreTest.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`ProcessGroupUCCTest.cpp_docs.md_docs.md`](./ProcessGroupUCCTest.cpp_docs.md_docs.md)
- [`ProcessGroupNCCLTest.cpp_docs.md_docs.md`](./ProcessGroupNCCLTest.cpp_docs.md_docs.md)
- [`HashStoreTest.cpp_kw.md_docs.md`](./HashStoreTest.cpp_kw.md_docs.md)
- [`CUDATest.hpp_docs.md_docs.md`](./CUDATest.hpp_docs.md_docs.md)
- [`ProcessGroupNCCLErrorsTest.cpp_kw.md_docs.md`](./ProcessGroupNCCLErrorsTest.cpp_kw.md_docs.md)
- [`HashStoreTest.cpp_docs.md_docs.md`](./HashStoreTest.cpp_docs.md_docs.md)
- [`CUDATest.hpp_kw.md_docs.md`](./CUDATest.hpp_kw.md_docs.md)
- [`StoreTestCommon.hpp_docs.md_docs.md`](./StoreTestCommon.hpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `FileStoreTest.cpp_docs.md_docs.md`
- **Keyword Index**: `FileStoreTest.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

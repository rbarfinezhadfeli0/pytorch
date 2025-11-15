# Documentation: `c10/test/util/LeftRight_test.cpp`

## File Metadata

- **Path**: `c10/test/util/LeftRight_test.cpp`
- **Size**: 6,893 bytes (6.73 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/LeftRight.h>
#include <gtest/gtest.h>
#include <vector>

using c10::LeftRight;
using std::vector;

TEST(LeftRightTest, givenInt_whenWritingAndReading_thenChangesArePresent) {
  LeftRight<int> obj;

  obj.write([](int& obj) { obj = 5; });
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);

  // check changes are also present in background copy
  obj.write([](int&) {}); // this switches to the background copy
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);
}

TEST(LeftRightTest, givenVector_whenWritingAndReading_thenChangesArePresent) {
  LeftRight<vector<int>> obj;

  obj.write([](vector<int>& obj) { obj.push_back(5); });
  vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);

  obj.write([](vector<int>& obj) { obj.push_back(6); });
  read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5, 6}), read);
}

TEST(LeftRightTest, givenVector_whenWritingReturnsValue_thenValueIsReturned) {
  LeftRight<vector<int>> obj;

  auto a = obj.write([](vector<int>&) -> int { return 5; });
  static_assert(std::is_same_v<int, decltype(a)>);
  EXPECT_EQ(5, a);
}

TEST(LeftRightTest, readsCanBeConcurrent) {
  LeftRight<int> obj;
  std::atomic<int> num_running_readers{0};

  std::thread reader1([&]() {
    obj.read([&](const int&) {
      ++num_running_readers;
      while (num_running_readers.load() < 2) {
      }
    });
  });

  std::thread reader2([&]() {
    obj.read([&](const int&) {
      ++num_running_readers;
      while (num_running_readers.load() < 2) {
      }
    });
  });

  // the threads only finish after both entered the read function.
  // if LeftRight didn't allow concurrency, this would cause a deadlock.
  reader1.join();
  reader2.join();
}

TEST(LeftRightTest, writesCanBeConcurrentWithReads_readThenWrite) {
  LeftRight<int> obj;
  std::atomic<bool> reader_running{false};
  std::atomic<bool> writer_running{false};

  std::thread reader([&]() {
    obj.read([&](const int&) {
      reader_running = true;
      while (!writer_running.load()) {
      }
    });
  });

  std::thread writer([&]() {
    // run read first, write second
    while (!reader_running.load()) {
    }

    obj.write([&](int&) { writer_running = true; });
  });

  // the threads only finish after both entered the read function.
  // if LeftRight didn't allow concurrency, this would cause a deadlock.
  reader.join();
  writer.join();
}

TEST(LeftRightTest, writesCanBeConcurrentWithReads_writeThenRead) {
  LeftRight<int> obj;
  std::atomic<bool> writer_running{false};
  std::atomic<bool> reader_running{false};

  std::thread writer([&]() {
    obj.read([&](const int&) {
      writer_running = true;
      while (!reader_running.load()) {
      }
    });
  });

  std::thread reader([&]() {
    // run write first, read second
    while (!writer_running.load()) {
    }

    obj.read([&](const int&) { reader_running = true; });
  });

  // the threads only finish after both entered the read function.
  // if LeftRight didn't allow concurrency, this would cause a deadlock.
  writer.join();
  reader.join();
}

TEST(LeftRightTest, writesCannotBeConcurrentWithWrites) {
  LeftRight<int> obj;
  std::atomic<bool> first_writer_started{false};
  std::atomic<bool> first_writer_finished{false};

  std::thread writer1([&]() {
    obj.write([&](int&) {
      first_writer_started = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      first_writer_finished = true;
    });
  });

  std::thread writer2([&]() {
    // make sure the other writer runs first
    while (!first_writer_started.load()) {
    }

    obj.write([&](int&) {
      // expect the other writer finished before this one starts
      EXPECT_TRUE(first_writer_finished.load());
    });
  });

  writer1.join();
  writer2.join();
}

namespace {
class MyException : public std::exception {};
} // namespace

TEST(LeftRightTest, whenReadThrowsException_thenThrowsThrough) {
  LeftRight<int> obj;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(obj.read([](const int&) { throw MyException(); }), MyException);
}

TEST(LeftRightTest, whenWriteThrowsException_thenThrowsThrough) {
  LeftRight<int> obj;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(obj.write([](int&) { throw MyException(); }), MyException);
}

TEST(
    LeftRightTest,
    givenInt_whenWriteThrowsExceptionOnFirstCall_thenResetsToOldState) {
  LeftRight<int> obj;

  obj.write([](int& obj) { obj = 5; });

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([](int& obj) {
        obj = 6;
        throw MyException();
      }),
      MyException);

  // check reading it returns old value
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);

  // check changes are also present in background copy
  obj.write([](int&) {}); // this switches to the background copy
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(5, read);
}

// note: each write is executed twice, on the foreground and background copy.
// We need to test a thrown exception in either call is handled correctly.
TEST(
    LeftRightTest,
    givenInt_whenWriteThrowsExceptionOnSecondCall_thenKeepsNewState) {
  LeftRight<int> obj;

  obj.write([](int& obj) { obj = 5; });
  bool write_called = false;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([&](int& obj) {
        obj = 6;
        if (write_called) {
          // this is the second time the write callback is executed
          throw MyException();
        } else {
          write_called = true;
        }
      }),
      MyException);

  // check reading it returns new value
  int read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(6, read);

  // check changes are also present in background copy
  obj.write([](int&) {}); // this switches to the background copy
  read = obj.read([](const int& obj) { return obj; });
  EXPECT_EQ(6, read);
}

TEST(LeftRightTest, givenVector_whenWriteThrowsException_thenResetsToOldState) {
  LeftRight<vector<int>> obj;

  obj.write([](vector<int>& obj) { obj.push_back(5); });

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      obj.write([](vector<int>& obj) {
        obj.push_back(6);
        throw MyException();
      }),
      MyException);

  // check reading it returns old value
  vector<int> read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);

  // check changes are also present in background copy
  obj.write([](vector<int>&) {}); // this switches to the background copy
  read = obj.read([](const vector<int>& obj) { return obj; });
  EXPECT_EQ((vector<int>{5}), read);
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `TEST`

**Classes/Structs**: `MyException`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/LeftRight.h`
- `gtest/gtest.h`
- `vector`


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
python c10/test/util/LeftRight_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/util`):

- [`bfloat16_test.cpp_docs.md`](./bfloat16_test.cpp_docs.md)
- [`complex_test_common.h_docs.md`](./complex_test_common.h_docs.md)
- [`TypeIndex_test.cpp_docs.md`](./TypeIndex_test.cpp_docs.md)
- [`generic_math_test.cpp_docs.md`](./generic_math_test.cpp_docs.md)
- [`Half_test.cpp_docs.md`](./Half_test.cpp_docs.md)
- [`nofatal_test.cpp_docs.md`](./nofatal_test.cpp_docs.md)
- [`small_vector_test.cpp_docs.md`](./small_vector_test.cpp_docs.md)
- [`exception_test.cpp_docs.md`](./exception_test.cpp_docs.md)
- [`string_view_test.cpp_docs.md`](./string_view_test.cpp_docs.md)
- [`Enumerate_test.cpp_docs.md`](./Enumerate_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `LeftRight_test.cpp_docs.md`
- **Keyword Index**: `LeftRight_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

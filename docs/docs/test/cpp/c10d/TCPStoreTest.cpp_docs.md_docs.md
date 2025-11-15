# Documentation: `docs/test/cpp/c10d/TCPStoreTest.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/c10d/TCPStoreTest.cpp_docs.md`
- **Size**: 12,926 bytes (12.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/c10d/TCPStoreTest.cpp`

## File Metadata

- **Path**: `test/cpp/c10d/TCPStoreTest.cpp`
- **Size**: 10,267 bytes (10.03 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/irange.h>
#include "StoreTestCommon.hpp"

#include <cstdlib>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

constexpr int64_t kShortStoreTimeoutMillis = 100;
constexpr int defaultTimeout = 20;

c10::intrusive_ptr<c10d::TCPStore> _createServer(
    bool useLibUV,
    int numWorkers = 1,
    int timeout = defaultTimeout) {
  return c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ 0,
          /* isServer */ true,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(timeout),
          /* multiTenant */ false,
          /* masterListenFd */ std::nullopt,
          /* useLibUV*/ useLibUV});
}

// Different ports for different tests.
void testHelper(bool useLibUV, const std::string& prefix = "") {
  constexpr auto numThreads = 16;
  constexpr auto numWorkers = numThreads + 1;

  auto serverTCPStore = _createServer(useLibUV, numWorkers);

  auto serverStore =
      c10::make_intrusive<c10d::PrefixStore>(prefix, serverTCPStore);
  // server store
  auto serverThread = std::thread([&serverStore, &serverTCPStore] {
    // Wait for all workers to join.
    serverTCPStore->waitForWorkers();

    // Basic set/get on the server store
    c10d::test::set(*serverStore, "key0", "value0");
    c10d::test::set(*serverStore, "key1", "value1");
    c10d::test::set(*serverStore, "key2", "value2");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::check(*serverStore, "key1", "value1");
    c10d::test::check(*serverStore, "key2", "value2");
    serverStore->add("counter", 1);
    auto numKeys = serverStore->getNumKeys();
    // We expect 5 keys since 3 are added above, 'counter' is added by the
    // helper thread, and the init key to coordinate workers.
    EXPECT_EQ(numKeys, 5);

    // Check compareSet, does not check return value
    c10d::test::compareSet(
        *serverStore, "key0", "wrongExpectedValue", "newValue");
    c10d::test::check(*serverStore, "key0", "value0");
    c10d::test::compareSet(*serverStore, "key0", "value0", "newValue");
    c10d::test::check(*serverStore, "key0", "newValue");

    auto delSuccess = serverStore->deleteKey("key0");
    // Ensure that the key was successfully deleted
    EXPECT_TRUE(delSuccess);
    auto delFailure = serverStore->deleteKey("badKeyName");
    // The key was not in the store so the delete operation should have failed
    // and returned false.
    EXPECT_FALSE(delFailure);
    numKeys = serverStore->getNumKeys();
    EXPECT_EQ(numKeys, 4);
    auto timeout = std::chrono::milliseconds(kShortStoreTimeoutMillis);
    serverStore->setTimeout(timeout);
    EXPECT_THROW(serverStore->get("key0"), c10::Error);
  });

  // Hammer on TCPStore
  std::vector<std::thread> threads;
  constexpr auto numIterations = 1000;
  c10d::test::Semaphore sem1, sem2;

  c10d::TCPStoreOptions opts{};
  opts.port = serverTCPStore->getPort();
  opts.numWorkers = numWorkers;

  // Each thread will have a client store to send/recv data
  std::vector<c10::intrusive_ptr<c10d::TCPStore>> clientTCPStores;
  std::vector<c10::intrusive_ptr<c10d::PrefixStore>> clientStores;
  for (const auto i : c10::irange(numThreads)) {
    clientTCPStores.push_back(
        c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts));
    clientStores.push_back(
        c10::make_intrusive<c10d::PrefixStore>(prefix, clientTCPStores[i]));
  }

  std::string expectedCounterRes =
      std::to_string(numThreads * numIterations + 1);

  for (const auto i : c10::irange(numThreads)) {
    threads.emplace_back([=, &sem1, &sem2, &clientStores, &expectedCounterRes] {
      for ([[maybe_unused]] const auto j : c10::irange(numIterations)) {
        clientStores[i]->add("counter", 1);
      }
      // Let each thread set and get key on its client store
      std::string key = "thread_" + std::to_string(i);
      for (const auto j : c10::irange(numIterations)) {
        std::string val = "thread_val_" + std::to_string(j);
        c10d::test::set(*clientStores[i], key, val);
        c10d::test::check(*clientStores[i], key, val);
      }

      sem1.post();
      sem2.wait();
      // Check the counter results
      c10d::test::check(*clientStores[i], "counter", expectedCounterRes);
      // Now check other threads' written data
      for (const auto j : c10::irange(numThreads)) {
        if (j == i) {
          continue;
        }
        std::string key = "thread_" + std::to_string(i);
        std::string val = "thread_val_" + std::to_string(numIterations - 1);
        c10d::test::check(*clientStores[i], key, val);
      }
    });
  }

  sem1.wait(numThreads);
  sem2.post(numThreads);

  for (auto& thread : threads) {
    thread.join();
  }

  serverThread.join();

  // Clear the store to test that client disconnect won't shutdown the store
  clientStores.clear();
  clientTCPStores.clear();

  // Check that the counter has the expected value
  c10d::test::check(*serverStore, "counter", expectedCounterRes);

  // Check that each threads' written data from the main thread
  for (const auto i : c10::irange(numThreads)) {
    std::string key = "thread_" + std::to_string(i);
    std::string val = "thread_val_" + std::to_string(numIterations - 1);
    c10d::test::check(*serverStore, key, val);
  }
}

TEST(TCPStoreTest, testHelper) {
  testHelper(false);
}

TEST(TCPStoreTest, testHelperUV) {
  testHelper(true);
}

TEST(TCPStoreTest, testHelperPrefix) {
  testHelper(false, "testPrefixNoUV");
}

TEST(TCPStoreTest, testHelperPrefixUV) {
  testHelper(true, "testPrefixUV");
}

TEST(TCPStoreTest, testCleanShutdown) {
  int numWorkers = 2;

  auto serverTCPStore = std::make_unique<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ 0,
          /* isServer */ true,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(defaultTimeout)});
  c10d::test::set(*serverTCPStore, "key", "val");

  auto clientTCPStore = c10::make_intrusive<c10d::TCPStore>(
      "127.0.0.1",
      c10d::TCPStoreOptions{
          /* port */ serverTCPStore->getPort(),
          /* isServer */ false,
          numWorkers,
          /* waitWorkers */ false,
          /* timeout */ std::chrono::seconds(defaultTimeout)});
  clientTCPStore->get("key");

  auto clientThread = std::thread([&clientTCPStore] {
    EXPECT_THROW(clientTCPStore->get("invalid_key"), c10::DistNetworkError);
  });

  // start server shutdown during a client request
  serverTCPStore = nullptr;

  clientThread.join();
}

TEST(TCPStoreTest, testLibUVPartialRead) {
  int numWorkers = 2; // thread 0 creates both server and client

  // server part
  c10d::TCPStoreOptions server_opts{
      0,
      true, // is master
      numWorkers,
      false, // don't wait otherwise client thread won't spawn
      std::chrono::seconds(defaultTimeout)};
  server_opts.useLibUV = true;

  auto serverTCPStore =
      std::make_unique<c10d::TCPStore>("127.0.0.1", server_opts);

  // client part
  c10d::TCPStoreOptions client_opts{
      serverTCPStore->getPort(),
      false, // is master
      numWorkers,
      false, // wait workers
      std::chrono::seconds(defaultTimeout)};
  client_opts.useLibUV = true;
  auto clientTCPStore =
      c10::make_intrusive<c10d::TCPStore>("127.0.0.1", client_opts);
  auto clientThread = std::thread([&clientTCPStore] {
    std::string keyPrefix(
        "/default_pg/0//b7dc24de75e482ba2ceb9f9ee20732c25c0166d8//cuda//");
    std::string value("v");
    std::vector<uint8_t> valueBuf(value.begin(), value.end());

    // split store->set(key, valueBuf) into two requests
    for (int i = 0; i < 10; ++i) {
      std::string key = keyPrefix + std::to_string(i);
      clientTCPStore->_splitSet(key, valueBuf);

      // check the result on server
      c10d::test::check(*clientTCPStore, key, "v");
    }
  });

  clientThread.join();
}

TEST(TCPStoreTest, testLibUVSetAndWait) {
  int numWorkers = 128; // thread 0 creates both server and client
  std::vector<std::thread> threads;

  // server part
  c10d::TCPStoreOptions server_opts{
      0,
      true, // is master
      numWorkers,
      false, // don't wait otherwise client thread won't spawn
      std::chrono::seconds(defaultTimeout)};
  server_opts.useLibUV = true;

  auto serverTCPStore =
      std::make_unique<c10d::TCPStore>("127.0.0.1", server_opts);

  // client part
  c10d::TCPStoreOptions client_opts{
      serverTCPStore->getPort(),
      false, // is master
      numWorkers,
      false, // wait workers
      std::chrono::seconds(defaultTimeout)};
  client_opts.useLibUV = true;

  for (const auto i : c10::irange(numWorkers)) {
    threads.emplace_back([=, &client_opts] {
      auto clientTCPStore =
          c10::make_intrusive<c10d::TCPStore>("127.0.0.1", client_opts);
      std::string key("k_" + std::to_string(i));
      std::string value("v_" + std::to_string(i));
      std::vector<uint8_t> valueBuf(value.begin(), value.end());
      clientTCPStore->set(key, valueBuf);
      std::vector<std::string> all_keys;
      for (const auto j : c10::irange(numWorkers)) {
        all_keys.push_back("k_" + std::to_string(j));
      }
      clientTCPStore->wait(all_keys);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

void testMultiTenantStores(bool libUV) {
  c10d::TCPStoreOptions opts{};
  opts.isServer = true;
  opts.multiTenant = true;
  opts.useLibUV = libUV;

  // Construct two server stores on the same port.
  auto store1 = c10::make_intrusive<c10d::TCPStore>("localhost", opts);
  auto store2 = c10::make_intrusive<c10d::TCPStore>("localhost", opts);

  // Assert that the two stores share the same server.
  c10d::test::set(*store1, "key0", "value0");
  c10d::test::check(*store2, "key0", "value0");

  // Dispose the second instance and assert that the server is still alive.
  store2.reset();

  c10d::test::set(*store1, "key0", "value0");
  c10d::test::check(*store1, "key0", "value0");
}

TEST(TCPStoreTest, testMultiTenantStores) {
  testMultiTenantStores(false);
}

TEST(TCPStoreTest, testMultiTenantStoresUV) {
  testMultiTenantStores(true);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `two`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/c10d`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `StoreTestCommon.hpp`
- `cstdlib`
- `string`
- `thread`
- `gtest/gtest.h`
- `torch/csrc/distributed/c10d/PrefixStore.hpp`
- `torch/csrc/distributed/c10d/TCPStore.hpp`


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
python test/cpp/c10d/TCPStoreTest.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/c10d`):

- [`ProcessGroupUCCTest.cpp_docs.md`](./ProcessGroupUCCTest.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`ProcessGroupNCCLErrorsTest.cpp_docs.md`](./ProcessGroupNCCLErrorsTest.cpp_docs.md)
- [`FileStoreTest.cpp_docs.md`](./FileStoreTest.cpp_docs.md)
- [`ProcessGroupMPITest.cpp_docs.md`](./ProcessGroupMPITest.cpp_docs.md)
- [`ProcessGroupNCCLTest.cpp_docs.md`](./ProcessGroupNCCLTest.cpp_docs.md)
- [`HashStoreTest.cpp_docs.md`](./HashStoreTest.cpp_docs.md)
- [`TestUtils.hpp_docs.md`](./TestUtils.hpp_docs.md)
- [`StoreTestCommon.hpp_docs.md`](./StoreTestCommon.hpp_docs.md)


## Cross-References

- **File Documentation**: `TCPStoreTest.cpp_docs.md`
- **Keyword Index**: `TCPStoreTest.cpp_kw.md`
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
python docs/test/cpp/c10d/TCPStoreTest.cpp_docs.md
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

- **File Documentation**: `TCPStoreTest.cpp_docs.md_docs.md`
- **Keyword Index**: `TCPStoreTest.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

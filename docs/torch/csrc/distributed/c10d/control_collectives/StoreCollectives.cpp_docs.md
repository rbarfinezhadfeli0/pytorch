# Documentation: StoreCollectives.cpp

## File Metadata
- **Path**: `torch/csrc/distributed/c10d/control_collectives/StoreCollectives.cpp`
- **Size**: 5649 bytes
- **Lines**: 222
- **Extension**: .cpp
- **Type**: Regular file

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/StoreCollectives.hpp>
#include <chrono>
#include <exception>
#include <vector>

namespace {
std::string getRankKey(const std::string& key, int rank) {
  return fmt::format("{}/{}", key, rank);
}
} // namespace

namespace c10d {

StoreCollectives::StoreCollectives(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int worldSize)
    : store_(std::move(store)), rank_(rank), worldSize_(worldSize) {}

void StoreCollectives::barrier(
    const std::string& key,
    std::chrono::milliseconds timeout,
    bool blocking) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  auto num_members_key = fmt::format("{}/num_members", key);
  auto last_members_key = fmt::format("{}/last_members", key);

  auto idx = store_->add(num_members_key, 1);
  store_->set(getRankKey(key, rank_), "joined");

  if (idx == worldSize_) {
    store_->set(last_members_key, "<val_ignored>");
  } else if (blocking) {
    try {
      store_->wait({last_members_key});
    } catch (const std::exception& e) {
      std::string msg = "barrier failed -- missing ranks: ";
      for (int i = 0; i < worldSize_; i++) {
        if (i == rank_) {
          continue;
        }
        auto rank_key = getRankKey(key, i);
        if (!store_->check({rank_key})) {
          msg += fmt::format("{}, ", i);
        }
      }
      TORCH_CHECK(false, msg, e.what());
    }
  }
}

void StoreCollectives::broadcastSend(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  store_->set(key, data);
}

std::vector<uint8_t> StoreCollectives::broadcastRecv(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  return store_->get(key);
}

void StoreCollectives::gatherSend(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  auto rank_key = getRankKey(key, rank_);
  store_->set(rank_key, data);
}

std::vector<std::vector<uint8_t>> StoreCollectives::gatherRecv(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  std::vector<std::string> keys;
  keys.reserve(worldSize_);

  for (int i = 0; i < worldSize_; i++) {
    if (i == rank_) {
      continue;
    }
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }

  std::vector<std::vector<uint8_t>> results;
  results.reserve(worldSize_);

  try {
    results = store_->multiGet(keys);
  } catch (const std::exception& e) {
    std::string msg = "gather failed -- missing ranks: ";
    for (int i = 0; i < worldSize_; i++) {
      if (i == rank_) {
        continue;
      }
      auto rank_key = getRankKey(key, i);
      if (!store_->check({rank_key})) {
        msg += fmt::format("{}, ", i);
      }
    }
    TORCH_CHECK(false, msg, e.what());
  }

  // insert local data
  results.insert(results.begin() + rank_, data);
  return results;
}

std::vector<uint8_t> StoreCollectives::scatterSend(
    const std::string& key,
    const std::vector<std::vector<uint8_t>>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  std::vector<std::string> keys;
  keys.reserve(worldSize_);
  for (int i = 0; i < worldSize_; i++) {
    if (i == rank_) {
      continue;
    }
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }
  auto local = data.at(rank_);

  std::vector<std::vector<uint8_t>> toSend{data};

  toSend.erase(toSend.begin() + rank_);

  store_->multiSet(keys, toSend);

  return local;
}

std::vector<uint8_t> StoreCollectives::scatterRecv(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  auto rank_key = getRankKey(key, rank_);
  return store_->get(rank_key);
}

std::vector<std::vector<uint8_t>> StoreCollectives::allGather(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  auto localKey = getRankKey(key, rank_);
  store_->set(localKey, data);

  std::vector<std::string> keys;
  keys.reserve(worldSize_);

  for (int i = 0; i < worldSize_; i++) {
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }

  try {
    return store_->multiGet(keys);
  } catch (const std::exception& e) {
    std::string msg = "all_gather failed -- missing ranks: ";
    for (int i = 0; i < worldSize_; i++) {
      if (i == rank_) {
        continue;
      }
      auto rank_key = getRankKey(key, i);
      if (!store_->check({rank_key})) {
        msg += fmt::format("{}, ", i);
      }
    }
    TORCH_CHECK(false, msg, e.what());
  }
}

int64_t StoreCollectives::allSum(
    const std::string& key,
    int64_t value,
    std::chrono::milliseconds timeout) {
  enforceUnique(key);
  StoreTimeoutGuard g{*store_, timeout};

  store_->add(key, value);

  barrier(key + "/barrier", timeout);

  return store_->add(key, 0);
}

void StoreCollectives::enforceUnique(const std::string& key) {
  auto it = seenKeys_.find(key);
  TORCH_INTERNAL_ASSERT(
      it == seenKeys_.end(), "Key ", key, " has already been used.");
  seenKeys_.emplace(key);
}

} // namespace c10d

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough


## Key Components

The file contains 539 words across 222 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5649 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*

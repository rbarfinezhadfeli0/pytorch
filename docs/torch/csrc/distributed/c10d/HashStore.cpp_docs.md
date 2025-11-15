# Documentation: `torch/csrc/distributed/c10d/HashStore.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/HashStore.cpp`
- **Size**: 5,811 bytes (5.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/HashStore.hpp>

#include <unistd.h>
#include <cstdint>

#include <chrono>

#include <c10/util/Exception.h>

namespace c10d {

c10::intrusive_ptr<Store> HashStore::clone() {
  return c10::intrusive_ptr<Store>::unsafe_reclaim_from_nonowning(this);
}

void HashStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  std::unique_lock<std::mutex> lock(m_);
  map_[key] = data;
  cv_.notify_all();
}

std::vector<uint8_t> HashStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if ((it == map_.end() && expectedValue.empty()) ||
      (it != map_.end() && it->second == expectedValue)) {
    // if the key does not exist and currentValue arg is empty or
    // the key does exist and current value is what is expected, then set it
    map_[key] = desiredValue;
    cv_.notify_all();
    return desiredValue;
  } else if (it == map_.end()) {
    // if the key does not exist
    return expectedValue;
  }
  // key exists but current value is not expected
  return it->second;
}

std::vector<uint8_t> HashStore::get(const std::string& key) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if (it != map_.end()) {
    return it->second;
  }
  // Slow path: wait up to any timeout_.
  auto pred = [&]() { return map_.find(key) != map_.end(); };
  if (timeout_ == kNoTimeout) {
    cv_.wait(lock, pred);
  } else {
    if (!cv_.wait_for(lock, timeout_, pred)) {
      C10_THROW_ERROR(DistStoreError, "Wait timeout");
    }
  }
  return map_[key];
}

void HashStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  std::unique_lock<std::mutex> lock(m_);
  waitLocked(lock, keys, timeout);
}

void HashStore::waitLocked(
    std::unique_lock<std::mutex>& lock,
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  auto pred = [&]() { return checkLocked(lock, keys); };

  if (timeout == kNoTimeout) {
    cv_.wait(lock, pred);
  } else {
    if (!cv_.wait_until(lock, end, pred)) {
      C10_THROW_ERROR(DistStoreError, "Wait timeout");
    }
  }
}

int64_t HashStore::add(const std::string& key, int64_t i) {
  std::unique_lock<std::mutex> lock(m_);
  const auto& value = map_[key];
  int64_t ti = i;
  if (!value.empty()) {
    auto buf = reinterpret_cast<const char*>(value.data());
    auto len = value.size();
    ti += std::stoll(std::string(buf, len));
  }

  auto str = std::to_string(ti);
  const uint8_t* strB = reinterpret_cast<const uint8_t*>(str.c_str());
  map_[key] = std::vector<uint8_t>(strB, strB + str.size());
  return ti;
}

int64_t HashStore::getNumKeys() {
  std::unique_lock<std::mutex> lock(m_);
  return static_cast<int64_t>(map_.size());
}

bool HashStore::deleteKey(const std::string& key) {
  std::unique_lock<std::mutex> lock(m_);
  auto numDeleted = map_.erase(key);
  return (numDeleted == 1);
}

bool HashStore::check(const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> lock(m_);

  return checkLocked(lock, keys);
}

bool HashStore::checkLocked(
    const std::unique_lock<std::mutex>& lock,
    const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    auto foundKV = map_.find(key) != map_.end();
    auto foundQueue =
        queues_.find(key) != queues_.end() && !queues_[key].empty();
    if (!foundKV && !foundQueue) {
      return false;
    }
  }
  return true;
}

void HashStore::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    map_[key] = value;
  } else {
    it->second.insert(it->second.end(), value.begin(), value.end());
  }
  cv_.notify_all();
}

std::vector<std::vector<uint8_t>> HashStore::multiGet(
    const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> lock(m_);
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  std::vector<std::vector<uint8_t>> res;
  res.reserve(keys.size());

  for (auto& key : keys) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      res.emplace_back(it->second);
    } else {
      auto pred = [&]() { return map_.find(key) != map_.end(); };
      if (timeout_ == kNoTimeout) {
        cv_.wait(lock, pred);
      } else {
        if (!cv_.wait_until(lock, deadline, pred)) {
          C10_THROW_ERROR(DistStoreError, "Wait timeout");
        }
      }
      res.emplace_back(map_[key]);
    }
  }
  return res;
}

void HashStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  std::unique_lock<std::mutex> lock(m_);

  for (auto i : ::c10::irange(keys.size())) {
    map_[keys[i]] = values[i];
  }
  cv_.notify_all();
}

bool HashStore::hasExtendedApi() const {
  return true;
}

void HashStore::queuePush(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  std::unique_lock<std::mutex> lock(m_);

  queues_[key].push_back(value);

  cv_.notify_one();
}

std::vector<uint8_t> HashStore::queuePop(const std::string& key, bool block) {
  std::unique_lock<std::mutex> lock(m_);

  if (block) {
    waitLocked(lock, {key}, timeout_);
  }

  auto& queue = queues_[key];
  TORCH_CHECK_WITH(DistQueueEmptyError, !queue.empty(), "queue is empty");

  auto val = queue.front();
  queue.pop_front();
  return val;
}

int64_t HashStore::queueLen(const std::string& key) {
  std::unique_lock<std::mutex> lock(m_);

  auto it = queues_.find(key);
  if (it == queues_.end()) {
    return 0;
  }
  return static_cast<int64_t>(it->second.size());
}

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/HashStore.hpp`
- `unistd.h`
- `cstdint`
- `chrono`
- `c10/util/Exception.h`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`Store.hpp_docs.md`](./Store.hpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `HashStore.cpp_docs.md`
- **Keyword Index**: `HashStore.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `torch/csrc/distributed/c10d/PrefixStore.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/PrefixStore.cpp`
- **Size**: 4,054 bytes (3.96 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <utility>

namespace c10d {

PrefixStore::PrefixStore(std::string prefix, c10::intrusive_ptr<Store> store)
    : prefix_(std::move(prefix)), store_(std::move(store)) {}

c10::intrusive_ptr<Store> PrefixStore::clone() {
  return c10::make_intrusive<PrefixStore>(prefix_, store_->clone());
}

std::string PrefixStore::joinKey(const std::string& key) {
  return prefix_ + "/" + key;
}

std::vector<std::string> PrefixStore::joinKeys(
    const std::vector<std::string>& keys) {
  std::vector<std::string> joinedKeys;
  joinedKeys.reserve(keys.size());
  for (const auto& key : keys) {
    joinedKeys.emplace_back(joinKey(key));
  }
  return joinedKeys;
}

void PrefixStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->set(joinKey(key), value);
}

std::vector<uint8_t> PrefixStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  return store_->compareSet(joinKey(key), expectedValue, desiredValue);
}

std::vector<uint8_t> PrefixStore::get(const std::string& key) {
  return store_->get(joinKey(key));
}

int64_t PrefixStore::add(const std::string& key, int64_t value) {
  return store_->add(joinKey(key), value);
}

bool PrefixStore::deleteKey(const std::string& key) {
  return store_->deleteKey(joinKey(key));
}

int64_t PrefixStore::getNumKeys() {
  return store_->getNumKeys();
}

bool PrefixStore::check(const std::vector<std::string>& keys) {
  auto joinedKeys = joinKeys(keys);
  return store_->check(joinedKeys);
}

void PrefixStore::wait(const std::vector<std::string>& keys) {
  auto joinedKeys = joinKeys(keys);
  store_->wait(joinedKeys);
}

void PrefixStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  auto joinedKeys = joinKeys(keys);
  store_->wait(joinedKeys, timeout);
}

const std::chrono::milliseconds& PrefixStore::getTimeout() const noexcept {
  return store_->getTimeout();
}

void PrefixStore::setTimeout(const std::chrono::milliseconds& timeout) {
  store_->setTimeout(timeout);
}

void PrefixStore::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->append(joinKey(key), value);
}

std::vector<std::vector<uint8_t>> PrefixStore::multiGet(
    const std::vector<std::string>& keys) {
  std::vector<std::string> prefixed_keys;
  prefixed_keys.reserve(keys.size());
  for (auto& key : keys) {
    prefixed_keys.push_back(joinKey(key));
  }
  return store_->multiGet(prefixed_keys);
}

void PrefixStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  std::vector<std::string> prefixed_keys;
  prefixed_keys.reserve(keys.size());
  for (auto& key : keys) {
    prefixed_keys.push_back(joinKey(key));
  }
  store_->multiSet(prefixed_keys, values);
}

// Returns true if this store support append, multiGet and multiSet
bool PrefixStore::hasExtendedApi() const {
  return store_->hasExtendedApi();
}

void PrefixStore::queuePush(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->queuePush(joinKey(key), value);
}

std::vector<uint8_t> PrefixStore::queuePop(const std::string& key, bool block) {
  return store_->queuePop(joinKey(key), block);
}

int64_t PrefixStore::queueLen(const std::string& key) {
  return store_->queueLen(joinKey(key));
}

c10::intrusive_ptr<Store> PrefixStore::getUnderlyingStore() {
  return store_;
}

c10::intrusive_ptr<Store> PrefixStore::getUnderlyingNonPrefixStore() {
  c10::intrusive_ptr<Store> store = store_;

  while (store) {
    // Attempt to dynamically cast to PrefixStore
    PrefixStore* asPrefixStore = dynamic_cast<PrefixStore*>(store.get());
    if (asPrefixStore) {
      store = asPrefixStore->getUnderlyingStore();
    } else {
      break; // We've reached a non-PrefixStore
    }
  }

  TORCH_CHECK(
      store != nullptr, "Underlying Non-PrefixStore shouldn't be null.");
  return store;
}

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

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

- `torch/csrc/distributed/c10d/PrefixStore.hpp`
- `utility`


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

- **File Documentation**: `PrefixStore.cpp_docs.md`
- **Keyword Index**: `PrefixStore.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

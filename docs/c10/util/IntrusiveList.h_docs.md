# Documentation: `c10/util/IntrusiveList.h`

## File Metadata

- **Path**: `c10/util/IntrusiveList.h`
- **Size**: 4,495 bytes (4.39 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/Exception.h>

namespace c10 {

template <typename T>
class IntrusiveList;

class IntrusiveListHook {
  template <typename P, typename T>
  friend class ListIterator;

  template <typename T>
  friend class IntrusiveList;

  IntrusiveListHook* next_{nullptr};
  IntrusiveListHook* prev_{nullptr};

  void link_before(IntrusiveListHook* next_node) {
    next_ = next_node;
    prev_ = next_node->prev_;
    next_node->prev_ = this;
    prev_->next_ = this;
  }

 public:
  IntrusiveListHook() : next_(this), prev_(this) {}

  IntrusiveListHook(const IntrusiveListHook&) = delete;
  IntrusiveListHook& operator=(const IntrusiveListHook&) = delete;
  IntrusiveListHook(IntrusiveListHook&&) = delete;
  IntrusiveListHook& operator=(IntrusiveListHook&&) = delete;

  void unlink() {
    TORCH_CHECK(is_linked());
    next_->prev_ = prev_;
    prev_->next_ = next_;
    next_ = this;
    prev_ = this;
  }

  ~IntrusiveListHook() {
    if (is_linked()) {
      unlink();
    }
  }

  bool is_linked() const {
    return next_ != this;
  }
};

template <typename P, typename T>
class ListIterator {
  static_assert(std::is_same_v<std::remove_const_t<P>, IntrusiveListHook>);
  static_assert(std::is_base_of_v<IntrusiveListHook, T>);
  P* ptr_;

  friend class IntrusiveList<T>;

 public:
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = std::conditional_t<std::is_const_v<P>, const T, T>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit ListIterator(P* ptr) : ptr_(ptr) {}
  ~ListIterator() = default;

  ListIterator(const ListIterator&) = default;
  ListIterator& operator=(const ListIterator&) = default;
  ListIterator(ListIterator&&) = default;
  ListIterator& operator=(ListIterator&&) = default;

  template <
      typename Q,
      class = std::enable_if_t<std::is_const_v<P> && !std::is_const_v<Q>>>
  ListIterator(const ListIterator<Q, T>& rhs) : ptr_(rhs.ptr_) {}

  template <
      typename Q,
      class = std::enable_if_t<std::is_const_v<P> && !std::is_const_v<Q>>>
  ListIterator& operator=(const ListIterator<Q, T>& rhs) {
    ptr_ = rhs.ptr_;
    return *this;
  }

  template <typename Q>
  bool operator==(const ListIterator<Q, T>& other) const {
    return ptr_ == other.ptr_;
  }

  template <typename Q>
  bool operator!=(const ListIterator<Q, T>& other) const {
    return !(*this == other);
  }

  auto& operator*() const {
    return static_cast<reference>(*ptr_);
  }

  ListIterator& operator++() {
    TORCH_CHECK(ptr_);
    ptr_ = ptr_->next_;
    return *this;
  }

  ListIterator& operator--() {
    TORCH_CHECK(ptr_);
    ptr_ = ptr_->prev_;
    return *this;
  }

  auto* operator->() const {
    return static_cast<pointer>(ptr_);
  }
};

template <typename T>
class IntrusiveList {
  static_assert(std::is_base_of_v<IntrusiveListHook, T>);

 public:
  IntrusiveList() = default;
  IntrusiveList(const std::initializer_list<std::reference_wrapper<T>>& items) {
    for (auto& item : items) {
      insert(this->end(), item);
    }
  }
  ~IntrusiveList() {
    while (head_.is_linked()) {
      head_.next_->unlink();
    }
  }
  IntrusiveList(const IntrusiveList&) = delete;
  IntrusiveList& operator=(const IntrusiveList&) = delete;
  IntrusiveList(IntrusiveList&&) = delete;
  IntrusiveList& operator=(IntrusiveList&&) = delete;

  using iterator = ListIterator<IntrusiveListHook, T>;
  using const_iterator = ListIterator<const IntrusiveListHook, T>;

  auto begin() const {
    return ++const_iterator{&head_};
  }

  auto begin() {
    return ++iterator{&head_};
  }

  auto end() const {
    return const_iterator{&head_};
  }

  auto end() {
    return iterator{&head_};
  }

  auto rbegin() const {
    return std::reverse_iterator{end()};
  }

  auto rbegin() {
    return std::reverse_iterator{end()};
  }

  auto rend() const {
    return std::reverse_iterator{begin()};
  }

  auto rend() {
    return std::reverse_iterator{begin()};
  }

  auto iterator_to(const T& n) const {
    return const_iterator{&n};
  }

  auto iterator_to(T& n) {
    return iterator{&n};
  }

  iterator insert(iterator pos, T& n) {
    n.link_before(pos.ptr_);
    return iterator{&n};
  }

  size_t size() const {
    size_t ret = 0;
    for ([[maybe_unused]] auto& _ : *this) {
      ret++;
    }
    return ret;
  }

  bool empty() const {
    return !head_.is_linked();
  }

 private:
  IntrusiveListHook head_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 7 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `IntrusiveList`, `IntrusiveListHook`, `ListIterator`, `IntrusiveList`, `ListIterator`, `IntrusiveList`, `IntrusiveList`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

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

Files in the same folder (`c10/util`):

- [`CallOnce.h_docs.md`](./CallOnce.h_docs.md)
- [`Unicode.cpp_docs.md`](./Unicode.cpp_docs.md)
- [`logging_is_not_google_glog.h_docs.md`](./logging_is_not_google_glog.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`complex_math.h_docs.md`](./complex_math.h_docs.md)
- [`order_preserving_flat_hash_map.h_docs.md`](./order_preserving_flat_hash_map.h_docs.md)
- [`flags_use_gflags.cpp_docs.md`](./flags_use_gflags.cpp_docs.md)
- [`flags_use_no_gflags.cpp_docs.md`](./flags_use_no_gflags.cpp_docs.md)
- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`typeid.cpp_docs.md`](./typeid.cpp_docs.md)


## Cross-References

- **File Documentation**: `IntrusiveList.h_docs.md`
- **Keyword Index**: `IntrusiveList.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

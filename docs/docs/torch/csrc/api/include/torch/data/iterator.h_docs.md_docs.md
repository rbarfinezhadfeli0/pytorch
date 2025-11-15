# Documentation: `docs/torch/csrc/api/include/torch/data/iterator.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/data/iterator.h_docs.md`
- **Size**: 7,726 bytes (7.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/data/iterator.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/data/iterator.h`
- **Size**: 5,305 bytes (5.18 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/utils/variadic.h>
#include <torch/types.h>

#include <c10/util/Exception.h>

#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace torch::data {
namespace detail {
// For increased safety and more separated logic, this implementation of
// `Iterator` consists of a `ValidIterator` and a `SentinelIterator`. A
// `ValidIterator` yields new batches until the `DataLoader` is exhausted. While
// the `DataLoader` is not exhausted, `ValidIterator`s compare equal if they are
// the same object. When the `ValidIterator` becomes exhausted, it compares
// equal to the `SentinelIterator`, but not before. Half the code here is to
// implement double dispatch for the comparison. Got damnit, C++.

template <typename Batch>
struct ValidIterator;

template <typename Batch>
struct SentinelIterator;

/// Base class for the `ValidIterator` and `SentinelIterator`
template <typename Batch>
struct IteratorImpl {
  virtual ~IteratorImpl() = default;
  virtual void next() = 0;
  virtual Batch& get() = 0;
  virtual bool operator==(const IteratorImpl& other) const = 0;
  virtual bool operator==(const ValidIterator<Batch>& other) const = 0;
  virtual bool operator==(const SentinelIterator<Batch>& other) const = 0;
};

template <typename Batch>
struct ValidIterator : public IteratorImpl<Batch> {
  using BatchProducer = std::function<std::optional<Batch>()>;

  explicit ValidIterator(BatchProducer next_batch)
      : next_batch_(std::move(next_batch)) {}

  /// Fetches the next batch.
  void next() override {
    // If we didn't get the very first batch yet, get it now.
    lazy_initialize();
    TORCH_CHECK(
        batch_.has_value(), "Attempted to increment iterator past the end");
    // Increment to the next batch.
    batch_ = next_batch_();
  }

  /// Returns the current batch. The precondition for this operation to not
  /// throw an exception is that it has been compared to the `SentinelIterator`
  /// and did not compare equal.
  Batch& get() override {
    // If we didn't get the very first batch yet, get it now.
    lazy_initialize();
    TORCH_CHECK(
        batch_.has_value(),
        "Attempted to dereference iterator that was past the end");
    return batch_.value();
  }

  /// Does double dispatch.
  bool operator==(const IteratorImpl<Batch>& other) const override {
    return other == *this;
  }

  /// A `ValidIterator` is equal to the `SentinelIterator` iff. the
  /// `ValidIterator` has reached the end of the dataloader.
  bool operator==(const SentinelIterator<Batch>& /* unused */) const override {
    lazy_initialize();
    return !batch_;
  }

  /// Returns true if the memory address of `other` equals that of `this`.
  bool operator==(const ValidIterator<Batch>& other) const override {
    return &other == this;
  }

  /// Gets the very first batch if it has not yet been fetched.
  void lazy_initialize() const {
    if (!initialized_) {
      batch_ = next_batch_();
      initialized_ = true;
    }
  }

  BatchProducer next_batch_;
  mutable std::optional<Batch> batch_;
  mutable bool initialized_ = false;
};

template <typename Batch>
struct SentinelIterator : public IteratorImpl<Batch> {
  void next() override {
    TORCH_CHECK(
        false,
        "Incrementing the DataLoader's past-the-end iterator is not allowed");
  }

  Batch& get() override {
    TORCH_CHECK(
        false,
        "Dereferencing the DataLoader's past-the-end iterator is not allowed");
  }

  /// Does double dispatch.
  bool operator==(const IteratorImpl<Batch>& other) const override {
    return other == *this;
  }

  /// Calls the comparison operator between `ValidIterator` and
  /// `SentinelIterator`.
  bool operator==(const ValidIterator<Batch>& other) const override {
    return other == *this;
  }

  /// Sentinel iterators always compare equal.
  bool operator==(const SentinelIterator<Batch>& other) const override {
    return true;
  }
};
} // namespace detail

template <typename Batch>
class Iterator {
 public:
  // Type aliases to make the class recognized as a proper iterator.
  using difference_type = std::ptrdiff_t;
  using value_type = Batch;
  using pointer = Batch*;
  using reference = Batch&;
  using iterator_category = std::input_iterator_tag;

  explicit Iterator(std::unique_ptr<detail::IteratorImpl<Batch>> impl)
      : impl_(std::move(impl)) {}

  /// Increments the iterator.
  /// Only permitted for valid iterators (not past the end).
  Iterator& operator++() {
    impl_->next();
    return *this;
  }

  /// Returns the current batch.
  /// Only permitted for valid iterators (not past the end).
  Batch& operator*() {
    return impl_->get();
  }

  /// Returns a pointer to the current batch.
  /// Only permitted for valid iterators (not past the end).
  Batch* operator->() {
    return &impl_->get();
  }

  /// Compares two iterators for equality.
  bool operator==(const Iterator& other) const {
    return *impl_ == *other.impl_;
  }

  /// Compares two iterators for inequality.
  bool operator!=(const Iterator& other) const {
    return !(*this == other);
  }

 private:
  /// Points either to a `ValidIterator` or to a `SentinelIterator`.
  std::shared_ptr<detail::IteratorImpl<Batch>> impl_;
};
} // namespace torch::data

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`

**Classes/Structs**: `ValidIterator`, `SentinelIterator`, `for`, `IteratorImpl`, `ValidIterator`, `SentinelIterator`, `Iterator`, `recognized`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/data`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/utils/variadic.h`
- `torch/types.h`
- `c10/util/Exception.h`
- `functional`
- `iterator`
- `memory`
- `type_traits`
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

Files in the same folder (`torch/csrc/api/include/torch/data`):

- [`transforms.h_docs.md`](./transforms.h_docs.md)
- [`example.h_docs.md`](./example.h_docs.md)
- [`dataloader.h_docs.md`](./dataloader.h_docs.md)
- [`worker_exception.h_docs.md`](./worker_exception.h_docs.md)
- [`datasets.h_docs.md`](./datasets.h_docs.md)
- [`samplers.h_docs.md`](./samplers.h_docs.md)
- [`dataloader_options.h_docs.md`](./dataloader_options.h_docs.md)


## Cross-References

- **File Documentation**: `iterator.h_docs.md`
- **Keyword Index**: `iterator.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/data`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/data`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/api/include/torch/data`):

- [`transforms.h_docs.md_docs.md`](./transforms.h_docs.md_docs.md)
- [`transforms.h_kw.md_docs.md`](./transforms.h_kw.md_docs.md)
- [`samplers.h_docs.md_docs.md`](./samplers.h_docs.md_docs.md)
- [`samplers.h_kw.md_docs.md`](./samplers.h_kw.md_docs.md)
- [`datasets.h_kw.md_docs.md`](./datasets.h_kw.md_docs.md)
- [`dataloader.h_docs.md_docs.md`](./dataloader.h_docs.md_docs.md)
- [`dataloader.h_kw.md_docs.md`](./dataloader.h_kw.md_docs.md)
- [`example.h_docs.md_docs.md`](./example.h_docs.md_docs.md)
- [`iterator.h_kw.md_docs.md`](./iterator.h_kw.md_docs.md)
- [`dataloader_options.h_docs.md_docs.md`](./dataloader_options.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `iterator.h_docs.md_docs.md`
- **Keyword Index**: `iterator.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

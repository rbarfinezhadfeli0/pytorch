# Documentation: `docs/torch/csrc/api/include/torch/data/datasets/shared.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/data/datasets/shared.h_docs.md`
- **Size**: 4,794 bytes (4.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/data/datasets/shared.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/data/datasets/shared.h`
- **Size**: 2,595 bytes (2.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/data/datasets/base.h>

#include <memory>
#include <utility>

namespace torch::data::datasets {

/// A dataset that wraps another dataset in a shared pointer and implements the
/// `BatchDataset` API, delegating all calls to the shared instance. This is
/// useful when you want all worker threads in the dataloader to access the same
/// dataset instance. The dataset must take care of synchronization and
/// thread-safe access itself.
///
/// Use `torch::data::datasets::make_shared_dataset()` to create a new
/// `SharedBatchDataset` like you would a `std::shared_ptr`.
template <typename UnderlyingDataset>
class SharedBatchDataset : public BatchDataset<
                               SharedBatchDataset<UnderlyingDataset>,
                               typename UnderlyingDataset::BatchType,
                               typename UnderlyingDataset::BatchRequestType> {
 public:
  using BatchType = typename UnderlyingDataset::BatchType;
  using BatchRequestType = typename UnderlyingDataset::BatchRequestType;

  /// Constructs a new `SharedBatchDataset` from a `shared_ptr` to the
  /// `UnderlyingDataset`.
  /* implicit */ SharedBatchDataset(
      std::shared_ptr<UnderlyingDataset> shared_dataset)
      : dataset_(std::move(shared_dataset)) {}

  /// Calls `get_batch` on the underlying dataset.
  BatchType get_batch(BatchRequestType request) override {
    return dataset_->get_batch(std::move(request));
  }

  /// Returns the `size` from the underlying dataset.
  std::optional<size_t> size() const override {
    return dataset_->size();
  }

  /// Accesses the underlying dataset.
  UnderlyingDataset& operator*() {
    return *dataset_;
  }

  /// Accesses the underlying dataset.
  const UnderlyingDataset& operator*() const {
    return *dataset_;
  }

  /// Accesses the underlying dataset.
  UnderlyingDataset* operator->() {
    return dataset_.get();
  }

  /// Accesses the underlying dataset.
  const UnderlyingDataset* operator->() const {
    return dataset_.get();
  }

  /// Calls `reset()` on the underlying dataset.
  void reset() {
    dataset_->reset();
  }

 private:
  std::shared_ptr<UnderlyingDataset> dataset_;
};

/// Constructs a new `SharedBatchDataset` by creating a
/// `shared_ptr<UnderlyingDatase>`. All arguments are forwarded to
/// `make_shared<UnderlyingDataset>`.
template <typename UnderlyingDataset, typename... Args>
SharedBatchDataset<UnderlyingDataset> make_shared_dataset(Args&&... args) {
  return std::make_shared<UnderlyingDataset>(std::forward<Args>(args)...);
}
} // namespace torch::data::datasets

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `SharedBatchDataset`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/data/datasets`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/data/datasets/base.h`
- `memory`
- `utility`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/api/include/torch/data/datasets`):

- [`base.h_docs.md`](./base.h_docs.md)
- [`mnist.h_docs.md`](./mnist.h_docs.md)
- [`tensor.h_docs.md`](./tensor.h_docs.md)
- [`map.h_docs.md`](./map.h_docs.md)
- [`chunk.h_docs.md`](./chunk.h_docs.md)
- [`stateful.h_docs.md`](./stateful.h_docs.md)


## Cross-References

- **File Documentation**: `shared.h_docs.md`
- **Keyword Index**: `shared.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/data/datasets`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/data/datasets`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/api/include/torch/data/datasets`):

- [`map.h_kw.md_docs.md`](./map.h_kw.md_docs.md)
- [`stateful.h_docs.md_docs.md`](./stateful.h_docs.md_docs.md)
- [`shared.h_kw.md_docs.md`](./shared.h_kw.md_docs.md)
- [`map.h_docs.md_docs.md`](./map.h_docs.md_docs.md)
- [`tensor.h_kw.md_docs.md`](./tensor.h_kw.md_docs.md)
- [`base.h_kw.md_docs.md`](./base.h_kw.md_docs.md)
- [`mnist.h_kw.md_docs.md`](./mnist.h_kw.md_docs.md)
- [`chunk.h_docs.md_docs.md`](./chunk.h_docs.md_docs.md)
- [`base.h_docs.md_docs.md`](./base.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `shared.h_docs.md_docs.md`
- **Keyword Index**: `shared.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

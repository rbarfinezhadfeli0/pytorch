# Documentation: `docs/torch/csrc/api/include/torch/data/samplers/distributed.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/data/samplers/distributed.h_docs.md`
- **Size**: 6,339 bytes (6.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/data/samplers/distributed.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/data/samplers/distributed.h`
- **Size**: 4,060 bytes (3.96 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/data/samplers/base.h>

#include <cstddef>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::data::samplers {

/// A `Sampler` that selects a subset of indices to sample from and defines a
/// sampling behavior. In a distributed setting, this selects a subset of the
/// indices depending on the provided num_replicas and rank parameters. The
/// `Sampler` performs a rounding operation based on the `allow_duplicates`
/// parameter to decide the local sample count.
template <typename BatchRequest = std::vector<size_t>>
class DistributedSampler : public Sampler<BatchRequest> {
 public:
  DistributedSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true)
      : size_(size),
        num_replicas_(num_replicas),
        rank_(rank),

        allow_duplicates_(allow_duplicates) {}

  /// Set the epoch for the current enumeration. This can be used to alter the
  /// sample selection and shuffling behavior.
  void set_epoch(size_t epoch) {
    epoch_ = epoch;
  }

  size_t epoch() const {
    return epoch_;
  }

 protected:
  size_t local_sample_count() {
    if (allow_duplicates_) {
      return (size_ + num_replicas_ - 1) / num_replicas_;
    } else {
      return size_ / num_replicas_;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t size_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t num_replicas_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t rank_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t epoch_{0};
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool allow_duplicates_;
};

/// Select samples randomly. The sampling order is shuffled at each `reset()`
/// call.
class TORCH_API DistributedRandomSampler : public DistributedSampler<> {
 public:
  DistributedRandomSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// Resets the `DistributedRandomSampler` to a new set of indices.
  void reset(std::optional<size_t> new_size = std::nullopt) override;

  /// Returns the next batch of indices.
  std::optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `DistributedRandomSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `DistributedRandomSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `DistributedRandomSampler`.
  size_t index() const noexcept;

 private:
  void populate_indices();

  size_t begin_index_;
  size_t end_index_;
  size_t sample_index_;
  std::vector<size_t> all_indices_;
};

/// Select samples sequentially.
class TORCH_API DistributedSequentialSampler : public DistributedSampler<> {
 public:
  DistributedSequentialSampler(
      size_t size,
      size_t num_replicas = 1,
      size_t rank = 0,
      bool allow_duplicates = true);

  /// Resets the `DistributedSequentialSampler` to a new set of indices.
  void reset(std::optional<size_t> new_size = std::nullopt) override;

  /// Returns the next batch of indices.
  std::optional<std::vector<size_t>> next(size_t batch_size) override;

  /// Serializes the `DistributedSequentialSampler` to the `archive`.
  void save(serialize::OutputArchive& archive) const override;

  /// Deserializes the `DistributedSequentialSampler` from the `archive`.
  void load(serialize::InputArchive& archive) override;

  /// Returns the current index of the `DistributedSequentialSampler`.
  size_t index() const noexcept;

 private:
  void populate_indices();

  size_t begin_index_;
  size_t end_index_;
  size_t sample_index_;
  std::vector<size_t> all_indices_;
};

} // namespace torch::data::samplers

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OutputArchive`, `InputArchive`, `DistributedSampler`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/data/samplers`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/data/samplers/base.h`
- `cstddef`
- `vector`


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

Files in the same folder (`torch/csrc/api/include/torch/data/samplers`):

- [`sequential.h_docs.md`](./sequential.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`base.h_docs.md`](./base.h_docs.md)
- [`custom_batch_request.h_docs.md`](./custom_batch_request.h_docs.md)
- [`stream.h_docs.md`](./stream.h_docs.md)
- [`random.h_docs.md`](./random.h_docs.md)


## Cross-References

- **File Documentation**: `distributed.h_docs.md`
- **Keyword Index**: `distributed.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/data/samplers`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/data/samplers`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch/data/samplers`):

- [`sequential.h_docs.md_docs.md`](./sequential.h_docs.md_docs.md)
- [`serialize.h_docs.md_docs.md`](./serialize.h_docs.md_docs.md)
- [`stream.h_kw.md_docs.md`](./stream.h_kw.md_docs.md)
- [`custom_batch_request.h_docs.md_docs.md`](./custom_batch_request.h_docs.md_docs.md)
- [`serialize.h_kw.md_docs.md`](./serialize.h_kw.md_docs.md)
- [`custom_batch_request.h_kw.md_docs.md`](./custom_batch_request.h_kw.md_docs.md)
- [`random.h_kw.md_docs.md`](./random.h_kw.md_docs.md)
- [`sequential.h_kw.md_docs.md`](./sequential.h_kw.md_docs.md)
- [`base.h_kw.md_docs.md`](./base.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `distributed.h_docs.md_docs.md`
- **Keyword Index**: `distributed.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

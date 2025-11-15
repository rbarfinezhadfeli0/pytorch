# Documentation: `torch/csrc/api/include/torch/data/datasets/map.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/data/datasets/map.h`
- **Size**: 4,072 bytes (3.98 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/types.h>

#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace torch::data::datasets {
namespace detail {
template <bool C, typename T>
using optional_if_t = std::conditional_t<C, std::optional<T>, T>;
} // namespace detail

/// A `MapDataset` is a dataset that applies a transform to a source dataset.
template <typename SourceDataset, typename AppliedTransform>
class MapDataset : public BatchDataset<
                       MapDataset<SourceDataset, AppliedTransform>,
                       detail::optional_if_t<
                           SourceDataset::is_stateful,
                           typename AppliedTransform::OutputBatchType>,
                       typename SourceDataset::BatchRequestType> {
 public:
  using DatasetType = SourceDataset;
  using TransformType = AppliedTransform;
  using BatchRequestType = typename SourceDataset::BatchRequestType;
  using OutputBatchType = detail::optional_if_t<
      SourceDataset::is_stateful,
      typename AppliedTransform::OutputBatchType>;

  MapDataset(DatasetType dataset, TransformType transform)
      : dataset_(std::move(dataset)), transform_(std::move(transform)) {}

  /// Gets a batch from the source dataset and applies the transform to it,
  /// returning the result.
  OutputBatchType get_batch(BatchRequestType indices) override {
    return get_batch_impl(std::move(indices));
  }

  /// Returns the size of the source dataset.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  std::optional<size_t> size() const noexcept override {
    return dataset_.size();
  }

  /// Calls `reset()` on the underlying dataset.
  /// NOTE: Stateless datasets do not have a reset() method, so a call to this
  /// method will only compile for stateful datasets (which have a reset()
  /// method).
  void reset() {
    dataset_.reset();
  }

  /// Returns the underlying dataset.
  const SourceDataset& dataset() noexcept {
    return dataset_;
  }

  /// Returns the transform being applied.
  const AppliedTransform& transform() noexcept {
    return transform_;
  }

 private:
  /// The implementation of `get_batch()` for the stateless case, which simply
  /// applies the transform to the output of `get_batch()` from the dataset.
  template <
      typename D = SourceDataset,
      typename = std::enable_if_t<!D::is_stateful>>
  OutputBatchType get_batch_impl(BatchRequestType indices) {
    return transform_.apply_batch(dataset_.get_batch(std::move(indices)));
  }

  /// The implementation of `get_batch()` for the stateful case. Here, we follow
  /// the semantics of `Optional.map()` in many functional languages, which
  /// applies a transformation to the optional's content when the optional
  /// contains a value, and returns a new optional (of a different type)  if the
  /// original optional returned by `get_batch()` was empty.
  template <typename D = SourceDataset>
  std::enable_if_t<D::is_stateful, OutputBatchType> get_batch_impl(
      BatchRequestType indices) {
    if (auto batch = dataset_.get_batch(std::move(indices))) {
      return transform_.apply_batch(std::move(*batch));
    }
    return std::nullopt;
  }

  /// The underlying dataset being transformed.
  SourceDataset dataset_;

  // The transformation that is applied to batches received from the dataset.
  AppliedTransform transform_;
};

/// Creates a `MapDataset` with the given dataset and transform.
template <typename DatasetType, typename TransformType>
MapDataset<DatasetType, TransformType> map(
    DatasetType dataset,
    TransformType transform) {
  static_assert(
      std::is_same_v<
          std::conditional_t<
              DatasetType::is_stateful,
              typename DatasetType::BatchType::value_type,
              typename DatasetType::BatchType>,
          typename TransformType::InputBatchType>,
      "BatchType type of dataset does not match input type of transform");
  return {std::move(dataset), std::move(transform)};
}

} // namespace torch::data::datasets

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`

**Classes/Structs**: `MapDataset`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/data/datasets`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/data/datasets/base.h`
- `torch/types.h`
- `c10/util/ArrayRef.h`
- `cstddef`
- `type_traits`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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
- [`shared.h_docs.md`](./shared.h_docs.md)
- [`chunk.h_docs.md`](./chunk.h_docs.md)
- [`stateful.h_docs.md`](./stateful.h_docs.md)


## Cross-References

- **File Documentation**: `map.h_docs.md`
- **Keyword Index**: `map.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

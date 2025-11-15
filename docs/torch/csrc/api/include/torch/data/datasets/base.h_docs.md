# Documentation: `torch/csrc/api/include/torch/data/datasets/base.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/data/datasets/base.h`
- **Size**: 3,176 bytes (3.10 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/data/example.h>
#include <torch/types.h>

#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch::data::datasets {
template <typename S, typename T>
class MapDataset;
template <typename D, typename T>
MapDataset<D, T> map(D, T); // NOLINT
} // namespace torch::data::datasets

namespace torch::data::datasets {
namespace detail {
template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};
} // namespace detail

/// A dataset that can yield data only in batches.
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = ArrayRef<size_t>>
class BatchDataset {
 public:
  using SelfType = Self;
  using BatchType = Batch;
  using BatchRequestType = BatchRequest;
  constexpr static bool is_stateful = detail::is_optional<BatchType>::value;

  virtual ~BatchDataset() = default;

  /// Returns a batch of data given an index.
  virtual Batch get_batch(BatchRequest request) = 0;

  /// Returns the size of the dataset, or an empty std::optional if it is
  /// unsized.
  virtual std::optional<size_t> size() const = 0;

  /// Creates a `MapDataset` that applies the given `transform` to this dataset.
  template <typename TransformType>
  MapDataset<Self, TransformType> map(TransformType transform) & {
    return datasets::map(static_cast<Self&>(*this), std::move(transform));
  }

  /// Creates a `MapDataset` that applies the given `transform` to this dataset.
  template <typename TransformType>
  MapDataset<Self, TransformType> map(TransformType transform) && {
    return datasets::map(
        std::move(static_cast<Self&>(*this)), std::move(transform));
  }
};

/// A dataset that can yield data in batches, or as individual examples.
///
/// A `Dataset` is a `BatchDataset`, because it supports random access and
/// therefore batched access is implemented (by default) by calling the random
/// access indexing function for each index in the requested batch of indices.
/// This can be customized.
template <typename Self, typename SingleExample = Example<>>
class Dataset : public BatchDataset<Self, std::vector<SingleExample>> {
 public:
  using ExampleType = SingleExample;

  /// Returns the example at the given index.
  virtual ExampleType get(size_t index) = 0;

  /// Returns a batch of data.
  /// The default implementation calls `get()` for every requested index
  /// in the batch.
  std::vector<ExampleType> get_batch(ArrayRef<size_t> indices) override {
    std::vector<ExampleType> batch;
    batch.reserve(indices.size());
    for (const auto i : indices) {
      batch.push_back(get(i));
    }
    return batch;
  }
};

/// A `StreamDataset` represents a dataset that is a potentially infinite
/// stream. It takes as batch index only a number, which is the batch size, and
/// yields that many elements from the stream.
template <typename Self, typename Batch = std::vector<Example<>>>
using StreamDataset = BatchDataset<Self, Batch, /*BatchRequest=*/size_t>;
} // namespace torch::data::datasets

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`

**Classes/Structs**: `MapDataset`, `is_optional`, `is_optional`, `BatchDataset`, `Dataset`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/data/datasets`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/data/example.h`
- `torch/types.h`
- `c10/util/ArrayRef.h`
- `cstddef`
- `cstdint`
- `type_traits`
- `utility`
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

Files in the same folder (`torch/csrc/api/include/torch/data/datasets`):

- [`mnist.h_docs.md`](./mnist.h_docs.md)
- [`tensor.h_docs.md`](./tensor.h_docs.md)
- [`shared.h_docs.md`](./shared.h_docs.md)
- [`map.h_docs.md`](./map.h_docs.md)
- [`chunk.h_docs.md`](./chunk.h_docs.md)
- [`stateful.h_docs.md`](./stateful.h_docs.md)


## Cross-References

- **File Documentation**: `base.h_docs.md`
- **Keyword Index**: `base.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

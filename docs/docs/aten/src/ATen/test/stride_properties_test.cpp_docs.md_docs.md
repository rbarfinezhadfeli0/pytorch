# Documentation: `docs/aten/src/ATen/test/stride_properties_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/stride_properties_test.cpp_docs.md`
- **Size**: 6,515 bytes (6.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/stride_properties_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/stride_properties_test.cpp`
- **Size**: 3,953 bytes (3.86 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

// TODO: failing sizes {4, 1, 4, 1}
std::vector<std::vector<int64_t>> sizes = {{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 1, 1, 4}, {1, 4, 1, 4}, {1, 4, 4, 1}};

inline bool CheckStrideIndices(const Tensor& t, at::MemoryFormat format) {
  size_t n_dim = t.dim();
  std::vector<size_t> stride_indices(n_dim);
  if (format == at::MemoryFormat::ChannelsLast) {
    // stride_indices_ should be {1, n-1, n-2, ..., 2, 0}
    std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2);
    stride_indices[0] = 1;
    stride_indices[n_dim - 1] = 0;
  } else if (format == at::MemoryFormat::Contiguous) {
    // stride_indices_ should be {n-1, n-2, n-3, ..., 0}
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
  } else {
    TORCH_INTERNAL_ASSERT(false, "not recognized memory format");
  }

  // testing computeStrideProps with `IValue ival(t)` somehow doesn't work on CI
  // with onnx; The function works fine within, but stride properties is somehow
  // altered in ival->type()->cast<TensorType>();
  auto tt = TensorType::create(std::nullopt, std::nullopt, t.sizes(), t.strides(), std::nullopt);
  TORCH_INTERNAL_ASSERT(tt->stride_properties().isComplete(), "complete stride properties is needed for the test");

  auto index_iter = stride_indices.begin();
  for (const auto& opt_stride : *tt->stride_properties().sizes()) {
    if (*index_iter++ != opt_stride->stride_index_.value()) {
      return false;
    }
  }

  return true;
}

TEST(StridePropertiesTest, StrideIndicesTest) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (const auto& size : sizes) {
    Tensor t = at::rand(size);
    for (auto memory_format : {at::MemoryFormat::ChannelsLast, at::MemoryFormat::Contiguous}) {
      t.resize_(size, memory_format);
      EXPECT_TRUE(CheckStrideIndices(t, memory_format));
    }
  }
}

TEST(StridePropertiesTest, ZeroStrideIndicesEagerConsistencyTest) {
  auto permuted_tensor = at::rand({6, 3, 1, 5, 2}).permute({0, 3, 2, 1, 4}); // permute dim-1 & dim-3
  auto tensor = permuted_tensor.expand({6, 5, 4, 3, 2}); // expand dim-2

  auto temp = TensorType::create(std::nullopt, std::nullopt, tensor.sizes(), tensor.strides(), std::nullopt);

  // TensorIterator would preserve stride order, this is the eager reference
  auto eager_tensor = tensor.relu();
  auto ref_type = TensorType::create(std::nullopt, std::nullopt, eager_tensor.sizes(), eager_tensor.strides(), std::nullopt);

  TORCH_INTERNAL_ASSERT(temp->stride_properties().isComplete() &&
      temp->stride_properties().isComplete(), "complete stride properties is needed for the test");
  auto ref_iter = (*(ref_type->stride_properties().sizes())).begin();
  for (const auto& opt_stride : *temp->stride_properties().sizes()) {
    EXPECT_TRUE(opt_stride->stride_index_.value() == (*ref_iter)->stride_index_.value());
    ref_iter++;
  }
}

TEST(StridePropertiesTest, ExpandedStrideIndicesTest) {
  Tensor t = at::rand({1});
  // note: expand with dimension of size 1 is tricky as stride is different
  // depending on the order of the unsqueezed dimension.
  t = t.expand({4, 4, 4});
  EXPECT_TRUE(CheckStrideIndices(t, at::MemoryFormat::Contiguous));
}

TEST(StridePropertiesTest, SlicedStrideIndicesTest) {
  // Sliced tensor shouldn't have changed stride order
  Tensor t = at::rand({16, 4}).slice(1, 0, 4, 4);

  auto temp = TensorType::create(std::nullopt, std::nullopt, t.sizes(), t.strides(), std::nullopt);
  TORCH_INTERNAL_ASSERT(temp->stride_properties().isComplete() &&
      temp->stride_properties().isComplete(), "complete stride properties is needed for the test");
  std::vector<size_t> stride_indices(2);
  std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);

  auto index_iter = stride_indices.begin();
  for (const auto& opt_stride : *temp->stride_properties().sizes()) {
    EXPECT_TRUE(*index_iter++ == opt_stride->stride_index_.value());
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/stride_properties_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `stride_properties_test.cpp_docs.md`
- **Keyword Index**: `stride_properties_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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
python docs/aten/src/ATen/test/stride_properties_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `stride_properties_test.cpp_docs.md_docs.md`
- **Keyword Index**: `stride_properties_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

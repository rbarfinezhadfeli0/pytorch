# Documentation: `test/cpp/lazy/test_lazy_graph_executor.cpp`

## File Metadata

- **Path**: `test/cpp/lazy/test_lazy_graph_executor.cpp`
- **Size**: 3,252 bytes (3.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <test/cpp/lazy/test_lazy_ops_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include <vector>

namespace torch {
namespace lazy {
namespace {

class LazyGraphExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executor_ = LazyGraphExecutor::Get();
  }

  using CachedComputationType = LazyGraphExecutor::CachedComputation;

  std::shared_ptr<CachedComputationType> GetCachedComputation(hash_t hash) {
    return executor_->GetComputationCache()->Get(hash);
  }

  void EnsureComputationIsCached(
      std::vector<LazyTensorPtr>& tensors,
      hash_t hash) {
    // Force computation to be cached by syncing the tensors.
    executor_->SyncTensorsGraph(
        &tensors, /* devices */ {}, /* wait */ true, /* sync_ltc_data */ true);

    // Ensure that the computation cache entry exists.
    auto cached_computation = GetCachedComputation(hash);
    EXPECT_NE(cached_computation, nullptr)
        << "Computation should be cached after sync";
  }

  LazyGraphExecutor* executor_;
};

TEST_F(LazyGraphExecutorTest, TestClearComputationCache) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor tensor_a =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));
    torch::Tensor tensor_b =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));

    torch::Tensor xla_tensor_a = CopyToDevice(tensor_a, device);
    torch::Tensor xla_tensor_b = CopyToDevice(tensor_b, device);
    torch::Tensor result = xla_tensor_a + xla_tensor_b;

    std::vector<LazyTensorPtr> tensors{TryGetLtcTensor(result)};
    hash_t hash = executor_->GetGraphHash(tensors);
    EnsureComputationIsCached(tensors, hash);
    EXPECT_EQ(executor_->GetComputationCache()->Numel(), 1);

    // Clear the entire computation cache.
    executor_->ClearComputationCache();

    // Ensure that there are no cache entries.
    EXPECT_EQ(executor_->GetComputationCache()->Numel(), 0);
    auto cached_computation = GetCachedComputation(hash);
    EXPECT_EQ(cached_computation, nullptr)
        << "Cache entry should be null after clearing";
  });
}

TEST_F(LazyGraphExecutorTest, TestRemoveSpecificCacheEntry) {
  ForEachDevice([&](const torch::Device& device) {
    torch::Tensor tensor_a =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));
    torch::Tensor tensor_b =
        torch::rand({2, 2}, at::TensorOptions(torch::kFloat));

    torch::Tensor xla_tensor_a = CopyToDevice(tensor_a, device);
    torch::Tensor xla_tensor_b = CopyToDevice(tensor_b, device);
    torch::Tensor result = xla_tensor_a + xla_tensor_b;

    std::vector<LazyTensorPtr> tensors{TryGetLtcTensor(result)};
    hash_t hash = executor_->GetGraphHash(tensors);
    EnsureComputationIsCached(tensors, hash);

    // Remove a specific cache entry.
    executor_->RemoveFromComputationCache(hash);

    // Ensure that the cache entry has been removed.
    auto cached_computation = GetCachedComputation(hash);
    EXPECT_EQ(cached_computation, nullptr)
        << "Cache entry should be null after removal";

    // Attempting to remove again should not do anything.
    executor_->RemoveFromComputationCache(hash);
  });
}

} // namespace
} // namespace lazy
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `lazy`, `torch`

**Classes/Structs**: `LazyGraphExecutorTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `test/cpp/lazy/test_lazy_ops_util.h`
- `torch/csrc/lazy/core/lazy_graph_executor.h`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/lazy/test_lazy_graph_executor.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/lazy`):

- [`test_backend_device.cpp_docs.md`](./test_backend_device.cpp_docs.md)
- [`test_lazy_ops_util.cpp_docs.md`](./test_lazy_ops_util.cpp_docs.md)
- [`test_trie_cache.cpp_docs.md`](./test_trie_cache.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_lazy_ops_util.h_docs.md`](./test_lazy_ops_util.h_docs.md)
- [`test_misc.cpp_docs.md`](./test_misc.cpp_docs.md)
- [`test_ir.cpp_docs.md`](./test_ir.cpp_docs.md)
- [`test_util.cpp_docs.md`](./test_util.cpp_docs.md)
- [`test_shape.cpp_docs.md`](./test_shape.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_lazy_graph_executor.cpp_docs.md`
- **Keyword Index**: `test_lazy_graph_executor.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

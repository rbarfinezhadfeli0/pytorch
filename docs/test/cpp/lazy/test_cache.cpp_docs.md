# Documentation: `test/cpp/lazy/test_cache.cpp`

## File Metadata

- **Path**: `test/cpp/lazy/test_cache.cpp`
- **Size**: 2,752 bytes (2.69 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class CacheNode : public Node {
 public:
  explicit CacheNode(const std::string& str)
      : Node(OpKind(), /* num_outputs */ 1), hash_(Hash(str)), str_(str) {}
  ~CacheNode() override = default;

  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of test node");
  }

  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of test node");
  }

  hash_t hash() const override {
    return hash_;
  }
  hash_t shapeHash() const override {
    return hash_;
  }

 private:
  hash_t hash_;
  std::string str_;
};

TEST(CacheTest, BasicTest) {
  std::shared_ptr<CacheNode> a = std::make_shared<CacheNode>("a");
  std::shared_ptr<CacheNode> b = std::make_shared<CacheNode>("b");
  std::shared_ptr<CacheNode> c = std::make_shared<CacheNode>("c");
  Cache<hash_t, CacheNode, HashReducer> cache(2);

  cache.Add(a->hash(), a);
  EXPECT_EQ(cache.Get(a->hash()), a);
  EXPECT_EQ(cache.Get(b->hash()), nullptr);
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  cache.Add(b->hash(), b);
  EXPECT_EQ(cache.Get(a->hash()), a);
  EXPECT_EQ(cache.Get(b->hash()), b);
  EXPECT_EQ(cache.Get(c->hash()), nullptr);

  cache.Add(c->hash(), c);
  EXPECT_EQ(cache.Get(a->hash()), nullptr); // a has been evicted
  EXPECT_EQ(cache.Get(b->hash()), b);
  EXPECT_EQ(cache.Get(c->hash()), c);

  cache.Erase(c->hash());
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  EXPECT_EQ(cache.Get(b->hash()), b);
  EXPECT_EQ(cache.Get(c->hash()), nullptr); // c has been removed

  cache.Clear();
  EXPECT_EQ(cache.Get(a->hash()), nullptr);
  EXPECT_EQ(cache.Get(b->hash()), nullptr);
  EXPECT_EQ(cache.Get(c->hash()), nullptr);
}

class CacheNodeWithShape : public TsNode {
 public:
  explicit CacheNodeWithShape(const Shape& shape)
      : TsNode(OpKind(), shape, /* num_outputs */ 1, /* seed */ 0) {}
};

TEST(CacheTest, ShapeCacheTestForDynamicShape) {
  // enable dynamic shape
  FLAGS_ltc_enable_dynamic_shapes = true;

  CacheNodeWithShape nodes[] = {
      CacheNodeWithShape(Shape(c10::kFloat, {2, 4})),
      CacheNodeWithShape(Shape(c10::kFloat, {4, 2}))};

  /*
   * Make sure the cached shape for node (2, 4) is not used for node (4, 2)
   */
  for (auto& node : nodes) {
    EXPECT_EQ(node.shape(), node.computeShape([&]() { return node.shape(); }));
  }

  // reset the flag
  FLAGS_ltc_enable_dynamic_shapes = false;
}

} // namespace lazy
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `lazy`, `torch`

**Classes/Structs**: `CacheNode`, `CacheNodeWithShape`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/Exception.h`
- `torch/csrc/lazy/core/cache.h`
- `torch/csrc/lazy/core/hash.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/shape.h`
- `torch/csrc/lazy/ts_backend/ts_node.h`


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
python test/cpp/lazy/test_cache.cpp
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
- [`test_lazy_graph_executor.cpp_docs.md`](./test_lazy_graph_executor.cpp_docs.md)
- [`test_ir.cpp_docs.md`](./test_ir.cpp_docs.md)
- [`test_util.cpp_docs.md`](./test_util.cpp_docs.md)
- [`test_shape.cpp_docs.md`](./test_shape.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_cache.cpp_docs.md`
- **Keyword Index**: `test_cache.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

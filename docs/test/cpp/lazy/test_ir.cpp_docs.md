# Documentation: `test/cpp/lazy/test_ir.cpp`

## File Metadata

- **Path**: `test/cpp/lazy/test_ir.cpp`
- **Size**: 5,567 bytes (5.44 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/dynamic_ir.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/generated/LazyIr.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <memory>

namespace torch {
namespace lazy {

class TestLeafNode : public Node {
 public:
  static OpKind ClassOpKind() {
    return OpKind();
  }

  explicit TestLeafNode(size_t param)
      : Node(ClassOpKind(), /* num_outputs */ 1), hash_(Hash(param)) {}
  ~TestLeafNode() override = default;

  const std::vector<Output>& operands() const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operands of leaf node");
  }

  const Output& operand(size_t i) const override {
    TORCH_INTERNAL_ASSERT(false, "Can't access operand[i] of leaf node");
  }

  hash_t hash() const override {
    return hash_;
  }
  hash_t shapeHash() const override {
    return hash_;
  }

 private:
  hash_t hash_;
};

TEST(IrTest, BasicTest) {
  NodePtr node1 = MakeNode<TestLeafNode>(1);
  NodePtr node2 = MakeNode<TestLeafNode>(2);
  EXPECT_NE(node1->hash(), node2->hash());

  EXPECT_EQ(node1->num_outputs(), 1);

  const TestLeafNode* leafptr = NodeCast<TestLeafNode>(node1.get());
  EXPECT_TRUE(leafptr != nullptr);
}

TEST(IrTest, MetaDataTest) {
  bool restore_FLAGS_torch_lazy_ir_debug = FLAGS_torch_lazy_ir_debug;
  FLAGS_torch_lazy_ir_debug = false;
  NodePtr node = MakeNode<TestLeafNode>(1);
  auto metaWithoutDebug = node->metadata();
  EXPECT_EQ(metaWithoutDebug.scope.size(), 0);
  EXPECT_EQ(metaWithoutDebug.frame_info.size(), 0);

  FLAGS_torch_lazy_ir_debug = true;
  node = MakeNode<TestLeafNode>(1);
  auto metaWithEmptyDebug = node->metadata();
  EXPECT_EQ(metaWithEmptyDebug.scope.size(), 0);
  EXPECT_EQ(metaWithEmptyDebug.frame_info.size(), 1);

  {
    ScopePusher scope("TestScope");
    node = MakeNode<TestLeafNode>(1);
    auto metaWithScope = node->metadata();
    EXPECT_EQ(metaWithScope.scope, "TestScope.1");
    EXPECT_EQ(metaWithScope.frame_info.size(), 1);
  }

  SourceLocation dummySourceLocation;
  dummySourceLocation.file = "file";
  dummySourceLocation.function = "function";
  dummySourceLocation.line = 10;
  GetPythonFramesFunction() = [&]() -> std::vector<SourceLocation> {
    return {dummySourceLocation};
  };
  node = MakeNode<TestLeafNode>(1);
  auto metaWithSourceLoc = node->metadata();
  EXPECT_EQ(metaWithSourceLoc.scope.size(), 0);
  EXPECT_EQ(metaWithSourceLoc.frame_info.size(), 1);
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].file, "file");
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].function, "function");
  EXPECT_EQ(metaWithSourceLoc.frame_info[0].line, 10);
  FLAGS_torch_lazy_ir_debug = restore_FLAGS_torch_lazy_ir_debug;
}

TEST(IrTest, TsNodeTest) {
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);
  NodePtr node2 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);
  EXPECT_EQ(node1->hash(), node2->hash());

  EXPECT_EQ(node1->num_outputs(), 1);

  const TsNode* leafptr = dynamic_cast<const TsNode*>(node1.get());
  EXPECT_TRUE(leafptr != nullptr);
}

TEST(IrTest, DimensionNodeTest) {
  const size_t DIM0 = 5;
  const size_t DIM1 = 8;
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      Shape(c10::kFloat, {DIM0, DIM1}),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);

  auto size0 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 0));
  auto size1 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 1));

  ASSERT_EQ(DIM0, size0->getStaticValue());
  ASSERT_EQ(DIM1, size1->getStaticValue());

  NodePtr size0_np = size0;
  auto size0_dn = std::dynamic_pointer_cast<DimensionNode>(size0_np);
  ASSERT_EQ(DIM0, size0_dn->getStaticValue());

  auto add_dim = std::dynamic_pointer_cast<SizeAdd>(
      MakeNode<SizeAdd>(Value{size0}, Value{size1}));
  ASSERT_EQ(DIM0 + DIM1, add_dim->getStaticValue());

  auto mul_dim = std::dynamic_pointer_cast<SizeMul>(
      MakeNode<SizeMul>(Value{size0}, Value{size1}));
  ASSERT_EQ(DIM0 * DIM1, mul_dim->getStaticValue());
}

TEST(IrTest, DimensionIsDynamicTest) {
  const size_t DIM0 = 5;
  const size_t DIM1 = 8;
  const auto shape = Shape(c10::kFloat, {DIM0, DIM1});
  NodePtr node1 = MakeNode<TsNode>(
      OpKind(at::aten::view),
      shape.with_symbolic_dims(std::vector<bool>{true, false}),
      /*num_outputs*/ 1,
      /*hash_seed*/ kHashSeed);

  auto size0 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 0));
  auto size1 =
      std::dynamic_pointer_cast<SizeNode>(MakeNode<SizeNode>(Value{node1}, 1));

  ASSERT_EQ(true, size0->isSymbolic());
  ASSERT_EQ(false, size1->isSymbolic());

  auto add_dim = std::dynamic_pointer_cast<SizeAdd>(
      MakeNode<SizeAdd>(Value{size0}, Value{size1}));
  ASSERT_EQ(true, add_dim->isSymbolic());

  add_dim = std::dynamic_pointer_cast<SizeAdd>(
      MakeNode<SizeAdd>(Value{size1}, Value{size1}));
  ASSERT_EQ(false, add_dim->isSymbolic());

  auto mul_dim = std::dynamic_pointer_cast<SizeMul>(
      MakeNode<SizeMul>(Value{size0}, Value{size0}));
  ASSERT_EQ(true, mul_dim->isSymbolic());
}

} // namespace lazy
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `lazy`, `torch`

**Classes/Structs**: `TestLeafNode`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/core/ScalarType.h`
- `c10/util/Exception.h`
- `torch/csrc/lazy/core/config.h`
- `torch/csrc/lazy/core/debug_util.h`
- `torch/csrc/lazy/core/dynamic_ir.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/ir_builder.h`
- `torch/csrc/lazy/core/ir_metadata.h`
- `torch/csrc/lazy/generated/LazyIr.h`
- `torch/csrc/lazy/ts_backend/dynamic_ir.h`
- `torch/csrc/lazy/ts_backend/ts_node.h`
- `memory`


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
python test/cpp/lazy/test_ir.cpp
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
- [`test_util.cpp_docs.md`](./test_util.cpp_docs.md)
- [`test_shape.cpp_docs.md`](./test_shape.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_ir.cpp_docs.md`
- **Keyword Index**: `test_ir.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

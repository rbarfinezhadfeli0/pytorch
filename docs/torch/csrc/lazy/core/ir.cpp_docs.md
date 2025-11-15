# Documentation: `torch/csrc/lazy/core/ir.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/core/ir.cpp`
- **Size**: 4,666 bytes (4.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/env.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/cache.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

// Enables caching on for dynamic shapes (aka disable hash on shapes)
// clang-format off
C10_DEFINE_bool(
    ltc_enable_dynamic_shapes,
    false,
    "Whether dynamic shape is enabled")

namespace torch::lazy {
static const torch::lazy::Output kNullOutput = torch::lazy::Output();

size_t Output::Hasher::operator()(const Output& output) const {
  return StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

hash_t Output::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

hash_t Output::shapeHash() const {
  return HashCombine(node->shapeHash(), Hash(index));
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

bool Output::operator==(const Value& rhs) const {
  // Either side could be kNullValue which has node as nullptr
  return (!node == !rhs.node) &&
      (!node || (node->hash() == rhs.node->hash() && index == rhs.index));
}

hash_t Value::hash() const {
  return HashCombine(node->hash(), Hash(index));
}

hash_t Value::shapeHash() const {
  return HashCombine(node->shapeHash(), Hash(index));
}

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

hash_t OpKind::hash() const {
  return StringHash(op.toQualString());
}

bool Node::enableDynamicShape() {
  static bool enabled = c10::utils::has_env("LTC_ENABLE_DYNAMIC_SHAPES");
  return enabled || FLAGS_ltc_enable_dynamic_shapes;
}

Node::Node(OpKind op, size_t num_outputs)
    : op_(op), num_outputs_(num_outputs), metadata_(GetMetaDataIfDebugging()) {}

Node::Node(
    OpKind op,
    OpList operands,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    std::vector<Shape>&& shapes,
    size_t num_outputs)
    : Node(op, num_outputs) {
  // Move shapes into node
  shapes_.insert(
      shapes_.end(),
      std::make_move_iterator(shapes.begin()),
      std::make_move_iterator(shapes.end()));

  for (auto& operand : operands) {
    // Ideally, optional operands should be filtered by the leaf node classes,
    // but it's just much easier to do it here.
    // TODO(alanwaketan): Find a way to move the below logic to the leaf node
    // classes.
    if (!operand) {
      continue;
    }

    AddOperand(operand.node, operand.index);
  }
}

Node::Node(OpKind op, OpList operands, size_t num_outputs)
    : Node(op, operands, std::vector<Shape>{}, num_outputs) {}

Node::Node(OpKind op, Shape shape, size_t num_outputs) : Node(op, num_outputs) {
  shapes_.push_back(std::move(shape));
}

// Retrieves the full shape of the IR Node.
c10::ArrayRef<Shape> Node::shapes() const {
  return shapes_;
}

// Retrieves the shape of the output at a given index.
const Shape& Node::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

// Add the shape computed by the shape_fn

void Node::addComputedShape(const std::function<Shape()>& shape_fn) {
  shapes_.push_back(computeShape(shape_fn));
}

using ShapeCache = Cache<hash_t, Shape, HashReducer>;

// Compute the shape using the provided shape_fn.
Shape Node::computeShape(const std::function<Shape()>& shape_fn) {
  static ShapeCache* cache = new ShapeCache(FLAGS_torch_lazy_shape_cache_size);

  auto hash = shapeHash();
  auto shape = cache->Get(hash);
  if (shape == nullptr) {
    shape = cache->Add(hash, std::make_shared<Shape>(shape_fn()));
  }
  return *shape;
}

const std::vector<Output>& Node::operands() const {
  return operands_as_outputs_;
}

const Output& Node::operand(size_t i) const {
  return operands_as_outputs_.at(i);
}

const Output& Node::nullable_operand(size_t i) const {
  // We use kNullOutput instead of kNullValue here to avoid implicit casting,
  // which would prevent this method from returning a reference.
  return i < operands_as_outputs_.size() ? operand(i) : kNullOutput;
}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << shapes() << " " << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata().scope.empty()) {
    ss << ", scope=" << metadata().scope;
  }
  EmitShortFrameInfo(ss, metadata().frame_info);
  return ss.str();
}

void Node::AddOperand(const NodePtr& node, size_t index) {
  TORCH_CHECK_LT(index, node->num_outputs());
  operands_.push_back(node);
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/env.h`
- `torch/csrc/lazy/backend/backend_interface.h`
- `torch/csrc/lazy/core/cache.h`
- `torch/csrc/lazy/core/config.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/ir_metadata.h`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `ir.cpp_docs.md`
- **Keyword Index**: `ir.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

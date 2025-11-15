# Documentation: `docs/torch/csrc/jit/passes/peephole_dict_idioms.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/peephole_dict_idioms.cpp_docs.md`
- **Size**: 9,964 bytes (9.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/peephole_dict_idioms.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/peephole_dict_idioms.cpp`
- **Size**: 7,142 bytes (6.97 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/peephole_dict_idioms.h>

namespace torch::jit {

namespace {

class DictNodeImplBase {
 public:
  virtual ~DictNodeImplBase() = default;

  virtual bool contains(const IValue&) const = 0;
  virtual size_t size() const = 0;
  virtual Value* get(const IValue&) const = 0;

  bool canOptimize() {
    return !has_overlap_ && !has_non_const_key_;
  }

 protected:
  bool has_overlap_ = false;
  bool has_non_const_key_ = false;
};

template <class KeyType>
class DictNodeImpl : public DictNodeImplBase {
 public:
  DictNodeImpl(
      std::function<KeyType(const IValue&)> ivalue_converter,
      Node* dict_creation_node)
      : ivalue_converter_(std::move(ivalue_converter)) {
    for (size_t i = 0; i < dict_creation_node->inputs().size(); i += 2) {
      auto key_opt = toIValue(dict_creation_node->input(i));

      // Key is not constant if we cannot convert to IValue
      if (key_opt == std::nullopt) {
        has_non_const_key_ = true;
        continue;
      }

      KeyType key = ivalue_converter_(*key_opt);
      if (dict_.find(key) == dict_.end()) {
        dict_.emplace(key, dict_creation_node->input(i + 1));
      } else {
        has_overlap_ = true;
      }
    }
  }

  bool contains(const IValue& ivalue) const override {
    auto key = ivalue_converter_(ivalue);
    return dict_.find(key) != dict_.end();
  }

  size_t size() const override {
    return dict_.size();
  }

  Value* get(const IValue& ivalue) const override {
    auto val = ivalue_converter_(ivalue);
    auto loc = dict_.find(val);
    if (loc != dict_.end()) {
      return loc->second;
    }
    TORCH_CHECK(false, "Cannot get non-existent key");
  }

 private:
  std::unordered_map<KeyType, Value*> dict_;
  std::function<KeyType(const IValue&)> ivalue_converter_;
};

class DictNode {
 public:
  explicit DictNode(Node* dict_creation_node) {
    auto dict_type = dict_creation_node->output()->type();
    auto key_value_types = dict_type->containedTypes();
    TORCH_CHECK(
        key_value_types.size() == 2, "Dict must have 2 contained types");
    const auto& key_type = key_value_types[0];

    switch (key_type->kind()) {
      case TypeKind::IntType: {
        auto ivalue_converter = [](const IValue& ival) { return ival.toInt(); };
        impl_ = std::make_unique<DictNodeImpl<int64_t>>(
            std::move(ivalue_converter), dict_creation_node);
        break;
      }

      case TypeKind::FloatType: {
        auto ivalue_converter = [](const IValue& ival) {
          return ival.toDouble();
        };
        impl_ = std::make_unique<DictNodeImpl<double>>(
            std::move(ivalue_converter), dict_creation_node);
        break;
      }

      case TypeKind::StringType: {
        auto ivalue_converter = [](const IValue& ival) {
          return *ival.toString();
        };
        impl_ = std::make_unique<DictNodeImpl<std::string>>(
            std::move(ivalue_converter), dict_creation_node);
        break;
      }

      default:
        impl_ = nullptr;
    }
  }

  bool canOptimize() const {
    if (impl_) {
      return impl_->canOptimize();
    }
    return false;
  }

  size_t size() const {
    if (impl_) {
      return impl_->size();
    }
    return 0;
  }

  std::optional<Value*> getOrNullopt(const IValue& key) const {
    if (impl_ && impl_->contains(key)) {
      return impl_->get(key);
    }
    return std::nullopt;
  }

 private:
  std::unique_ptr<DictNodeImplBase> impl_;
};

bool isDict(Value* v) {
  return v->type()->castRaw<DictType>() != nullptr;
}

class PeepholeOptimizeDictIdiomsImpl {
 public:
  explicit PeepholeOptimizeDictIdiomsImpl(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb_(std::make_unique<AliasDb>(graph_)) {}

  bool run() {
    collectMutatedDicts(graph_->block());
    return runBlock(graph_->block());
  }

 private:
  void checkForMutatedDicts(Value* v) {
    if (isDict(v) && aliasDb_->hasWriters(v)) {
      mutated_dicts_.insert(v);
    }
  }

  void collectMutatedDicts(Block* b) {
    for (Value* v : b->inputs()) {
      checkForMutatedDicts(v);
    }
    for (Node* n : b->nodes()) {
      for (Value* v : n->outputs()) {
        checkForMutatedDicts(v);
      }
      for (Block* block : n->blocks()) {
        collectMutatedDicts(block);
      }
    }
  }

  const DictNode& getDictNode(Node* creation_node) {
    auto cached = dict_cache_.find(creation_node);
    if (cached == dict_cache_.end()) {
      cached =
          dict_cache_.emplace(creation_node, DictNode(creation_node)).first;
    }

    return cached->second;
  }

  std::optional<Value*> getValueFromDict(Node* dict_creation_node, Value* key) {
    const DictNode& dict_node = getDictNode(dict_creation_node);
    auto key_opt = toIValue(key);
    // Key is not constant if we cannot convert to IValue
    if (key_opt == std::nullopt) {
      return std::nullopt;
    }
    IValue key_ival = *key_opt;
    if (dict_node.canOptimize()) {
      return dict_node.getOrNullopt(key_ival);
    }
    return std::nullopt;
  }

  std::optional<int64_t> computeLen(Node* dict_creation_node) {
    const DictNode& dict_node = getDictNode(dict_creation_node);
    if (dict_node.canOptimize()) {
      return static_cast<int64_t>(dict_node.size());
    }
    return std::nullopt;
  }

  bool optimizeLen(Node* len_node, Node* creation_node) {
    if (creation_node->kind() == prim::DictConstruct) {
      auto len = computeLen(creation_node);
      if (len != std::nullopt) {
        WithInsertPoint guard(len_node);
        len_node->output()->replaceAllUsesWith(graph_->insertConstant(len));
        return true;
      }
    }
    return false;
  }

  bool optimizeGetItem(Node* getitem_node, Node* creation_node) {
    if (creation_node->kind() == prim::DictConstruct) {
      auto key = getitem_node->input(1);
      auto value = getValueFromDict(creation_node, key);
      if (value != std::nullopt) {
        getitem_node->output()->replaceAllUsesWith(*value);
        return true;
      }
    }
    return false;
  }

  bool runBlock(Block* block) {
    bool changed = false;
    for (Node* node : block->nodes()) {
      for (Block* b : node->blocks()) {
        changed |= runBlock(b);
      }

      // only optimizing dict ops
      if (node->inputs().empty() || !isDict(node->input(0))) {
        continue;
      }

      auto first_input = node->input(0);

      // only optimizing ops with unmutated inputs
      if (mutated_dicts_.count(first_input)) {
        continue;
      }

      if (node->kind() == aten::len) {
        changed |= optimizeLen(node, first_input->node());
      } else if (node->kind() == aten::__getitem__) {
        changed |= optimizeGetItem(node, first_input->node());
      }
    }
    return changed;
  }

  std::shared_ptr<Graph> graph_;
  std::unordered_set<Value*> mutated_dicts_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::unordered_map<Node*, DictNode> dict_cache_;
};

} // namespace

bool PeepholeOptimizeDictIdioms(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeDictIdiomsImpl opt(graph);
  return opt.run();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 25 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `bool`

**Classes/Structs**: `DictNodeImplBase`, `KeyType`, `DictNodeImpl`, `DictNode`, `PeepholeOptimizeDictIdiomsImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/alias_analysis.h`
- `torch/csrc/jit/passes/peephole_dict_idioms.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/csrc/jit/passes`):

- [`inline_fork_wait.h_docs.md`](./inline_fork_wait.h_docs.md)
- [`subgraph_rewrite.cpp_docs.md`](./subgraph_rewrite.cpp_docs.md)
- [`value_refinement_utils.cpp_docs.md`](./value_refinement_utils.cpp_docs.md)
- [`create_autodiff_subgraphs.cpp_docs.md`](./create_autodiff_subgraphs.cpp_docs.md)
- [`update_differentiable_graph_requires_grad.h_docs.md`](./update_differentiable_graph_requires_grad.h_docs.md)
- [`inplace_check.h_docs.md`](./inplace_check.h_docs.md)
- [`common_subexpression_elimination.h_docs.md`](./common_subexpression_elimination.h_docs.md)
- [`dtype_analysis.cpp_docs.md`](./dtype_analysis.cpp_docs.md)
- [`canonicalize.h_docs.md`](./canonicalize.h_docs.md)
- [`add_if_then_else.h_docs.md`](./add_if_then_else.h_docs.md)


## Cross-References

- **File Documentation**: `peephole_dict_idioms.cpp_docs.md`
- **Keyword Index**: `peephole_dict_idioms.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/jit/passes`):

- [`peephole_dict_idioms.h_docs.md_docs.md`](./peephole_dict_idioms.h_docs.md_docs.md)
- [`remove_redundant_profiles.h_kw.md_docs.md`](./remove_redundant_profiles.h_kw.md_docs.md)
- [`loop_unrolling.cpp_kw.md_docs.md`](./loop_unrolling.cpp_kw.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`guard_elimination.h_docs.md_docs.md`](./guard_elimination.h_docs.md_docs.md)
- [`frozen_conv_add_relu_fusion.cpp_docs.md_docs.md`](./frozen_conv_add_relu_fusion.cpp_docs.md_docs.md)
- [`hoist_conv_packed_params.h_kw.md_docs.md`](./hoist_conv_packed_params.h_kw.md_docs.md)
- [`lift_closures.h_kw.md_docs.md`](./lift_closures.h_kw.md_docs.md)
- [`frozen_conv_folding.h_kw.md_docs.md`](./frozen_conv_folding.h_kw.md_docs.md)
- [`frozen_graph_optimizations.h_docs.md_docs.md`](./frozen_graph_optimizations.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `peephole_dict_idioms.cpp_docs.md_docs.md`
- **Keyword Index**: `peephole_dict_idioms.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

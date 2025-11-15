# Documentation: `docs/torch/csrc/jit/passes/onnx/naming.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/onnx/naming.cpp_docs.md`
- **Size**: 8,719 bytes (8.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/onnx/naming.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/naming.cpp`
- **Size**: 5,992 bytes (5.85 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/onnx/naming.h>
#include <torch/csrc/onnx/onnx.h>

#include <utility>

namespace torch::jit::onnx {

namespace ONNXScopeName {

using NameFunc = std::string (*)(const torch::jit::ScopePtr& scope);

const std::string name_separator = "::";

namespace {

std::string nameFromRoot(
    const torch::jit::ScopePtr& scope,
    const std::string& layer_separator,
    NameFunc name_func) {
  std::string out = (*name_func)(scope);
  if (scope->isRoot()) {
    return out;
  }
  auto parent = scope->parent();
  while (isCompatibleScope(parent)) {
    out = std::string((*name_func)(parent)).append(layer_separator).append(out);
    parent = parent->parent();
  }
  return out;
}

std::pair<std::string, std::string> parseNameFromScope(
    const torch::jit::ScopePtr& scope) {
  std::string full_name = scope->name().toUnqualString();
  auto pos = full_name.find(name_separator);
  TORCH_CHECK(
      pos != std::string::npos,
      "Scope name (" + full_name + ") does not contain '" + name_separator +
          "'");
  return std::make_pair(full_name.substr(0, pos), full_name.substr(pos + 2));
}

} // namespace

std::string createFullScopeName(
    const std::string& class_name,
    const std::string& variable_name) {
  return std::string(class_name).append(name_separator).append(variable_name);
}

std::string variableName(const torch::jit::ScopePtr& scope) {
  return parseNameFromScope(scope).second;
}

std::string variableNameFromRoot(
    const torch::jit::ScopePtr& scope,
    const std::string& layer_separator) {
  return nameFromRoot(scope, layer_separator, &variableName);
}

std::string className(const torch::jit::ScopePtr& scope) {
  return parseNameFromScope(scope).first;
}

std::string classNameFromRoot(
    const torch::jit::ScopePtr& scope,
    const std::string& layer_separator) {
  return nameFromRoot(scope, layer_separator, &className);
}

bool isCompatibleScope(const torch::jit::ScopePtr& scope) {
  return !scope->isRoot() && !scope->isBlank() &&
      (std::string(scope->name().toUnqualString()).find(name_separator) !=
       std::string::npos);
}
} // namespace ONNXScopeName

namespace {

class NodeNameGenerator {
 public:
  NodeNameGenerator(std::shared_ptr<Graph> g) : graph_(std::move(g)) {}
  virtual ~NodeNameGenerator() = 0;
  void PopulateNodeNames();

 protected:
  virtual void CreateNodeName(Node* n) = 0;
  void PopulateNodeNames(Block* /*b*/);
  void UpdateOutputsNames(Node* n);
  bool IsGraphOutput(const Value* v, const std::shared_ptr<Graph>& graph) const;

 protected:
  std::string CreateUniqueName(
      std::unordered_map<std::string, size_t>& base_name_count,
      std::string base_name);

  std::unordered_map<const Node*, std::string> node_names_;
  std::unordered_map<std::string, size_t> base_node_name_counts_;
  std::shared_ptr<Graph> graph_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::string layer_separator_ = "/";
};
NodeNameGenerator::~NodeNameGenerator() = default;

class ScopedNodeNameGenerator : public NodeNameGenerator {
 public:
  ScopedNodeNameGenerator(std::shared_ptr<Graph> g)
      : NodeNameGenerator(std::move(g)) {}

 protected:
  void CreateNodeName(Node* n) override;

 private:
  std::string GetFullScopeName(const ScopePtr& scope);
  std::unordered_map<ScopePtr, std::string> full_scope_names_;
  std::unordered_map<std::string, size_t> base_scope_name_counts_;
};

std::string NodeNameGenerator::CreateUniqueName(
    std::unordered_map<std::string, size_t>& base_name_count,
    std::string base_name) {
  if (base_name_count.find(base_name) == base_name_count.end()) {
    base_name_count[base_name] = 0;
  } else {
    auto count = ++base_name_count[base_name];
    base_name += "_";
    base_name += std::to_string(count);
  }
  return base_name;
}

bool NodeNameGenerator::IsGraphOutput(
    const Value* v,
    const std::shared_ptr<Graph>& graph) const {
  for (const auto* graph_output : graph->outputs()) {
    if (v == graph_output) {
      return true;
    }
  }
  return false;
}

void NodeNameGenerator::UpdateOutputsNames(Node* n) {
  if (node_names_.find(n) != node_names_.end()) {
    auto node_name = node_names_[n];
    for (auto i : c10::irange(n->outputs().size())) {
      auto output = n->output(i);
      if (!IsGraphOutput(output, graph_)) {
        auto output_name = node_name;
        output_name.append("_output_").append(std::to_string(i));
        output->setDebugName(output_name);
      }
    }
  }
}

void NodeNameGenerator::PopulateNodeNames() {
  PopulateNodeNames(graph_->block());
}

void NodeNameGenerator::PopulateNodeNames(Block* b) {
  for (auto* n : b->nodes()) {
    for (auto* sub_block : n->blocks()) {
      PopulateNodeNames(sub_block);
    }
    CreateNodeName(n);
    UpdateOutputsNames(n);
  }
}

void ScopedNodeNameGenerator::CreateNodeName(Node* n) {
  if (node_names_.find(n) == node_names_.end()) {
    if (!ONNXScopeName::isCompatibleScope(n->scope())) {
      return;
    }
    if (n->mustBeNone()) {
      // JIT IR does not allow attribute for None node.
      return;
    }
    auto name = GetFullScopeName(n->scope());
    name += layer_separator_;
    name += n->kind().toUnqualString();
    node_names_[n] = CreateUniqueName(base_node_name_counts_, name);
  }
  n->s_(Symbol::attr(::torch::onnx::kOnnxNodeNameAttribute), node_names_[n]);
}

std::string ScopedNodeNameGenerator::GetFullScopeName(const ScopePtr& scope) {
  if (full_scope_names_.find(scope) == full_scope_names_.end()) {
    auto full_scope_name =
        ONNXScopeName::variableNameFromRoot(scope, layer_separator_);
    full_scope_names_[scope] =
        CreateUniqueName(base_scope_name_counts_, full_scope_name);
  }
  return full_scope_names_[scope];
}

} // namespace

void AssignScopedNamesForNodeAndValue(std::shared_ptr<Graph>& graph) {
  auto node_name_generator = std::make_unique<ScopedNodeNameGenerator>(graph);
  node_name_generator->PopulateNodeNames();
}

} // namespace torch::jit::onnx

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `void`, `ONNXScopeName`, `torch`, `std`

**Classes/Structs**: `NodeNameGenerator`, `ScopedNodeNameGenerator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/onnx/naming.h`
- `torch/csrc/onnx/onnx.h`
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

Files in the same folder (`torch/csrc/jit/passes/onnx`):

- [`remove_inplace_ops_for_onnx.cpp_docs.md`](./remove_inplace_ops_for_onnx.cpp_docs.md)
- [`list_model_parameters.cpp_docs.md`](./list_model_parameters.cpp_docs.md)
- [`preprocess_for_onnx.h_docs.md`](./preprocess_for_onnx.h_docs.md)
- [`remove_inplace_ops_for_onnx.h_docs.md`](./remove_inplace_ops_for_onnx.h_docs.md)
- [`constant_fold.cpp_docs.md`](./constant_fold.cpp_docs.md)
- [`eliminate_unused_items.cpp_docs.md`](./eliminate_unused_items.cpp_docs.md)
- [`cast_all_constant_to_floating.h_docs.md`](./cast_all_constant_to_floating.h_docs.md)
- [`list_model_parameters.h_docs.md`](./list_model_parameters.h_docs.md)
- [`shape_type_inference.cpp_docs.md`](./shape_type_inference.cpp_docs.md)
- [`constant_map.cpp_docs.md`](./constant_map.cpp_docs.md)


## Cross-References

- **File Documentation**: `naming.cpp_docs.md`
- **Keyword Index**: `naming.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/jit/passes/onnx`):

- [`constant_map.cpp_kw.md_docs.md`](./constant_map.cpp_kw.md_docs.md)
- [`deduplicate_initializers.h_docs.md_docs.md`](./deduplicate_initializers.h_docs.md_docs.md)
- [`shape_type_inference.h_docs.md_docs.md`](./shape_type_inference.h_docs.md_docs.md)
- [`function_substitution.h_kw.md_docs.md`](./function_substitution.h_kw.md_docs.md)
- [`eliminate_unused_items.h_kw.md_docs.md`](./eliminate_unused_items.h_kw.md_docs.md)
- [`prepare_division_for_onnx.cpp_docs.md_docs.md`](./prepare_division_for_onnx.cpp_docs.md_docs.md)
- [`fixup_onnx_controlflow.cpp_kw.md_docs.md`](./fixup_onnx_controlflow.cpp_kw.md_docs.md)
- [`constant_fold.cpp_docs.md_docs.md`](./constant_fold.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`onnx_log.cpp_kw.md_docs.md`](./onnx_log.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `naming.cpp_docs.md_docs.md`
- **Keyword Index**: `naming.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

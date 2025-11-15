# Documentation: `docs/torch/csrc/jit/passes/quantization/dedup_module_uses.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/quantization/dedup_module_uses.cpp_docs.md`
- **Size**: 7,227 bytes (7.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/quantization/dedup_module_uses.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/quantization/dedup_module_uses.cpp`
- **Size**: 4,494 bytes (4.39 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/quantization/dedup_module_uses.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

#include <stack>

namespace torch::jit {
namespace {
class ModuleUseDeduper {
 public:
  ModuleUseDeduper(Module& module) : module_(module) {}
  void dedup() {
    for (auto& method : module_.get_methods()) {
      const auto& graph = method.graph();
      findModuleUses(graph.get());
    }
    dedupModuleUses();
  }

 private:
  // Analyze the code to record information represents
  // uses of the module, which we'll use later to actually perform the dedup
  // operation Please see the comments of member variables of the class for more
  // information
  void findModuleUses(Graph* graph) {
    GRAPH_DUMP("Finding module uses for ", graph);

    std::stack<Block*> blocks_to_visit;
    blocks_to_visit.push(graph->block());
    Value* self = graph->inputs()[0];
    while (!blocks_to_visit.empty()) {
      Block* b = blocks_to_visit.top();
      blocks_to_visit.pop();
      for (Node* n : b->nodes()) {
        for (Block* subblock : n->blocks()) {
          blocks_to_visit.push(subblock);
        }
        if (n->kind() != prim::CallMethod) {
          continue;
        }
        Value* instance = n->inputs()[0];
        // boundary_val is the value we get when we trace back
        // the GetAttr access chain until we hit the input of graph
        // or a node that is not prim::GetAttr
        auto path = getModuleAccessPath(instance, self);

        // path.size() == 0 means we're calling a method
        // on self, we don't need to dedup uses of self
        if (path.empty()) {
          continue;
        }
        value_to_path_map_[instance] = path;
        auto m = findChildModule(module_, path);
        // If we fail to insert the module to the unique_modules_ set,
        // which means there are uses of this module before this point,
        // we'll have to rewrite the use
        if (!unique_modules_.insert(m._ivalue()).second) {
          uses_to_rewrite_.push_back(instance);
          GRAPH_DEBUG("Found use to rewrite: ", instance->debugName());
        }
      }
    }
  }

  // Deduplicate module uses given the information we recorded before
  void dedupModuleUses() {
    for (Value* v : uses_to_rewrite_) {
      const auto& path = value_to_path_map_.at(v);
      const auto& m = findChildModule(module_, path);
      // add a clone of the child module to the parent of the duplicated module
      const auto& child_name = addChildModule(module_, m, path);
      TORCH_INTERNAL_ASSERT(v->node()->kind() == prim::GetAttr);
      // change the name in GetAttr call
      auto original_name = v->node()->s(attr::name);
      v->node()->s_(attr::name, child_name);
      GRAPH_UPDATE(
          "Module use dedup: changing use of original module ",
          original_name,
          " to ",
          child_name);
    }
  }

  std::string addChildModule(
      Module& module,
      const Module& child_module,
      const std::vector<std::string>& path) {
    TORCH_INTERNAL_ASSERT(
        !path.empty(), "path must have at least one element.");
    // Parent module of the leaf child module corresponding to
    // the path
    auto parent_of_leaf = findChildModule(
        module, std::vector<std::string>(path.begin(), path.end() - 1));

    // Original name of the child module
    const std::string& original_name = path[path.size() - 1];
    int uid = 0;
    std::string child_name = original_name + "_" + std::to_string(uid++);
    while (parent_of_leaf.hasattr(child_name)) {
      child_name = original_name + "_" + std::to_string(uid++);
    }
    parent_of_leaf.register_module(child_name, child_module.deepcopy());
    return child_name;
  }

  Module module_;
  // Map from value of module instance to the list of names of submodules
  // starting from the top level module, e.g. ["sub1", "sub2", "relu"]
  // Also this is a cache of calling `getModuleAccessPath` of the value
  std::unordered_map<Value*, std::vector<std::string>> value_to_path_map_;
  // Set of unique modules that are used in the graphs
  std::unordered_set<ModulePtr> unique_modules_;
  // Values that represent the module instance(the use of the module)
  // that we'll need to rewrite as a use of a cloned module
  // instance
  std::vector<Value*> uses_to_rewrite_;
};

} // namespace

void DedupModuleUses(Module& module) {
  ModuleUseDeduper d(module);
  d.dedup();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`

**Classes/Structs**: `ModuleUseDeduper`, `for`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/quantization/dedup_module_uses.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/quantization/helper.h`
- `stack`


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

Files in the same folder (`torch/csrc/jit/passes/quantization`):

- [`quantization_type.cpp_docs.md`](./quantization_type.cpp_docs.md)
- [`insert_observers.cpp_docs.md`](./insert_observers.cpp_docs.md)
- [`insert_quant_dequant.h_docs.md`](./insert_quant_dequant.h_docs.md)
- [`register_packed_params.h_docs.md`](./register_packed_params.h_docs.md)
- [`finalize.cpp_docs.md`](./finalize.cpp_docs.md)
- [`helper.cpp_docs.md`](./helper.cpp_docs.md)
- [`finalize.h_docs.md`](./finalize.h_docs.md)
- [`insert_observers.h_docs.md`](./insert_observers.h_docs.md)
- [`fusion_passes.h_docs.md`](./fusion_passes.h_docs.md)
- [`quantization_patterns.h_docs.md`](./quantization_patterns.h_docs.md)


## Cross-References

- **File Documentation**: `dedup_module_uses.cpp_docs.md`
- **Keyword Index**: `dedup_module_uses.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/quantization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/passes/quantization`):

- [`dedup_module_uses.h_kw.md_docs.md`](./dedup_module_uses.h_kw.md_docs.md)
- [`insert_observers.cpp_kw.md_docs.md`](./insert_observers.cpp_kw.md_docs.md)
- [`insert_quant_dequant.cpp_kw.md_docs.md`](./insert_quant_dequant.cpp_kw.md_docs.md)
- [`finalize.cpp_kw.md_docs.md`](./finalize.cpp_kw.md_docs.md)
- [`register_packed_params.h_kw.md_docs.md`](./register_packed_params.h_kw.md_docs.md)
- [`helper.cpp_docs.md_docs.md`](./helper.cpp_docs.md_docs.md)
- [`fusion_passes.h_kw.md_docs.md`](./fusion_passes.h_kw.md_docs.md)
- [`finalize.cpp_docs.md_docs.md`](./finalize.cpp_docs.md_docs.md)
- [`quantization_type.h_docs.md_docs.md`](./quantization_type.h_docs.md_docs.md)
- [`insert_observers.cpp_docs.md_docs.md`](./insert_observers.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `dedup_module_uses.cpp_docs.md_docs.md`
- **Keyword Index**: `dedup_module_uses.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

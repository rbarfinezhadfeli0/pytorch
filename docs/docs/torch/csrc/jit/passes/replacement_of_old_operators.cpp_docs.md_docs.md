# Documentation: `docs/torch/csrc/jit/passes/replacement_of_old_operators.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/replacement_of_old_operators.cpp_docs.md`
- **Size**: 6,715 bytes (6.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/replacement_of_old_operators.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/replacement_of_old_operators.cpp`
- **Size**: 3,628 bytes (3.54 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/replacement_of_old_operators.h>

#include <c10/util/Exception.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/operator_upgraders/upgraders.h>
#include <torch/csrc/jit/operator_upgraders/utils.h>
#include <torch/csrc/jit/operator_upgraders/version_map.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace torch::jit {

struct OldOpsReplacerWithUpgraders {
  OldOpsReplacerWithUpgraders(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    if (!graph_->get_op_version().has_value()) {
      return;
    }

    auto current_version = graph_->get_op_version().value();
    DepthFirstGraphNodeIterator graph_it(graph_);
    Node* node = graph_it.next();
    while (node) {
      // load the schema name for this op
      std::optional<std::string> schema_name = std::nullopt;
      if (auto op_schema = node->maybeSchema()) {
        schema_name = getFullSchemaName(*op_schema);
      } else {
        schema_name = node->getHistoricSchemaName();
      }

      if (schema_name.has_value()) {
        // this implies there was a version bump because of this operator
        auto version_entry =
            get_operator_version_map().find(schema_name.value());
        if (version_entry != get_operator_version_map().end()) {
          const auto& entry = version_entry->second;
          auto upgrader_entry = findUpgrader(entry, current_version);
          if (!upgrader_entry.has_value()) {
            if (!isOpSymbolCurrent(schema_name.value(), current_version)) {
              TORCH_INTERNAL_ASSERT(
                  false,
                  "Upgrader must be present for ",
                  schema_name.value(),
                  ". The upgrader might have deprecated");
            }
            node = graph_it.next();
            continue;
          }
          auto upgrader_entry_val = upgrader_entry.value();
          auto upgrader_name = upgrader_entry_val.upgrader_name;
          auto upgrader_graph_entry = dump_upgraders_map().find(upgrader_name);
          TORCH_INTERNAL_ASSERT(
              upgrader_graph_entry != dump_upgraders_map().end(),
              "Corresponding upgrader graph for ",
              upgrader_name,
              " must exist.",
              " This upgrader"
              " might be deprecated.");

          auto upgrader_graph = upgrader_graph_entry->second;
          // inline the upgrader function body
          WithInsertPoint guard(node);
          auto new_outputs = insertGraph(
              *node->owningGraph(), *upgrader_graph, node->inputs());
          const auto& old_outputs = node->outputs();
          TORCH_INTERNAL_ASSERT(new_outputs.size() == old_outputs.size());
          for (const auto i : c10::irange(old_outputs.size())) {
            TORCH_INTERNAL_ASSERT(
                new_outputs[i]->type() == old_outputs[i]->type())
            old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
          }
          node->removeAllInputs();
          node->destroy();
        }
      }
      node = graph_it.next();
    }

    // now that we updated the graph, we want to bump the
    // graph version too.
    graph_->set_op_version(caffe2::serialize::kProducedFileFormatVersion);
  }

  std::shared_ptr<Graph> graph_;
};

TORCH_API void ReplaceOldOperatorsWithUpgraders(std::shared_ptr<Graph> graph) {
  OldOpsReplacerWithUpgraders(std::move(graph)).run();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OldOpsReplacerWithUpgraders`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/replacement_of_old_operators.h`
- `c10/util/Exception.h`
- `caffe2/serialize/versions.h`
- `torch/csrc/jit/frontend/schema_matching.h`
- `torch/csrc/jit/ir/irparser.h`
- `torch/csrc/jit/operator_upgraders/upgraders.h`
- `torch/csrc/jit/operator_upgraders/utils.h`
- `torch/csrc/jit/operator_upgraders/version_map.h`
- `torch/csrc/jit/runtime/graph_iterator.h`
- `limits`
- `string`
- `unordered_map`
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

- **File Documentation**: `replacement_of_old_operators.cpp_docs.md`
- **Keyword Index**: `replacement_of_old_operators.cpp_kw.md`
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

- **File Documentation**: `replacement_of_old_operators.cpp_docs.md_docs.md`
- **Keyword Index**: `replacement_of_old_operators.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

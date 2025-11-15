# Documentation: `docs/torch/csrc/jit/passes/canonicalize_graph_fuser_ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/canonicalize_graph_fuser_ops.cpp_docs.md`
- **Size**: 6,617 bytes (6.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/canonicalize_graph_fuser_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/canonicalize_graph_fuser_ops.cpp`
- **Size**: 3,825 bytes (3.74 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch::jit {

struct ChunkOutput {
  ChunkOutput(Value* v, size_t o) : val(v), offset(o) {}
  Value* val;
  size_t offset;
};

static std::optional<std::vector<ChunkOutput>> getChunkOutputs(Node* chunk) {
  std::vector<ChunkOutput> outputs;
  for (auto list_use : chunk->output()->uses()) {
    if (list_use.user->matches(
            "aten::select(t[] list, int idx) -> t", attr::idx) &&
        list_use.user->output()->type()->cast<TensorType>()) {
      outputs.emplace_back(
          list_use.user->output(),
          list_use.user->get<int64_t>(attr::idx).value());
    } else if (list_use.user->kind() == prim::ListUnpack) {
      // This sometimes happens if the sizes can't be evenly divided by the
      // number of chunks
      if (static_cast<int64_t>(list_use.user->outputs().size()) !=
          chunk->get<int64_t>(attr::chunks).value()) {
        return std::nullopt;
      }
      auto unpack_outputs = list_use.user->outputs();
      for (const auto i : c10::irange(unpack_outputs.size())) {
        outputs.emplace_back(unpack_outputs[i], i);
      }
    } else {
      return std::nullopt;
    }
  }
  return outputs;
}

static void CanonicalizeOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      CanonicalizeOps(sub);
    if (it->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        it->matches("aten::mul(Tensor self, Tensor other) -> Tensor") ||
        it->matches("aten::div(Tensor self, Tensor other) -> Tensor")) {
      // Replace rank 0 Tensor constants with scalar constants.
      if (auto other = it->get<at::Tensor>(attr::other)) {
        if (other->dim() == 0) {
          WithInsertPoint insert_guard{*it};
          auto graph = it->owningGraph();
          auto new_other = graph->insertConstant(other->item());
          std::vector<Value*> inputs = it->inputs().vec();
          inputs.at(1) = new_other;
          Value* new_output =
              graph->insertNode(graph->create(it->kind(), inputs))->output();
          new_output->node()->copyMetadata(*it);
          new_output->copyMetadata(it->output());
          it->output()->replaceAllUsesWith(new_output);
        }
      }
    } else if (it->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      // Replace aten::chunk (which returns a list) with ConstantChunk with the
      // outputs unpacked.
      if (auto orig_outputs = getChunkOutputs(*it)) {
        WithInsertPoint guard(*it);
        auto* self = it->namedInput(attr::self);
        auto* graph = it->owningGraph();
        const auto chunks = it->get<int64_t>(attr::chunks).value();
        const auto dim = it->get<int64_t>(attr::dim).value();
        auto* node =
            graph->insertNode(graph->create(prim::ConstantChunk, chunks));
        node->addInput(self);
        node->i_(attr::chunks, chunks)->i_(attr::dim, dim);
        node->copyMetadata(*it);
        for (const auto& orig_out : *orig_outputs) {
          orig_out.val->replaceAllUsesWith(node->outputs()[orig_out.offset]);
          node->outputs()[orig_out.offset]->setType(orig_out.val->type());
        }
      }
    }
  }
}

void CanonicalizeOps(const std::shared_ptr<Graph>& graph) {
  CanonicalizeOps(graph->block());
  GRAPH_DUMP("After CanonicalizeOps: ", graph);
  EliminateDeadCode(graph);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ChunkOutput`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`


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

- **File Documentation**: `canonicalize_graph_fuser_ops.cpp_docs.md`
- **Keyword Index**: `canonicalize_graph_fuser_ops.cpp_kw.md`
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

- **File Documentation**: `canonicalize_graph_fuser_ops.cpp_docs.md_docs.md`
- **Keyword Index**: `canonicalize_graph_fuser_ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

# Documentation: `docs/torch/csrc/jit/passes/inline_forked_closures.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/inline_forked_closures.cpp_docs.md`
- **Size**: 5,771 bytes (5.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/inline_forked_closures.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/inline_forked_closures.cpp`
- **Size**: 3,105 bytes (3.03 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/inline_forked_closures.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>

namespace torch::jit {

// Closure nodes are emitted as a tuple of (function %, context tuple %)
// Inside the closure the closure is then unpacked so that all closed over
// values are set. A function closing over a and b would look like:
// def foo(context):
//  a, b = context
//
// To fork the closure, we need to set each value in the context tuple
// as an explicit input to the fork node, and then within the closure
// subgraph, replace the context unpacking value with the new graph input.
// fork(foo) ->
// def foo(a, b):
static void inlineForkedClosure(Node* fork_closure, NodeKind genKind) {
  Node* function_context_node = fork_closure->input()->node();

  if (function_context_node->inputs().size() != 2 ||
      function_context_node->inputs().at(0)->node()->kind() != prim::Closure ||
      function_context_node->inputs().at(1)->node()->kind() !=
          prim::TupleConstruct) {
    throw ErrorReport(fork_closure->sourceRange()) << "Cannot fork this value";
  }

  Node* function = function_context_node->inputs().at(0)->node();
  Node* context = function_context_node->inputs().at(1)->node();
  auto fork_graph = function->g(attr::Subgraph)->copy();
  auto g = fork_closure->owningGraph();
  Node* fork_node = g->create(genKind, 1)
                        ->insertAfter(fork_closure)
                        ->setSourceRange(fork_closure->sourceRange());

  if (fork_graph->inputs().size() != 1 ||
      !fork_graph->inputs().at(0)->type()->cast<TupleType>()) {
    throw ErrorReport(fork_node->sourceRange())
        << "Cannot fork lambda with parameters";
  }
  auto fork_graph_context = fork_graph->inputs().at(0);
  AT_ASSERT(fork_graph_context->uses().size() == 1);
  auto fork_graph_unpack = fork_graph_context->uses().at(0).user;

  for (size_t i = 0; i < context->inputs().size(); ++i) {
    auto cont_input = context->inputs().at(i);
    fork_node->addInput(cont_input);
    auto inp = fork_graph->insertInput(i)->copyMetadata(cont_input);
    fork_graph_unpack->outputs().at(i)->replaceAllUsesWith(inp);
  }
  fork_graph_unpack->destroy();
  fork_graph->eraseInput(fork_graph->inputs().size() - 1);
  fork_node->output()->copyMetadata(fork_closure->output());
  fork_closure->output()->replaceAllUsesWith(fork_node->output());
  fork_closure->destroy();
  fork_node->g_(attr::Subgraph, fork_graph);
  runCleanupPasses(fork_graph);
}

static void inlineForkedClosures(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    switch (n->kind()) {
      case prim::forkClosure: {
        inlineForkedClosure(n, prim::fork);
      } break;
      case prim::awaitableClosure: {
        inlineForkedClosure(n, prim::awaitable);
      } break;
      default: {
        for (Block* b : n->blocks()) {
          inlineForkedClosures(b);
        }
      } break;
    }
  }
}

void inlineForkedClosures(std::shared_ptr<Graph>& to_clean) {
  inlineForkedClosures(to_clean->block());
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/inline_forked_closures.h`
- `torch/csrc/jit/frontend/ir_emitter.h`


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

- **File Documentation**: `inline_forked_closures.cpp_docs.md`
- **Keyword Index**: `inline_forked_closures.cpp_kw.md`
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

- **File Documentation**: `inline_forked_closures.cpp_docs.md_docs.md`
- **Keyword Index**: `inline_forked_closures.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

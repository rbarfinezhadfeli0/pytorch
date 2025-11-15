# Documentation: `torch/csrc/jit/passes/lift_closures.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/lift_closures.cpp`
- **Size**: 2,733 bytes (2.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/lift_closures.h>

#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir.h>

#include <utility>

namespace torch::jit {

// Closures are initially emitted as prim::Closure nodes with a single block.
// Here, we convert the block to a subgraph, adding all closed over variables
// as a context tuple input to the closure node.
// At this point the closure has already undergone conversion to SSA,
// so closed over variables will just be value * that are not set in the
// closure block.
// Within the closure subgraph, the context tuple is unpacked and the unpacked
// values are used for closed over values.
static void liftClosure(Node* closure) {
  auto block = closure->blocks().at(0);
  auto subgraph = std::make_shared<Graph>();
  // closures/forks can be nested, so use closure owning graph
  auto g = closure->owningGraph();
  Node* pack_context =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(closure);
  Value* context = subgraph->addInput("context");
  // cannot use createTupleUnpack because the type is not known yet
  Node* unpack_context =
      subgraph->insertNode(subgraph->create(prim::TupleUnpack, {context}, 0));

  std::unordered_map<Value*, Value*> captures;
  auto env = [&](Value* v) -> Value* {
    auto it = captures.find(v);
    if (it != captures.end()) {
      return it->second;
    }
    pack_context->addInput(v);
    Value* r = unpack_context->addOutput()->copyMetadata(v);
    captures[v] = r;
    return r;
  };
  subgraph->block()->cloneFrom(block, env);
  auto context_type = TupleType::create(
      fmap(pack_context->inputs(), [](Value* v) { return v->type(); }));
  context->setType(context_type);
  pack_context->output()->setType(context_type);
  auto closure_tuple =
      g->create(prim::TupleConstruct, {}, 1)->insertAfter(pack_context);
  closure->output()->replaceAllUsesWith(closure_tuple->output());
  closure_tuple->addInput(closure->output());
  closure_tuple->addInput(pack_context->output());
  closure_tuple->output()->setType(
      TupleType::create({closure->output()->type(), std::move(context_type)}));
  closure->eraseBlock(0);
  closure->g_(attr::Subgraph, std::move(subgraph));
  runCleanupPasses(closure->g(attr::Subgraph));
}

static void liftClosures(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    switch (n->kind()) {
      case prim::Closure: {
        liftClosure(n);
      } break;
      default: {
        for (Block* b : n->blocks()) {
          liftClosures(b);
        }
      }
    }
  }
}

void liftClosures(const std::shared_ptr<Graph>& to_clean) {
  liftClosures(to_clean->block());
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `torch/csrc/jit/passes/lift_closures.h`
- `torch/csrc/jit/frontend/ir_emitter.h`
- `torch/csrc/jit/ir/ir.h`
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

- **File Documentation**: `lift_closures.cpp_docs.md`
- **Keyword Index**: `lift_closures.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

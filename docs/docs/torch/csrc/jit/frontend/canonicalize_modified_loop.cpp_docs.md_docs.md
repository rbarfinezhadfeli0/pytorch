# Documentation: `docs/torch/csrc/jit/frontend/canonicalize_modified_loop.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/frontend/canonicalize_modified_loop.cpp_docs.md`
- **Size**: 4,977 bytes (4.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/frontend/canonicalize_modified_loop.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/canonicalize_modified_loop.cpp`
- **Size**: 2,368 bytes (2.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>

namespace torch::jit {

// Transforms a Loop that has both a trip count specified and a loop
// body condition so that the iter count is no longer specified
// and it is recognizable as a python while loop.
static void canonicalizeModifiedLoop(Node* n) {
  LoopView loop(n);
  if (loop.loopType() != LoopView::ModifiedLoop) {
    return;
  }

  auto g = n->owningGraph();
  WithInsertPoint node_insert(n);
  auto zero = g->insertConstant(0);
  auto one = g->insertConstant(1);
  auto max_trip_count = loop.maxTripCount();
  auto condition = g->insert(aten::gt, {max_trip_count, zero});
  loop.replaceMaxTripCount(
      g->insertConstant(std::numeric_limits<int64_t>::max()));

  auto inp_condition = toIValue(loop.inputCond());
  if (inp_condition == std::nullopt || inp_condition->toBool() == false) {
    condition = g->insert(aten::__and__, {condition, loop.inputCond()});
  }
  loop.replaceInputCondition(condition);
  n->addOutput()->setType(IntType::get());
  WithInsertPoint loop_insert(loop.bodyBlock());
  n->addInput(zero);
  auto new_iter = loop.bodyBlock()->addInput()->setType(IntType::get());
  // unset unique name for jitter, its replacement does not have a name
  loop.currentTripCount()->setDebugName("")->replaceAllUsesWith(new_iter);
  auto inc_iter = g->insert(aten::add, {new_iter, one});
  loop.bodyBlock()->registerOutput(inc_iter);
  auto less_than_max_trip = g->insert(aten::lt, {inc_iter, max_trip_count});
  auto loop_continue = loop.nextCond();
  auto new_condition =
      g->insert(aten::__and__, {less_than_max_trip, loop_continue});
  loop.bodyBlock()->eraseOutput(0);
  loop.bodyBlock()->insertOutput(0, new_condition);
}

static void canonicalizeModifiedLoops(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      canonicalizeModifiedLoops(b);
    }
    if (n->kind() == prim::Loop) {
      canonicalizeModifiedLoop(n);
    }
  }
}

// Transforms loops so that they can be represented as python
// for or while loops
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph) {
  canonicalizeModifiedLoops(graph->block());
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `functional`
- `memory`
- `string`
- `torch/csrc/Export.h`
- `torch/csrc/jit/frontend/canonicalize_modified_loop.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/ir_views.h`


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

Files in the same folder (`torch/csrc/jit/frontend`):

- [`schema_matching.cpp_docs.md`](./schema_matching.cpp_docs.md)
- [`source_range.h_docs.md`](./source_range.h_docs.md)
- [`exit_transforms.h_docs.md`](./exit_transforms.h_docs.md)
- [`function_schema_parser.h_docs.md`](./function_schema_parser.h_docs.md)
- [`inline_loop_condition.h_docs.md`](./inline_loop_condition.h_docs.md)
- [`mini_environment.h_docs.md`](./mini_environment.h_docs.md)
- [`tree_views.cpp_docs.md`](./tree_views.cpp_docs.md)
- [`function_schema_parser.cpp_docs.md`](./function_schema_parser.cpp_docs.md)
- [`tracer.cpp_docs.md`](./tracer.cpp_docs.md)


## Cross-References

- **File Documentation**: `canonicalize_modified_loop.cpp_docs.md`
- **Keyword Index**: `canonicalize_modified_loop.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/frontend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/frontend`):

- [`strtod.h_kw.md_docs.md`](./strtod.h_kw.md_docs.md)
- [`tree_views.cpp_docs.md_docs.md`](./tree_views.cpp_docs.md_docs.md)
- [`function_schema_parser.cpp_docs.md_docs.md`](./function_schema_parser.cpp_docs.md_docs.md)
- [`tree.h_kw.md_docs.md`](./tree.h_kw.md_docs.md)
- [`versioned_symbols.cpp_kw.md_docs.md`](./versioned_symbols.cpp_kw.md_docs.md)
- [`parser.cpp_kw.md_docs.md`](./parser.cpp_kw.md_docs.md)
- [`lexer.h_kw.md_docs.md`](./lexer.h_kw.md_docs.md)
- [`parser.cpp_docs.md_docs.md`](./parser.cpp_docs.md_docs.md)
- [`convert_to_ssa.h_docs.md_docs.md`](./convert_to_ssa.h_docs.md_docs.md)
- [`error_report.cpp_kw.md_docs.md`](./error_report.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `canonicalize_modified_loop.cpp_docs.md_docs.md`
- **Keyword Index**: `canonicalize_modified_loop.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

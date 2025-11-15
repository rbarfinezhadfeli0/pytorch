# Documentation: `docs/torch/csrc/jit/passes/erase_number_types.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/erase_number_types.cpp_docs.md`
- **Size**: 5,077 bytes (4.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/erase_number_types.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/erase_number_types.cpp`
- **Size**: 2,338 bytes (2.28 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/erase_number_types.h>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <ATen/ScalarOps.h>

namespace torch::jit {

static void SetNumTypeToTensorType(Value* v) {
  if (v->type()->isSubtypeOf(*NumberType::get())) {
    v->setType(TensorType::fromNumberType(*v->type()));
  } else if (v->type()->isSubtypeOf(*BoolType::get())) {
    v->setType(TensorType::fromBoolType());
  }
}

void EraseNumberTypesOnBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto inp : it->inputs()) {
      SetNumTypeToTensorType(inp);
    }
    for (auto sub : it->blocks()) {
      EraseNumberTypesOnBlock(sub);
    }
    switch (it->kind()) {
      case prim::Constant: {
        // remove primitive constants, replacing with tensor equivalent
        // ONNX does not support non-tensor constants
        if (it->output()->type()->isSubtypeOf(*NumberType::get()) ||
            it->output()->type()->isSubtypeOf(*BoolType::get())) {
          at::Scalar s;
          if (it->output()->type()->isSubtypeOf(*BoolType::get())) {
            s = *constant_as<bool>(it->output());
          } else {
            s = *constant_as<at::Scalar>(it->output());
          }

          WithInsertPoint guard(*it);
          Value* r = block->owningGraph()->insertConstant(
              scalar_to_tensor(s), std::nullopt, it->scope());
          r->copyMetadata(it->output());
          it->output()->replaceAllUsesWith(r);
          it.destroyCurrent();
        }
      } break;
      case aten::Bool:
      case aten::Float:
      case aten::Int:
      case aten::FloatImplicit:
      case aten::IntImplicit:
      case aten::ScalarImplicit:
      case prim::NumToTensor: {
        it->output()->replaceAllUsesWith(it->inputs()[0]);
        it.destroyCurrent();
      } break;
      default: {
        for (auto o : it->outputs()) {
          SetNumTypeToTensorType(o);
        }
      } break;
    }
  }
}

void EraseNumberTypes(const std::shared_ptr<Graph>& graph) {
  for (auto inp : graph->inputs()) {
    SetNumTypeToTensorType(inp);
  }
  EraseNumberTypesOnBlock(graph->block());
  GRAPH_DUMP("After EraseNumberTypes: ", graph);
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

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/erase_number_types.h`
- `torch/csrc/jit/ir/constants.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/dead_code_elimination.h`
- `ATen/ScalarOps.h`


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

- **File Documentation**: `erase_number_types.cpp_docs.md`
- **Keyword Index**: `erase_number_types.cpp_kw.md`
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

- **File Documentation**: `erase_number_types.cpp_docs.md_docs.md`
- **Keyword Index**: `erase_number_types.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

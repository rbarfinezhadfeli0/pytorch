# Documentation: `docs/torch/nativert/graph/passes/pass_manager/GraphPasses.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/graph/passes/pass_manager/GraphPasses.cpp_docs.md`
- **Size**: 5,262 bytes (5.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/graph/passes/pass_manager/GraphPasses.cpp`

## File Metadata

- **Path**: `torch/nativert/graph/passes/pass_manager/GraphPasses.cpp`
- **Size**: 3,002 bytes (2.93 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/graph/passes/pass_manager/GraphPasses.h>

#include <torch/nativert/graph/passes/SubgraphRewriter.h>
#include <torch/nativert/graph/passes/pass_manager/GraphPassRegistry.h>

namespace torch::nativert {

void register_base_passes() {
  GraphPassRegistry::add_pass("EmptyPass", [](Graph*) { return false; });

  GraphPassRegistry::add_pass(
      "LinearDynamicFp16UnpackedWeight", [](Graph* graph) {
        std::string p = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.aten.linear.default(input=%i, weight=%w, bias=%b)
    return (%out_0))";

        std::string p_1 = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight.default(X=%i, weight=%w, bias=%b)
    return (%out_0))";

        std::string p_new = R"(
    graph(%i, %w, %b):
    %pw = torch.ops.quantized.linear_prepack_fp16.default(W=%w, B=%b)
    %out_0 = torch.ops.quantized.linear_dynamic_fp16.default(X=%i, W_prepack=%pw)
    return (%out_0))";

        SubgraphRewriter rewriter("LinearDynamicFp16UnpackedWeight");
        rewriter.registerRewritePattern(p, p_new);
        rewriter.registerRewritePattern(p_1, p_new);
        return rewriter.run(graph);
      });

  GraphPassRegistry::add_pass(
      "LinearReluDynamicFp16UnpackedWeight", [](Graph* graph) {
        std::string p = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.aten.linear.default(input=%i, weight=%w, bias=%b)
    %out_1 = torch.ops.aten.relu.default(self=%out_0)
    return (%out_1))";

        std::string p_1 = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight.default(X=%i, weight=%w, bias=%b)
    %out_1 = torch.ops.aten.relu.default(self=%out_0)
    return (%out_1))";

        std::string p_new = R"(
    graph(%i, %w, %b):
    %pw = torch.ops.quantized.linear_prepack_fp16.default(W=%w, B=%b)
    %out_0 = torch.ops.quantized.linear_relu_dynamic_fp16.default(X=%i, W_prepack=%pw)
    return (%out_0))";

        SubgraphRewriter rewriter("LinearReluDynamicFp16UnpackedWeight");
        rewriter.registerRewritePattern(p, p_new);
        rewriter.registerRewritePattern(p_1, p_new);
        return rewriter.run(graph);
      });

  GraphPassRegistry::add_pass("CleanUpDeadNodes", [](Graph* graph) {
    return graph->cleanupDeadNodes();
  });

  GraphPassRegistry::add_pass("RemoveDetach", [](Graph* graph) {
    std::vector<Node*> nodesToDestroy;

    for (auto& node : graph->nodes()) {
      if (node.target() == "torch.ops.aten.detach.default") {
        nodesToDestroy.push_back(&node);
        graph->replaceAllUses(node.outputs()[0], node.inputs()[0].value);
      }
    }

    VLOG(1) << "[GraphPasses] Removed " << nodesToDestroy.size()
            << " aten.detach nodes";

    const bool mutated = !nodesToDestroy.empty();

    for (Node* node : nodesToDestroy) {
      node->destroy();
    }

    graph->renumberValues();
    graph->finalize();
    graph->lint();

    return mutated;
  });
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/graph/passes/pass_manager`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/graph/passes/pass_manager/GraphPasses.h`
- `torch/nativert/graph/passes/SubgraphRewriter.h`
- `torch/nativert/graph/passes/pass_manager/GraphPassRegistry.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/nativert/graph/passes/pass_manager`):

- [`PassPipeline.h_docs.md`](./PassPipeline.h_docs.md)
- [`PassManager.h_docs.md`](./PassManager.h_docs.md)
- [`GraphPassRegistry.h_docs.md`](./GraphPassRegistry.h_docs.md)
- [`PassManager.cpp_docs.md`](./PassManager.cpp_docs.md)
- [`GraphPasses.h_docs.md`](./GraphPasses.h_docs.md)


## Cross-References

- **File Documentation**: `GraphPasses.cpp_docs.md`
- **Keyword Index**: `GraphPasses.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/graph/passes/pass_manager`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/graph/passes/pass_manager`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/nativert/graph/passes/pass_manager`):

- [`PassManager.cpp_docs.md_docs.md`](./PassManager.cpp_docs.md_docs.md)
- [`PassManager.h_kw.md_docs.md`](./PassManager.h_kw.md_docs.md)
- [`GraphPasses.h_kw.md_docs.md`](./GraphPasses.h_kw.md_docs.md)
- [`PassManager.h_docs.md_docs.md`](./PassManager.h_docs.md_docs.md)
- [`GraphPasses.cpp_kw.md_docs.md`](./GraphPasses.cpp_kw.md_docs.md)
- [`GraphPassRegistry.h_kw.md_docs.md`](./GraphPassRegistry.h_kw.md_docs.md)
- [`PassPipeline.h_docs.md_docs.md`](./PassPipeline.h_docs.md_docs.md)
- [`GraphPasses.h_docs.md_docs.md`](./GraphPasses.h_docs.md_docs.md)
- [`GraphPassRegistry.h_docs.md_docs.md`](./GraphPassRegistry.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `GraphPasses.cpp_docs.md_docs.md`
- **Keyword Index**: `GraphPasses.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

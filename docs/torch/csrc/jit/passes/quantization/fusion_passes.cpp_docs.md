# Documentation: `torch/csrc/jit/passes/quantization/fusion_passes.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/quantization/fusion_passes.cpp`
- **Size**: 2,582 bytes (2.52 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/quantization/fusion_passes.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch::jit {

namespace {
void fuseQuantizeAddReluImpl(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter fused_add_relu_rewriter;
  std::string quantized_add_relu_pattern = R"(
    graph(%a_quant, %b_quant, %scale, %zero_point):
         %add_out = quantized::add(%a_quant, %b_quant, %scale, %zero_point)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_relu_pattern = R"(
    graph(%a_quant, %b_quant, %scale, %zero_point):
         %r = quantized::add_relu(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_relu_pattern, fused_add_relu_pattern);
  std::string quantized_add_out_relu_pattern = R"(
    graph(%a_quant, %b_quant, %out_quant):
         %add_out = quantized::add_out(%a_quant, %b_quant, %out_quant)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_out_relu_pattern = R"(
    graph(%a_quant, %b_quant, %out_quant):
         %r = quantized::add_relu_out(%a_quant, %b_quant, %out_quant)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_out_relu_pattern, fused_add_out_relu_pattern);
  std::string quantized_add_scalar_relu_pattern = R"(
    graph(%a_quant, %b_scalar):
         %add_out = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_scalar_relu_pattern = R"(
    graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_scalar_relu_pattern, fused_add_scalar_relu_pattern);
  std::string quantized_add_scalar_out_relu_pattern = R"(
    graph(%a_quant, %b_scalar, %out_quant):
         %add_out = quantized::add_scalar_out(%a_quant, %b_scalar, %out_quant)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_scalar_out_relu_pattern = R"(
    graph(%a_quant, %b_scalar, %out_quant):
         %r = quantized::add_scalar_relu_out(%a_quant, %b_scalar, %out_quant)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_scalar_out_relu_pattern, fused_add_scalar_out_relu_pattern);
  fused_add_relu_rewriter.runOnGraph(graph);
}
} // namespace

void FuseQuantizedAddRelu(std::shared_ptr<Graph>& graph) {
  fuseQuantizeAddReluImpl(graph);
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/quantization/fusion_passes.h`
- `torch/csrc/jit/passes/subgraph_rewrite.h`


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

- **File Documentation**: `fusion_passes.cpp_docs.md`
- **Keyword Index**: `fusion_passes.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

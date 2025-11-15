# Documentation: `docs/torch/csrc/jit/passes/quantization/finalize.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/quantization/finalize.h_docs.md`
- **Size**: 4,809 bytes (4.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/quantization/finalize.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/quantization/finalize.h`
- **Size**: 2,296 bytes (2.24 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/quantization/quantization_type.h>

namespace torch::jit {

/** \brief Backend specific pass to fuse dequantize - op - quantize calls
 * as quantized_op calls.
 *
 * Right now this is a fusion for fbgemm backend and only works for quantized
 * conv op, we'll extend to more ops and more backends in the future.
 *
 * Currently supported fusion:
 * q(conv2d(dq(a), dq(w), dq(b))) --> to_nchw(fbgemm_conv2d(prepack(to_nhwc(a)),
 *                                                          prepack(to_nhwc(w)),
 *                                                          prepack(to_nhwc(b))))
 *
 * q(linear(dq(a), dq(w), dq(b))) --> to_nchw(fbgemm_linear(prepack(to_nhwc(a)),
 *                                                          prepack(to_nhwc(w)),
 *                                                          prepack(to_nhwc(b))))
 *
 * \param graph the graph we want to apply fusion
 */
TORCH_API void QuantFusion(
    std::shared_ptr<Graph>& graph,
    QuantType quant_type = QuantType::STATIC);

/** \brief Insert prepack and unpack function in graph
 *  We want add pack/unpack functions for quantized weight because later we want
 * to fold the packed weight as an attribute of the module, in order to reduce
 * the cost of packing the weight on the fly in quantized models.
 *
 *  Each quantized op has it's corresponding prepack/unpack function,
 *  right now, we only need to do prepack/unpack for quantized::linear
 * and quantized::conv2d.
 */
TORCH_API void InsertPrepackUnpack(std::shared_ptr<Graph>& graph);

/** \brief Insert pack and unpack function in all graphs
 *   of module
 *
 *   Go through graphs of all the methods of all child modules
 *   and call InsertPrepackUnpack on the graph.
 */
TORCH_API void InsertPrepackUnpack(Module& module);

TORCH_API script::Module Finalize(
    script::Module& module,
    QuantType quant_type = QuantType::STATIC,
    const std::vector<std::string>& preserved_attrs =
        std::vector<std::string>());

TORCH_API void FoldQuantizedPrepackingOps(Module& module);

TORCH_API Module FinalizeOnDevicePTQ(
    Module& module,
    QuantType quant_type,
    const std::string& method_name);
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

This file is located in `torch/csrc/jit/passes/quantization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/passes/quantization/quantization_type.h`


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
- [`insert_observers.h_docs.md`](./insert_observers.h_docs.md)
- [`fusion_passes.h_docs.md`](./fusion_passes.h_docs.md)
- [`quantization_patterns.h_docs.md`](./quantization_patterns.h_docs.md)


## Cross-References

- **File Documentation**: `finalize.h_docs.md`
- **Keyword Index**: `finalize.h_kw.md`
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

- **File Documentation**: `finalize.h_docs.md_docs.md`
- **Keyword Index**: `finalize.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

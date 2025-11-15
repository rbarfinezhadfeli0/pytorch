# Documentation: `docs/torch/csrc/jit/tensorexpr/external_functions.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/external_functions.h_docs.md`
- **Size**: 6,016 bytes (5.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/external_functions.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/external_functions.h`
- **Size**: 3,433 bytes (3.35 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>
#include <ATen/Functions.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/Export.h>
#include <cstdint>
#include <vector>

#define FOR_ALL_EXTERNAL_FUNCTIONS(_)   \
  _(nnc_aten_adaptive_avg_pool2d)       \
  _(nnc_aten_addmm)                     \
  _(nnc_aten_conv2d)                    \
  _(nnc_aten_conv1d)                    \
  _(nnc_aten_conv1d_out)                \
  _(nnc_aten_dequantize)                \
  _(nnc_aten_dequantize_out)            \
  _(nnc_aten_embedding)                 \
  _(nnc_aten_matmul)                    \
  _(nnc_aten_mv)                        \
  _(nnc_aten_mm)                        \
  _(nnc_aten_mean)                      \
  _(nnc_aten_max_red)                   \
  _(nnc_aten_max_red_out)               \
  _(nnc_aten_quantized_conv1d)          \
  _(nnc_aten_quantized_conv1d_out)      \
  _(nnc_aten_quantized_conv2d)          \
  _(nnc_aten_quantized_conv2d_out)      \
  _(nnc_aten_quantized_conv2d_relu)     \
  _(nnc_aten_quantized_conv2d_relu_out) \
  _(nnc_aten_quantized_linear)          \
  _(nnc_aten_quantized_linear_out)      \
  _(nnc_aten_quantized_linear_relu)     \
  _(nnc_aten_quantized_add)             \
  _(nnc_aten_quantized_cat)             \
  _(nnc_aten_quantized_mul)             \
  _(nnc_aten_quantized_mul_out)         \
  _(nnc_aten_quantized_mul_scalar)      \
  _(nnc_aten_quantized_mul_scalar_out)  \
  _(nnc_aten_quantized_relu)            \
  _(nnc_aten_quantized_sigmoid)         \
  _(nnc_aten_quantized_sigmoid_out)     \
  _(nnc_aten_quantize_per_tensor)       \
  _(nnc_aten_quantize_per_tensor_out)   \
  _(nnc_aten_triangular_solve)          \
  _(nnc_aten_upsample_nearest2d)        \
  _(nnc_aten_upsample_nearest2d_out)    \
  _(nnc_prepacked_conv2d_clamp_run)     \
  _(nnc_prepacked_linear_clamp_run)

#define DECLARE_EXTERNAL_FUNCTION(NAME) \
  TORCH_API void NAME(                  \
      int64_t bufs_num,                 \
      void** buf_data,                  \
      int64_t* buf_ranks,               \
      int64_t* buf_dims,                \
      int64_t* buf_strides,             \
      int8_t* buf_dtypes,               \
      int64_t args_num,                 \
      int64_t* extra_args);

namespace torch::jit::tensorexpr {
struct QIData final {
  double scale;
  int64_t zero;
  c10::ScalarType scalarType;
};
std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg =
        std::nullopt);

std::vector<at::Tensor> constructTensors2(
    int64_t bufs_in_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg =
        std::nullopt,
    size_t bufs_out_num = 0);

#ifdef C10_MOBILE
extern "C" {
#endif
void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept;

FOR_ALL_EXTERNAL_FUNCTIONS(DECLARE_EXTERNAL_FUNCTION)
#if AT_MKLDNN_ENABLED()
DECLARE_EXTERNAL_FUNCTION(nnc_mkldnn_prepacked_conv_run)
#endif

TORCH_API void nnc_aten_free(size_t bufs_num, void** ptrs) noexcept;

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace torch::jit::tensorexpr

#undef DECLARE_EXTERNAL_FUNCTION

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `QIData`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/Functions.h`
- `c10/macros/Macros.h`
- `torch/csrc/Export.h`
- `cstdint`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `external_functions.h_docs.md`
- **Keyword Index**: `external_functions.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/jit/tensorexpr`):

- [`loopnest.h_kw.md_docs.md`](./loopnest.h_kw.md_docs.md)
- [`expr.h_docs.md_docs.md`](./expr.h_docs.md_docs.md)
- [`block_codegen.h_kw.md_docs.md`](./block_codegen.h_kw.md_docs.md)
- [`ir_cloner.cpp_kw.md_docs.md`](./ir_cloner.cpp_kw.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`tensorexpr_init.h_docs.md_docs.md`](./tensorexpr_init.h_docs.md_docs.md)
- [`lowerings.cpp_kw.md_docs.md`](./lowerings.cpp_kw.md_docs.md)
- [`graph_opt.h_kw.md_docs.md`](./graph_opt.h_kw.md_docs.md)
- [`eval.h_kw.md_docs.md`](./eval.h_kw.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `external_functions.h_docs.md_docs.md`
- **Keyword Index**: `external_functions.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

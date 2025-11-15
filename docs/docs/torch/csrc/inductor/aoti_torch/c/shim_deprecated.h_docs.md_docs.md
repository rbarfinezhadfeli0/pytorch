# Documentation: `docs/torch/csrc/inductor/aoti_torch/c/shim_deprecated.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_torch/c/shim_deprecated.h_docs.md`
- **Size**: 8,574 bytes (8.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_torch/c/shim_deprecated.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_torch/c/shim_deprecated.h`
- **Size**: 6,551 bytes (6.40 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#ifndef AOTI_TORCH_SHIM_DEPRECATED
#define AOTI_TORCH_SHIM_DEPRECATED

#include <torch/csrc/inductor/aoti_torch/c/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

[[deprecated(
    "aoti_torch__embedding_bag is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__embedding_bag(
    AtenTensorHandle weight,
    AtenTensorHandle indices,
    AtenTensorHandle offsets,
    int32_t scale_grad_by_freq,
    int32_t mode,
    int32_t sparse,
    AtenTensorHandle per_sample_weights, // optional argument
    int32_t include_last_offset,
    int32_t padding_idx,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
);

[[deprecated(
    "aoti_torch__fft_c2c is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,
    const int64_t* dim_ptr,
    int64_t dim_size,
    int64_t normalization,
    int32_t forward,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch__scaled_mm is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle bias,
    int32_t* out_dtype,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle scale_result,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1);

[[deprecated(
    "aoti_torch__scaled_mm_v2 is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_mm_v2(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle bias,
    AtenTensorHandle scale_result,
    int32_t* out_dtype,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0);

[[deprecated(
    "aoti_torch_addmm_out is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha);

[[deprecated(
    "aoti_torch_bmm is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

[[deprecated(
    "aoti_torch_convolution is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias, // optional argument
    const int64_t* stride_ptr,
    int64_t stride_size,
    const int64_t* padding_ptr,
    int64_t padding_size,
    const int64_t* dilation_ptr,
    int64_t dilation_size,
    int transposed,
    const int64_t* output_padding_ptr,
    int64_t output_padding_size,
    int64_t groups,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch_mm_out is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2);

[[deprecated(
    "aoti_torch_nonzero is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_nonzero(AtenTensorHandle self, AtenTensorHandle* out);

[[deprecated(
    "aoti_torch_repeat_interleave_Tensor is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,
    int64_t* output_size,
    AtenTensorHandle* out);

[[deprecated(
    "aoti_torch_view_as_real is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch_view_dtype is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,
    int32_t dtype,
    AtenTensorHandle* ret // returns new reference
);

[[deprecated(
    "aoti_torch__scaled_dot_product_flash_attention is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    double scale,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_flash_attention_v2(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    int is_causal,
    int return_debug_mask,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
);

[[deprecated(
    "aoti_torch__scaled_dot_product_efficient_attention is deprecated and will be removed in future versions.")]]
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch__scaled_dot_product_efficient_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    AtenTensorHandle attn_bias, // optional argument
    int compute_log_sumexp,
    double dropout_p,
    int is_causal,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
);

#ifdef __cplusplus
} // extern "C"

#endif
#endif // AOTI_TORCH_SHIM_DEPRECATED

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_torch/c`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_torch/c/macros.h`


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

Files in the same folder (`torch/csrc/inductor/aoti_torch/c`):

- [`macros.h_docs.md`](./macros.h_docs.md)
- [`shim_cpu.h_docs.md`](./shim_cpu.h_docs.md)
- [`shim.h_docs.md`](./shim.h_docs.md)
- [`shim_xpu.h_docs.md`](./shim_xpu.h_docs.md)
- [`shim_mps.h_docs.md`](./shim_mps.h_docs.md)


## Cross-References

- **File Documentation**: `shim_deprecated.h_docs.md`
- **Keyword Index**: `shim_deprecated.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_torch/c`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_torch/c`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/inductor/aoti_torch/c`):

- [`shim_deprecated.h_kw.md_docs.md`](./shim_deprecated.h_kw.md_docs.md)
- [`shim_cpu.h_docs.md_docs.md`](./shim_cpu.h_docs.md_docs.md)
- [`shim_xpu.h_docs.md_docs.md`](./shim_xpu.h_docs.md_docs.md)
- [`macros.h_kw.md_docs.md`](./macros.h_kw.md_docs.md)
- [`shim.h_kw.md_docs.md`](./shim.h_kw.md_docs.md)
- [`shim_mps.h_kw.md_docs.md`](./shim_mps.h_kw.md_docs.md)
- [`macros.h_docs.md_docs.md`](./macros.h_docs.md_docs.md)
- [`shim_mps.h_docs.md_docs.md`](./shim_mps.h_docs.md_docs.md)
- [`shim_cpu.h_kw.md_docs.md`](./shim_cpu.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `shim_deprecated.h_docs.md_docs.md`
- **Keyword Index**: `shim_deprecated.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

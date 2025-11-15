# Documentation: `docs/torch/csrc/inductor/aoti_torch/c/shim_xpu.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_torch/c/shim_xpu.h_docs.md`
- **Size**: 8,070 bytes (7.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_torch/c/shim_xpu.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_torch/c/shim_xpu.h`
- **Size**: 5,982 bytes (5.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#ifndef AOTI_TORCH_SHIM_XPU
#define AOTI_TORCH_SHIM_XPU

#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef USE_XPU
#ifdef __cplusplus
extern "C" {
#endif

struct XPUGuardOpaque;
using XPUGuardHandle = XPUGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_xpu_guard(
    int32_t device_index,
    XPUGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_xpu_guard(XPUGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_guard_set_index(XPUGuardHandle guard, int32_t device_index);

struct XPUStreamGuardOpaque;
using XPUStreamGuardHandle = XPUStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_xpu_stream_guard(
    void* stream,
    int32_t device_index,
    XPUStreamGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_xpu_stream_guard(XPUStreamGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_stream(int32_t device_index, void** ret_stream);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_device(int32_t* device_index);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_set_current_xpu_device(const int32_t& device_index);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_current_sycl_queue(void** ret);

#if AT_MKLDNN_ENABLED()

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_mkldnn__convolution_pointwise_binary(
    AtenTensorHandle X,
    AtenTensorHandle other,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_xpu_mkldnn__convolution_pointwise(
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* attr,
    const double** scalars,
    int64_t scalars_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_mkldnn__convolution_pointwise_binary_(
    AtenTensorHandle other,
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_xpu__qlinear_pointwise_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle* B,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    const char* post_op_name,
    const double** post_op_args,
    int64_t post_op_args_len_,
    const char* post_op_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu__qlinear_pointwise_binary_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle* other,
    AtenTensorHandle* B,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    double other_scale,
    int64_t other_zero_point,
    const char* binary_post_op,
    double binary_alpha,
    const char* unary_post_op,
    const double** unary_post_op_args,
    int64_t unary_post_op_args_len_,
    const char* unary_post_op_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_xpu__qconv_pointwise_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle* B,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    const char* attr,
    const double** post_op_args,
    int64_t post_op_args_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu__qconv2d_pointwise_binary_tensor(
    AtenTensorHandle X,
    AtenTensorHandle act_scale,
    AtenTensorHandle act_zero_point,
    AtenTensorHandle onednn_weight,
    AtenTensorHandle weight_scales,
    AtenTensorHandle weight_zero_points,
    AtenTensorHandle accum,
    AtenTensorHandle* B,
    const int64_t* stride_args,
    int64_t stride_len_,
    const int64_t* padding_args,
    int64_t padding_len_,
    const int64_t* dilation_args,
    int64_t dilation_len_,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    const int32_t* output_dtype,
    double accum_scale,
    int64_t accum_zero_point,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

#endif // AT_MKLDNN_ENABLED()
#ifdef __cplusplus
} // extern "C"
#endif

#endif // USE_XPU
#endif // AOTI_TORCH_SHIM_XPU

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `XPUGuardOpaque`, `XPUStreamGuardOpaque`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_torch/c`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `torch/csrc/inductor/aoti_torch/c/shim.h`


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
- [`shim_deprecated.h_docs.md`](./shim_deprecated.h_docs.md)
- [`shim_mps.h_docs.md`](./shim_mps.h_docs.md)


## Cross-References

- **File Documentation**: `shim_xpu.h_docs.md`
- **Keyword Index**: `shim_xpu.h_kw.md`
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
- [`macros.h_kw.md_docs.md`](./macros.h_kw.md_docs.md)
- [`shim.h_kw.md_docs.md`](./shim.h_kw.md_docs.md)
- [`shim_deprecated.h_docs.md_docs.md`](./shim_deprecated.h_docs.md_docs.md)
- [`shim_mps.h_kw.md_docs.md`](./shim_mps.h_kw.md_docs.md)
- [`macros.h_docs.md_docs.md`](./macros.h_docs.md_docs.md)
- [`shim_mps.h_docs.md_docs.md`](./shim_mps.h_docs.md_docs.md)
- [`shim_cpu.h_kw.md_docs.md`](./shim_cpu.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `shim_xpu.h_docs.md_docs.md`
- **Keyword Index**: `shim_xpu.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

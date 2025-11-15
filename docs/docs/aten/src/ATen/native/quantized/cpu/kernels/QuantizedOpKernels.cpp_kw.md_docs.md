# Documentation: `docs/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp_kw.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp_kw.md`
- **Size**: 10,959 bytes (10.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp`

## File Information

- **Original File**: [aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp](../../../../../../../../aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp)
- **Documentation**: [`QuantizedOpKernels.cpp_docs.md`](./QuantizedOpKernels.cpp_docs.md)
- **Folder**: `aten/src/ATen/native/quantized/cpu/kernels`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`_fake_quant_per_channel_cachemask_cpu_helper`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`_fake_quantize_tensor_helper`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`_qadaptive_avg_pool_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`_qadd_tensor_cpu_impl`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`_qavg_pool_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`_qmaxpool_2d_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`_qmul_tensor_cpu_impl`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`check_tensor_memory_format`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`constexpr`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`cpu_masked_fill_kernel_quantized_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`dequantize_per_channel_affine_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`dequantize_tensor_arm`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`dequantize_tensor_per_channel_affine_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`dequantize_tensor_per_channel_float_qparams_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`dequantize_tensor_per_tensor_affine_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`dequantize_tensor_per_tensor_affine_sub_byte_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`do_avg_pool_nhwc_on_AVX_n`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`do_avg_pool_on_AVX_n`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`do_bn_compute`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`do_quantized_bilinear_on_AVX_n`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`fake_quant_per_channel_cachemask_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`fake_quantize_learnable_channel_grad_kernel_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`fake_quantize_learnable_tensor_grad_kernel_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`fake_quantize_tensor_cachemask_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`fake_quantize_tensor_cachemask_tensor_qparams_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`for`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`hsum`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`hsum_sq`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`if`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`index_put_kernel_quantized_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`leaky_qrelu_out_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`masked_fill_kernel_quantized_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`q_batch_norm_cpu_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`q_batch_norm_cpu_kernel_impl`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`q_batch_norm_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qadaptive_avg_pool2d_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qadaptive_avg_pool3d_ndhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qadd_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qadd_scalar_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qadd_tensor_cpu_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qavg_pool2d_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qavg_pool3d_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qcat_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qclamp_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qclamp_max_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qclamp_min_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qelu_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qgelu_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qhardsigmoid_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qhardswish_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qmaxpool_2d_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qmaxpool_3d_nthwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qmean_inner_dim_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qmul_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qmul_tensor_cpu_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qprelu_out_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qrelu_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qsigmoid_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qstd_inner_dim_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qtanh_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qthreshold_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qtopk_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_arm`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_arm_q8`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_per_channel_affine_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_per_channel_float_qparams_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_per_channel_impl`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_per_tensor_affine_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_per_tensor_affine_sub_byte_cpu`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`qupsample_bilinear2d_nhwc_kernel`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)

### Includes

- **`ATen/Dispatch.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/Functions.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/Parallel.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/core/List.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/core/Tensor.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/Activation.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/TensorIterator.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/TopKImpl.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/UpSample.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/cpu/IndexKernelUtils.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/cpu/Loops.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/cpu/utils.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/quantized/AffineQuantizer.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/quantized/FakeQuantAffine.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/quantized/IndexKernel.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/native/quantized/cpu/QuantizedOps.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/ops/_empty_affine_quantized.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/ops/empty.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`ATen/quantized/Quantizer.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`arm_neon.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`c10/util/Unroll.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`c10/util/irange.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`cmath`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`fbgemm/QuantUtils.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`omp.h`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)

### Namespaces

- **`at`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)
- **`quantize_tensor_arm_intrinsics`**: [QuantizedOpKernels.cpp_docs.md](./QuantizedOpKernels.cpp_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/kernels`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/kernels`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/kernels`):

- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`QuantizedOpKernels.cpp_docs.md_docs.md`](./QuantizedOpKernels.cpp_docs.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `QuantizedOpKernels.cpp_kw.md_docs.md`
- **Keyword Index**: `QuantizedOpKernels.cpp_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
